import copy
import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from llm_reasoner import DecisionResult, DecisionType, LLMReasoner

import numpy as np

import models
from ta import TrustedAuthority
from user import User
from utils import LHH, QUANTIZE_SCALE, SIG
from rule_engine import AgentSuggestion, RoundContext, RuleEngine, SuggestionType, VerificationLevel
from orchestrate_agent import (
    OrchestrationRuleEngine, OrchestrationDecision,
    build_orchestration_state, call_orchestrate_agent,
)

logger = logging.getLogger(__name__)


def build_window_token(edge_id: str, window_seq: int, base_version: int) -> int:
    material = f"{edge_id}:{window_seq}:{base_version}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass
class ClientUpload:
    user_id: str
    edge_id: str
    window_token: int
    base_version: int
    masked_model: np.ndarray
    update_norm: float
    score: float
    sample_count: int
    proof: Dict[str, float]


@dataclass
class PreparedBatch:
    edge_id: str
    window_token: int
    base_version: int
    user_ids: List[str]
    batch_size: int
    masked_sum: np.ndarray
    plaintext_sum: np.ndarray
    avg_model: np.ndarray
    mask_hash: object
    masked_hash: object
    signature: bytes
    risk_score: float
    staleness: int
    merge_weight: float


@dataclass
class BatchCommit:
    edge_id: str
    window_token: int
    base_version: int
    applied_version: int
    batch_size: int
    risk_score: float
    staleness: int
    merge_weight: float
    verification_passed: bool


@dataclass
class WindowState:
    token: int
    base_version: int
    created_tick: int
    last_flush_tick: int
    inflight: int = 0
    uploads: List[ClientUpload] = field(default_factory=list)


class AgentAuditor:
    def __init__(self):
        self.upload_events: List[dict] = []
        self.batch_events: List[dict] = []

    def record_upload(self, upload: ClientUpload, tick: int):
        self.upload_events.append({
            "tick": tick,
            "user_id": upload.user_id,
            "edge_id": upload.edge_id,
            "window_token": upload.window_token,
            "base_version": upload.base_version,
            "score": upload.score,
            "update_norm": upload.update_norm,
        })

    def record_commit(self, commit: BatchCommit, tick: int):
        self.batch_events.append({
            "tick": tick,
            "edge_id": commit.edge_id,
            "window_token": commit.window_token,
            "base_version": commit.base_version,
            "applied_version": commit.applied_version,
            "batch_size": commit.batch_size,
            "risk_score": commit.risk_score,
            "staleness": commit.staleness,
            "merge_weight": commit.merge_weight,
            "verification_passed": commit.verification_passed,
        })

    def summary(self) -> dict:
        accepted = sum(1 for item in self.batch_events if item["verification_passed"])
        return {
            "uploads": len(self.upload_events),
            "batches": len(self.batch_events),
            "accepted_batches": accepted,
        }


class AgentCoordinator:
    def __init__(self,
                 ta: TrustedAuthority,
                 initial_model_vector: np.ndarray,
                 async_weight: float = 0.6,
                 llm_reasoner: Optional[LLMReasoner] = None,
                 decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None,
                 quorum_q_es: int = 2,
                 orchestrator: Optional['OrchestrationRuleEngine'] = None):
        self.ta = ta
        self.global_model_vector = initial_model_vector.astype(np.float64).copy()
        self.global_version = 0
        self.async_weight = async_weight
        self.commit_count = 0
        self.llm_reasoner = llm_reasoner
        self.decision_hook = decision_hook
        self.quorum_q_es = quorum_q_es           # Cloud 窗口需要凑够几个 ES
        self.orchestrator = orchestrator          # 编排 RuleEngine
        self.quorum_timeout = 10                 # Cloud 窗口最多等多少 tick
        self._round_commits: List[str] = []      # 本轮已提交的 edge_id 列表
        self._pending_batches: List[PreparedBatch] = []  # 等待凑够 quorum 的批次
        self._quorum_tick_start: int = 0         # 当前 Cloud 窗口的开始 tick
        self._pending_changed: bool = False      # 缓冲是否有新批次入队
        self._last_agent_decision = None         # 缓存上次 Agent 建议
        self._last_agent_tick: int = -5          # 上次调 Agent 的 tick (初始-5确保首次触发)

    def get_global_snapshot(self) -> Tuple[np.ndarray, int]:
        return self.global_model_vector.copy(), self.global_version

    def orchestrate_round(self, edges: List['AgentEdge'], tick: int) -> OrchestrationDecision:
        """编排Agent: 收集ES状态 → Agent建议 → RuleEngine → 决策"""
        if self.orchestrator is None or self.llm_reasoner is None:
            committed = len(set(self._round_commits))
            close = committed >= self.quorum_q_es
            return OrchestrationDecision(
                close_round=close,
                commit_edges=list(set(self._round_commits)),
                agent_reason="No orchestrator → fallback to rule",
                agent_approved=True,
            )
        state = build_orchestration_state(
            edges=edges, coordinator=self, current_round=tick,
            quorum_q_es=self.quorum_q_es,
        )
        edgemap = {edge.id: edge for edge in edges}
        decision = call_orchestrate_agent(
            reasoner=self.llm_reasoner, state=state,
            edgemap=edgemap, rule_engine=self.orchestrator,
        )
        if self.decision_hook:
            logger.info(f"[orchestrate] close={decision.close_round} "
                        f"commit={decision.commit_edges} approved={decision.agent_approved}")
        return decision

    def prepare_batch(self,
                      edge_id: str,
                      window_token: int,
                      base_version: int,
                      user_ids: List[str],
                      masked_sum: np.ndarray,
                      masked_hash: object,
                      signature: bytes,
                      risk_score: float) -> PreparedBatch:
        batch_size = len(user_ids)
        if batch_size == 0:
            raise ValueError("Cannot prepare an empty batch.")

        global_mask = self.ta.compute_global_mask(user_ids, window_token)
        plaintext_sum = masked_sum - global_mask
        avg_model = plaintext_sum.astype(np.float64) / QUANTIZE_SCALE / batch_size

        staleness = max(0, self.global_version - base_version)
        merge_weight = self.async_weight / (1.0 + staleness)
        merge_weight = float(np.clip(merge_weight, 0.05, 1.0))

        return PreparedBatch(
            edge_id=edge_id,
            window_token=window_token,
            base_version=base_version,
            user_ids=list(user_ids),
            batch_size=batch_size,
            masked_sum=masked_sum,
            plaintext_sum=plaintext_sum,
            avg_model=avg_model,
            mask_hash=LHH.Hash(self.ta.get_lhh_pp(), global_mask),
            masked_hash=masked_hash,
            signature=signature,
            risk_score=risk_score,
            staleness=staleness,
            merge_weight=merge_weight,
        )

    def _decide_reject_llm(self, batch: PreparedBatch) -> Optional[bool]:
        """用 LLM 决定是否接受批次。返回 None 表示 LLM 不可用。"""
        if self.llm_reasoner is None:
            return None

        info = {
            "edge_id": batch.edge_id,
            "batch_size": batch.batch_size,
            "risk_score": batch.risk_score,
            "staleness": batch.staleness,
            "merge_weight": batch.merge_weight,
            "global_version": self.global_version,
        }
        result = self.llm_reasoner.decide(
            decision_type=DecisionType.COORDINATOR_REJECT_BATCH,
            agent_id="coordinator",
            round_info=info,
        )
        if self.decision_hook:
            self.decision_hook(DecisionType.COORDINATOR_REJECT_BATCH, result)
        if result.decision == "reject":
            return False
        elif result.decision == "accept":
            return True
        return None

    def enqueue_batch(self, batch: PreparedBatch, verification_passed: bool) -> BatchCommit:
        """存入缓冲队列。不再立即合并。等 quorum 达标后统一处理。"""
        self._pending_batches.append(batch)
        self._pending_changed = True  # 标记缓冲有变化 → 下次 process_quorum 触发 Agent
        if not verification_passed:
            return BatchCommit(
                edge_id=batch.edge_id, window_token=batch.window_token,
                base_version=batch.base_version, applied_version=self.global_version,
                batch_size=batch.batch_size, risk_score=batch.risk_score,
                staleness=batch.staleness, merge_weight=batch.merge_weight,
                verification_passed=False,
            )
        # quorum 还没到 → 暂不合并，等 process_quorum()
        return None  # 不产生 commit

    def process_quorum(self, tick: int, edges: List['AgentEdge'] = None) -> List[BatchCommit]:
        """
        有 Agent → 调编排 Agent 决定是否关轮、补谁
        无 Agent → 硬编码: 够 Q_ES 个 edge 或超时 → 批量合并
        """
        if not self._pending_batches:
            return []

        unique_edges = set(b.edge_id for b in self._pending_batches)
        timed_out = (tick - self._quorum_tick_start) >= self.quorum_timeout

        # ── 有 Agent: 只在缓冲变化或超时逼近时触发 ──
        time_left = self.quorum_timeout - (tick - self._quorum_tick_start)
        near_timeout = time_left <= 2  # 还剩 ≤2 tick 就超时
        need_agent = (
            self._pending_changed                     # 新批次入队了
            or near_timeout                           # 快超时了
            or (tick - self._last_agent_tick) >= 5    # 距上次 ≥5 tick
        )

        if need_agent and edges and self.llm_reasoner:
            self._last_agent_decision = self.orchestrate_round(edges, tick)
            self._last_agent_tick = tick
            self._pending_changed = False

        decision = self._last_agent_decision
        if decision is not None and decision.close_round:
            # Agent 说关轮 → 只提 commit_edges
            commit_set = set(decision.commit_edges)
            comp = decision.compensation_weights
            logger.info(
                f"[Orchestrate] Agent says close. commit={decision.commit_edges} "
                f"approved={decision.agent_approved}"
            )
            if not decision.agent_approved:
                logger.info(f"[Orchestrate] Override: {decision.override_reason}")
        elif decision is not None:
            # Agent 说再等等
            if not timed_out:
                logger.info(f"[Orchestrate] Agent says wait → skip")
                return []
            logger.info("[Orchestrate] Agent says wait but timeout → forced close")
            commit_set = unique_edges
            comp = {}
        else:
            # 无 Agent 建议 → 硬编码
            if len(unique_edges) < self.quorum_q_es and not timed_out:
                return []
            commit_set = unique_edges
            comp = {}

        # 批量合并
        commits = []
        for batch in self._pending_batches:
            if batch.edge_id not in commit_set:
                continue  # Agent 没选这个 ES → 跳过，但保留缓冲中的其他批次

            staleness = max(0, self.global_version - batch.base_version)
            merge_weight = self.async_weight / (1.0 + staleness)
            merge_weight = float(np.clip(merge_weight, 0.05, 1.0))

            # 补偿权重
            boost = comp.get(batch.edge_id, 1.0)
            merge_weight = min(1.0, merge_weight * boost)

            self.global_model_vector = (
                (1.0 - merge_weight) * self.global_model_vector
                + merge_weight * batch.avg_model
            )
            self._round_commits.append(batch.edge_id)
            self.global_version += 1
            self.commit_count += 1

            commits.append(BatchCommit(
                edge_id=batch.edge_id, window_token=batch.window_token,
                base_version=batch.base_version, applied_version=self.global_version,
                batch_size=batch.batch_size, risk_score=batch.risk_score,
                staleness=staleness, merge_weight=merge_weight,
                verification_passed=True,
            ))

        # 清空缓冲，重置计时和缓存
        self._pending_batches = []
        self._pending_changed = False
        self._last_agent_decision = None
        self._quorum_tick_start = tick
        return commits

    def commit_prepared_batch(self, batch: PreparedBatch, verification_passed: bool) -> BatchCommit:
        """保留兼容性: 委托给 enqueue_batch"""
        return self.enqueue_batch(batch, verification_passed) or BatchCommit(
            edge_id=batch.edge_id, window_token=batch.window_token,
            base_version=batch.base_version, applied_version=self.global_version,
            batch_size=batch.batch_size, risk_score=batch.risk_score,
            staleness=batch.staleness, merge_weight=batch.merge_weight,
            verification_passed=verification_passed,
        )


class AgentClient:
    def __init__(self,
                 user: User,
                 train_loader,
                 rng: np.random.Generator,
                 latency_range: Tuple[int, int] = (1, 3),
                 dropout_rate: float = 0.0,
                 norm_bound: float = 500.0,
                 llm_reasoner: Optional[LLMReasoner] = None,
                 decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None):
        """
        Args:
            llm_reasoner: 可选的 LLM 决策引擎。不传则使用随机逻辑。
            decision_hook: 每次 LLM 决策后的回调（用于日志/审计）。
        """
        self.user = user
        self.train_loader = train_loader
        self.sample_count = len(train_loader.dataset)
        self.rng = rng
        self.latency_range = latency_range
        self.dropout_rate = dropout_rate
        self.norm_bound = norm_bound
        self.llm_reasoner = llm_reasoner
        self.decision_hook = decision_hook
        self._pending_upload: Optional[ClientUpload] = None
        self._pending_window: Optional[Tuple[str, int]] = None
        self._ticks_left = 0
        self._round_accuracies: List[float] = []  # 历史准确率，供 LLM 参考

    def is_idle(self) -> bool:
        return self._pending_upload is None

    def start_task(self,
                   global_model_vector: np.ndarray,
                   model_meta: list,
                   edge_id: str,
                   window_token: int,
                   base_version: int,
                   local_epochs: int,
                   lr: float):
        if not self.is_idle():
            return

        self.user.set_global_model(global_model_vector, model_meta)
        self.user.local_train(self.train_loader, local_epochs, lr)

        local_vector, _ = models.flatten_parameters(self.user.model)
        model_int = (local_vector * QUANTIZE_SCALE).astype(np.int64)
        mask_int = self.user._generate_ta_mask_int(window_token, model_int.shape[0])
        masked_model = model_int + mask_int

        delta = local_vector - global_model_vector
        update_norm = float(np.linalg.norm(delta))
        score = 1.0 / (1.0 + update_norm)
        proof = {
            "norm_ok": float(update_norm <= self.norm_bound),
            "sample_count": float(self.sample_count),
            "update_norm": update_norm,
        }

        self._pending_upload = ClientUpload(
            user_id=self.user.id,
            edge_id=edge_id,
            window_token=window_token,
            base_version=base_version,
            masked_model=masked_model,
            update_norm=update_norm,
            score=score,
            sample_count=self.sample_count,
            proof=proof,
        )
        self._pending_window = (edge_id, window_token)
        self._ticks_left = int(self.rng.integers(self.latency_range[0], self.latency_range[1] + 1))

    def record_accuracy(self, accuracy: float):
        """供外部记录准确率历史（LLM 决策上下文用）"""
        self._round_accuracies.append(accuracy)

    def _decide_dropout_llm(self) -> Optional[str]:
        """用 LLM 决定是否掉线。返回 None 表示 LLM 不可用，让调用方走随机逻辑。"""
        if self.llm_reasoner is None:
            return None

        info = {
            "round": self._pending_upload.window_token if self._pending_upload else -1,
            "update_norm": self._pending_upload.update_norm if self._pending_upload else 0,
            "staleness": self._ticks_left,
            "sample_count": self.sample_count,
            "recent_accuracy": self._round_accuracies[-1] if self._round_accuracies else None,
        }
        result = self.llm_reasoner.decide(
            decision_type=DecisionType.CLIENT_DROPOUT,
            agent_id=self.user.id,
            round_info=info,
        )
        if self.decision_hook:
            self.decision_hook(DecisionType.CLIENT_DROPOUT, result)
        return result.decision

    def tick(self) -> Tuple[str, Optional[ClientUpload], Optional[Tuple[str, int]]]:
        """客户端 tick — 训完就传，或随机掉线。不调 Agent。"""
        if self._pending_upload is None:
            return "idle", None, None

        self._ticks_left -= 1
        if self._ticks_left > 0:
            return "waiting", None, None

        # 随机掉线
        if float(self.rng.random()) < self.dropout_rate:
            window_info = self._pending_window
            self._pending_upload = None
            self._pending_window = None
            return "drop", None, window_info

        upload = self._pending_upload
        self._pending_upload = None
        self._pending_window = None
        return "upload", upload, None


class AgentEdge:
    def __init__(self,
                 edge_id: str,
                 ta: TrustedAuthority,
                 coordinator: AgentCoordinator,
                 auditor: AgentAuditor,
                 min_batch_size: int = 2,
                 flush_timeout: int = 2,
                 enable_verification: bool = True,
                 llm_reasoner: Optional[LLMReasoner] = None,
                 decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None,
                 rule_engine=None):
        self.id = edge_id
        self.ta = ta
        self.coordinator = coordinator
        self.auditor = auditor
        self.min_batch_size = min_batch_size
        self.flush_timeout = flush_timeout
        self.enable_verification = enable_verification
        self.llm_reasoner = llm_reasoner
        self.decision_hook = decision_hook
        self.rule_engine = rule_engine

        self.lhh_pp = ta.get_lhh_pp()
        self.sign_priv_key = ta.register_edge_server(self.id, require_verification=self.enable_verification)

        self._window_seq = 0
        self._active_token: Optional[int] = None
        self._windows: Dict[int, WindowState] = {}

    def _open_new_window(self, base_version: int, tick: int) -> WindowState:
        token = build_window_token(self.id, self._window_seq, base_version)
        self._window_seq += 1
        state = WindowState(
            token=token,
            base_version=base_version,
            created_tick=tick,
            last_flush_tick=tick,
        )
        self._windows[token] = state
        self._active_token = token
        return state

    def assign_window(self, current_version: int, tick: int) -> Tuple[int, int]:
        state = None
        if self._active_token is not None:
            state = self._windows.get(self._active_token)
            if state is not None and state.base_version != current_version:
                state = None

        if state is None:
            state = self._open_new_window(current_version, tick)

        state.inflight += 1
        return state.token, state.base_version

    def note_dropout(self, window_token: int):
        state = self._windows.get(window_token)
        if state is not None and state.inflight > 0:
            state.inflight -= 1

    def receive_upload(self, upload: ClientUpload, tick: int):
        state = self._windows.get(upload.window_token)
        if state is None:
            state = WindowState(
                token=upload.window_token,
                base_version=upload.base_version,
                created_tick=tick,
                last_flush_tick=tick,
            )
            self._windows[upload.window_token] = state

        state.uploads.append(upload)
        if state.inflight > 0:
            state.inflight -= 1
        self.auditor.record_upload(upload, tick)

    def _compute_risk_score(self, uploads: List[ClientUpload]) -> float:
        norms = np.array([item.update_norm for item in uploads], dtype=np.float64)
        if norms.size == 0:
            return 0.0
        dispersion = norms.std() / (norms.mean() + 1e-8) if norms.size > 1 else 0.0
        proof_fail = 1.0 - np.mean([item.proof.get("norm_ok", 0.0) for item in uploads])
        return float(dispersion + proof_fail)

    def _build_masked_hash(self, masked_sum: np.ndarray) -> Tuple[object, bytes]:
        if self.enable_verification:
            masked_hash = LHH.Hash(self.lhh_pp, masked_sum)
            signature = SIG.sign(pickle.dumps(masked_hash), self.sign_priv_key)
        else:
            masked_hash = None
            signature = b"verification_disabled"
        return masked_hash, signature

    def _verify_prepared_batch(self, batch: PreparedBatch) -> bool:
        if not self.enable_verification:
            return True

        pub_key = self.ta1.get_all_es_public_keys()[self.id]
        if not SIG.verify(pickle.dumps(batch.masked_hash), batch.signature, pub_key):
            return False

        plain_hash = LHH.Hash(self.lhh_pp, batch.plaintext_sum)
        return LHH.Mul(plain_hash, batch.mask_hash) == batch.masked_hash

    def _flush_window(self, token: int, tick: int) -> Optional[BatchCommit]:
        state = self._windows.get(token)
        if state is None or not state.uploads:
            return None

        uploads = list(state.uploads)
        state.uploads = []
        state.last_flush_tick = tick

        masked_sum = np.sum(np.stack([item.masked_model for item in uploads], axis=0), axis=0)
        masked_hash, signature = self._build_masked_hash(masked_sum)
        risk_score = self._compute_risk_score(uploads)
        user_ids = [item.user_id for item in uploads]

        prepared = self.coordinator.prepare_batch(
            edge_id=self.id,
            window_token=state.token,
            base_version=state.base_version,
            user_ids=user_ids,
            masked_sum=masked_sum,
            masked_hash=masked_hash,
            signature=signature,
            risk_score=risk_score,
        )
        verification_passed = self._verify_prepared_batch(prepared)
        # 存入 Cloud 缓冲队列 — 等 quorum 达标后统一处理
        commit = self.coordinator.enqueue_batch(prepared, verification_passed)
        if commit is not None:
            self.auditor.record_commit(commit, tick)

        if state.inflight == 0 and not state.uploads and token != self._active_token:
            self._windows.pop(token, None)
        return commit

    def _decide_flush_llm(self, state: WindowState, tick: int) -> Optional[bool]:
        """LLM 建议 + RuleEngine 兜底 → 最终决策"""
        norms = [u.update_norm for u in state.uploads] if state.uploads else [0.0]
        info = {
            "pending_clients": len(state.uploads),
            "min_batch_size": self.min_batch_size,
            "ticks_since_flush": tick - state.last_flush_tick,
            "flush_timeout": self.flush_timeout,
            "avg_risk_score": float(np.mean(norms)) if norms else 0.0,
        }

        agent_says = None
        if self.llm_reasoner:
            try:
                result = self.llm_reasoner.decide(
                    decision_type=DecisionType.EDGE_FLUSH,
                    agent_id=self.id,
                    round_info=info,
                )
                if self.decision_hook:
                    self.decision_hook(DecisionType.EDGE_FLUSH, result)
                agent_says = result.decision
            except Exception:
                pass

        # RuleEngine 校验
        if self.rule_engine is not None and agent_says is not None:
            from rule_engine import AgentSuggestion, RoundContext, SuggestionType
            ctx = RoundContext(
                pending_clients=len(state.uploads),
                min_batch_size=self.min_batch_size,
                ticks_since_flush=tick - state.last_flush_tick,
                flush_timeout=self.flush_timeout,
                risk_score=float(np.mean(norms)) if norms else 0.0,
            )
            suggestion = AgentSuggestion(
                type=SuggestionType.FLUSH_TIMING,
                decision=agent_says,
            )
            validated = self.rule_engine.validate(suggestion, ctx)
            if validated.final_decision == "flush":
                return True
            elif validated.final_decision == "wait":
                return False

        if agent_says == "flush":
            return True
        elif agent_says == "wait":
            return False
        return None

    def tick(self, tick: int) -> List[BatchCommit]:
        """Edge tick — 纯硬编码: 够了就发车，超时就发车"""
        commits: List[BatchCommit] = []
        for token in list(self._windows.keys()):
            state = self._windows[token]
            should_flush = False

            if state.uploads and len(state.uploads) >= self.min_batch_size:
                should_flush = True
            elif state.uploads and (tick - state.last_flush_tick) >= self.flush_timeout:
                should_flush = True

            if should_flush:
                commit = self._flush_window(token, tick)
                if commit is not None:
                    commits.append(commit)

            if state.inflight == 0 and not state.uploads and token != self._active_token:
                self._windows.pop(token, None)

        active_state = self._windows.get(self._active_token) if self._active_token is not None else None
        if active_state is None or active_state.base_version != self.coordinator.global_version:
            self._active_token = None

        return commits
