import copy
import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import models
from llm_reasoner import DecisionResult, DecisionType, LLMReasoner
from rule_engine import AgentSuggestion, RoundContext, RuleEngine, SuggestionType
from ta import TrustedAuthority
from user import User
from utils import LHH, QUANTIZE_SCALE, SIG

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


class AsyncAuditor:
    def __init__(self):
        self.upload_events: List[dict] = []
        self.batch_events: List[dict] = []

    def record_upload(self, upload: ClientUpload, tick: int):
        self.upload_events.append(
            {
                "tick": tick,
                "user_id": upload.user_id,
                "edge_id": upload.edge_id,
                "window_token": upload.window_token,
                "base_version": upload.base_version,
                "score": upload.score,
                "update_norm": upload.update_norm,
            }
        )

    def record_commit(self, commit: BatchCommit, tick: int):
        self.batch_events.append(
            {
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
            }
        )

    def summary(self) -> dict:
        accepted = sum(1 for item in self.batch_events if item["verification_passed"])
        rejected = len(self.batch_events) - accepted
        return {
            "uploads": len(self.upload_events),
            "batches": len(self.batch_events),
            "accepted_batches": accepted,
            "rejected_batches": rejected,
        }


class AsyncCoordinator:
    def __init__(
        self,
        ta: TrustedAuthority,
        initial_model_vector: np.ndarray,
        async_weight: float = 0.6,
        llm_reasoner: Optional[LLMReasoner] = None,
        decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None,
        rule_engine: Optional[RuleEngine] = None,
        max_staleness: int = 3,
        reject_risk_threshold: float = 0.8,
    ):
        self.ta = ta
        self.global_model_vector = initial_model_vector.astype(np.float64).copy()
        self.global_version = 0
        self.async_weight = async_weight
        self.commit_count = 0
        self.llm_reasoner = llm_reasoner
        self.decision_hook = decision_hook
        self.rule_engine = rule_engine
        self.max_staleness = max_staleness
        self.reject_risk_threshold = reject_risk_threshold

    def get_global_snapshot(self) -> Tuple[np.ndarray, int]:
        return self.global_model_vector.copy(), self.global_version

    def prepare_batch(
        self,
        edge_id: str,
        window_token: int,
        base_version: int,
        user_ids: List[str],
        masked_sum: np.ndarray,
        masked_hash: object,
        signature: bytes,
        risk_score: float,
    ) -> PreparedBatch:
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
            mask_hash=LHH.Hash(self.ta_primary.get_lhh_pp(), global_mask),
            masked_hash=masked_hash,
            signature=signature,
            risk_score=risk_score,
            staleness=staleness,
            merge_weight=merge_weight,
        )

    def _decide_batch_llm(self, batch: PreparedBatch) -> Optional[str]:
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
        decision = str(result.decision).lower()
        if decision in {"accept", "reject"}:
            return decision
        return None

    def _hardcoded_batch_fallback(self, batch: PreparedBatch) -> str:
        if batch.batch_size <= 0:
            return "reject"
        if batch.staleness > self.max_staleness:
            return "reject"
        if batch.risk_score > self.reject_risk_threshold:
            return "reject"
        return "accept"

    def commit_prepared_batch(
        self, batch: PreparedBatch, verification_passed: bool
    ) -> BatchCommit:
        if not verification_passed:
            return BatchCommit(
                edge_id=batch.edge_id,
                window_token=batch.window_token,
                base_version=batch.base_version,
                applied_version=self.global_version,
                batch_size=batch.batch_size,
                risk_score=batch.risk_score,
                staleness=batch.staleness,
                merge_weight=batch.merge_weight,
                verification_passed=False,
            )

        agent_decision = self._decide_batch_llm(batch)
        if agent_decision is None:
            agent_decision = self._hardcoded_batch_fallback(batch)

        final_decision = agent_decision
        if self.rule_engine is not None:
            suggestion = AgentSuggestion(
                type=SuggestionType.BATCH_ACCEPTANCE,
                decision=agent_decision,
                reasoning="LLM-or-fallback decision",
                confidence=1.0,
            )
            ctx = RoundContext(
                batch_size=batch.batch_size,
                staleness=batch.staleness,
                risk_score=batch.risk_score,
            )
            validated = self.rule_engine.validate(suggestion, ctx)
            final_decision = str(validated.final_decision).lower()

        if final_decision == "reject":
            return BatchCommit(
                edge_id=batch.edge_id,
                window_token=batch.window_token,
                base_version=batch.base_version,
                applied_version=self.global_version,
                batch_size=batch.batch_size,
                risk_score=batch.risk_score,
                staleness=batch.staleness,
                merge_weight=batch.merge_weight,
                verification_passed=False,
            )

        self.global_model_vector = (
            (1.0 - batch.merge_weight) * self.global_model_vector
            + batch.merge_weight * batch.avg_model
        )
        self.global_version += 1
        self.commit_count += 1

        return BatchCommit(
            edge_id=batch.edge_id,
            window_token=batch.window_token,
            base_version=batch.base_version,
            applied_version=self.global_version,
            batch_size=batch.batch_size,
            risk_score=batch.risk_score,
            staleness=batch.staleness,
            merge_weight=batch.merge_weight,
            verification_passed=True,
        )


class AsyncClient:
    def __init__(
        self,
        user: User,
        train_loader,
        rng: np.random.Generator,
        latency_range: Tuple[int, int] = (1, 3),
        dropout_rate: float = 0.0,
        norm_bound: float = 500.0,
        llm_reasoner: Optional[LLMReasoner] = None,
        decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None,
    ):
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
        self._round_accuracies: List[float] = []

    def is_idle(self) -> bool:
        return self._pending_upload is None

    def start_task(
        self,
        global_model_vector: np.ndarray,
        model_meta: list,
        edge_id: str,
        window_token: int,
        base_version: int,
        local_epochs: int,
        lr: float,
    ):
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
        self._round_accuracies.append(accuracy)

    def _decide_dropout_llm(self) -> Optional[str]:
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
        if self._pending_upload is None:
            return "idle", None, None

        self._ticks_left -= 1
        if self._ticks_left > 0:
            return "waiting", None, None

        llm_decision = self._decide_dropout_llm()
        if llm_decision == "drop":
            window_info = self._pending_window
            self._pending_upload = None
            self._pending_window = None
            return "drop", None, window_info
        elif llm_decision == "participate":
            upload = self._pending_upload
            self._pending_upload = None
            self._pending_window = None
            return "upload", upload, None

        if float(self.rng.random()) < self.dropout_rate:
            window_info = self._pending_window
            self._pending_upload = None
            self._pending_window = None
            return "drop", None, window_info

        upload = self._pending_upload
        self._pending_upload = None
        self._pending_window = None
        return "upload", upload, None


class AsyncEdge:
    def __init__(
        self,
        edge_id: str,
        ta: TrustedAuthority,
        coordinator: AsyncCoordinator,
        auditor: AsyncAuditor,
        min_batch_size: int = 2,
        flush_timeout: int = 2,
        enable_verification: bool = True,
        llm_reasoner: Optional[LLMReasoner] = None,
        decision_hook: Optional[Callable[[DecisionType, DecisionResult], None]] = None,
        rule_engine: Optional[RuleEngine] = None,
    ):
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

        self.lhh_pp = ta_primary.get_lhh_pp()
        self.sign_priv_key = ta_primary.register_edge_server(self.id, require_verification=self.enable_verification)

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

        pub_key = self.ta_primary.get_all_es_public_keys()[self.id]
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
        commit = self.coordinator.commit_prepared_batch(prepared, verification_passed)
        self.auditor.record_commit(commit, tick)

        if state.inflight == 0 and not state.uploads and token != self._active_token:
            self._windows.pop(token, None)
        return commit

    def _build_flush_context(self, state: WindowState, tick: int) -> tuple[dict, RoundContext]:
        risk_score = self._compute_risk_score(state.uploads)
        ticks_since_flush = tick - state.last_flush_tick
        info = {
            "pending_clients": len(state.uploads),
            "min_batch_size": self.min_batch_size,
            "ticks_since_flush": ticks_since_flush,
            "flush_timeout": self.flush_timeout,
            "risk_score": risk_score,
        }
        ctx = RoundContext(
            pending_clients=info["pending_clients"],
            min_batch_size=self.min_batch_size,
            ticks_since_flush=ticks_since_flush,
            flush_timeout=self.flush_timeout,
            risk_score=risk_score,
        )
        return info, ctx

    def _fallback_flush_decision(self, state: WindowState, tick: int) -> str:
        if not state.uploads:
            return "wait"
        if len(state.uploads) >= self.min_batch_size:
            return "flush"
        if (tick - state.last_flush_tick) >= self.flush_timeout:
            return "flush"
        return "wait"

    def _decide_flush_action(self, state: WindowState, tick: int) -> bool:
        info, ctx = self._build_flush_context(state, tick)
        decision = None
        reasoning = "rule-based fallback"

        if self.llm_reasoner:
            try:
                result = self.llm_reasoner.decide(
                    decision_type=DecisionType.EDGE_FLUSH,
                    agent_id=self.id,
                    round_info=info,
                )
                if self.decision_hook:
                    self.decision_hook(DecisionType.EDGE_FLUSH, result)
                llm_decision = str(result.decision).lower()
                if llm_decision in {"flush", "wait"}:
                    decision = llm_decision
                    reasoning = result.reasoning or "llm suggestion"
            except Exception:
                pass

        if decision is None:
            decision = self._fallback_flush_decision(state, tick)

        if self.rule_engine is not None:
            suggestion = AgentSuggestion(
                type=SuggestionType.FLUSH_TIMING,
                decision=decision,
                reasoning=reasoning,
                confidence=1.0,
            )
            validated = self.rule_engine.validate(suggestion, ctx)
            decision = str(validated.final_decision).lower()

        return decision == "flush"

    def tick(self, tick: int) -> List[BatchCommit]:
        commits: List[BatchCommit] = []
        for token in list(self._windows.keys()):
            state = self._windows[token]
            should_flush = self._decide_flush_action(state, tick)

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
