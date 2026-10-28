"""
编排Agent: Cloud层调度决策

职责:
  1. 打包所有ES的当前状态
  2. 发送给 LLM Agent → 获取结构化JSON建议
  3. RuleEngine 校验建议合法性
  4. 返回最终调度决策

部署位置: Cloud (AgentCoordinator 内部)

Agent 不看梯度、不看掩码、不看种子。只看 ES 架构元数据。
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── 状态数据结构 ─────────────────────────────────

@dataclass
class EdgeStatus:
    """单个 Edge 的当前状态（发给 Agent 的最小信息集）"""
    edge_id: str
    clients_ready: int = 0        # 窗口里已经上传完成的客户端数
    clients_inflight: int = 0     # 领了任务还在训的客户端数
    need_for_quorum: int = 0      # 还差几个到 K_user
    estimated_ticks: int = 0      # 预计最近一个客户端到达的剩余 tick 数
    history_rate: float = 0.0     # 最近 H 窗口内被纳入全局更新的比例
    target_rate: float = 0.0      # Q_ES / N_ES 目标参与率
    compensation_weight: float = 1.0  # 当前补偿权重
    lagging: bool = False         # 是否明显掉队


@dataclass
class OrchestrationState:
    """本轮全局编排状态"""
    current_round: int = 0
    total_edges: int = 0
    quorum_q_es: int = 0          # Cloud 需要凑够几个 ES
    edges_committed: int = 0      # 本轮已提交的 ES 数
    edges: List[EdgeStatus] = field(default_factory=list)
    global_accuracy: Optional[float] = None
    global_version: int = 0


@dataclass
class OrchestrationDecision:
    """经过 RuleEngine 校验后的最终调度决策"""
    priority_edges: List[str] = field(default_factory=list)
    defer_edges: List[str] = field(default_factory=list)
    commit_edges: List[str] = field(default_factory=list)
    close_round: bool = False
    compensation_weights: Dict[str, float] = field(default_factory=dict)
    agent_reason: str = ""
    agent_approved: bool = False
    override_reason: str = ""


# ─── 编排RuleEngine ──────────────────────────────

class OrchestrationRuleEngine:
    """编排层硬约束: 确保 Agent 建议不违反安全/公平底线"""

    def __init__(self, quorum_q_es: int = 2, total_edges: int = 4,
                 min_participation_rate: float = 0.15,
                 max_consecutive_exclusion: int = 3,
                 max_compensation: float = 1.5):
        self.quorum_q_es = quorum_q_es
        self.total_edges = total_edges
        self.min_participation_rate = min_participation_rate
        self.max_consecutive_exclusion = max_consecutive_exclusion
        self.max_compensation = max_compensation
        self._consecutive_exclusion: Dict[str, int] = {}  # 跟踪连续落选

    def validate(self, suggestion: dict, state: OrchestrationState,
                 edgemap: Dict[str, 'AgentEdge']) -> OrchestrationDecision:
        """
        校验 Agent 的 JSON 建议 → 返回最终决策

        规则:
          R1: close_round 必须有 >= Q_ES 个 commit_edges
          R2: commit_edges 必须在 priority_edges 里
          R3: 连续落选 > max_consecutive 的 ES 强制进 priority
          R4: compensation_weight 不能超过 max_compensation
          R5: commit_edges 中的 ES 必须满足 K_user (窗口够人了)
        """
        overrides = []
        approved = True

        priority = suggestion.get("priority_edges", [])
        defer_ = suggestion.get("defer_edges", [])
        commit = suggestion.get("commit_edges", [])
        close = suggestion.get("close_round", False)
        comp = suggestion.get("compensation_weights", {})
        reason = suggestion.get("reason", "")

        # R1: 要关轮必须有足够 ES
        if close and len(commit) < self.quorum_q_es:
            overrides.append(
                f"close_round=true but commit_edges ({len(commit)}) < Q_ES ({self.quorum_q_es})")
            close = False
            approved = False

        # R2: commit 必须在 priority 或 规则放行
        for eid in commit:
            if eid not in priority:
                # 但如果该 ES 连续落选太多，强制接受
                excl = self._consecutive_exclusion.get(eid, 0)
                if excl < self.max_consecutive_exclusion:
                    overrides.append(f"commit_edges includes non-priority ES {eid}")
                    approved = False
                # else: 规则放行 — 太多次没参与了，这次强制接收

        # R3: 连续落选 > max_consecutive → 强制进 priority
        for eid, count in self._consecutive_exclusion.items():
            if count >= self.max_consecutive_exclusion and eid not in priority:
                priority.insert(0, eid)
                overrides.append(
                    f"ES {eid} excluded {count} consecutive rounds → forced priority")
                approved = False

        # R4: 补偿权重上限
        for eid, w in comp.items():
            if w > self.max_compensation:
                comp[eid] = self.max_compensation
                overrides.append(
                    f"ES {eid} compensation ({w:.2f}) > max ({self.max_compensation}) → capped")
                approved = False

        # R5: commit 中的 ES 必须满足本地阈值
        for eid in commit:
            edge = edgemap.get(eid)
            if edge is not None:
                ready = sum(1 for w in edge._windows.values()
                            for _ in w.uploads)
                if ready < edge.min_batch_size:
                    overrides.append(
                        f"ES {eid} has {ready} clients < min_batch {edge.min_batch_size}")
                    approved = False

        # 更新连续落选计数
        committed_set = set(commit) if close else set()
        for es_status in state.edges:
            eid = es_status.edge_id
            if eid in committed_set:
                self._consecutive_exclusion[eid] = 0
            else:
                self._consecutive_exclusion[eid] = (
                    self._consecutive_exclusion.get(eid, 0) + 1)

        override_reason = "; ".join(overrides) if overrides else ""

        return OrchestrationDecision(
            priority_edges=priority,
            defer_edges=defer_,
            commit_edges=commit,
            close_round=close,
            compensation_weights=comp,
            agent_reason=reason,
            agent_approved=approved,
            override_reason=override_reason,
        )


# ─── 状态收集函数 ──────────────────────────────

def build_orchestration_state(
    edges: List['AgentEdge'],
    coordinator: 'AgentCoordinator',
    current_round: int,
    quorum_q_es: int,
    global_accuracy: Optional[float] = None,
    history_window: int = 5,
) -> OrchestrationState:
    """打包所有 ES 的当前状态，准备发给 Agent"""
    total = len(edges)
    target_rate = quorum_q_es / max(total, 1)
    edge_statuses = []

    for edge in edges:
        # 统计窗口状态
        clients_ready = 0
        clients_inflight = 0
        for win in edge._windows.values():
            clients_ready += len(win.uploads)
            clients_inflight += win.inflight

        K_user = edge.min_batch_size
        need = max(0, K_user - clients_ready)

        # 预估到达时间: 取最小剩余tick（简化）
        estimated = 1 if clients_inflight > 0 else K_user * 2

        # 历史参与率
        history_rate = edge.auditor.get_participation_rate(
            edge.id, window=history_window)

        # 补偿权重
        compensation = 1.0
        if history_rate < target_rate:
            compensation = min(1.5, 1.0 + (target_rate - history_rate) / target_rate)

        edge_statuses.append(EdgeStatus(
            edge_id=edge.id,
            clients_ready=clients_ready,
            clients_inflight=clients_inflight,
            need_for_quorum=need,
            estimated_ticks=estimated,
            history_rate=history_rate,
            target_rate=target_rate,
            compensation_weight=compensation,
            lagging=(need >= K_user),  # 差太多 → 掉队
        ))

    # 统计已提交
    committed = coordinator.commit_count

    return OrchestrationState(
        current_round=current_round,
        total_edges=total,
        quorum_q_es=quorum_q_es,
        edges_committed=committed,
        edges=edge_statuses,
        global_accuracy=global_accuracy,
        global_version=coordinator.global_version,
    )


# ─── Agent 调用封装 ──────────────────────────────

def build_agent_prompt(state: OrchestrationState) -> str:
    """构建发给 LLM Agent 的编排 prompt"""
    edges_text = ""
    for es in state.edges:
        lag = "LAGGING" if es.lagging else "normal"
        edges_text += (
            f"  - {es.edge_id}: ready={es.clients_ready}, inflight={es.clients_inflight}, "
            f"need={es.need_for_quorum} to reach local quorum, "
            f"est_ticks={es.estimated_ticks}, "
            f"history_rate={es.history_rate:.0%}, target={es.target_rate:.0%}, "
            f"compensation={es.compensation_weight:.2f}, status={lag}\n"
        )

    prompt = f"""你是一个联邦学习编排智能体。请决定本轮优先推进哪些边缘服务器。

当前轮次状态:
- 边缘服务器总数: {state.total_edges}
- Cloud quorum (Q_ES): 需要凑够 {state.quorum_q_es} 个 ES 才能关轮
- 本轮已提交: {state.edges_committed}
- 全局版本: {state.global_version}
- 全局准确率: {state.global_accuracy or '无'}

各边缘服务器状态:
{edges_text}

决策规则:
- "priority_edges": 本轮优先推进哪些 ES（按优先级排序，越靠前越优先）
- "defer_edges": 哪些 ES 本轮明显掉队，本轮不建议继续等待
- "commit_edges": 哪些 ES 已经或即将满足本地阈值，本轮提交
- "close_round": 如果足够多的 ES 已准备好满足 Q_ES，则设为 true
- "compensation_weights": 对于历史参与率低的 lagging 边缘，给予适度补偿权重 (1.0-1.5)

策略: 优先推进接近本地阈值的 ES。对于历史参与率低于目标值的 ES，适当提高优先级。
但如果某个 ES 明显赶不上本轮（need >= 本地阈值），不要因为它拖住整轮——标记为 defer。

请只返回一个 JSON 对象，键为:
priority_edges, defer_edges, commit_edges, close_round, compensation_weights, reason
reason 请用中文简要说明决策理由。"""

    return prompt


def call_orchestrate_agent(
    reasoner,  # LLMReasoner instance
    state: OrchestrationState,
    edgemap: Dict[str, 'AgentEdge'],
    rule_engine: OrchestrationRuleEngine,
) -> OrchestrationDecision:
    """完整编排流程: 收集状态 → Agent建议 → RuleEngine → 最终决策"""

    # 1. 调 Agent
    try:
        from llm_reasoner import DecisionType
        prompt = build_agent_prompt(state)

        # ====== 显示 Agent 输入 ======
        print(chr(10) + '=' * 50)
        print('[编排Agent] 输入:')
        for es in state.edges:
            lag = '掉队' if es.lagging else '正常'
            print(f'  {es.edge_id}: ready={es.clients_ready}, need={es.need_for_quorum}, hist={es.history_rate:.0%}, inflight={es.clients_inflight}, {lag}')
        print(f'  Q_ES={state.quorum_q_es}, ver={state.global_version}')
        print('-' * 50)
        # 直接把编排状态作为 context 传给 Agent
        result = reasoner.decide(
            decision_type=DecisionType.EDGE_FLUSH,
            agent_id="orchestrator",
            round_info={
                "state": {es.edge_id: {
                    "ready": es.clients_ready,
                    "need": es.need_for_quorum,
                    "history": es.history_rate,
                    "compensation": es.compensation_weight,
                } for es in state.edges},
                "quorum": state.quorum_q_es,
                "total": state.total_edges,
            },
            extra={"full_state": state},
        )

        # ====== 显示 Agent 输出 ======
        raw = result.decision if isinstance(result.decision, str) else str(result.decision)
        print(f'[编排Agent] 原始输出: {raw[:400]}')
        print(f'[编排Agent] 推理: {result.reasoning[:200] if result.reasoning else "无"}')
        print('=' * 50)
        # 2. 解析 JSON
        agent_json = _parse_json_safe(raw)

        if agent_json is None:
            # JSON 解析失败 → 用规则
            return _fallback_decision(state, edgemap, rule_engine)

        # 3. RuleEngine 校验
        decision = rule_engine.validate(agent_json, state, edgemap)
        decision.agent_reason = agent_json.get("reason", "")

        return decision

    except Exception as e:
        logger.warning(f"Orchestrate agent failed: {e} → falling back to rules")
        return _fallback_decision(state, edgemap, rule_engine)


def _parse_json_safe(text: str) -> Optional[dict]:
    """安全解析 JSON — Agent 可能返回多余文本"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return json.loads(text)
    except (json.JSONDecodeError, Exception):
        return None


def _fallback_decision(
    state: OrchestrationState,
    edgemap: Dict[str, 'AgentEdge'],
    rule_engine: OrchestrationRuleEngine,
) -> OrchestrationDecision:
    """Agent 不可用时的硬编码回退"""
    ready = []
    near = []
    lagging = []
    for es in state.edges:
        if es.need_for_quorum == 0:
            ready.append(es.edge_id)
        elif es.need_for_quorum <= 2:
            near.append(es.edge_id)
        else:
            lagging.append(es.edge_id)

    priority = ready + near + lagging
    commit = ready
    close = len(ready) >= state.quorum_q_es

    # 补偿权重
    comp = {}
    for es in state.edges:
        if es.history_rate < es.target_rate:
            comp[es.edge_id] = min(1.5, 1.0 + (es.target_rate - es.history_rate) / es.target_rate)

    return OrchestrationDecision(
        priority_edges=priority,
        defer_edges=lagging,
        commit_edges=commit,
        close_round=close,
        compensation_weights=comp,
        agent_reason="Fallback — agent unavailable",
        agent_approved=True,
    )


# ─── AgentAuditor 补充方法 ──────────────────────

def patch_auditor_for_orchestration(auditor):
    """给 AgentAuditor 加历史参与率查询方法"""
    if not hasattr(auditor, '_commit_records'):
        auditor._commit_records: List[Dict[str, Any]] = []

    def get_participation_rate(edge_id: str, window: int = 5) -> float:
        """最近 window 次提交中，该 Edge 参与的比例"""
        recent = auditor._commit_records[-window * 4:]  # 取足够多
        commits = [r for r in recent if r.get('type') == 'commit']
        commits = commits[-window:]
        if not commits:
            return 0.0
        participated = sum(1 for c in commits if c.get('edge_id') == edge_id)
        return participated / len(commits)

    auditor.get_participation_rate = get_participation_rate

    # 记录每次提交
    old_record = auditor.record_commit
    def record_commit_with_tracking(commit, tick):
        auditor._commit_records.append({
            'type': 'commit', 'edge_id': commit.edge_id,
            'tick': tick, 'verified': commit.verification_passed,
        })
        old_record(commit, tick)
    auditor.record_commit = record_commit_with_tracking

    return auditor
