"""
RuleEngine — LLM Agent 的安全约束层

设计原则:
   Agent 只输出"建议"，RuleEngine 执行硬约束校验。
   即使 Agent 被 prompt 注入、被恶意控制、或 LLM 提供商被攻击，
   系统安全不变式也不会被打破。

   Agent ∉ 可信计算基 (TCB)
   RuleEngine ∈ TCB

用法:
    from rule_engine import RuleEngine, AgentSuggestion

    engine = RuleEngine(verification_round_interval=5)
    suggestion = engine.validate(agent_suggestion, round_context)
    if suggestion.approved:
        execute(suggestion)
    else:
        execute_fallback(round_context)  # 原随机规则
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── 建议类型 ───────────────────────────────────────

class SuggestionType(str, Enum):
    CLIENT_SELECTION = "client_selection"          # 选哪些用户参与
    FLUSH_TIMING = "flush_timing"                  # 何时聚合
    VERIFICATION_INTENSITY = "verification_intensity"  # 验证强度
    BATCH_ACCEPTANCE = "batch_acceptance"          # 是否接受异步批次


class VerificationLevel(str, Enum):
    FULL = "full"        # LHH + Ed25519 全验
    LIGHT = "light"       # 只验 Ed25519 签名，跳过 LHH
    SKIP = "skip"         # 全跳过 (快速模式)


@dataclass
class AgentSuggestion:
    """Agent 的原始建议"""
    type: SuggestionType
    decision: Any
    reasoning: str = ""
    confidence: float = 0.5


@dataclass
class ValidatedDecision:
    """经过规则引擎校验后的决策"""
    type: SuggestionType
    original_suggestion: Any   # Agent 的原始建议
    final_decision: Any        # 规则引擎修正后的最终决策
    approved: bool             # 是否完全采纳 Agent 建议
    override_reason: str = ""  # 如果不采纳，原因是什么


# ─── 上下文 ─────────────────────────────────────────

@dataclass
class RoundContext:
    current_round: int = 0
    online_users: int = 0
    total_users: int = 0
    new_user_count: int = 0           # 本轮新用户数
    batch_size: int = 0              # 异步批次大小
    rounds_since_last_full_verify: int = 0
    delta_norm: float = 0.0           # 聚合结果变化幅度
    pending_clients: int = 0
    min_batch_size: int = 2
    ticks_since_flush: int = 0
    flush_timeout: int = 2
    risk_score: float = 0.0
    staleness: int = 0


# ─── 规则引擎 ──────────────────────────────────────

class RuleEngine:
    """
    安全约束层 — 夹在 Agent 建议和协议执行之间。

    Agent 说 "skip verification" → RuleEngine 检查 → 本轮是新用户多? 拒绝, 强制FULL
    Agent 说 "选 [U1,U2]" → RuleEngine 检查 → U2 是新人且老用户<3? 补一个老用户
    Agent 说 "flush now" → RuleEngine 检查 → 只有1个客户端? 拒绝, 继续等
    """

    def __init__(
        self,
        verification_round_interval: int = 5,    # 每 N 轮至少做一次 FULL 验证
        min_veteran_ratio: float = 0.5,          # 新用户比例超过此值 → 强制 FULL verify
        min_online_for_skip: int = 4,            # 在线用户少于此数 → 不能 skip verify
        delta_threshold: float = 0.3,            # 聚合结果变化超此值 → 强制 FULL verify
        new_user_buddy_count: int = 3,           # 新人至少要 N 个老用户陪跑
    min_batch_for_flush: int = 2,            # 绝对最小批大小
    max_staleness: int = 3,                  # 异步批次最大陈旧度
    reject_risk_threshold: float = 0.8,      # 批次风险阈值
    ):
        self.verify_interval = verification_round_interval
        self.min_veteran_ratio = min_veteran_ratio
        self.min_online_for_skip = min_online_for_skip
        self.delta_threshold = delta_threshold
        self.new_user_buddy_count = new_user_buddy_count
        self.min_batch_for_flush = min_batch_for_flush
        self.max_staleness = max_staleness
        self.reject_risk_threshold = reject_risk_threshold

    # ── 三个入口 ──────────────────────────────────

    def validate(
        self, suggestion: AgentSuggestion, ctx: RoundContext
    ) -> ValidatedDecision:
        """统一入口"""
        if suggestion.type == SuggestionType.CLIENT_SELECTION:
            return self._validate_client_selection(suggestion, ctx)
        elif suggestion.type == SuggestionType.FLUSH_TIMING:
            return self._validate_flush_timing(suggestion, ctx)
        elif suggestion.type == SuggestionType.VERIFICATION_INTENSITY:
            return self._validate_verification(suggestion, ctx)
        elif suggestion.type == SuggestionType.BATCH_ACCEPTANCE:
            return self._validate_batch_acceptance(suggestion, ctx)
        else:
            return ValidatedDecision(
                type=suggestion.type,
                original_suggestion=suggestion.decision,
                final_decision=suggestion.decision,
                approved=True,
            )

    # ── 客户端选择校验 ─────────────────────────────

    def _validate_client_selection(
        self, suggestion: AgentSuggestion, ctx: RoundContext
    ) -> ValidatedDecision:
        """
        Agent 建议: 选 [U1, U2, ...]
        规则:
          1. 不能全选新人
          2. 新人必须配至少 new_user_buddy_count 个老用户
          3. 建议为空 → 回退到随机选择
        """
        selected: List[str] = suggestion.decision if isinstance(suggestion.decision, list) else []
        new_users_in_selection = selected[:ctx.new_user_count]  # 简化

        overrides = []

        # 规则 1: 不能全空
        if not selected or len(selected) == 0:
            overrides.append("Agent proposed empty set → fallback to random")

        # 规则 2: 新人陪跑
        new_count = ctx.new_user_count
        veteran_count = len(selected) - new_count if new_count < len(selected) else 0
        if new_count > 0 and veteran_count < self.new_user_buddy_count:
            overrides.append(
                f"New users ({new_count}) need ≥{self.new_user_buddy_count} veterans, "
                f"only {veteran_count} present → must add veterans"
            )

        if overrides:
            reason = "; ".join(overrides)
            logger.warning(f"RuleEngine: client_selection overridden — {reason}")
            return ValidatedDecision(
                type=SuggestionType.CLIENT_SELECTION,
                original_suggestion=selected,
                final_decision=None,  # None = 回退到原随机逻辑
                approved=False,
                override_reason=reason,
            )

        return ValidatedDecision(
            type=SuggestionType.CLIENT_SELECTION,
            original_suggestion=selected,
            final_decision=selected,
            approved=True,
        )

    # ── 聚合时机校验 ────────────────────────────────

    def _validate_flush_timing(
        self, suggestion: AgentSuggestion, ctx: RoundContext
    ) -> ValidatedDecision:
        """
        Agent 建议: "flush" 或 "wait"
        规则:
          1. pending_clients < min_batch_for_flush → 强制 wait
          2. ticks_since_flush >= flush_timeout → 强制 flush (Agent 不能让你永远等)
          3. risk_score > 0.8 AND Agent说flush → 拦截，强制 wait 一轮
        """
        agent_says = suggestion.decision  # "flush" or "wait"
        overrides = []

        # 规则 1: 绝对最小批次
        if ctx.pending_clients < self.min_batch_for_flush:
            if agent_says == "flush":
                overrides.append(
                    f"pending clients ({ctx.pending_clients}) < "
                    f"min ({self.min_batch_for_flush}) → forced wait"
                )
            return ValidatedDecision(
                type=SuggestionType.FLUSH_TIMING,
                original_suggestion=agent_says,
                final_decision="wait",
                approved=False,
                override_reason="; ".join(overrides) or "hard minimum not met",
            )

        # 规则 2: 超时强制 flush
        if ctx.ticks_since_flush >= ctx.flush_timeout:
            if agent_says == "wait":
                overrides.append("flush timeout reached → forced flush")
            return ValidatedDecision(
                type=SuggestionType.FLUSH_TIMING,
                original_suggestion=agent_says,
                final_decision="flush",
                approved=(agent_says == "flush"),
                override_reason="; ".join(overrides) or "",
            )

        # 规则 3: 高风险拦截
        if ctx.risk_score > 0.8 and agent_says == "flush":
            overrides.append(f"risk_score ({ctx.risk_score:.2f}) > 0.8 → forced wait one more tick")

        if overrides:
            logger.warning(f"RuleEngine: flush_timing overridden — {'; '.join(overrides)}")
            return ValidatedDecision(
                type=SuggestionType.FLUSH_TIMING,
                original_suggestion=agent_says,
                final_decision="wait",
                approved=False,
                override_reason="; ".join(overrides),
            )

        return ValidatedDecision(
            type=SuggestionType.FLUSH_TIMING,
            original_suggestion=agent_says,
            final_decision=agent_says,
            approved=True,
        )

    # ── 验证强度校验 ──────────────────────────────

    def _validate_verification(
        self, suggestion: AgentSuggestion, ctx: RoundContext
    ) -> ValidatedDecision:
        """
        Agent 建议: VerificationLevel.FULL / LIGHT / SKIP
        规则:
          1. rounds_since_last_full_verify >= interval → 强制 FULL
          2. new_user_ratio > 0.5 → 强制 FULL
          3. delta_norm > threshold → 强制 FULL
          4. online_users < min_online_for_skip → 至少 LIGHT
          5. Agent 说 SKIP 但上面都不触发 → 允许 SKIP
        """
        agent_level = suggestion.decision  # VerificationLevel
        overrides = []

        # 计算实际比例
        new_ratio = ctx.new_user_count / max(ctx.online_users, 1)

        # 规则 1: 定期强制全验
        if ctx.rounds_since_last_full_verify >= self.verify_interval:
            overrides.append(
                f"{ctx.rounds_since_last_full_verify} rounds since last full verify "
                f"(≥{self.verify_interval}) → forced FULL"
            )

        # 规则 2: 新用户太多
        if new_ratio > (1 - self.min_veteran_ratio):
            overrides.append(
                f"new user ratio ({new_ratio:.0%}) > threshold → forced FULL"
            )

        # 规则 3: 聚合结果变化大
        if ctx.delta_norm > self.delta_threshold:
            overrides.append(
                f"delta_norm ({ctx.delta_norm:.3f}) > threshold ({self.delta_threshold}) → forced FULL"
            )

        # 规则 4: 在线用户太少
        if ctx.online_users < self.min_online_for_skip and agent_level == VerificationLevel.SKIP:
            overrides.append(
                f"online users ({ctx.online_users}) < min ({self.min_online_for_skip}) "
                f"→ SKIP denied, at least LIGHT"
            )
            final = VerificationLevel.FULL if overrides[:3] else VerificationLevel.LIGHT
            logger.warning(f"RuleEngine: verification overridden — {'; '.join(overrides)}")
            return ValidatedDecision(
                type=SuggestionType.VERIFICATION_INTENSITY,
                original_suggestion=agent_level,
                final_decision=final,
                approved=False,
                override_reason="; ".join(overrides),
            )

        if overrides:  # 规则 1-3 触发 → FULL
            logger.warning(f"RuleEngine: verification overridden — {'; '.join(overrides)}")
            return ValidatedDecision(
                type=SuggestionType.VERIFICATION_INTENSITY,
                original_suggestion=agent_level,
                final_decision=VerificationLevel.FULL,
                approved=False,
                override_reason="; ".join(overrides),
            )

        # Agent 建议通过
        return ValidatedDecision(
            type=SuggestionType.VERIFICATION_INTENSITY,
            original_suggestion=agent_level,
            final_decision=agent_level,
            approved=True,
        )

    def _validate_batch_acceptance(
        self, suggestion: AgentSuggestion, ctx: RoundContext
    ) -> ValidatedDecision:
        """
        Agent 建议: "accept" / "reject"
        规则:
          1. 批次大小为 0 -> 强制 reject
          2. 陈旧度过大 -> 强制 reject
          3. 风险分数过高 -> 强制 reject
          4. 否则允许 Agent 在安全边界内做选择
        """
        agent_says = str(suggestion.decision).lower()
        overrides = []

        if ctx.batch_size <= 0:
            overrides.append("empty batch -> forced reject")
        if ctx.staleness > self.max_staleness:
            overrides.append(
                f"staleness ({ctx.staleness}) > max ({self.max_staleness}) -> forced reject"
            )
        if ctx.risk_score > self.reject_risk_threshold:
            overrides.append(
                f"risk_score ({ctx.risk_score:.3f}) > threshold ({self.reject_risk_threshold}) -> forced reject"
            )

        if overrides:
            logger.warning(f"RuleEngine: batch_acceptance overridden — {'; '.join(overrides)}")
            return ValidatedDecision(
                type=SuggestionType.BATCH_ACCEPTANCE,
                original_suggestion=agent_says,
                final_decision="reject",
                approved=False,
                override_reason="; ".join(overrides),
            )

        final = "accept" if agent_says == "accept" else "reject"
        return ValidatedDecision(
            type=SuggestionType.BATCH_ACCEPTANCE,
            original_suggestion=agent_says,
            final_decision=final,
            approved=(final == agent_says),
        )

    # ── 统计 ───────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "verify_interval": self.verify_interval,
            "min_veteran_ratio": self.min_veteran_ratio,
            "min_online_for_skip": self.min_online_for_skip,
            "delta_threshold": self.delta_threshold,
            "new_user_buddy_count": self.new_user_buddy_count,
        }
