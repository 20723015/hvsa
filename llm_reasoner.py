"""
LLMReasoner — 统一的 LLM Agent 接入层。

用法:
    from llm_reasoner import LLMReasoner, DecisionType

    reasoner = LLMReasoner(backend="anthropic", api_key="...")
    decision = reasoner.decide(
        agent_type="AgentClient",
        context={"round": 5, "update_norm": 0.32, "staleness": 1},
    )

设计原则:
    - 所有 Agent 通过统一的 `decide()` 方法调用
    - 内置多后端支持 (Anthropic, OpenAI, 模拟)
    - 内置缓存减少 API 调用
    - LLM 失败时自动降级到随机/规则 fallback
    - 可序列化，实验结果可复现
"""

import json
import logging
import os
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """所有 Agent 需要 LLM 决策的类型"""
    CLIENT_DROPOUT = "client_dropout"          # 客户端是否掉线
    CLIENT_LATENCY = "client_latency"          # 客户端延迟多久上传
    EDGE_FLUSH = "edge_flush"                  # 边缘服务器是否立即聚合
    EDGE_RISK_SCORE = "edge_risk_score"        # 边缘服务器风险评估
    COORDINATOR_MERGE_WEIGHT = "coordinator_merge_weight"  # 协调器合并权重
    COORDINATOR_REJECT_BATCH = "coordinator_reject_batch"  # 协调器是否拒绝批次


@dataclass
class DecisionContext:
    """传给 LLM 的上下文"""
    decision_type: DecisionType
    agent_id: str
    round_info: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "agent_id": self.agent_id,
            "round_info": self.round_info,
            "history": self.history[-5:],  # 只保留最近5条历史
            "extra": self.extra,
        }


@dataclass
class DecisionResult:
    """LLM 的决策结果"""
    decision: Any
    reasoning: str = ""
    confidence: float = 1.0
    backend_used: str = "unknown"
    from_cache: bool = False


# ─── 后端抽象 ──────────────────────────────────────────────

class LLMBackend(ABC):
    @abstractmethod
    def decide(self, context: DecisionContext) -> DecisionResult:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class MockBackend(LLMBackend):
    """模拟后端：用随机数代替 LLM，用于测试和降级"""
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def name(self) -> str:
        return "mock"

    def decide(self, context: DecisionContext) -> DecisionResult:
        dt = context.decision_type
        if dt == DecisionType.CLIENT_DROPOUT:
            decision = self._rng.random() < 0.1  # 10% 掉线率
            return DecisionResult(
                decision="drop" if decision else "participate",
                reasoning=f"Mock random decision (threshold=0.1, draw={decision:.3f})",
                confidence=0.5,
                backend_used=self.name(),
            )
        elif dt == DecisionType.EDGE_FLUSH:
            decision = self._rng.random() < 0.5
            return DecisionResult(
                decision="flush" if decision else "wait",
                reasoning=f"Mock random flush decision",
                confidence=0.5,
                backend_used=self.name(),
            )
        else:
            # 默认：返回一个安全值
            return DecisionResult(
                decision=None,
                reasoning="Mock fallback",
                confidence=0.0,
                backend_used=self.name(),
            )


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API 后端 (兼容 DeepSeek 等第三方 Anthropic-compatible API)

    Args:
        api_key: API 密钥。
        model: 模型名称。
        base_url: API 端点。默认 Anthropic 官方；DeepSeek 用 'https://api.deepseek.com/anthropic'。
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514",
                 max_retries: int = 2, base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.base_url = base_url

    def name(self) -> str:
        prefix = "deepseek" if self.base_url and "deepseek" in self.base_url else "anthropic"
        return f"{prefix}/{self.model}"

    def decide(self, context: DecisionContext) -> DecisionResult:
        if not self.api_key:
            logger.warning("AnthropicBackend: No API key set, falling through")
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed: pip install anthropic")

        prompt = self._build_prompt(context)

        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                client = anthropic.Anthropic(**kwargs)
                response = client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=0.2,
                    system="You are a federated learning decision optimizer. "
                           "Respond with a single JSON object with keys: 'decision', 'reasoning', 'confidence'.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                return self._parse_response(text, context)

            except Exception as e:
                logger.warning(f"AnthropicBackend attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("AnthropicBackend: All retries exhausted")

    def _build_prompt(self, context: DecisionContext) -> str:
        """根据决策类型构建 prompt"""
        dt = context.decision_type

        if dt == DecisionType.CLIENT_DROPOUT:
            return self._prompt_dropout(context)
        elif dt == DecisionType.EDGE_FLUSH:
            return self._prompt_flush(context)
        elif dt == DecisionType.COORDINATOR_REJECT_BATCH:
            return self._prompt_reject(context)
        else:
            return f"Decide on {dt.value} with context: {json.dumps(context.to_prompt_dict(), indent=2)}"

    def _prompt_dropout(self, ctx: DecisionContext) -> str:
        info = ctx.round_info
        return f"""你是联邦学习客户端 {ctx.agent_id}。你应该参与本轮训练还是掉线？

当前状态:
- 全局轮次: {info.get('round', '?')}
- 全局版本: {info.get('global_version', '?')}
- 你的更新范数: {info.get('update_norm', '?')}
- 陈旧度: {info.get('staleness', '?')}
- 你的样本数: {info.get('sample_count', '?')}
- 最近准确率: {info.get('recent_accuracy', '?')}

掉线理由: 更新已过时、数据质量低、或节省资源。
参与理由: 更新新鲜且有价值。

请只返回 JSON:
{{"decision": "participate" 或 "drop", "reasoning": "用中文简述理由", "confidence": 0.0-1.0}}"""

    def _prompt_flush(self, ctx: DecisionContext) -> str:
        info = ctx.round_info
        return f"""你是边缘服务器 {ctx.agent_id}。你应该立即聚合(flush)还是继续等待(wait)？

当前状态:
- 窗口内客户端数: {info.get('pending_clients', 0)}
- 最小批大小: {info.get('min_batch_size', 2)}
- 距上次flush: {info.get('ticks_since_flush', 0)} ticks
- flush超时: {info.get('flush_timeout', 2)} ticks
- 平均风险分: {info.get('avg_risk_score', 0):.3f}

flush理由: 客户端够了、超时了、或风险可接受。
wait理由: 客户端太少、风险偏高、或刚flush过。

请只返回 JSON:
{{"decision": "flush" 或 "wait", "reasoning": "用中文简述理由", "confidence": 0.0-1.0}}"""

    def _prompt_reject(self, ctx: DecisionContext) -> str:
        info = ctx.round_info
        return f"""你是全局协调器。你应该接受(ACCEPT)还是拒绝(REJECT)这个异步批次？

批次详情:
- 边缘服务器: {info.get('edge_id', '?')}
- 批次大小: {info.get('batch_size', '?')}
- 风险分: {info.get('risk_score', 0):.3f}
- 陈旧度: staleness={info.get('staleness', 0)} 个版本落后
- 合并权重: {info.get('merge_weight', 0):.3f}
- 全局版本: {info.get('global_version', '?')}

请只返回 JSON:
{{"decision": "accept" 或 "reject", "reasoning": "用中文简述理由", "confidence": 0.0-1.0}}"""

    def _parse_response(self, text: str, context: DecisionContext) -> DecisionResult:
        """解析 LLM 的 JSON 响应"""
        try:
            # 提取 JSON (处理 LLM 可能返回额外的文本)
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                text = text[start:end]
            data = json.loads(text)
            return DecisionResult(
                decision=data.get("decision"),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 1.0),
                backend_used=self.name(),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {text[:200]}... error: {e}")
            raise


class OpenAIBackend(LLMBackend):
    """OpenAI API 后端"""
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", max_retries: int = 2):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.max_retries = max_retries

    def name(self) -> str:
        return f"openai/{self.model}"

    def decide(self, context: DecisionContext) -> DecisionResult:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed: pip install openai")

        prompt = AnthropicBackend()._build_prompt(context)  # 复用 prompt 构建

        for attempt in range(self.max_retries + 1):
            try:
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "Respond with JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message.content or ""
                return AnthropicBackend()._parse_response(text, context)  # 复用解析

            except Exception as e:
                logger.warning(f"OpenAIBackend attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("OpenAIBackend: All retries exhausted")


# ─── 主入口 ──────────────────────────────────────────────

class LLMReasoner:
    """
    统一的 LLM Agent 接入层。

    Args:
        backend: 后端名称 ("anthropic", "openai", "mock") 或 LLMBackend 实例
        api_key: API 密钥 (None 则从环境变量读取)
        model: 模型名称
        fallback: LLM 失败时是否降级到 mock
        cache_size: 缓存大小 (0=不缓存)
        mock_seed: mock 后端的随机种子
    """

    def __init__(
        self,
        backend: Union[str, LLMBackend] = "mock",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback: bool = True,
        cache_size: int = 128,
        mock_seed: Optional[int] = None,
    ):
        self._backend = self._resolve_backend(backend, api_key, model, mock_seed)
        self.fallback = fallback
        self._cache: Dict[str, DecisionResult] = {}
        self._cache_size = cache_size
        self._call_count = 0
        self._cache_hit_count = 0
        self._fallback_count = 0
        self._history: List[Dict[str, Any]] = []  # 全局历史

    @staticmethod
    def _resolve_backend(
        backend: Union[str, LLMBackend],
        api_key: Optional[str],
        model: Optional[str],
        mock_seed: Optional[int],
    ) -> LLMBackend:
        if isinstance(backend, LLMBackend):
            return backend
        if backend == "anthropic":
            return AnthropicBackend(api_key=api_key, model=model or "claude-sonnet-4-20250514")
        elif backend == "deepseek":
            return AnthropicBackend(api_key=api_key, model=model or "deepseek-chat",
                                    base_url="https://api.deepseek.com/anthropic")
        elif backend == "openai":
            return OpenAIBackend(api_key=api_key, model=model or "gpt-4o-mini")
        else:
            return MockBackend(seed=mock_seed)

    def _cache_key(self, context: DecisionContext) -> str:
        """生成缓存键"""
        d = context.to_prompt_dict()
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def decide(
        self,
        decision_type: DecisionType,
        agent_id: str,
        round_info: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        """
        统一的决策接口。

        Args:
            decision_type: 决策类型
            agent_id: Agent ID
            round_info: 当前轮次信息
            extra: 额外上下文

        Returns:
            DecisionResult
        """
        context = DecisionContext(
            decision_type=decision_type,
            agent_id=agent_id,
            round_info=round_info or {},
            history=self._history,
            extra=extra or {},
        )

        self._call_count += 1

        # 缓存查找
        if self._cache_size > 0:
            key = self._cache_key(context)
            if key in self._cache:
                self._cache_hit_count += 1
                result = self._cache[key]
                result.from_cache = True
                return result

        # 调用后端
        try:
            result = self._backend.decide(context)
        except Exception as e:
            logger.warning(f"LLMReasoner: Backend {self._backend.name()} failed: {e}")
            if self.fallback:
                self._fallback_count += 1
                logger.info("LLMReasoner: Falling back to mock backend")
                mock = MockBackend()
                result = mock.decide(context)
            else:
                raise

        # 写缓存
        if self._cache_size > 0:
            key = self._cache_key(context)
            if len(self._cache) < self._cache_size:
                self._cache[key] = result

        # 记录历史
        self._history.append({
            "decision_type": decision_type.value,
            "agent_id": agent_id,
            "result_decision": str(result.decision),
            "confidence": result.confidence,
            "backend": result.backend_used,
        })

        return result

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            "backend": self._backend.name(),
            "calls": self._call_count,
            "cache_hits": self._cache_hit_count,
            "cache_hit_rate": self._cache_hit_count / max(self._call_count, 1),
            "fallbacks": self._fallback_count,
            "cache_size": len(self._cache),
        }

    def reset_history(self):
        self._history = []
        self._call_count = 0
        self._cache_hit_count = 0
        self._fallback_count = 0

    @property
    def backend_name(self) -> str:
        return self._backend.name()
