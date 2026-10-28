"""
研究点2 类型定义: 组包 + Checkpoint 结果 + 版本化协议配置

这些 dataclass 专用于 buffer/checkpoint 半异步协议，
不破坏研究点1的同步 round-based 类型。
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class ProtocolConfig:
    """版本化协议配置 (研究点 2, 策略绑定的配置对象).

    全部策略参数在系统初始化时由 TA/本地管理员预置并本地持有 (TA 与
    各 ES 持有同一实例), 不属于任何 Cloud 可下发的协议消息; checkpoint
    manifest 携带其规范 JSON 序列化的 SHA-256 摘要 (digest), ES 校验
    "声明摘要 == 本地配置摘要" (fail-closed), 云端声明值只作审计。

    字段分三层:
    - 调度层: K / s_max
    - 权重与策略层: 权重来源 (由 participation_debias / sigma_target /
      staleness_sigma 唯一确定)、σ 策略及其参数
    - 补偿层: 生效 gamma_vcm (σ-agg 非 V-PDA 路径恒为 0)、VCM mode、
      momentum、UR-VCM 可信域比率、V-PDA 参数
    """

    # 调度层
    K: int = 3
    s_max: int = 3
    # 权重与策略层
    participation_debias: str = 'none'      # none | ipw
    sigma_target: str = 'vcm'               # vcm | agg
    staleness_sigma: float = 0.0
    staleness_metric: str = 'version'       # version | rel2 | wait
    sigma_policy: str = 'fixed'             # fixed | alignment
    sigma_alignment_threshold: float = 0.72
    sigma_alignment_ema: float = 0.8
    sigma_alignment_warmup: int = 2
    sigma_alignment_boost_threshold: float = 0.9
    sigma_alignment_high: float = -0.25
    # 补偿层 (gamma_vcm 为本 run 生效值)
    gamma_vcm: float = 0.0
    vcm_mode: str = 'full'
    momentum: float = 0.0
    vcm_trust_ratio: float = 1.0
    # V-PDA 参数
    debias_kappa: float = 0.5
    debias_min: float = 0.5
    debias_max: float = 2.0
    debias_scale: int = 65536
    # 配置版本
    config_version: str = 'p2-v1'

    def digest(self) -> str:
        """规范 JSON 序列化 (键排序) 的 SHA-256 摘要."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @property
    def expected_weight_source(self) -> str:
        """本配置唯一确定的权重来源 (来源绑定的配置侧推导)."""
        if self.participation_debias == 'ipw':
            return 'vpda'
        if self.sigma_target == 'agg' and self.staleness_sigma > 0:
            return 'staleness'
        return 'none'


def canonical_digest(obj) -> str:
    """任意可 JSON 化对象的规范 SHA-256 (键排序), 用于 vote/receipt 载荷."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                         default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def tokens_digest(tokens: List[str]) -> str:
    """选中组包 token 列表的规范摘要 (排序后逐项拼接再哈希)."""
    return canonical_digest(sorted(str(t) for t in tokens))


def array_digest(arr) -> str:
    """数组内容摘要 (dtype + 原始字节), 用于把聚合结果绑定进 vote/receipt."""
    if arr is None:
        return 'none'
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.dtype.str.encode() + a.tobytes()).hexdigest()


def result_digest(checkpoint_version: int, token_digest: str,
                  raw_sum_int, M_Q, final_model) -> str:
    """ES 验证过的聚合结果的规范摘要 (结果摘要绑定, P0-1).

    vote 与 receipt 均绑定该摘要: Cloud 若在取得诚实 receipt 后替换
    raw_sum / M_Q / final_model, commit 时摘要不匹配即被拒绝; ES 也只会
    接受与本地已验证结果摘要一致的 receipt。final_model 由 ES 依据已
    验证 raw sum + 公开规则确定性重构, 一并绑定可防止"换模型"攻击。
    """
    return canonical_digest({
        'v': int(checkpoint_version),
        'tokens': token_digest,
        'raw_sum': array_digest(raw_sum_int),
        'M_Q': array_digest(M_Q),
        'final': array_digest(final_model)})


@dataclass(frozen=True)
class VerificationResult:
    """带阶段与原因的验证结果 (布尔接口的详细化包装).

    stage ∈ {config_digest, version, history, policy_decl, policy_alpha,
    policy_source, policy_weights, signature, base_hash, lhh, reconstruct, ok}
    —— 用于日志归因与安全证据的"检出阶段"统计 (AttackOutcome.detection_stage)。
    """
    passed: bool
    stage: str
    reason: str = ''

    def __bool__(self) -> bool:  # 便于真值使用
        return self.passed


@dataclass(frozen=True)
class CheckpointVote:
    """ES 对一次 checkpoint 候选的带签名投票 (验证通过后生成).

    载荷 = (voter, checkpoint_version, token_digest, config_digest,
    result_digest, passed, reason); TA 聚合全部 vote 并验签后才可签发
    receipt。result_digest 绑定 ES 实际验证过的聚合结果 (raw sum / M_Q /
    final model), 使投票不可被移植到另一个候选上。
    """
    voter: str
    checkpoint_version: int
    token_digest: str
    config_digest: str
    passed: bool
    reason: str = ''
    result_digest: str = ''
    signature: bytes = b''

    def payload(self) -> dict:
        return {'voter': self.voter,
                'checkpoint_version': self.checkpoint_version,
                'token_digest': self.token_digest,
                'config_digest': self.config_digest,
                'result_digest': self.result_digest,
                'passed': self.passed,
                'reason': self.reason}

    def payload_digest(self) -> str:
        return canonical_digest(self.payload())


@dataclass(frozen=True)
class CheckpointReceipt:
    """TA 签发的 checkpoint 提交凭证 (权威授权对象).

    TA 在验证全部 vote、检查 token 已注册且未消费、调度约束满足后签发;
    签发与 ledger 落地 (token 消费 + 版本记录) 在同一原子事务内完成。
    Cloud 必须持有效 receipt 才能提交; ES 只有看到有效 receipt 才推进
    版本与策略状态。receipt 本身只能使用一次 (ledger 主键约束)。
    """
    receipt_id: str            # ta_id:v{version}:{token_digest[:16]}
    checkpoint_version: int    # 本次消费的版本号 v (提交后全局变为 v+1)
    token_digest: str
    selected_tokens: List[str]
    trigger_mode: str = 'normal'   # normal | timeout
    result_digest: str = ''        # 结果摘要绑定 (P0-1)
    signature: bytes = b''

    def payload(self) -> dict:
        return {'receipt_id': self.receipt_id,
                'checkpoint_version': self.checkpoint_version,
                'token_digest': self.token_digest,
                'selected_tokens': sorted(self.selected_tokens),
                'trigger_mode': self.trigger_mode,
                'result_digest': self.result_digest}

    def payload_digest(self) -> str:
        return canonical_digest(self.payload())


@dataclass
class AttackOutcome:
    """单次攻击注入的结构化结果 (攻击实验统计的最小单元).

    requested/applicable/attempted 分离, 使"配置不适用的攻击"不再混入
    未检出样本; detected/accepted 记录验证器裁决; detection_stage/reason
    来自 VerificationResult; 前后状态摘要用于断言"失败不推进任何状态"。
    """
    attack: str
    requested: bool
    applicable: bool = False
    attempted: bool = False
    detected: bool = False
    accepted: bool = False
    detection_stage: str = ''
    reason_code: str = ''
    state_before: dict = field(default_factory=dict)
    state_after: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroupPackage:
    """组级掩码包 (Π_k).

    对应文档 Sec.4.2: ES 封装的组级掩码聚合包，
    存入 Cloud buffer 时不包含解掩码所需材料。
    """
    edge_id: str
    package_token: str                     # 如 "edge_1:v0:p2"
    base_version: int                      # r_k: 该包基于的全局版本
    base_hash: bytes                       # h_base(r_k): W(base_version) 的哈希
    online_users: List[str]                # U_online^k(r_k)
    n_k: int                               # |U_online^k(r_k)|
    S_k: int                               # Σ s_i^k: 组内总样本量
    masked_sum_int: np.ndarray             # W̄_k(r_k): int64 掩码和
    H_k: object                            # LHH.Hash result (NIST P-256 Point)
    signature: bytes                       # Ed25519 σ_k(r_k)
    created_tick: int = 0                  # 模拟时钟: 组包生成时刻
    upd_norm: float = 0.0                  # 组内样本量加权平均本地更新范数 m̂_k (矩校准 VCM 元数据)


@dataclass
class CheckpointResult:
    """Checkpoint 聚合结果.

    对应文档 Sec.4.5-4.6: Cloud 在一次 checkpoint 中
    恢复的数据以及验证所需材料。
    """
    version: int                           # v: checkpoint 序号
    selected_packages: List[GroupPackage]  # Q(v)
    N_Q: int                               # 总参与客户端数
    S_Q: int                               # Σ S_k: 总样本量
    raw_sum_int: np.ndarray                # W_sum^raw(v): int64
    raw_sum_float: np.ndarray              # 解量化后的 raw sum (float64)
    M_Q: np.ndarray                        # TA 返回的聚合掩码 (int64)
    H_mask_raw: object                     # H_mask^raw(v)
    corrected_sum_float: np.ndarray        # W_sum^corr(v): 补偿后 (float64)
    final_model: np.ndarray               # W(v+1): 新全局模型 (float64)
    compensation_per_package: dict = field(default_factory=dict)
    verification_passed: bool = False


@dataclass
class StagedCheckpoint:
    """stage 阶段暂存的 checkpoint 候选 (P0-2 committed 两阶段提交).

    stage 只做只读校验并记录"若提交成功要应用的变更", 不触碰 Cloud 权威
    状态 (buffer / 版本 / 模型历史)。TA finalize 成功后才由 publish 原子
    应用。可序列化 (journal) 以支持进程崩溃后恢复完成 —— 存在 journal 时
    重放为幂等操作。
    """
    base_version: int                      # stage 时刻的 Cloud 版本 v
    selected_packages: List['GroupPackage']
    final_model: np.ndarray
    raw_sum_int: np.ndarray
    M_Q: np.ndarray
    H_mask_raw: object
    corrected_sum: np.ndarray
    comp_map: dict = field(default_factory=dict)
    receipt: object = None                 # CheckpointReceipt | None (verify-off)
    stage_id: str = ''
