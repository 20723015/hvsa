import sys
import pickle
import hashlib
import logging
import math
from dataclasses import replace

import numpy as np

try:
    from utils import (LHH, QUANTIZE_SCALE, SIG, vpda_weights,
                       staleness_alpha, staleness_int_weights,
                       AlignmentSigmaPolicy)
except ImportError:
    print("错误: 无法从 utils.py 导入 LHH 或 SIG。")
    sys.exit(1)

from typing import TYPE_CHECKING, Optional, Union

from rule_engine import (
    AgentSuggestion, RoundContext, RuleEngine,
    SuggestionType, VerificationLevel,
)
from point2_types import (ProtocolConfig, VerificationResult,
                          CheckpointVote, tokens_digest, result_digest,
                          array_digest)

if TYPE_CHECKING:
    from user import User
    from cloud import CloudServer
    from ta import TrustedAuthority
    from ecdsa.ellipticcurve import Point

logger = logging.getLogger(__name__)


class EdgeServer:
    """
    边缘服务器

    职责:
    1. 收集用户掩码梯度，预聚合
    2. 计算 LHH 哈希 + Ed25519 签名
    3. 转发聚合结果给 Cloud，上报在线用户给 TA
    4. 验证 CS 广播的全局结果
    """

    def __init__(self, id: str, cloud_server: 'CloudServer',
                 ta: 'TrustedAuthority',
                 enable_verification: bool = True,
                 n_min: int = 2,
                 llm_reasoner=None, rule_engine=None,
                 debias_kappa: float = 0.5,
                 debias_min: float = 0.5,
                 debias_max: float = 2.0,
                 debias_scale: int = 65536,
                 vcm_trust_ratio: float = 1.0,
                 gamma_vcm: float = 0.0,
                 vcm_mode: str = 'full',
                 momentum: float = 0.0,
                 staleness_metric: str = 'version',
                 staleness_sigma: float = 0.0,
                 sigma_policy: str = 'fixed',
                 sigma_alignment_threshold: float = 0.72,
                 sigma_alignment_ema: float = 0.8,
                 sigma_alignment_warmup: int = 2,
                 sigma_alignment_boost_threshold: float = 0.90,
                 sigma_alignment_high: float = -0.25,
                 participation_debias: str = 'none',
                 sigma_target: str = 'vcm',
                 policy_config: Optional['ProtocolConfig'] = None):
        self.id = id
        self.cloud_server = cloud_server
        self.ta = ta
        self.enable_verification = enable_verification
        self.n_min = n_min
        self.llm_reasoner = llm_reasoner
        self.rule_engine = rule_engine
        self._full_verify_rounds: list[int] = []

        # V-PDA 参与去偏: ES 本地维护的组参与历史 (从每个验证通过的
        # checkpoint 的选中集合独立累计, 云端无法伪造)
        self._debias_kappa = debias_kappa
        self._debias_min = debias_min
        self._debias_max = debias_max
        self._debias_scale = int(debias_scale)
        # UR-VCM 可信域比率 ρ (ES 独立持有协议参数, 云端改 ρ 会被重算抓出)
        self._vcm_trust_ratio = float(vcm_trust_ratio)
        # edge_id -> 成功提交 checkpoint 入选累计次数 A_k (当前轮不提前计入)
        self._p2_participation_counts: dict[str, int] = {}

        # ---- 策略绑定: ES 独立持有的聚合策略协议参数 ----
        # 权重/补偿策略不允许由 Cloud 在 bundle 中自由声明: ES 只接受
        # "可由共识量 + 本地协议配置独立复算"的策略, 声明值与本地配置
        # 不一致一律拒绝 (fail-closed)。
        self._gamma_vcm = float(gamma_vcm)          # 本 checkpoint 生效的 γ
        self._vcm_mode = str(vcm_mode)
        self._momentum = float(momentum)
        self._staleness_metric = str(staleness_metric)   # version | rel2 | wait
        self._staleness_sigma = float(staleness_sigma)
        self._sigma_policy = str(sigma_policy)           # fixed | alignment
        self._alignment_policy = (AlignmentSigmaPolicy(
            sigma_fallback=staleness_sigma,
            sigma_high=sigma_alignment_high,
            ema_beta=sigma_alignment_ema,
            warmup=sigma_alignment_warmup,
            boost_threshold=sigma_alignment_boost_threshold,
            mid_threshold=sigma_alignment_threshold,
        ) if sigma_policy == 'alignment' else None)
        # 版本化协议配置: main/TA 预置同一实例并传入; 未传入时 (旧调用方)
        # 从散参数派生等价配置, digest 校验语义不变。配置由 TA/本地管理员
        # 预置, 不由 Cloud 下发; manifest 携带 digest, ES 校验声明 == 本地。
        if policy_config is None:
            policy_config = ProtocolConfig(
                participation_debias=participation_debias,
                sigma_target=sigma_target,
                staleness_sigma=float(staleness_sigma),
                staleness_metric=str(staleness_metric),
                sigma_policy=str(sigma_policy),
                sigma_alignment_threshold=float(sigma_alignment_threshold),
                sigma_alignment_ema=float(sigma_alignment_ema),
                sigma_alignment_warmup=int(sigma_alignment_warmup),
                sigma_alignment_boost_threshold=float(
                    sigma_alignment_boost_threshold),
                sigma_alignment_high=float(sigma_alignment_high),
                gamma_vcm=float(gamma_vcm), vcm_mode=str(vcm_mode),
                momentum=float(momentum),
                vcm_trust_ratio=float(vcm_trust_ratio),
                debias_kappa=float(debias_kappa), debias_min=float(debias_min),
                debias_max=float(debias_max), debias_scale=int(debias_scale),
            )
        self._policy_config = policy_config
        self._policy_digest = policy_config.digest()
        # 策略来源绑定: 本 run 的权重来源由协议配置唯一确定 (与云端同源
        # 接线), 不是云端可选项。ES 先核对声明来源 == 期望来源再复算 ——
        # 把预定来源降级为 none (静默退回等权) 或在合法规则间择优声明,
        # 都在来源核对即被拒。
        self._expected_weight_source = policy_config.expected_weight_source
        # 已提交共识版本 (由 commit 通知推进) 与本地已提交模型历史:
        # 绑定 bundle 中的 checkpoint_version, 防 Cloud 谎报版本以配合伪造权重
        self._p2_committed_version: int = 0
        self._p2_committed_history: dict[int, np.ndarray] = {}
        # Aσ 两阶段状态: 验证时暂存本 checkpoint 的 cos, 仅在 commit 通知
        # 到达时落地 (与云端"提交才落地"的语义一致, 攻击候选不推进状态)
        self._p2_pending_cos: Optional[float] = None
        # 最近一次验证通过的候选摘要 (结果摘要绑定, P0-1)
        self._p2_last_verified: Optional[dict] = None

        self._current_round = -1
        self._masked_gradients = []
        self._online_users = []

        # 1. 从 TA 获取 LHH 公共参数
        try:
            self.lhh_pp = ta.get_lhh_pp()
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 无法从 TA 获取 LHH 参数: {e}")
            sys.exit(1)

        # 2. 从 TA 注册并获取签名私钥
        try:
            self.my_sign_priv_key = ta.register_edge_server(self.id, require_verification=self.enable_verification)
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 无法从 TA 获取签名私钥: {e}")
            sys.exit(1)

        # 3. 获取所有 ES 的公钥
        try:
            self.all_es_pub_keys = ta.get_all_es_public_keys()
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 无法从 TA 获取公钥列表: {e}")
            sys.exit(1)

        # 4. TA receipt 验签公钥 (权威提交授权的签名方)
        self._ta_receipt_pub_key = ta.get_receipt_public_key()

        mode_str = "验证模式" if self.enable_verification else "快速实验模式"
        logger.info(f"EdgeServer {self.id} 已初始化 [{mode_str}]。")

    def start_new_round(self, round_t: int, base_model_hash: bytes = b''):
        self._current_round = round_t
        self._masked_gradients = []
        self._online_users = []
        # 本轮训练基模型的哈希 (签名元数据字段, 对齐 Sec.3.3.3 meta_k)
        self._p1_base_hash = base_model_hash

    def _decide_verification_level(self) -> Optional[VerificationLevel]:
        if self.rule_engine is None or not self.enable_verification:
            return None

        ctx = RoundContext(
            current_round=self._current_round,
            online_users=len(self._online_users),
            total_users=len(self._online_users),
            new_user_count=0,
            rounds_since_last_full_verify=(
                self._current_round - self._full_verify_rounds[-1]
                if self._full_verify_rounds else 999
            ),
        )

        agent_level = VerificationLevel.FULL
        if self.llm_reasoner:
            try:
                from llm_reasoner import DecisionType
                result = self.llm_reasoner.decide(
                    decision_type=DecisionType.COORDINATOR_REJECT_BATCH,
                    agent_id=self.id,
                    round_info={"round": self._current_round, "online": len(self._online_users)},
                )
                dec = str(result.decision).lower()
                if "skip" in dec:
                    agent_level = VerificationLevel.SKIP
                elif "light" in dec:
                    agent_level = VerificationLevel.LIGHT
            except Exception:
                pass

        suggestion = AgentSuggestion(
            type=SuggestionType.VERIFICATION_INTENSITY,
            decision=agent_level,
            reasoning=f"Agent suggested {agent_level}",
        )
        validated = self.rule_engine.validate(suggestion, ctx)
        return validated.final_decision

    def receive_masked_gradient(self, msg_bytes: bytes):
        try:
            data = pickle.loads(msg_bytes)
            user_id = data['user_id']
            round_t = data['round']
            gradient_int = data['masked_gradient']

            if round_t != self._current_round:
                return
            if user_id in self._online_users:
                return

            self._masked_gradients.append(gradient_int)
            self._online_users.append(user_id)

        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 处理接收到的梯度时出错: {e}")

    def trigger_aggregation_and_forward(self):
        if not self._online_users:
            logger.warning(f"EdgeServer {self.id} (R{self._current_round}): 无数据，跳过。")
            return

        # 1. 本地预聚合
        try:
            pre_aggregated_gradient = np.sum(np.array(self._masked_gradients), axis=0)
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 预聚合失败! {e}")
            return

        # 2. Agent 决策: 验证强度
        verif_level = self._decide_verification_level()

        if verif_level == VerificationLevel.FULL or (
            verif_level is None and self.enable_verification):
            h_local = LHH.Hash(self.lhh_pp, pre_aggregated_gradient)
            try:
                n_k = len(self._online_users)
                S_k = n_k  # RP1 等权: 每客户端权重 1 (对齐论文 Phase 3 meta_k)
                user_digest = hashlib.sha256(
                    ",".join(sorted(self._online_users)).encode()).digest()
                # 签名绑定全量元数据 meta_k = (edge_id, round, base_hash,
                # digest(U_online), n_k, S_k, H_k)，对齐 Sec.3.3.3。
                meta_k = (self.id, self._current_round, self._p1_base_hash,
                          user_digest, n_k, S_k, h_local)
                signature = SIG.sign(pickle.dumps(meta_k), self.my_sign_priv_key)
            except Exception as e:
                logger.error(f"EdgeServer {self.id}: 签名失败: {e}")
                return
        elif verif_level == VerificationLevel.LIGHT:
            h_local = None
            sig_msg = pickle.dumps((self.id, self._current_round, pre_aggregated_gradient[:10]))
            try:
                signature = SIG.sign(sig_msg, self.my_sign_priv_key)
            except Exception as e:
                logger.error(f"EdgeServer {self.id}: LIGHT签名失败: {e}")
                return
        else:
            h_local = None
            signature = b'accuracy_only_fast_mode'

        if verif_level == VerificationLevel.FULL:
            self._full_verify_rounds.append(self._current_round)

        # 3. 发送给 CloudServer (附带签名元数据字段, 供 ES 验证时重建 meta_k)
        cs_msg = {
            "edge_id": self.id,
            "round": self._current_round,
            "pre_aggregated_gradient": pre_aggregated_gradient,
            "lhh_hash": h_local,
            "signature": signature,
            "online_users": list(self._online_users),
            "base_hash": self._p1_base_hash,
        }
        try:
            self.cloud_server.receive_pre_aggregation(pickle.dumps(cs_msg))
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 发送至 CloudServer 失败: {e}")

        # 4. 发送在线用户列表给 TA
        ta_msg = {
            "edge_id": self.id,
            "round": self._current_round,
            "online_users": self._online_users
        }
        try:
            self.ta.receive_online_list(pickle.dumps(ta_msg))
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 发送至 TA 失败: {e}")

    def verify_global_result(self,
                             global_plaintext_model_int: Union[np.ndarray, None],
                             es_hash_list: Union[list, None]) -> bool:
        if not self.enable_verification:
            return True

        if global_plaintext_model_int is None or es_hash_list is None:
            logger.error(f"EdgeServer {self.id}: 验证失败! 收到空结果。")
            return False

        # The protocol wire domain is a signed 64-bit vector of the public
        # dimension. LHH.Hash alone also accepts floats (via int truncation),
        # flattened matrices and arbitrary Python integers. Those representations
        # must not reach verification/model reconstruction: adding the EC group
        # order to an object-array coordinate preserves its hash but changes the
        # reconstructed model. The int64 domain is much smaller than that order.
        public_dim = getattr(self.lhh_pp, 'dim', None)
        if public_dim is None and isinstance(self.lhh_pp, list):
            public_dim = len(self.lhh_pp)
        if (not isinstance(global_plaintext_model_int, np.ndarray)
                or global_plaintext_model_int.dtype.kind != 'i'
                or global_plaintext_model_int.dtype.itemsize != 8
                or global_plaintext_model_int.shape != (public_dim,)):
            logger.error(f"EdgeServer {self.id}: 聚合结果必须为公开维度的一维 int64 向量")
            return False

        # 1. 刷新 ES 公钥列表
        try:
            self.all_es_pub_keys = self.ta.get_all_es_public_keys()
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 无法获取本轮 ES 验证名单: {e}")
            return False

        # RP1 is synchronous: the evidence must cover the full registered ES
        # roster exactly once. Signatures authenticate individual entries but
        # do not establish completeness or prevent duplicating a valid entry.
        # Without this check the cloud could remove/add H_k and simultaneously
        # subtract/add its known masked group sum, preserving the LHH equation.
        try:
            expected_edges = set(self.all_es_pub_keys)
            evidence_edges = [item['edge_id'] for item in es_hash_list]
            if (not expected_edges or len(evidence_edges) != len(expected_edges)
                    or set(evidence_edges) != expected_edges):
                logger.error(f"EdgeServer {self.id}: ES 证据名单缺失、重复或含未知组")
                return False
        except (TypeError, KeyError):
            logger.error(f"EdgeServer {self.id}: ES 证据名单格式错误")
            return False

        # 2. 验证签名并自行聚合哈希 (RHS)
        aggregated_h_all_rhs = None

        for item in es_hash_list:
            try:
                eid = item['edge_id']
                h_val = item['lhh_hash']
                sig = item['signature']

                if eid not in self.all_es_pub_keys:
                    logger.error(f"EdgeServer {self.id}: 未知 EdgeServer {eid}")
                    return False

                pub_key = self.all_es_pub_keys[eid]
                # 显式交叉核对 round / base_hash (对齐点2验证器; 签名兜底)
                online_users = item.get('online_users', [])
                rnd = item.get('round', self._current_round)
                base_hash = item.get('base_hash', b'')
                if rnd != self._current_round:
                    logger.error(f"EdgeServer {self.id}: {eid} round 不匹配 "
                                 f"(got {rnd}, expect {self._current_round})")
                    return False
                if base_hash != self._p1_base_hash:
                    logger.error(f"EdgeServer {self.id}: {eid} base_hash 不匹配")
                    return False
                # 重建被签名的 meta_k = (edge_id, round, base_hash,
                # digest(U_online), n_k, S_k, H_k)，对齐 Sec.3.3.3 / Sec.3.3.6 Phase 5a。
                n_k = len(online_users)
                S_k = n_k  # RP1 等权
                user_digest = hashlib.sha256(
                    ",".join(sorted(online_users)).encode()).digest()
                meta_k = (eid, rnd, base_hash, user_digest, n_k, S_k, h_val)
                h_bytes = pickle.dumps(meta_k)

                if not SIG.verify(h_bytes, sig, pub_key):
                    logger.error(f"EdgeServer {self.id}: EdgeServer {eid} 签名无效！")
                    return False

                if aggregated_h_all_rhs is None:
                    aggregated_h_all_rhs = h_val
                else:
                    aggregated_h_all_rhs = LHH.Mul(aggregated_h_all_rhs, h_val)

            except Exception as e:
                logger.error(f"EdgeServer {self.id}: 验证出错: {e}")
                return False

        # 3. 从 TA 获取 H_mask
        try:
            _, h_mask = self.ta.get_verification_info()
            if h_mask is None:
                raise ValueError("TA 未完成 H_mask 计算")
        except Exception as e:
            logger.error(f"EdgeServer {self.id}: 无法获取 H_mask: {e}")
            return False

        # 4. 计算 G_sum 的哈希
        h_plain = LHH.Hash(self.lhh_pp, global_plaintext_model_int)

        # 5. 计算 LHS = H_sum * H_mask (LHH 同态性: H(x)·H(y)=H(x+y))
        lhs = LHH.Mul(h_plain, h_mask)

        # 6. 最终比较
        if lhs == aggregated_h_all_rhs:
            return True
        else:
            logger.error(f"EdgeServer {self.id}: 哈希不匹配 (LHS != RHS)。")
            return False

    # ============================================================
    #  研究点2: buffer/checkpoint 半异步方法
    # ============================================================

    def p2_start_package(self, base_version: int, base_hash: bytes,
                         package_token: str):
        """point2: 启动一组新的组包训练.

        Args:
            base_version: 本次组包基于的全局模型版本 r_k
            base_hash: h_base(r_k)
            package_token: 唯一标识 e.g. "edge_0:v0:p1"
        """
        self._p2_base_version = base_version
        self._p2_base_hash = base_hash
        self._p2_package_token = package_token
        self._p2_online_users: list[str] = []
        self._p2_masked_gradients: list[np.ndarray] = []
        self._p2_user_samples: dict[str, int] = {}    # user_id → s_i
        self._p2_user_norms: dict[str, float] = {}    # user_id → ‖Δw_i‖ (矩校准元数据)
        self._p2_ready = False

    def p2_receive_masked_gradient(self, msg_bytes: bytes, package_token: str,
                                    base_version: int, sample_size: int = 0,
                                    upd_norm: float = 0.0):
        """point2: 接收用户上传，校验 package_token 和 base_version."""
        try:
            data = pickle.loads(msg_bytes)
            user_id = data['user_id']
            gradient_int = data['masked_gradient']

            if package_token != self._p2_package_token:
                return
            if base_version != self._p2_base_version:
                return
            if user_id in self._p2_online_users:
                return

            self._p2_masked_gradients.append(gradient_int)
            self._p2_online_users.append(user_id)
            self._p2_user_samples[user_id] = sample_size
            self._p2_user_norms[user_id] = float(upd_norm)

        except Exception as e:
            logger.error(f"EdgeServer {self.id}: p2接收梯度出错: {e}")

    def p2_seal_group_package(self, tick: int = 0) -> Optional['GroupPackage']:
        """point2: 封组，生成 GroupPackage (Π_k).

        对应文档 Sec.4.2: 聚合掩码和 → LHH → 签名 → 上报 TA.
        n_min 隐私前提: 组内在线 < n_min 时不封组 (Sec.3.4.1 Lemma 3.2).
        """
        from point2_types import GroupPackage

        n_k = len(self._p2_online_users)
        if n_k < self.n_min:
            logger.warning(f"EdgeServer {self.id}: p2 组内在线 {n_k} < n_min={self.n_min}, 不封组")
            return None

        pre_aggregated = np.sum(np.array(self._p2_masked_gradients), axis=0)
        # 性能模式 (--verify 关) 跳过 LHH: H_k 仅被 ES 验证消费,
        # 攻击实验必须开 --verify, 此时与完整验证行为一致
        H_k = (LHH.Hash(self.lhh_pp, pre_aggregated)
               if self.enable_verification else None)

        S_k = sum(self._p2_user_samples.values())

        # 组级一阶矩 m̂_k = Σ s_i·‖Δw_i‖ / S_k (样本量加权平均本地更新范数;
        # 仅标量元数据, 不泄露单用户方向, 与掩码域聚合兼容)
        if S_k > 0 and self._p2_user_norms:
            upd_norm = sum(s_i * self._p2_user_norms.get(u, 0.0)
                           for u, s_i in self._p2_user_samples.items()) / S_k
        else:
            upd_norm = 0.0

        # 签名: (edge_id, package_token, base_hash, digest(users), n_k, S_k, H_k)
        import hashlib
        user_digest = hashlib.sha256(
            ",".join(sorted(self._p2_online_users)).encode()
        ).digest()
        sig_msg = pickle.dumps((
            self.id, self._p2_package_token, self._p2_base_hash,
            user_digest, n_k, S_k, H_k
        ))
        assert isinstance(self.my_sign_priv_key, bytes), \
            f"sign_priv_key type={type(self.my_sign_priv_key)}"
        signature = SIG.sign(sig_msg, self.my_sign_priv_key)

        # 上报 TA
        self.ta.p2_receive_package_online_list(
            package_token=self._p2_package_token,
            edge_id=self.id,
            base_version=self._p2_base_version,
            online_users=self._p2_online_users,
        )
        # 向 TA 权威 ledger 注册 token (合法 token 登记, 授权时校验前置)
        self.ta.p2_register_package_token(
            package_token=self._p2_package_token, edge_id=self.id,
            base_version=self._p2_base_version)

        pkg = GroupPackage(
            edge_id=self.id,
            package_token=self._p2_package_token,
            base_version=self._p2_base_version,
            base_hash=self._p2_base_hash,
            online_users=list(self._p2_online_users),
            n_k=n_k,
            S_k=S_k,
            masked_sum_int=pre_aggregated,
            H_k=H_k,
            signature=signature,
            created_tick=tick,
            upd_norm=upd_norm,
        )

        logger.info(f"EdgeServer {self.id}: p2组包 {self._p2_package_token} "
                     f"已封装 n={n_k} S={S_k} base_v={self._p2_base_version}")
        return pkg

    def p2_note_committed_checkpoint(self, selected_packages: list,
                                     committed_version: int = None,
                                     final_model: np.ndarray = None,
                                     receipt=None) -> None:
        """commit 成功后由协调器通知: 校验 TA receipt 并推进共识状态.

        只记录真正成功提交的 checkpoint; 验证失败/commit 被拒的候选不计入。
        ES **只有看到有效 CheckpointReceipt 才推进**版本与策略状态:
        - 通知版本 ≤ 本地已提交版本 → 幂等跳过 (崩溃恢复重发通知不重复
          计入参与历史/Aσ 状态);
        - receipt 为 None (兼容路径: 单测/verify-off) → 跳过验签;
        - receipt 非 None → 验证 TA 签名、token/版本匹配, 且 committed
          版本必须严格等于本地已提交版本 +1 (拒绝重放通知与跳版本);
        - 推进内容: 参与历史 A_k (权重独立复算依据)、已提交模型历史
          (版本绑定)、Aσ 策略状态 (两阶段落地)。
        """
        if (committed_version is not None
                and int(committed_version) <= self._p2_committed_version):
            logger.info(f"EdgeServer {self.id}: commit 通知 "
                        f"v{committed_version} ≤ 已提交 "
                        f"v{self._p2_committed_version}, 幂等跳过")
            return
        if receipt is not None:
            payload = bytes.fromhex(receipt.payload_digest())
            if not SIG.verify(payload, receipt.signature,
                              self._ta_receipt_pub_key):
                logger.error(f"EdgeServer {self.id}: receipt 签名无效 "
                             f"({receipt.receipt_id}) → 不推进状态")
                return
            if receipt.checkpoint_version + 1 != int(committed_version or -1):
                logger.error(f"EdgeServer {self.id}: receipt 版本不匹配 "
                             f"(receipt v{receipt.checkpoint_version}, "
                             f"通知 v{committed_version}) → 不推进状态")
                return
            if tokens_digest([p.package_token for p in selected_packages]) \
                    != receipt.token_digest:
                logger.error(f"EdgeServer {self.id}: receipt token 集合与"
                             f"通知不符 → 不推进状态")
                return
            # 结果摘要绑定 (P0-1): receipt 必须对应本地实际验证过的候选,
            # 且通知的 final_model 与验证时一致 —— 偷换模型在此被拒绝
            lv = self._p2_last_verified
            if (lv is None or lv.get('v') != receipt.checkpoint_version
                    or lv.get('result_digest') != receipt.result_digest):
                logger.error(f"EdgeServer {self.id}: receipt 结果摘要与本地"
                             f"已验证候选不符 → 不推进状态")
                return
            if (final_model is not None
                    and array_digest(final_model) != lv.get('final_digest')):
                logger.error(f"EdgeServer {self.id}: 通知模型与已验证模型"
                             f"不符 → 不推进状态")
                return
            # 权威确认: 核对完整 receipt (签名 + prepare 记录 + 已 finalize),
            # 只查 receipt_id 无法发现"同 id 不同内容"的伪造/replay
            if not self.ta.p2_confirm_receipt(receipt):
                logger.error(f"EdgeServer {self.id}: receipt 未经 TA ledger "
                             f"最终确认 → 不推进状态")
                return
        if committed_version is not None:
            if int(committed_version) != self._p2_committed_version + 1:
                logger.error(
                    f"EdgeServer {self.id}: 通知版本 {committed_version} "
                    f"非本地已提交版本 {self._p2_committed_version} +1 "
                    f"(重放/跳版本) → 不推进状态")
                return
        for pkg in selected_packages:
            self._p2_participation_counts[pkg.edge_id] = (
                self._p2_participation_counts.get(pkg.edge_id, 0) + 1)
        if committed_version is not None:
            self._p2_committed_version = int(committed_version)
            if final_model is not None:
                self._p2_committed_history[int(committed_version)] = \
                    np.asarray(final_model, dtype=np.float64).copy()
        if self._alignment_policy is not None:
            self._alignment_policy.commit_observation(self._p2_pending_cos)
            self._p2_pending_cos = None

    def p2_vote_checkpoint_with_result(
            self, bundle: dict) -> tuple[CheckpointVote, VerificationResult]:
        """只验证一次，并同时返回签名 vote 与详细诊断结果。

        vote 载荷 = (voter, checkpoint_version, token_digest,
        config_digest, passed, reason); 验证失败也生成 passed=False 的
        vote (携带失败阶段/原因), 由 TA 统一拒绝授权。
        """
        res = self.p2_verify_checkpoint_detailed(bundle)
        tokens = [p.package_token for p in bundle.get('selected_packages', [])]
        v = bundle.get('checkpoint_version', -1)
        tdig = tokens_digest(tokens)
        # 结果摘要绑定 (P0-1): ES 把"实际验证过的聚合结果"钉进 vote,
        # 使 vote/receipt 不能被移植到另一个候选 (如偷换 final_model)。
        rdig = result_digest(v, tdig, bundle.get('raw_sum_int'),
                             bundle.get('M_Q'), bundle.get('final_model'))
        if res.passed:
            self._p2_last_verified = {
                'v': v, 'result_digest': rdig,
                'final_digest': array_digest(bundle.get('final_model'))}
        vote = CheckpointVote(
            voter=self.id, checkpoint_version=v, token_digest=tdig,
            config_digest=self._policy_digest, passed=res.passed,
            reason=res.reason, result_digest=rdig)
        vote = replace(vote, signature=SIG.sign(
            bytes.fromhex(vote.payload_digest()), self.my_sign_priv_key))
        return vote, res

    def p2_vote_checkpoint(self, bundle: dict) -> CheckpointVote:
        """兼容接口：验证 bundle 并返回带签名的 CheckpointVote。"""
        vote, _ = self.p2_vote_checkpoint_with_result(bundle)
        return vote

    def _p2_recompute_debias_weights(self, pkgs: list) -> list:
        """按本地公开参与历史独立重算 V-PDA 整数权重 (拒绝云端伪造权重).

        均值在**全部组**上取 (协议口径: 公平基准是全体组, 非选中的子集);
        ES 从 TA 注册的 ES 公钥列表获知全体组, 用本地累计的 A_k 独立重算。
        """
        counts = [self._p2_participation_counts.get(p.edge_id, 0) for p in pkgs]
        all_counts = [self._p2_participation_counts.get(eid, 0)
                      for eid in self.all_es_pub_keys]
        _, ws = vpda_weights(counts, kappa=self._debias_kappa,
                             wmin=self._debias_min, wmax=self._debias_max,
                             scale=self._debias_scale, all_counts=all_counts)
        return ws

    def p2_verify_checkpoint(self, bundle: dict) -> bool:
        """布尔兼容包装; 实验入口使用 p2_verify_checkpoint_detailed."""
        return self.p2_verify_checkpoint_detailed(bundle).passed

    def p2_verify_checkpoint_detailed(self, bundle: dict) -> VerificationResult:
        """point2: 验证 checkpoint 结果.

        对应文档 Sec.4.6 (含策略绑定扩展):
        1. 版本绑定: bundle 声明的 checkpoint 版本 == ES 本地已提交共识版本
        2. 验签名
        3. 验证 LHH: H_sum^raw * H_mask^raw = ∏ H_k^{w_k}
        4. 策略绑定: 权重 (V-PDA/σ) 与补偿系数 α 均由 ES 依据本地协议配置
           + 公开共识量独立复算, 云端声明值不一致即拒绝 —— 恶意云端不能
           自行选择权重, 也不能把伪造权重改标为其他来源绕过复算; 权重
           来源本身亦由 ES 本地协议配置唯一确定 (来源绑定), 降级为 none
           或在合法来源间择优即拒
        5. 公开重算补偿后最终模型

        Args:
            bundle: {
                'checkpoint_version': int,
                'selected_packages': list[GroupPackage],
                'raw_sum_int': np.ndarray,
                'M_Q': np.ndarray,
                'H_mask_raw': object,
                'final_model': np.ndarray,
                'gamma_vcm': float,
                'vcm_mode': str,
                'momentum': float,
                'global_model_history': dict[int, np.ndarray],
                'weights': list[int] | None,   # V-PDA/σ-agg 整数权重
                'weight_source': str | None,   # none | vpda | staleness
                'alphas': list[float] | None,  # VCM 补偿衰减系数
            }
        """
        if not self.enable_verification:
            return True

        # Aσ 两阶段: 清掉上一轮遗留的暂存 cos (验证失败/未提交的候选
        # 不应推进策略状态)
        self._p2_pending_cos = None

        pkgs = bundle['selected_packages']
        raw_sum_int = bundle['raw_sum_int']
        H_mask_raw = bundle['H_mask_raw']
        final_model = bundle['final_model']
        history = bundle.get('global_model_history', {})

        # 0a. 配置摘要绑定 (fail-closed): manifest 必须携带协议配置 digest,
        #     且与 ES 本地预置配置一致 —— 缺失或错配都说明云端在另选策略
        declared_digest = bundle.get('policy_config_digest')
        if declared_digest != self._policy_digest:
            logger.error(f"EdgeServer {self.id}: policy_config_digest 缺失"
                         f"/错配 (声明={str(declared_digest)[:16]}…, 本地="
                         f"{self._policy_digest[:16]}…) → 拒绝")
            return VerificationResult(
                False, 'config_digest',
                f"policy_config_digest 缺失/错配 "
                f"(声明={str(declared_digest)[:16]}, 本地={self._policy_digest[:16]})")

        # 0b. 版本绑定: bundle 声明的 checkpoint 版本必须等于 ES 本地已提交
        #    共识版本 (由 commit 通知推进)。谎报版本会同时改变 σ 权重与
        #    VCM 补偿的取值, 必须先钉死。
        v = bundle['checkpoint_version']
        if v != self._p2_committed_version:
            logger.error(f"EdgeServer {self.id}: checkpoint 版本错位 "
                         f"(声明 v{v}, 本地已提交 v{self._p2_committed_version})")
            return VerificationResult(
                False, 'version',
                f"checkpoint 版本错位 (声明 v{v}, 本地已提交 "
                f"v{self._p2_committed_version})")
        # W(v) 与 ES 本地已提交模型比对 (v=0 为初始模型, 不在本地历史内;
        # W(r_k) 已由组包 base_hash 签名绑定, 见 Step 1)。v>0 时 bundle
        # 必须提供 history[v] 且与本地一致 —— 省略该条目等价于静默关闭
        # VCM 补偿策略, 与省略 alphas 同型, 一律 fail-closed。
        if v > 0:
            local_wv = self._p2_committed_history.get(v)
            if v not in history or local_wv is None:
                logger.error(f"EdgeServer {self.id}: bundle 缺少共识历史 "
                             f"W({v}) 或本地未持有该版本 → fail-closed")
                return VerificationResult(
                    False, 'history', f"bundle 缺少共识历史 W({v})")
            if not np.allclose(np.asarray(history[v], dtype=np.float64),
                               local_wv, rtol=1e-12, atol=0.0):
                logger.error(f"EdgeServer {self.id}: 声明历史 W({v}) 与本地"
                             f"共识模型不一致")
                return VerificationResult(
                    False, 'history', f"声明历史 W({v}) 与本地共识模型不一致")

        # 声明的策略参数必须与 ES 本地协议配置一致 (fail-closed):
        # γ/momentum 直接进入最终模型重算, 声明不同值等于另选聚合策略。
        declared_gamma = bundle.get('gamma_vcm')
        if declared_gamma is None or not math.isclose(
                float(declared_gamma), self._gamma_vcm,
                rel_tol=1e-12, abs_tol=1e-12):
            logger.error(f"EdgeServer {self.id}: 声明 γ_vcm={declared_gamma} "
                         f"与协议配置 {self._gamma_vcm} 不符")
            return VerificationResult(
                False, 'policy_decl',
                f"声明 γ_vcm={declared_gamma} 与协议配置 {self._gamma_vcm} 不符")
        if self._gamma_vcm > 0:
            declared_mode = bundle.get('vcm_mode', 'full')
            if declared_mode != self._vcm_mode:
                logger.error(f"EdgeServer {self.id}: 声明 vcm_mode="
                             f"{declared_mode} 与协议配置 {self._vcm_mode} 不符")
                return VerificationResult(
                    False, 'policy_decl',
                    f"声明 vcm_mode={declared_mode} 与协议配置不符")
        declared_momentum = float(bundle.get('momentum', 0.0))
        if not math.isclose(declared_momentum, self._momentum,
                            rel_tol=1e-12, abs_tol=1e-12):
            logger.error(f"EdgeServer {self.id}: 声明 momentum="
                         f"{declared_momentum} 与协议配置 {self._momentum} 不符")
            return VerificationResult(
                False, 'policy_decl',
                f"声明 momentum={declared_momentum} 与协议配置不符")

        # 策略绑定: 独立复算补偿系数 alphas (策略规则见
        # _p2_recompute_policy_alphas) 与权重, 之后与云端声明值比对。
        # ok=False = 策略不可复算 (wait / 共识历史缺失) → fail-closed:
        # 云端省略 alphas 不能把衰减策略静默关闭。
        alphas_ok, alphas = self._p2_recompute_policy_alphas(pkgs, v, history,
                                                             raw_sum_int)
        if not alphas_ok:
            return VerificationResult(
                False, 'policy_alpha',
                "本地策略不可复算 α (wait 度量/共识历史缺失) → fail-closed")
        alphas_declared = bundle.get('alphas')
        if alphas is None:
            if alphas_declared is not None:
                logger.error(f"EdgeServer {self.id}: 本地策略 (σ="
                             f"{self._staleness_sigma}, "
                             f"{self._sigma_policy}) 不产生 α, "
                             f"云端却提供 alphas → 拒绝")
                return VerificationResult(
                    False, 'policy_alpha', "本地策略不产生 α, 云端却提供 alphas")
        else:
            if (alphas_declared is None or len(alphas_declared) != len(alphas)
                    or not np.allclose(alphas_declared, alphas,
                                       rtol=1e-9, atol=1e-12)):
                logger.error(
                    f"EdgeServer {self.id}: 云端 alphas 与公开策略规则复算值"
                    f"不一致 (云端伪造 α/σ*? local={alphas} "
                    f"cloud={alphas_declared})")
                return VerificationResult(
                    False, 'policy_alpha',
                    "云端 alphas 与公开策略规则复算值不一致")

        # 权重来源校验 (weight_source = none | vpda | staleness):
        #  0. 来源绑定: 声明来源必须等于本地协议配置唯一确定的来源 ——
        #     恶意云端不得把预定的 vpda/staleness 降级为 none (静默退回
        #     等权聚合), 也不得在两条合法规则之间择优声明
        #  1. vpda: ES 按完整公开参与历史独立重算, 与云端权重比对
        #  2. staleness: ES 按公开陈旧度规则 (版本差/参数漂移, σ 为本地协议
        #     配置) 独立重算整数权重 —— 云端伪造权重或把 V-PDA 权重改标为
        #     staleness 都会在此被拒
        #  3. 权重存在但来源缺失/非法, 或声明来源却无权重 → 一律拒绝
        weights = bundle.get('weights')
        src = bundle.get('weight_source')
        declared_src = src if src is not None else 'none'
        if declared_src != self._expected_weight_source:
            logger.error(
                f"EdgeServer {self.id}: 声明 weight_source={declared_src} "
                f"与协议配置确定的来源 {self._expected_weight_source} 不符 "
                f"(降级/择源攻击?) → 拒绝")
            return VerificationResult(
                False, 'policy_source',
                f"声明 weight_source={declared_src} 与协议配置确定的来源 "
                f"{self._expected_weight_source} 不符")
        if weights is None:
            if src not in (None, 'none'):
                logger.error(f"EdgeServer {self.id}: 声明 weight_source={src} "
                             f"但未提供权重 → 拒绝")
                return VerificationResult(
                    False, 'policy_source',
                    f"声明 weight_source={src} 但未提供权重")
            ws = [1] * len(pkgs)
        else:
            if src not in ('vpda', 'staleness'):
                logger.error(f"EdgeServer {self.id}: 提供了权重但 "
                             f"weight_source={src} 缺失/非法 → 拒绝")
                return VerificationResult(
                    False, 'policy_source',
                    f"提供了权重但 weight_source={src} 缺失/非法")
            ws = [int(w) for w in weights]
            if any(w <= 0 for w in ws):
                logger.error(f"EdgeServer {self.id}: 非法非正权重 {ws} → 拒绝")
                return VerificationResult(
                    False, 'policy_weights', f"非法非正权重 {ws}")
            if src == 'vpda':
                local_ws = self._p2_recompute_debias_weights(pkgs)
                if local_ws != ws:
                    logger.error(
                        f"EdgeServer {self.id}: V-PDA 权重与公开参与历史不匹配 "
                        f"(云端伪造权重? local={local_ws} cloud={ws})"
                    )
                    return VerificationResult(
                        False, 'policy_weights',
                        "V-PDA 权重与公开参与历史不匹配")
            else:  # staleness
                local_ws = self._p2_recompute_staleness_weights(pkgs, v,
                                                                history)
                if local_ws is None or local_ws != ws:
                    logger.error(
                        f"EdgeServer {self.id}: σ 陈旧度权重与公开规则复算值"
                        f"不匹配 (云端伪造/改标权重? local={local_ws} "
                        f"cloud={ws})")
                    return VerificationResult(
                        False, 'policy_weights',
                        "σ 陈旧度权重与公开规则复算值不匹配")

        # 1. 验签名 + 聚合 RHS
        try:
            self.all_es_pub_keys = self.ta.get_all_es_public_keys()
        except Exception:
            pass

        import hashlib
        aggregated_rhs = None
        for j, pkg in enumerate(pkgs):
            if pkg.edge_id not in self.all_es_pub_keys:
                logger.error(f"EdgeServer {self.id}: 未知 {pkg.edge_id}")
                return VerificationResult(
                    False, 'signature', f"未知 ES {pkg.edge_id}")

            # base_hash 负责把组包绑定到其训练基版本；签名只能证明
            # “该 Edge 签过这个值”，还需要检查它确实对应历史模型。
            base_model = history.get(pkg.base_version)
            if base_model is None:
                logger.error(f"EdgeServer {self.id}: 缺少 base version v{pkg.base_version}")
                return VerificationResult(
                    False, 'base_hash',
                    f"缺少 base version v{pkg.base_version} 的共识历史")
            expected_base_hash = hashlib.sha256(
                np.asarray(base_model, dtype=np.float64).tobytes()
            ).digest()
            if pkg.base_hash != expected_base_hash:
                logger.error(
                    f"EdgeServer {self.id}: {pkg.package_token} base_hash 不匹配"
                )
                return VerificationResult(
                    False, 'base_hash',
                    f"{pkg.package_token} base_hash 与历史模型不符")

            user_digest = hashlib.sha256(
                ",".join(sorted(pkg.online_users)).encode()
            ).digest()
            sig_msg = pickle.dumps((
                pkg.edge_id, pkg.package_token, pkg.base_hash,
                user_digest, pkg.n_k, pkg.S_k, pkg.H_k
            ))
            if not SIG.verify(sig_msg, pkg.signature,
                              self.all_es_pub_keys[pkg.edge_id]):
                logger.error(f"EdgeServer {self.id}: {pkg.edge_id} 签名无效")
                return VerificationResult(
                    False, 'signature', f"{pkg.edge_id} 组包签名无效")

            # 加权验证: RHS = ∏ H_k^{w_k}; 等权时 w_k=1 退化为 ∏ H_k
            term = (LHH.Exp(pkg.H_k, ws[j]) if weights is not None else pkg.H_k)
            aggregated_rhs = (term if aggregated_rhs is None
                              else LHH.Mul(aggregated_rhs, term))

        # 2. LHH 验证 raw sum (加权: H(Σ w_k·W̄_k) == ∏ H_k^{w_k})
        H_sum_raw = LHH.Hash(self.lhh_pp, raw_sum_int)
        lhs = LHH.Mul(H_sum_raw, H_mask_raw)
        if lhs != aggregated_rhs:
            logger.error(f"EdgeServer {self.id}: p2 LHH mismatch")
            return VerificationResult(False, 'lhh', "LHH 等式不成立 (聚合/掩码不一致)")

        # 3. 公开规则重算补偿后最终模型 (加权: denom=Σ w_k·S_k,
        #    C_k = w_k·α_k·S_k·γ·(W(v)−W(r_k)), 等权时与 G5 完全一致;
        #    vcm_mode=trust 时按 UR-VCM 可信域门控 η 重算)。
        #    γ/mode/α 一律使用 ES 本地配置与独立复算值 (已与声明值比对),
        #    不消费 bundle 中的策略参数。
        denom = sum(ws[j] * pkg.S_k for j, pkg in enumerate(pkgs))
        denom = max(1, denom)
        vcm_mode = self._vcm_mode
        raw_float = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE
        raw_norm = np.linalg.norm(raw_float)

        total_comp = np.zeros_like(raw_float)
        for j, pkg in enumerate(pkgs):
            r_k = pkg.base_version
            if r_k < v and r_k in history:
                G_k = history[v].astype(np.float64) - history[r_k].astype(np.float64)
                if vcm_mode == 'span' and (v - r_k) > 0:
                    G_k = G_k / (v - r_k)
                alpha = 1.0 if alphas is None else alphas[j]
                C_k = ws[j] * alpha * pkg.S_k * self._gamma_vcm * G_k
                if vcm_mode == 'clip' and raw_norm > 0:
                    max_norm = 0.1 * raw_norm
                    c_norm = np.linalg.norm(C_k)
                    if c_norm > max_norm:
                        C_k = C_k * (max_norm / c_norm)
                total_comp += C_k

        if vcm_mode == 'trust':
            # UR-VCM: R=原始聚合均值, B=所选组基模型样本量加权均值,
            # Δtrain=R−B, C̄=原始补偿均值, η=min(1, ρ‖Δtrain‖/(‖C̄‖+ε))
            R = raw_float / denom
            base_num = np.zeros_like(R)
            for j, pkg in enumerate(pkgs):
                W_rk = history[pkg.base_version]
                base_num += ws[j] * pkg.S_k * np.asarray(W_rk, dtype=np.float64)
            B = base_num / denom
            C_bar = total_comp / denom
            d_norm = float(np.linalg.norm(R - B))
            c_norm = float(np.linalg.norm(C_bar))
            eta = (min(1.0, self._vcm_trust_ratio * d_norm / (c_norm + 1e-12))
                   if c_norm > 0 else 1.0)
            expected_final = R + eta * C_bar
        else:
            expected_final = (raw_float + total_comp) / denom
        momentum = self._momentum
        if momentum > 0:
            W_v = history.get(v)
            if W_v is not None:
                expected_final = (momentum * W_v.astype(np.float64)
                                  + (1.0 - momentum) * expected_final)
        if not np.allclose(expected_final, final_model, rtol=1e-4, atol=1e-6):
            logger.error(f"EdgeServer {self.id}: p2 最终模型重算不匹配")
            return VerificationResult(False, 'reconstruct', "最终模型公开重算不匹配")

        res = VerificationResult(True, "ok", "p2 checkpoint 验证通过"); _ = logger.info(f"EdgeServer {self.id}: p2 checkpoint {v} 验证通过")
        return res
        return True

    def _p2_recompute_policy_alphas(self, pkgs: list, v: int,
                                    history: dict, raw_sum_int: np.ndarray
                                    ) -> tuple[bool, Optional[list]]:
        """按本地协议配置独立复算 VCM 补偿系数 α_k (策略绑定核心).

        返回 (ok, alphas), ok=False 表示"策略不可复算"——与"合法策略
        不产生 α"严格区分, 两者绝不能共用同一返回值 (否则云端在 wait 等
        不可复算度量下省略 alphas 即可把衰减策略静默关闭):

        - ok=True, alphas=None: 合法无衰减 (σ=0 等权补偿), 云端也不应
          提供 alphas
        - ok=True, alphas=list: 固定策略 σ>0 或 alignment 策略的复算值
        - ok=False: wait 度量 (含云端调度状态, 不可复算) 或 rel2 所需
          共识历史缺失 → fail-closed, 无论云端是否提供 alphas 一律拒绝

        alignment 策略由公开量复算 cos(C^(0), Δtrain), 经两阶段
        AlignmentSigmaPolicy 得 σ_eff, 再按 metric 复算 α_k。
        """
        if self._alignment_policy is not None:
            cos = self._p2_alignment_cos(pkgs, v, history, raw_sum_int)
            self._p2_pending_cos = cos
            sigma_eff = self._alignment_policy.tentative_eff(cos)
        elif self._staleness_sigma > 0:
            sigma_eff = self._staleness_sigma
        else:
            return True, None

        alphas = []
        for pkg in pkgs:
            s_k = v - pkg.base_version
            if self._staleness_metric == 'rel2':
                Wv = history.get(v)
                Wr = history.get(pkg.base_version)
                W0 = history.get(0)
                if Wv is None or Wr is None or W0 is None:
                    logger.error(f"EdgeServer {self.id}: rel2 度量缺少共识"
                                 f"历史 (v={v}, r_k={pkg.base_version}) → "
                                 f"策略不可复算, fail-closed")
                    return False, None
                dist = (float(np.linalg.norm(np.asarray(Wv, dtype=np.float64)
                                             - np.asarray(Wr, dtype=np.float64)))
                        / (float(np.linalg.norm(np.asarray(Wv, dtype=np.float64)
                                                - np.asarray(W0, dtype=np.float64)))
                           + 1e-12))
                alphas.append(staleness_alpha('rel2', s_k, sigma_eff, dist))
            elif self._staleness_metric == 'wait':
                logger.error(f"EdgeServer {self.id}: wait 陈旧度度量含云端"
                             f"调度状态, ES 无法独立复算 → 拒绝")
                return False, None
            else:
                alphas.append(staleness_alpha('version', s_k, sigma_eff))
        return True, alphas

    def _p2_alignment_cos(self, pkgs: list, v: int, history: dict,
                          raw_sum_int: np.ndarray) -> Optional[float]:
        """由 bundle 公开量复算 Aσ 方向一致性 cos(C^(0), Δtrain).

        与 main_point2 云端探测同一公式: R=原始聚合均值, B=基模型样本量
        加权均值, C^(0)=未衰减 Direct-VCM 补偿均值; cos 无效 (零补偿或零
        训练增量) 时返回 None (不推进策略状态)。
        """
        Wv = history.get(v)
        if Wv is None:
            return None
        S_policy = max(1, sum(pkg.S_k for pkg in pkgs))
        raw_policy = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE
        R = raw_policy / S_policy
        b_num = np.zeros_like(R)
        c0_num = np.zeros_like(R)
        for pkg in pkgs:
            Wr = history.get(pkg.base_version)
            if Wr is None:
                return None
            Wr = np.asarray(Wr, dtype=np.float64)
            b_num += pkg.S_k * Wr
            if pkg.base_version < v:
                c0_num += pkg.S_k * self._gamma_vcm * (
                    np.asarray(Wv, dtype=np.float64) - Wr)
        d_vec = R - b_num / S_policy
        c_vec = c0_num / S_policy
        d_norm = float(np.linalg.norm(d_vec))
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm <= 0.0 or d_norm <= 0.0:
            return None
        return float(np.dot(c_vec, d_vec) / (c_norm * d_norm + 1e-12))

    def _p2_recompute_staleness_weights(self, pkgs: list, v: int,
                                        history: dict):
        """按公开陈旧度规则独立复算 σ 整数权重 (σ-agg 掩码域权重).

        w_k = round(α_k·65536), α_k 由本地 σ 配置与共识量 (v, r_k, 历史)
        复算; 与 vpda 同为可策略绑定的权重来源。不可复算时返回 None。
        """
        alphas = []
        for pkg in pkgs:
            s_k = v - pkg.base_version
            if self._staleness_metric == 'rel2':
                Wv = history.get(v)
                Wr = history.get(pkg.base_version)
                W0 = history.get(0)
                if Wv is None or Wr is None or W0 is None:
                    return None
                dist = (float(np.linalg.norm(np.asarray(Wv, dtype=np.float64)
                                             - np.asarray(Wr, dtype=np.float64)))
                        / (float(np.linalg.norm(np.asarray(Wv, dtype=np.float64)
                                                - np.asarray(W0, dtype=np.float64)))
                           + 1e-12))
                alphas.append(staleness_alpha('rel2', s_k,
                                              self._staleness_sigma, dist))
            elif self._staleness_metric == 'wait':
                return None
            else:
                alphas.append(staleness_alpha('version', s_k,
                                              self._staleness_sigma))
        return staleness_int_weights(alphas)
