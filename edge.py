import sys
import pickle
import hashlib
import logging
import numpy as np

try:
    from utils import LHH, QUANTIZE_SCALE, SIG
except ImportError:
    print("错误: 无法从 utils.py 导入 LHH 或 SIG。")
    sys.exit(1)

from typing import TYPE_CHECKING, Optional, Union

from rule_engine import (
    AgentSuggestion, RoundContext, RuleEngine,
    SuggestionType, VerificationLevel,
)

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
                 llm_reasoner=None, rule_engine=None):
        self.id = id
        self.cloud_server = cloud_server
        self.ta = ta
        self.enable_verification = enable_verification
        self.n_min = n_min
        self.llm_reasoner = llm_reasoner
        self.rule_engine = rule_engine
        self._full_verify_rounds: list[int] = []

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

        # 1. 刷新 ES 公钥列表
        try:
            self.all_es_pub_keys = self.ta.get_all_es_public_keys()
        except Exception:
            pass

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
                    aggregated_h_all_rhs = LHH.Add(aggregated_h_all_rhs, h_val)

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

        # 5. 计算 LHS = H_sum * H_mask (同态加法)
        lhs = LHH.Add(h_plain, h_mask)

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
        self._p2_ready = False

    def p2_receive_masked_gradient(self, msg_bytes: bytes, package_token: str,
                                    base_version: int, sample_size: int = 0):
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
        H_k = LHH.Hash(self.lhh_pp, pre_aggregated)

        S_k = sum(self._p2_user_samples.values())

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
        )

        logger.info(f"EdgeServer {self.id}: p2组包 {self._p2_package_token} "
                     f"已封装 n={n_k} S={S_k} base_v={self._p2_base_version}")
        return pkg

    def p2_verify_checkpoint(self, bundle: dict) -> bool:
        """point2: 验证 checkpoint 结果.

        对应文档 Sec.4.6:
        1. 验签名
        2. 验证 LHH: H_sum^raw * H_mask^raw = ∏ H_k
        3. 公开重算补偿后最终模型

        Args:
            bundle: {
                'checkpoint_version': int,
                'selected_packages': list[GroupPackage],
                'raw_sum_int': np.ndarray,
                'M_Q': np.ndarray,
                'H_mask_raw': object,
                'final_model': np.ndarray,
                'gamma_vcm': float,
                'global_model_history': dict[int, np.ndarray],
            }
        """
        if not self.enable_verification:
            return True

        pkgs = bundle['selected_packages']
        raw_sum_int = bundle['raw_sum_int']
        H_mask_raw = bundle['H_mask_raw']
        final_model = bundle['final_model']
        gamma_vcm = bundle['gamma_vcm']
        history = bundle.get('global_model_history', {})

        # 1. 验签名 + 聚合 RHS
        try:
            self.all_es_pub_keys = self.ta.get_all_es_public_keys()
        except Exception:
            pass

        import hashlib
        aggregated_rhs = None
        for pkg in pkgs:
            if pkg.edge_id not in self.all_es_pub_keys:
                logger.error(f"EdgeServer {self.id}: 未知 {pkg.edge_id}")
                return False

            # base_hash 负责把组包绑定到其训练基版本；签名只能证明
            # “该 Edge 签过这个值”，还需要检查它确实对应历史模型。
            base_model = history.get(pkg.base_version)
            if base_model is None:
                logger.error(f"EdgeServer {self.id}: 缺少 base version v{pkg.base_version}")
                return False
            expected_base_hash = hashlib.sha256(
                np.asarray(base_model, dtype=np.float64).tobytes()
            ).digest()
            if pkg.base_hash != expected_base_hash:
                logger.error(
                    f"EdgeServer {self.id}: {pkg.package_token} base_hash 不匹配"
                )
                return False

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
                return False

            aggregated_rhs = (pkg.H_k if aggregated_rhs is None
                              else LHH.Add(aggregated_rhs, pkg.H_k))

        # 2. LHH 验证 raw sum
        H_sum_raw = LHH.Hash(self.lhh_pp, raw_sum_int)
        lhs = LHH.Add(H_sum_raw, H_mask_raw)
        if lhs != aggregated_rhs:
            logger.error(f"EdgeServer {self.id}: p2 LHH mismatch")
            return False

        # 3. 公开规则重算补偿后最终模型 (使用 S_k)
        S_Q = sum(pkg.S_k for pkg in pkgs)
        v = bundle['checkpoint_version']
        raw_float = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE

        total_comp = np.zeros_like(raw_float)
        for pkg in pkgs:
            r_k = pkg.base_version
            if r_k < v and r_k in history:
                G_k = history[v].astype(np.float64) - history[r_k].astype(np.float64)
                total_comp += pkg.S_k * gamma_vcm * G_k

        expected_final = (raw_float + total_comp) / S_Q
        if not np.allclose(expected_final, final_model, rtol=1e-4, atol=1e-6):
            logger.error(f"EdgeServer {self.id}: p2 最终模型重算不匹配")
            return False

        logger.info(f"EdgeServer {self.id}: p2 checkpoint {v} 验证通过")
        return True
