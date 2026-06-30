import sys
import pickle
import logging
import numpy as np

try:
    from utils import LHH, SIG
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
                 llm_reasoner=None, rule_engine=None):
        self.id = id
        self.cloud_server = cloud_server
        self.ta = ta
        self.enable_verification = enable_verification
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

    def start_new_round(self, round_t: int):
        self._current_round = round_t
        self._masked_gradients = []
        self._online_users = []

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
                h_bytes = pickle.dumps(h_local)
                signature = SIG.sign(h_bytes, self.my_sign_priv_key)
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

        # 3. 发送给 CloudServer
        cs_msg = {
            "edge_id": self.id,
            "round": self._current_round,
            "pre_aggregated_gradient": pre_aggregated_gradient,
            "lhh_hash": h_local,
            "signature": signature
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
                h_bytes = pickle.dumps(h_val)

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
