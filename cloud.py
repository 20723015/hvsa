import sys
import pickle
import logging
import numpy as np

try:
    from utils import LHH, QUANTIZE_SCALE
except ImportError:
    print("错误: 无法从 utils.py 导入 LHH 或 QUANTIZE_SCALE。")
    sys.exit(1)

logger = logging.getLogger(__name__)


class CloudServer:
    """
    云服务器

    职责:
    1. 聚合掩码梯度 (整数加法) -> X_all
    2. 收集所有 ES 的哈希和签名 -> H_list
    3. 接收 TA 的全局掩码 M_global
    4. 解掩码: G_sum = X_all - M_global
    5. 平均: G_avg = G_sum / N
    6. 返回 (G_avg, G_sum, H_list) 供主流程分发和验证
    """

    def __init__(self):
        self.id = "CS-01"
        self._current_round = -1
        self._gradient_dim = 0
        self._expected_edge_servers = 0

        self._sum_masked_gradients = None
        self._collected_hash_data = []
        self._global_mask = None
        self._num_online_users = 0
        self._received_edge_servers = set()
        self._mask_received = False

        self.final_averaged_model = None

        logger.info(f"CloudServer {self.id} 已初始化。")

    def start_new_round(self, round_t: int, gradient_dim: int, expected_edge_servers: int):
        self._current_round = round_t
        self._gradient_dim = gradient_dim
        self._expected_edge_servers = expected_edge_servers

        self._sum_masked_gradients = np.zeros(self._gradient_dim, dtype=np.int64)
        self._collected_hash_data = []
        self._global_mask = None
        self._num_online_users = 0
        self._received_edge_servers = set()
        self._mask_received = False
        self.final_averaged_model = None

        logger.info(f"CloudServer {self.id}: 已开始新一轮 (Round {round_t})。")

    def receive_pre_aggregation(self, msg_bytes: bytes):
        try:
            data = pickle.loads(msg_bytes)
            edge_id = data['edge_id']
            round_t = data['round']
            pre_agg_grad = data['pre_aggregated_gradient']
            lhh_hash = data['lhh_hash']
            signature = data['signature']

            if round_t != self._current_round:
                logger.warning(f"CloudServer {self.id}: 收到来自 {edge_id} 的过时数据。忽略。")
                return
            if edge_id in self._received_edge_servers:
                logger.warning(f"CloudServer {self.id}: 收到来自 {edge_id} 的重复提交。忽略。")
                return

            if self._sum_masked_gradients is None:
                self._sum_masked_gradients = pre_agg_grad
            else:
                self._sum_masked_gradients += pre_agg_grad

            entry = {
                'edge_id': edge_id,
                'lhh_hash': lhh_hash,
                'signature': signature
            }
            self._collected_hash_data.append(entry)
            self._received_edge_servers.add(edge_id)
            logger.info(
                f"CloudServer {self.id} (R{round_t}): 收到 {edge_id} 数据。"
                f"进度 [{len(self._received_edge_servers)}/{self._expected_edge_servers}]")

        except Exception as e:
            logger.error(f"CloudServer {self.id}: 处理预聚合数据时出错: {e}")
            import traceback
            traceback.print_exc()

    def receive_global_mask(self, msg_bytes: bytes):
        """
        接收 TA 发来的全局掩码 M_global。
        """
        try:
            data = pickle.loads(msg_bytes)
            round_t = data['round']
            global_mask = data['global_mask']

            if round_t != self._current_round:
                logger.warning(f"CloudServer {self.id}: 收到来自 TA 的过时掩码。")
                return

            self._global_mask = global_mask
            self._num_online_users = data.get('num_online_users', 0)
            self._mask_received = True

            logger.info(f"CloudServer {self.id} (R{round_t}): 收到 TA 全局掩码。")

        except Exception as e:
            logger.error(f"CloudServer {self.id}: 处理全局掩码时出错: {e}")

    def trigger_final_unmasking_and_averaging(self):
        """
        等 ES 数据和 TA 掩码都到齐后:
        G_sum = X_all - M_global，然后平均。
        """
        logger.info(f"CloudServer {self.id} (R{self._current_round}): 触发最终处理...")

        if len(self._received_edge_servers) < self._expected_edge_servers:
            logger.error(
                f"CloudServer {self.id}: ES 数据未收齐"
                f" ({len(self._received_edge_servers)}/{self._expected_edge_servers})。")
            return None, None, None

        if not self._mask_received or self._global_mask is None:
            logger.error("CloudServer: TA 掩码未到达。")
            return None, None, None

        if self._num_online_users == 0:
            return None, None, None

        try:
            # G_sum = X_all - M_global
            sum_of_models_int = self._sum_masked_gradients - self._global_mask

            # 反量化与平均
            sum_of_models_float = sum_of_models_int.astype(np.float64) / QUANTIZE_SCALE
            self.final_averaged_model = sum_of_models_float / self._num_online_users

            logger.info(f"CloudServer {self.id}: --- 聚合完成 ---")
            logger.info(f"  明文整数总和(前5): {sum_of_models_int[:5]}")

            return self.final_averaged_model, sum_of_models_int, self._collected_hash_data

        except Exception as e:
            logger.error(f"CloudServer {self.id}: 最终处理失败: {e}")
            return None, None, None
