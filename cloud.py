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

    # ============================================================
    #  研究点2: buffer/checkpoint 半异步方法
    # ============================================================

    def point2_reset(self, initial_model_vector: np.ndarray,
                     initial_version: int = 0):
        """初始化 point2 专用状态.

        Args:
            initial_model_vector: W(0) 初始全局模型 (float64)
            initial_version: v=0
        """
        self.p2_current_version: int = initial_version
        self.p2_global_model_history: dict[int, np.ndarray] = {
            initial_version: initial_model_vector.astype(np.float64).copy()
        }
        self.p2_checkpoint_log: list['CheckpointResult'] = []
        self.p2_package_buffer: dict[str, 'GroupPackage'] = {}
        # 每个 edge 上一次 selected 的 package_token (用于公平约束)
        self.p2_edge_last_selected: dict[str, str] = {}

    def p2_hash_current_model(self) -> bytes:
        """返回当前最新全局模型的哈希."""
        import hashlib
        v = self.p2_current_version
        W = self.p2_global_model_history[v]
        return hashlib.sha256(W.tobytes()).digest()

    def p2_receive_group_package(self, pkg: 'GroupPackage'):
        """point2: 接收组包入 buffer.

        对应文档 Sec.4.4: 每组最多一个待处理包，新包替换旧包.
        """
        edge_id = pkg.edge_id
        old = self.p2_package_buffer.get(edge_id)
        if old is not None and old.package_token != (
                self.p2_edge_last_selected.get(edge_id)):
            pass  # 旧包还没被消费 → 替换

        self.p2_package_buffer[edge_id] = pkg
        logger.info(f"CloudServer: p2 buffer ← {pkg.package_token} "
                     f"edge={edge_id} n={pkg.n_k} base_v={pkg.base_version} "
                     f"buffer_size={len(self.p2_package_buffer)}")

    def p2_prune_expired_packages(self, s_max: int) -> list[str]:
        """移除超过 s_max 陈旧度的组包.

        Returns:
            removed_tokens: 被移除的 package_token 列表
        """
        v = self.p2_current_version
        removed = []
        for edge_id, pkg in list(self.p2_package_buffer.items()):
            staleness = v - pkg.base_version
            if staleness > s_max:
                del self.p2_package_buffer[edge_id]
                removed.append(pkg.package_token)
                logger.info(f"CloudServer: p2 expired {pkg.package_token} "
                             f"staleness={staleness}>{s_max}")
        return removed

    def p2_select_checkpoint(self, K: int, s_max: int) -> list['GroupPackage']:
        """point2: 从 buffer 选择 K 个最早到达的合法组包.

        对应文档 Sec.4.4: 按到达顺序选最早的 K 个.
        """
        self.p2_prune_expired_packages(s_max)

        if len(self.p2_package_buffer) < K:
            return []

        sorted_pkgs = sorted(
            self.p2_package_buffer.values(),
            key=lambda p: p.created_tick
        )
        selected = sorted_pkgs[:K]

        logger.info(f"CloudServer: p2 checkpoint select K={K} from "
                     f"buffer_size={len(self.p2_package_buffer)} "
                     f"selected={[p.package_token for p in selected]}")
        return selected

    def p2_recover_raw_sum(self, selected_packages: list['GroupPackage'],
                           M_Q: np.ndarray) -> np.ndarray:
        """point2: 恢复 checkpoint 的 raw sum.

        W_sum^raw(v) = Σ W̄_k(r_k) - M_Q(v)
        """
        W_bar_Q = np.zeros(self._gradient_dim, dtype=np.int64)
        for pkg in selected_packages:
            W_bar_Q += pkg.masked_sum_int
        return W_bar_Q - M_Q

    def p2_apply_vcm_and_average(self, selected_packages: list['GroupPackage'],
                                 raw_sum_int: np.ndarray,
                                 gamma_vcm: float = 1.0) -> tuple:
        """point2: Direct-VCM 后置补偿 + 参数平均.

        对应文档 Sec.4.5:
        - raw_sum_float = raw_sum_int / QUANTIZE_SCALE
        - C_k(v) = n_k(r_k) · γ_vcm · (W(v) - W(r_k))
        - corrected = raw_sum_float + Σ C_k
        - W(v+1) = corrected / N_Q

        Returns:
            (corrected_sum_float, final_model, compensation_map)
        """
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]

        raw_float = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE
        total_comp = np.zeros_like(raw_float)
        comp_map = {}

        for pkg in selected_packages:
            r_k = pkg.base_version
            if r_k < v:
                W_rk = self.p2_global_model_history.get(r_k)
                if W_rk is not None:
                    G_k = W_v.astype(np.float64) - W_rk.astype(np.float64)
                    C_k = pkg.n_k * gamma_vcm * G_k
                    total_comp += C_k
                    comp_map[pkg.package_token] = float(np.linalg.norm(C_k))
            else:
                comp_map[pkg.package_token] = 0.0

        corrected = raw_float + total_comp
        N_Q = sum(pkg.n_k for pkg in selected_packages)
        final_model = corrected / N_Q

        logger.info(f"CloudServer: p2 VCM v={v} N_Q={N_Q} "
                     f"gamma_vcm={gamma_vcm} stale_packages={len(comp_map)-sum(1 for v in comp_map.values() if v==0)}")

        return corrected, final_model, comp_map

    def p2_commit_checkpoint(self, selected_packages: list['GroupPackage'],
                             final_model: np.ndarray,
                             raw_sum_int: np.ndarray,
                             M_Q: np.ndarray,
                             H_mask_raw: object,
                             corrected_sum: np.ndarray,
                             comp_map: dict) -> 'CheckpointResult':
        """point2: 提交 checkpoint, 更新全局状态.

        - 移除 buffer 中已消费的组包
        - 记录版本历史
        - v += 1
        - 返回 CheckpointResult
        """
        from point2_types import CheckpointResult

        # 从 buffer 移除
        for pkg in selected_packages:
            self.p2_package_buffer.pop(pkg.edge_id, None)
            self.p2_edge_last_selected[pkg.edge_id] = pkg.package_token

        v = self.p2_current_version
        new_v = v + 1
        self.p2_global_model_history[new_v] = final_model.copy()
        self.p2_current_version = new_v

        result = CheckpointResult(
            version=v,
            selected_packages=list(selected_packages),
            N_Q=sum(pkg.n_k for pkg in selected_packages),
            raw_sum_int=raw_sum_int,
            raw_sum_float=raw_sum_int.astype(np.float64) / QUANTIZE_SCALE,
            M_Q=M_Q,
            H_mask_raw=H_mask_raw,
            corrected_sum_float=corrected_sum,
            final_model=final_model,
            compensation_per_package=comp_map,
            verification_passed=False,
        )

        self.p2_checkpoint_log.append(result)
        logger.info(f"CloudServer: p2 checkpoint {v} committed → W({new_v})")

        return result
