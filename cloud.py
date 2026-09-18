import os
import sys
import pickle
import logging
import numpy as np

try:
    from utils import LHH, QUANTIZE_SCALE, SIG
except ImportError:
    print("错误: 无法从 utils.py 导入 LHH 或 QUANTIZE_SCALE。")
    sys.exit(1)

from point2_types import (CheckpointResult, StagedCheckpoint, result_digest,
                          tokens_digest)

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
                'round': round_t,
                'lhh_hash': lhh_hash,
                'signature': signature,
                # 签名元数据字段 (EdgeVerify 需据此重建被签名的 meta)
                'online_users': data.get('online_users', []),
                'base_hash': data.get('base_hash', b''),
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
        对齐安全模型 (Sec.3.3.4): Cloud 只恢复已验证的明文和 W_sum^raw，
        不生成规范全局模型；共识模型由 ES 在验证通过后按 W_sum^raw / n_online
        本地确定性重构。同步基线要求全部 ES 到齐 (全组同步 barrier)。
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

            logger.info(f"CloudServer {self.id}: --- 聚合完成 ---")
            logger.info(f"  明文整数总和(前5): {sum_of_models_int[:5]}")

            # 规范模型由 ES 重构，Cloud 不提供 (final_averaged_model = None)
            self.final_averaged_model = None
            return None, sum_of_models_int, self._collected_hash_data

        except Exception as e:
            logger.error(f"CloudServer {self.id}: 最终处理失败: {e}")
            return None, None, None

    # ============================================================
    #  研究点2: buffer/checkpoint 半异步方法
    # ============================================================

    def point2_reset(self, initial_model_vector: np.ndarray,
                     initial_version: int = 0,
                     stage_path: str = None):
        """初始化 point2 专用状态.

        Args:
            initial_model_vector: W(0) 初始全局模型 (float64)
            initial_version: v=0
            stage_path: 可选; staged checkpoint 的持久化 journal 路径
                        (崩溃恢复用; None = 仅内存暂存)
        """
        self.p2_current_version: int = initial_version
        self.p2_global_model_history: dict[int, np.ndarray] = {
            initial_version: initial_model_vector.astype(np.float64).copy()
        }
        self.p2_checkpoint_log: list['CheckpointResult'] = []
        self.p2_package_buffer: dict[str, 'GroupPackage'] = {}
        # 公平计数器: edge_id → 连续未被选中次数
        self.p2_defer_counters: dict[str, int] = {}
        # 组包到达时间记录: package_token → arrival_tick
        self.p2_arrival_ticks: dict[str, int] = {}
        # 累计过期组包数
        self.p2_expired_total: int = 0
        # 已成功提交 checkpoint 的 package_token；用于 exactly-once / replay 防护
        self.p2_aggregated_ids: set[str] = set()
        self.p2_duplicate_contributions: int = 0
        # 矩校准 VCM: 全局漂移范数 M̂ 的 EMA (DVAFL 式(15)-(16), λ=0.9);
        # 记录已更新到的版本, 避免同一 checkpoint 多次调用重复更新
        self._p2_drift_norm_ema: float | None = None
        self._p2_drift_ema_v: int = initial_version
        # moment1n: 原始比值 r_raw=m̂/M̂ 的 EMA (自归一化分母, 使比值中心=1,
        # 消除数据集相关的整体尺度, 只保留逐包相对调制)
        self._p2_mr_ema: float | None = None
        # 曲率感知 VCM 的加速度外推系数
        self._vcm_curve_lambda: float = getattr(self, '_vcm_curve_lambda', 0.5)
        # UR-VCM 可信域比率 ρ (vcm-mode=trust): η=min(1, ρ‖Δtrain‖/(‖C̄‖+ε))
        self._vcm_trust_ratio: float = getattr(self, '_vcm_trust_ratio', 1.0)
        # 最近一次聚合的 UR-VCM 诊断 (main_point2 记录到 npz)
        self.p2_last_vcm_diag: dict | None = None
        # 服务器优化器 (FADAS 门控 / FedAdam) 状态
        self._p2_adam_m: np.ndarray | None = None
        self._p2_adam_v: np.ndarray | None = None
        self._p2_adam_t: int = 0
        # TA receipt 验签公钥 (由 main 注入; 未注入 = 兼容路径, 不校验)
        self._p2_receipt_pub_key = None
        # 最近一次 checkpoint 选择的触发模式 (normal | timeout)
        self.p2_last_trigger_mode = 'normal'
        # staged checkpoint (P0-2 两阶段提交: stage → TA finalize → publish)
        self._p2_staged = None
        self._p2_stage_path = stage_path

    def p2_set_stage_path(self, stage_path: str) -> None:
        """设置 staged checkpoint journal 路径 (崩溃恢复)."""
        self._p2_stage_path = stage_path

    # --- stage journal (持久化, 支持崩溃恢复) ---
    def _p2_write_stage_journal(self, stage) -> None:
        if not self._p2_stage_path:
            return
        tmp = f"{self._p2_stage_path}.tmp"
        with open(tmp, 'wb') as f:
            pickle.dump(stage, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._p2_stage_path)

    def _p2_clear_stage_journal(self) -> None:
        if not self._p2_stage_path:
            return
        try:
            os.remove(self._p2_stage_path)
        except FileNotFoundError:
            pass

    def p2_staged_checkpoint(self):
        """返回内存中暂存的候选 (无则 None)."""
        return self._p2_staged

    def p2_load_staged_journal(self):
        """从 journal 载入上次未完成提交的 staged 候选 (无则 None)."""
        if not self._p2_stage_path or not os.path.exists(self._p2_stage_path):
            return None
        try:
            with open(self._p2_stage_path, 'rb') as f:
                stage = pickle.load(f)
            self._p2_staged = stage
            return stage
        except Exception as e:
            logger.error(f"CloudServer: 读取 stage journal 失败: {e}")
            return None

    def p2_discard_staged(self, stage=None, reason: str = '') -> None:
        """放弃暂存候选 (TA finalize 被协议拒绝/候选作废): 不推进任何状态."""
        discarded = stage or self._p2_staged
        self._p2_staged = None
        self._p2_clear_stage_journal()
        if discarded is not None:
            logger.warning(f"CloudServer: discarded staged checkpoint "
                           f"v{discarded.base_version} "
                           f"({[p.package_token for p in discarded.selected_packages]})"
                           + (f": {reason}" if reason else ""))

    def p2_complete_staged(self, stage=None) -> None:
        """提交事务全部完成 (finalize + publish + notify): 清理暂存与 journal.

        publish 之后仍保留 journal, 直到通知阶段结束 —— 否则"publish 完成
        但 notify 前崩溃"会丢失恢复所需的候选记录, 造成 Cloud/ES 版本分叉。
        """
        done = stage or self._p2_staged
        self._p2_staged = None
        self._p2_clear_stage_journal()
        if done is not None:
            logger.info(f"CloudServer: staged checkpoint {done.stage_id} "
                        f"事务完成, journal 已清理")

    def p2_set_receipt_pub_key(self, pub_key: bytes) -> None:
        """注入 TA receipt 验签公钥 (verify-on 提交必须持有效 receipt)."""
        self._p2_receipt_pub_key = pub_key

    def p2_hash_current_model(self) -> bytes:
        """返回当前最新全局模型的哈希."""
        import hashlib
        v = self.p2_current_version
        W = self.p2_global_model_history[v]
        return hashlib.sha256(W.tobytes()).digest()

    def p2_receive_group_package(self, pkg: 'GroupPackage', arrival_tick: int = 0):
        """point2: 接收组包入 buffer.

         对应文档 Sec.4.4: 每组最多一个待处理包；未消费旧包时拒绝新包.
        """
        edge_id = pkg.edge_id
        if pkg.package_token in self.p2_aggregated_ids:
            self.p2_duplicate_contributions += 1
            logger.warning(f"CloudServer: reject replayed package {pkg.package_token}")
            return False

        old = self.p2_package_buffer.get(edge_id)
        if old is not None:
            # 每个 edge 在前一个组包被消费或过期前不能产生新待处理包。
            # 这也保证验证失败时，原候选不会被新包覆盖。
            self.p2_duplicate_contributions += 1
            logger.warning(
                f"CloudServer: reject new package {pkg.package_token}; "
                f"pending package {old.package_token} still exists"
            )
            return False

        self.p2_package_buffer[edge_id] = pkg
        self.p2_arrival_ticks[pkg.package_token] = arrival_tick
        # 新到达的组包初始化公平计数器
        if edge_id not in self.p2_defer_counters:
            self.p2_defer_counters[edge_id] = 0
        logger.info(f"CloudServer: p2 buffer ← {pkg.package_token} "
                     f"edge={edge_id} n={pkg.n_k} S={pkg.S_k} base_v={pkg.base_version} "
                     f"buffer_size={len(self.p2_package_buffer)}")
        return True

    def p2_has_pending_package(self, edge_id: str) -> bool:
        """返回某 edge 是否仍有未消费的组包。"""
        return edge_id in self.p2_package_buffer

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
        self.p2_expired_total += len(removed)
        return removed

    def p2_select_checkpoint(self, K: int, s_max: int,
                            current_tick: int = 0,
                            delta_timeout: int = 0,
                            policy: str = 'fifo',
                            D_fair: int = 2) -> list['GroupPackage']:
        """point2: 从 buffer 选择 K 个合法组包.

        对应文档 Sec.4.4 + Sec 8.1 调度增强:
        - 常规: |B_legal| >= K → 按策略选 K 个
        - 超时: 最早组包等待 >= delta_timeout → 保守 checkpoint (1 <= n < K)
        - policy: fifo / fresh_first / fair_fresh_first
        """
        removed = self.p2_prune_expired_packages(s_max)
        # 移除过期包对应的公平计数器
        for token in removed:
            for eid, pkg in list(self.p2_package_buffer.items()):
                if pkg.package_token == token:
                    self.p2_defer_counters.pop(eid, None)
                    break

        if len(self.p2_package_buffer) == 0:
            return []

        # 按 created_tick 排序 (用于 FIFO 和 超时检测)
        sorted_by_arrival = sorted(
            self.p2_package_buffer.values(),
            key=lambda p: p.created_tick
        )

        # 超时触发检测
        timeout_triggered = False
        if delta_timeout > 0 and 1 <= len(self.p2_package_buffer) < K:
            earliest = sorted_by_arrival[0]
            wait_time = current_tick - earliest.created_tick
            if wait_time >= delta_timeout:
                timeout_triggered = True
                logger.info(f"CloudServer: p2 timeout triggered "
                            f"(wait={wait_time} >= {delta_timeout})")

        if not timeout_triggered and len(self.p2_package_buffer) < K:
            return []

        # 按策略选择
        if policy == 'fifo':
            selected = sorted_by_arrival[:K]
        elif policy == 'fresh_first':
            sorted_by_fresh = sorted(
                self.p2_package_buffer.values(),
                key=lambda p: (self.p2_current_version - p.base_version)
            )
            selected = sorted_by_fresh[:K]
        elif policy == 'fair_fresh_first':
            # 先选 c_k >= D_fair 的保底候选
            guaranteed = []
            for pkg in sorted_by_arrival:
                eid = pkg.edge_id
                defer = self.p2_defer_counters.get(eid, 0)
                if defer >= D_fair:
                    guaranteed.append(pkg)
            # 剩余名额按 fresh-first 填充
            remaining_slots = max(0, K - len(guaranteed))
            remaining_candidates = [p for p in sorted_by_arrival if p not in guaranteed]
            sorted_remaining = sorted(
                remaining_candidates,
                key=lambda p: (self.p2_current_version - p.base_version)
            )
            selected = guaranteed + sorted_remaining[:remaining_slots]
        else:
            selected = sorted_by_arrival[:K]

        # 更新公平计数器
        selected_edge_ids = {p.edge_id for p in selected}
        for pkg in sorted_by_arrival:
            eid = pkg.edge_id
            if eid in selected_edge_ids:
                self.p2_defer_counters[eid] = 0
            else:
                self.p2_defer_counters[eid] = self.p2_defer_counters.get(eid, 0) + 1

        trigger_mode = "timeout" if timeout_triggered else "normal"
        self.p2_last_trigger_mode = trigger_mode
        logger.info(f"CloudServer: p2 checkpoint select ({trigger_mode}, {policy}) "
                     f"K={K} from buffer={len(self.p2_package_buffer)} "
                     f"selected={[p.package_token for p in selected]}")
        return selected

    def p2_recover_raw_sum(self, selected_packages: list['GroupPackage'],
                           M_Q: np.ndarray, alphas=None,
                           weights=None) -> np.ndarray:
        """point2: 恢复 checkpoint 的 raw sum (等权).

        W_sum^raw(v) = Σ W̄_k(r_k) - M_Q(v).
        掩码域恒等权 (整数加法) 以保持 LHH 同态验证; alphas 仅用于
        明文域补偿缩放 (p2_apply_vcm_and_average), 不进入此步.

        weights (sigma-target=agg): 整数化权重 w_k, 恢复加权和
        Σ w_k·W̄_k − M_Q_weighted = Σ w_k·model_k (掩码同缩放, 整数域封闭,
        LHH 同态验证对整数标量乘仍成立). dtype=int64, 需防溢出
        (w_k ≤ 2^16, masked_k ~ S_k·‖w‖·QS, 实验参数下余量 >1e3 倍).
        """
        W_bar_Q = np.zeros(self._gradient_dim, dtype=np.int64)
        for j, pkg in enumerate(selected_packages):
            if weights is not None:
                W_bar_Q += int(weights[j]) * pkg.masked_sum_int
            else:
                W_bar_Q += pkg.masked_sum_int
        return W_bar_Q - M_Q

    def p2_weighted_average(self, selected_packages: list['GroupPackage'],
                            raw_sum_int: np.ndarray,
                            weights: list) -> tuple:
        """σ 聚合端加权平均 (sigma-target=agg, 无 VCM 补偿).

        raw_int = Σ w_k·S_k·w̄_k·QS (w̄_k 为组内样本量加权平均模型, 已扣加权掩码)
        W(v+1) = raw_int / (QS · Σ w_k·S_k)  —— σ 权重真实作用于聚合平均.
        σ=0 时 w_k 全等 → 退化为等权平均 (数学上 ≡ G3), 可作回归锚点.

        实现: 委托 p2_weighted_vcm_average (γ=0), 加权 VCM 项为零即纯加权平均。
        """
        return self.p2_weighted_vcm_average(selected_packages, raw_sum_int,
                                            weights, gamma_vcm=0.0)

    def p2_weighted_vcm_average(self, selected_packages: list['GroupPackage'],
                                raw_sum_int: np.ndarray,
                                weights: list,
                                gamma_vcm: float = 0.0,
                                vcm_mode: str = 'full',
                                alphas=None) -> tuple:
        """加权原始模型 + 加权 VCM (V-PDA / 一般整数权重路径).

        W(v+1) = [ Σ w_k·S_k·Ŵ_k + Σ w_k·α_k·S_k·γ·(W(v) − W(r_k)) ] / Σ w_k·S_k

        - raw_sum_int 为加权恢复: Σ w_k·masked_sum_k − M_Q (w 为整数化权重,
          掩码域整数标量乘封闭, LHH 同态保持)。
        - VCM 补偿项同样加权 (w_k 为明文域公开规则), γ=0 时退化为纯加权平均
          (等价历史 sigma-target=agg 语义)。
        - 支持 vcm_mode full/span/clip (与 edge 验证侧重算一致; moment*/curve
          为等权路径变体, 不与加权路径组合)。

        Returns:
            (corrected_sum_float, final_model, compensation_map)
        """
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]
        raw_float = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE
        denom = sum(int(w) * pkg.S_k for w, pkg in zip(weights, selected_packages))
        denom = max(1, denom)
        raw_norm = np.linalg.norm(raw_float)

        total_comp = np.zeros_like(raw_float)
        comp_map = {}
        for j, pkg in enumerate(selected_packages):
            alpha = 1.0 if alphas is None else alphas[j]
            w_k = int(weights[j])
            r_k = pkg.base_version
            if r_k < v:
                W_rk = self.p2_global_model_history.get(r_k)
                if W_rk is not None:
                    G_k = W_v.astype(np.float64) - W_rk.astype(np.float64)
                    if vcm_mode == 'span' and (v - r_k) > 0:
                        G_k = G_k / (v - r_k)
                    C_k = w_k * alpha * pkg.S_k * gamma_vcm * G_k
                    if vcm_mode == 'clip' and raw_norm > 0:
                        max_norm = 0.1 * raw_norm
                        c_norm = np.linalg.norm(C_k)
                        if c_norm > max_norm:
                            C_k = C_k * (max_norm / c_norm)
                    total_comp += C_k
                    comp_map[pkg.package_token] = float(np.linalg.norm(C_k))
            else:
                comp_map[pkg.package_token] = 0.0

        corrected = raw_float + total_comp
        if vcm_mode == 'trust':
            # UR-VCM (加权路径): R=Σw_kS_kŴ_k/Σw_kS_k, B=Σw_kS_kW(r_k)/Σw_kS_k,
            # C̄=Σw_kC_k/Σw_kS_k, W=R+ηC̄ (η 同公开公式)
            R = raw_float / denom
            base_num = np.zeros_like(R)
            for j, pkg in enumerate(selected_packages):
                W_rk = self.p2_global_model_history.get(pkg.base_version)
                if W_rk is not None:
                    base_num += int(weights[j]) * pkg.S_k * np.asarray(
                        W_rk, dtype=np.float64)
            B = base_num / denom
            C_bar = total_comp / denom
            d_norm = float(np.linalg.norm(R - B))
            c_norm = float(np.linalg.norm(C_bar))
            eta = (min(1.0, self._vcm_trust_ratio * d_norm / (c_norm + 1e-12))
                   if c_norm > 0 else 1.0)
            final_model = R + eta * C_bar
            corrected = final_model * denom
            self.p2_last_vcm_diag = dict(train_norm=d_norm, comp_norm=c_norm,
                                         eta=eta, gated=bool(eta < 1.0))
        else:
            final_model = corrected / denom
            self.p2_last_vcm_diag = None

        logger.info(f"CloudServer: p2 weighted-vcm v={v} denom={denom} "
                    f"gamma_vcm={gamma_vcm} weights={weights}"
                    + (f" diag={self.p2_last_vcm_diag}" if self.p2_last_vcm_diag else ""))

        return corrected, final_model, comp_map

    def p2_apply_vcm_and_average(self, selected_packages: list['GroupPackage'],
                                 raw_sum_int: np.ndarray,
                                 gamma_vcm: float = 1.0,
                                 vcm_mode: str = 'full',
                                 alphas=None) -> tuple:
        """point2: Direct-VCM 后置补偿 + 参数平均.

        对应文档 Sec.4.5 (终版，使用样本量 S_k):
        - raw_sum_float = raw_sum_int / QUANTIZE_SCALE
        - C_k(v) = S_k(r_k) · γ_vcm · (W(v) - W(r_k))
        - corrected = raw_sum_float + Σ C_k
        - W(v+1) = corrected / S_Q(v)

        alphas: 陈旧度补偿衰减 (与 selected_packages 一一对应, 默认全 1)。
        只缩放 VCM 补偿项 C_k (明文域公开规则, 不进入掩码域):
            C_k = α_k·S_k·γ·(W(v)-W(r_k)),  聚合 S_Q 保持等权。
        语义: 越陈旧的组包补偿强度越低 (旧组的版本差更难精确补), σ 由
        --staleness-sigma 控制 α_k = 1/(s_k+1)^σ。σ=0 时退化为原始 VCM。

        vcm_mode (模式变体, 默认 full 保持原始行为):
        - 'full': C_k = S_k·γ·(W(v)-W(r_k)), 原始 Direct-VCM
        - 'span': C_k 除以版本跨度 (v - r_k), 多版本累积差均摊
        - 'clip': C_k 范数截断到 raw_sum 范数的 10%
        - 'moment1'/'moment2' (矩校准, DVAFL TMC2026 式(15)-(18) 的组级 SecAgg 适配):
            补偿步长按 m̂_k/M̂ 缩放, m̂_k=组内样本量加权平均本地更新范数
            (用户上报的标量元数据), M̂=全局漂移范数 ‖W(v)-W(v-1)‖ 的 EMA(λ=0.9)。
            组更新相对全局漂移越弱, 补偿越保守 → 抑制 Non-IID 下 Direct 模式的
            补偿振荡 (DVAFL 消融: Direct 66.1% → 一阶 67.8% → 二阶 70.2%)。
            moment2 用 (m̂/M̂)²。M̂ 不可用 (v=0/首个 drift) 时退化为 Direct。

        Returns:
            (corrected_sum_float, final_model, compensation_map)
        """
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]

        # 曲率感知 VCM (curve): 补偿方向加入轨迹加速度外推项
        # A(v) = [W(v)−W(v−1)] − [W(v−1)−W(v−2)],  G_k += λ_c·A(v)
        # 动机: NI 下轨迹转弯处线性快进失准 (方向级误差), 加速度外推
        # 修正补偿方向; 全部来自公开全局轨迹, SecAgg 兼容
        curve_term = None
        if vcm_mode == 'curve' and self._vcm_curve_lambda > 0 \
                and v >= 2 and (v-1) in self.p2_global_model_history \
                and (v-2) in self.p2_global_model_history:
            g1 = W_v - self.p2_global_model_history[v-1]
            g0 = self.p2_global_model_history[v-1] - self.p2_global_model_history[v-2]
            curve_term = self._vcm_curve_lambda * (g1 - g0)

        # 更新全局漂移范数 EMA (每版本一次)
        if v > self._p2_drift_ema_v and v >= 1 and (v - 1) in self.p2_global_model_history:
            drift = float(np.linalg.norm(
                W_v - self.p2_global_model_history[v - 1]))
            self._p2_drift_norm_ema = (drift if self._p2_drift_norm_ema is None
                                       else 0.9 * self._p2_drift_norm_ema + 0.1 * drift)
            self._p2_drift_ema_v = v

        raw_float = raw_sum_int.astype(np.float64) / QUANTIZE_SCALE
        raw_norm = np.linalg.norm(raw_float)
        total_comp = np.zeros_like(raw_float)
        comp_map = {}

        for j, pkg in enumerate(selected_packages):
            alpha = 1.0 if alphas is None else alphas[j]
            r_k = pkg.base_version
            # 矩校准步长比: m̂_k/M̂ (组更新强度 / 全局漂移强度), 限幅 [0,2];
            # M̂ 冷启动或元数据缺失时退化为 Direct (比值 1)
            if (vcm_mode in ('moment1', 'moment2', 'moment1c', 'moment1n')
                    and self._p2_drift_norm_ema and pkg.upd_norm > 0):
                r_raw = pkg.upd_norm / self._p2_drift_norm_ema
                if vcm_mode == 'moment1n':
                    # 自归一化: 比值 = r_raw / r_raw 的历史 EMA (中心=1, 数据集
                    # 尺度无关, 保留逐包相对调制); 冷启动 (EMA 未就绪) 退化为 Direct
                    if self._p2_mr_ema is None:
                        self._p2_mr_ema = r_raw
                        ratio = 1.0
                    else:
                        ratio = min(2.0, max(0.5,
                            r_raw / self._p2_mr_ema))
                        self._p2_mr_ema = (0.9 * self._p2_mr_ema
                                           + 0.1 * r_raw)
                else:
                    ratio = min(2.0, max(0.0, r_raw))
                if vcm_mode == 'moment1c':
                    ratio = min(1.0, ratio)   # 保守校准: 只减不增
                if vcm_mode == 'moment2':
                    ratio *= ratio
            else:
                ratio = 1.0
            if r_k < v:
                W_rk = self.p2_global_model_history.get(r_k)
                if W_rk is not None:
                    G_k = W_v.astype(np.float64) - W_rk.astype(np.float64)
                    if curve_term is not None:
                        G_k = G_k + curve_term
                    if vcm_mode == 'span' and (v - r_k) > 0:
                        G_k = G_k / (v - r_k)
                    C_k = ratio * alpha * pkg.S_k * gamma_vcm * G_k
                    if vcm_mode == 'clip' and raw_norm > 0:
                        max_norm = 0.1 * raw_norm
                        c_norm = np.linalg.norm(C_k)
                        if c_norm > max_norm:
                            C_k = C_k * (max_norm / c_norm)
                    total_comp += C_k
                    comp_map[pkg.package_token] = float(np.linalg.norm(C_k))
            else:
                comp_map[pkg.package_token] = 0.0

        corrected = raw_float + total_comp
        S_Q = sum(pkg.S_k for pkg in selected_packages)
        if vcm_mode == 'trust':
            # UR-VCM (等权路径): R=原始聚合均值, B=所选组基模型样本量加权均值,
            # Δtrain=R−B (聚合训练增量, 而非完整模型范数),
            # C̄=原始补偿均值, η=min(1, ρ·‖Δtrain‖/(‖C̄‖+ε)), W=R+ηC̄。
            # C̄=0 → 恒等 (W=R); Δtrain=0 且 C̄≠0 → η=0 抑制补偿。
            R = raw_float / max(1, S_Q)
            base_num = np.zeros_like(R)
            for pkg in selected_packages:
                W_rk = self.p2_global_model_history.get(pkg.base_version)
                if W_rk is not None:
                    base_num += pkg.S_k * np.asarray(W_rk, dtype=np.float64)
            B = base_num / max(1, S_Q)
            C_bar = total_comp / max(1, S_Q)
            d_norm = float(np.linalg.norm(R - B))
            c_norm = float(np.linalg.norm(C_bar))
            eta = (min(1.0, self._vcm_trust_ratio * d_norm / (c_norm + 1e-12))
                   if c_norm > 0 else 1.0)
            final_model = R + eta * C_bar
            corrected = final_model * max(1, S_Q)
            self.p2_last_vcm_diag = dict(train_norm=d_norm, comp_norm=c_norm,
                                         eta=eta, gated=bool(eta < 1.0))
        else:
            final_model = corrected / S_Q
            self.p2_last_vcm_diag = None

        logger.info(f"CloudServer: p2 VCM v={v} S_Q={S_Q} "
                     f"gamma_vcm={gamma_vcm} stale_packages={len(comp_map)-sum(1 for v in comp_map.values() if v==0)}"
                     + (f" diag={self.p2_last_vcm_diag}" if self.p2_last_vcm_diag else ""))

        return corrected, final_model, comp_map

    def p2_server_lr_gate(self, final_model: np.ndarray,
                          tau_max: int) -> np.ndarray:
        """FADAS (ICML 2024, arXiv:2407.18365 Eq.4) 式 delay-adaptive lr 门控.

        W' = W(v) + η·(final − W(v)),  η = min{1, 1/τ_max},
        τ_max = checkpoint 内最坏陈旧度 max(v − r_k) (公开标量元数据).
        无 stale 包 (τ_max ≤ 1) 时恒等; 仅依赖聚合量与时间戳, SecAgg 兼容.
        """
        v = self.p2_current_version
        eta = min(1.0, 1.0 / max(1, int(tau_max)))
        W_v = self.p2_global_model_history[v]
        out = W_v + eta * (np.asarray(final_model, dtype=np.float64) - W_v)
        logger.info(f"CloudServer: p2 server-lr-gate v={v} tau_max={tau_max} eta={eta:.3f}")
        return out

    def p2_adam_update(self, final_model: np.ndarray, beta1: float = 0.9,
                       beta2: float = 0.99, tau: float = 1e-3,
                       eta: float = 1.0, gamma_tr: float = 1.0) -> np.ndarray:
        """FedAdam (ICLR 2021, arXiv:2003.00295) 服务器 Adam.

        Δ = final − W(v) 为 checkpoint 聚合伪梯度; √v̂ 归一化遏制异步动量
        的隐式偏差 (arXiv:2402.09247), 原文: 短视界初始收敛快于 FedAvgM.
        v_{-1} = τ² (Reddi et al. 建议). gamma_tr: 信赖域 — Adam 步范数
        上限 γ·‖Δ‖, 防首步 m̂/√v̂=sign(Δ) 的 ±η 爆炸 (短视界必需).
        """
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]
        delta = np.asarray(final_model, dtype=np.float64) - W_v
        if self._p2_adam_m is None:
            self._p2_adam_m = np.zeros_like(delta)
            self._p2_adam_v = np.full_like(delta, tau * tau)
        self._p2_adam_t += 1
        self._p2_adam_m = beta1 * self._p2_adam_m + (1 - beta1) * delta
        self._p2_adam_v = beta2 * self._p2_adam_v + (1 - beta2) * delta * delta
        m_hat = self._p2_adam_m / (1 - beta1 ** self._p2_adam_t)
        v_hat = self._p2_adam_v / (1 - beta2 ** self._p2_adam_t)
        out = W_v + eta * m_hat / (np.sqrt(v_hat) + tau)
        if gamma_tr < float('inf'):
            step = out - W_v
            step_norm = float(np.linalg.norm(step))
            max_norm = gamma_tr * float(np.linalg.norm(delta)) + 1e-12
            if step_norm > max_norm:
                out = W_v + step * (max_norm / step_norm)
        logger.info(f"CloudServer: p2 server-adam v={v} t={self._p2_adam_t} eta={eta}")
        return out

    def p2_server_trust_region(self, final_model: np.ndarray,
                               k: float = 2.0) -> np.ndarray:
        """服务器端聚合更新信赖域 (arXiv:2606.13287 裁剪-延迟解耦的聚合端实现).

        Δ = final − W(v); 维护历史 ‖Δ‖ 的中位数 M, 步长上限 k·M:
        W' = W(v) + Δ·min{1, k·M/‖Δ‖}.
        只消费聚合量 (SecAgg 兼容), 无动量/二阶矩状态, 短视界安全;
        目标: 拦截偶发的大步长坏 checkpoint 对轨迹的破坏 (NI 双峰发散).
        """
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]
        delta = np.asarray(final_model, dtype=np.float64) - W_v
        norm = float(np.linalg.norm(delta))
        if norm <= 0:
            return final_model
        hist = getattr(self, '_p2_delta_norm_hist', [])
        if hist:
            med = float(np.median(hist))
            cap = k * med
            if norm > cap:
                delta = delta * (cap / norm)
        hist.append(norm)
        self._p2_delta_norm_hist = hist[-20:]
        return W_v + delta

    def p2_apply_momentum(self, final_model: np.ndarray,
                          momentum_beta: float) -> np.ndarray:
        """Server momentum (FedAvgM 风格): W' = β·W(v) + (1−β)·final.

        - 平滑半异步 + VCM 补偿带来的模型轨迹抖动
        - 凸组合, 不离开参与者模型凸包, 不影响验证链算术性质
        - β=0 时恒等 (关闭)
        """
        if momentum_beta <= 0.0:
            return final_model
        v = self.p2_current_version
        W_v = self.p2_global_model_history[v]
        return (momentum_beta * W_v.astype(np.float64)
                + (1.0 - momentum_beta) * final_model.astype(np.float64))

    def p2_stage_checkpoint(self, selected_packages: list['GroupPackage'],
                            final_model: np.ndarray,
                            raw_sum_int: np.ndarray,
                            M_Q: np.ndarray,
                            H_mask_raw: object,
                            corrected_sum: np.ndarray,
                            comp_map: dict,
                            receipt=None) -> 'StagedCheckpoint':
        """point2 阶段 1 (stage): 只读校验候选并暂存, 不改任何权威状态.

        verify-on 路径必须携带有效 TA CheckpointReceipt (签名、版本、
        token 集合、结果摘要四方校验) —— Cloud 本地的 aggregated_ids/buffer
        只是非权威缓存, 重启或篡改不能绕过 TA ledger 的权威状态。

        与旧 `p2_commit_checkpoint` 的关键差异: 这里**不**修改 buffer、
        不推进版本、不写模型历史 —— 变更记录在 StagedCheckpoint 中, 只有
        TA finalize 成功后由 `p2_publish_staged` 应用; finalize 失败时
        Cloud/TA/ES 三者不分叉 (P0-2)。暂存同时写入可选 journal, 使进程
        崩溃后仍可恢复完成。
        """
        # receipt 权威校验 (设置过 TA receipt 公钥即强制执行)
        if self._p2_receipt_pub_key is not None:
            if receipt is None:
                raise ValueError("verify-on 提交必须携带 TA CheckpointReceipt")
            payload = bytes.fromhex(receipt.payload_digest())
            if not SIG.verify(payload, receipt.signature,
                              self._p2_receipt_pub_key):
                raise ValueError("receipt 签名无效 (非 TA 签发)")
            if receipt.checkpoint_version != self.p2_current_version:
                raise ValueError(
                    f"receipt 版本不匹配: receipt v{receipt.checkpoint_version}, "
                    f"cloud 当前 v{self.p2_current_version}")
            tdig = tokens_digest([p.package_token for p in selected_packages])
            if tdig != receipt.token_digest:
                raise ValueError("receipt token 集合与提交候选不符")
            # 结果摘要绑定 (P0-1): 提交的 raw sum / M_Q / final model 必须
            # 与 ES 验证并投票的候选逐字节一致 —— 取得诚实 receipt 后偷换
            # 聚合结果在此被拒绝
            rdig = result_digest(receipt.checkpoint_version, tdig,
                                 raw_sum_int, M_Q, final_model)
            if rdig != receipt.result_digest:
                raise ValueError("receipt 结果摘要与提交内容不符 "
                                 "(raw sum/M_Q/final model 被替换)")

        # 候选必须仍是 buffer 中的原包且未被消费 (只读检查): 避免重复提交
        # 或状态错位
        for pkg in selected_packages:
            if pkg.package_token in self.p2_aggregated_ids:
                raise ValueError(f"replayed package already aggregated: {pkg.package_token}")
            buffered = self.p2_package_buffer.get(pkg.edge_id)
            if buffered is None or buffered.package_token != pkg.package_token:
                raise ValueError(f"package is no longer pending: {pkg.package_token}")

        stage = StagedCheckpoint(
            base_version=self.p2_current_version,
            selected_packages=list(selected_packages),
            final_model=np.asarray(final_model, dtype=np.float64).copy(),
            raw_sum_int=np.asarray(raw_sum_int).copy(),
            M_Q=np.asarray(M_Q).copy(),
            H_mask_raw=H_mask_raw,
            corrected_sum=np.asarray(corrected_sum, dtype=np.float64).copy(),
            comp_map=dict(comp_map),
            receipt=receipt,
            stage_id=f"v{self.p2_current_version}:"
                     f"{tokens_digest([p.package_token for p in selected_packages])[:16]}")
        self._p2_staged = stage
        self._p2_write_stage_journal(stage)
        logger.info(f"CloudServer: p2 staged checkpoint {stage.stage_id} "
                    f"(v{stage.base_version} → v{stage.base_version + 1}, "
                    f"{len(stage.selected_packages)} 包, 待 TA finalize)")
        return stage

    def p2_publish_staged(self, stage=None) -> 'CheckpointResult':
        """point2 阶段 3 (publish): TA finalize 成功后应用暂存变更.

        幂等: 同一 stage 的版本已推进 (base+1) 时直接返回已记录的
        CheckpointResult —— 崩溃恢复重放不会重复消费 buffer、不会重复推进
        版本。仅在 TA ledger 已落地 (或 verify-off 无 receipt) 时由
        `commit_tx.run_commit_transaction` 调用。journal 保留到通知阶段
        结束 (见 `p2_complete_staged`), 以覆盖 "publish 后、notify 前崩溃"。
        """
        stage = stage or self._p2_staged or self.p2_load_staged_journal()
        if stage is None:
            raise ValueError("publish 拒绝: 无 staged checkpoint")

        base_v = int(stage.base_version)
        if self.p2_current_version == base_v + 1:
            for r in self.p2_checkpoint_log:
                if r.version == base_v:
                    self._p2_staged = None
                    logger.info(f"CloudServer: publish 幂等重放 {stage.stage_id} "
                                f"(v{base_v} 已落地, 不重复应用)")
                    return r
            raise ValueError("publish 拒绝: 版本已推进但无对应 checkpoint 日志 "
                             "(状态不一致)")
        if self.p2_current_version != base_v:
            raise ValueError(f"publish 拒绝: 版本漂移 (stage v{base_v}, "
                             f"cloud v{self.p2_current_version})")

        # 应用变更 (变更集合与旧 commit 一致, 但只在 TA 落地后执行)
        for pkg in stage.selected_packages:
            self.p2_package_buffer.pop(pkg.edge_id, None)
            self.p2_aggregated_ids.add(pkg.package_token)

        S_Q = sum(pkg.S_k for pkg in stage.selected_packages)
        new_v = base_v + 1
        self.p2_global_model_history[new_v] = np.asarray(
            stage.final_model, dtype=np.float64).copy()
        self.p2_current_version = new_v

        result = CheckpointResult(
            version=base_v,
            selected_packages=list(stage.selected_packages),
            N_Q=sum(pkg.n_k for pkg in stage.selected_packages),
            S_Q=S_Q,
            raw_sum_int=stage.raw_sum_int,
            raw_sum_float=(np.asarray(stage.raw_sum_int).astype(np.float64)
                           / QUANTIZE_SCALE),
            M_Q=stage.M_Q,
            H_mask_raw=stage.H_mask_raw,
            corrected_sum_float=stage.corrected_sum,
            final_model=np.asarray(stage.final_model, dtype=np.float64),
            compensation_per_package=stage.comp_map,
            verification_passed=False,
        )
        self.p2_checkpoint_log.append(result)
        self._p2_staged = None
        logger.info(f"CloudServer: p2 checkpoint {base_v} published → "
                    f"W({new_v}) N_Q={result.N_Q} S_Q={S_Q}")
        return result

    def p2_commit_checkpoint(self, selected_packages: list['GroupPackage'],
                             final_model: np.ndarray,
                             raw_sum_int: np.ndarray,
                             M_Q: np.ndarray,
                             H_mask_raw: object,
                             corrected_sum: np.ndarray,
                             comp_map: dict,
                             receipt=None) -> 'CheckpointResult':
        """兼容包装: stage + publish 立即执行 (旧调用点/单测).

        新协议路径请用 `commit_tx.run_commit_transaction` —— 它把 TA
        finalize 夹在 stage 与 publish 之间, finalize 失败时 Cloud 状态
        不推进 (P0-2)。本包装不具备该保护, 仅用于不涉及 receipt 的兼容
        调用与既有单测。
        """
        staged = self.p2_stage_checkpoint(
            selected_packages, final_model, raw_sum_int, M_Q, H_mask_raw,
            corrected_sum, comp_map, receipt=receipt)
        result = self.p2_publish_staged(staged)
        self.p2_complete_staged(staged)
        return result
