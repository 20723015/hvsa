import sys
import pickle
import logging
import numpy as np

try:
    from utils import KA, PRG, QUANTIZE_SCALE
except ImportError:
    print("错误: 无法从 utils.py 导入 KA/PRG。请确保 utils.py 存在。")
    sys.exit(1)

import torch

try:
    from models import train_local, flatten_parameters, unflatten_parameters
except ImportError:
    print("错误: 无法从 models.py 导入。请确保 models.py 存在。")
    sys.exit(1)

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from edge import EdgeServer
    from models import MLP_MNIST, CNN_CIFAR10

logger = logging.getLogger(__name__)


class User:
    """
    轻量级安全聚合方案的客户端。

    与 TA 通过 DH 协商唯一种子 s_i。
    掩码 = PRG(s_i)，TA 可精确取消掩码。
    """

    def __init__(self, id: str, model: 'torch.nn.Module'):
        self.id = id
        self.model = model

        # 生成一对长期 DH 密钥
        logger.info(f"User {self.id}: 正在生成长期 DH 密钥...")
        try:
            self._ka_pub_key, self._ka_priv_key = KA.gen()
        except Exception as e:
            logger.error(f"User {self.id}: DH 密钥对生成失败! {e}")
            raise

        self._ta_seed = None
        self._ta_registered = False

        logger.info(f"User {self.id} 已初始化。")

    def get_public_key_bytes(self) -> bytes:
        return self._ka_pub_key

    def register_with_ta(self, ta_pub_key: bytes):
        """
        【阶段 1: 初始化】
        使用 TA 的公钥，计算并存储共享的秘密种子。
        """
        try:
            seed = KA.agree(self._ka_priv_key, ta_pub_key)
            self._ta_seed = seed
            self._ta_registered = True
            logger.info(f"User {self.id}: 成功与 TA 协商秘密种子。")
        except Exception as e:
            logger.error(f"User {self.id}: 与 TA 协商秘密种子失败: {e}")
            raise

    def is_registered(self) -> bool:
        return self._ta_registered

    def local_train(self, train_loader, epochs: int, lr: float):
        logger.info(f"User {self.id}: 开始本地训练...")
        train_local(self.model, train_loader, epochs, lr)
        logger.info(f"User {self.id}: 本地训练完成。")

    def mask_and_upload_model(self, round_t: int, server: 'EdgeServer'):
        try:
            flat_model_params, _ = flatten_parameters(self.model)

            # 量化为整数
            model_int = (flat_model_params * QUANTIZE_SCALE).astype(np.int64)

            # 生成整数掩码
            ta_mask_int = self._generate_ta_mask_int(round_t, model_int.shape[0])

            # 整数加法
            masked_model_int = model_int + ta_mask_int

            msg = {
                "user_id": self.id,
                "round": round_t,
                "masked_gradient": masked_model_int
            }
            server.receive_masked_gradient(pickle.dumps(msg))

        except Exception as e:
            logger.error(f"User {self.id}: 上传失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_ta_mask_int(self, round_t: int, dim: int) -> np.ndarray:
        """mask = PRG(s_i)，TA 可精确取消。"""
        if not self._ta_registered:
            raise Exception(f"User {self.id}: 尚未完成 TA 注册")
        return PRG.mask_int(self._ta_seed, self.id, round_t, dim)

    def set_global_model(self, avg_flat_model: np.ndarray, model_meta: list):
        try:
            unflatten_parameters(self.model, avg_flat_model, model_meta)
            logger.info(f"User {self.id}: 已成功设置新的全局模型。")
        except Exception as e:
            logger.error(f"User {self.id}: 设置全局模型失败: {e}")

    # ============================================================
    #  研究点2: stale reuse + 本地缓存
    #  对应文档 Sec.4.2 + Sec.6
    # ============================================================

    def point2_init_cache(self):
        """初始化 point2 缓存状态."""
        self._p2_stale_flat: Optional[np.ndarray] = None      # w_i^{stale,k}: 因上传失败而缓存的本地模型
        self._p2_stale_base_flat: Optional[np.ndarray] = None  # W(r_stale): stale 模型的训练基 (delta 重锚定用)
        self._p2_stale_version: int = -1                    # t_i: stale 结果的产生版本
        self._p2_pending_plain_flat: Optional[np.ndarray] = None  # 本轮新训练的模型 (上传前)
        self._p2_pending_base_flat: Optional[np.ndarray] = None  # 本轮训练基 W(r_k) (delta 重锚定用)
        self._p2_last_training_base_version: int = -1        # 本轮训练基版本
        self._p2_sample_size: int = 0                         # |D_i^k|
        self._p2_upd_norm_ema: Optional[float] = None         # 自身更新范数 EMA (自剪裁用)

    def p2_set_sample_size(self, sample_size: int):
        """设置本地样本量."""
        self._p2_sample_size = sample_size

    def p2_confirm_upload_failed(self):
        """上传失败: 缓存本轮新训练的模型为 stale 候选.

        对应文档 Sec.4.2 L301-304: 上传失败时缓存 w_i^{new,k} 作为下次 stale 候选.
        """
        if self._p2_pending_plain_flat is not None:
            self._p2_stale_flat = self._p2_pending_plain_flat.astype(np.float64).copy()
            self._p2_stale_base_flat = (self._p2_pending_base_flat.astype(np.float64).copy()
                                        if self._p2_pending_base_flat is not None else None)
            self._p2_stale_version = self._p2_last_training_base_version
            self._p2_pending_plain_flat = None
            self._p2_pending_base_flat = None

    def p2_confirm_consumed(self):
        """Checkpoint 确认消费: 清除 stale 候选.

        对应文档 Sec.4.2 L305-307: 上传成功并被 ES 确认接收后清除 stale.
        """
        self._p2_stale_flat = None
        self._p2_stale_base_flat = None
        self._p2_stale_version = -1
        self._p2_pending_plain_flat = None
        self._p2_pending_base_flat = None

    def p2_get_stale_info(self) -> tuple:
        """返回 (has_stale, stale_version) 用于指标记录."""
        return (self._p2_stale_flat is not None, self._p2_stale_version)

    def p2_train_offline_and_cache(self,
                                   train_loader,
                                   local_epochs: int,
                                   lr: float,
                                   global_version: int,
                                   model_meta: list):
        """MAFSA 忠实语义: 掉线轮在本地旧模型上继续训练并缓存 (算力回收).

        对应 MAFSA Algorithm 1 掉线分支:
        - 不接收全局模型, 在 self.model (上一轮结束状态, 或已有 stale 缓存) 上继续训练
        - 完成后缓存为 stale 候选 (_p2_stale_flat); 恢复上线后由 mafsa/ours
          混合规则自动回收该成果 (expired update reuse)
        - 不产生 pending (本轮不上传)

        Args:
            train_loader: 本地数据
            local_epochs / lr: 训练参数
            global_version: 当前基版本 (记录 stale_version, 供 E_i 计算)
            model_meta: 模型结构元数据 (unflatten 用)
        """
        if self._p2_stale_flat is not None:
            # 掉线多轮: 在缓存旧模型之上继续训练 (累积掉线期间成果)
            unflatten_parameters(self.model, self._p2_stale_flat, model_meta)
        # else: self.model 保持上一轮训练结束状态, 直接继续训练
        train_local(self.model, train_loader, local_epochs, lr)
        new_flat, _ = flatten_parameters(self.model)
        self._p2_stale_flat = new_flat.astype(np.float64).copy()
        self._p2_stale_base_flat = None  # 非全局基, 不可 delta rebase
        self._p2_stale_version = global_version
        self._p2_pending_plain_flat = None

    def p2_make_dropout_reuse_masked(self,
                                     global_flat: np.ndarray,
                                     global_version: int,
                                     package_token: str) -> Optional[np.ndarray]:
        """训练前掉线时的 stale reuse: 用缓存旧模型重锚定后补交贡献.

        对应文档 Sec.4.2 (reuse 救掉线):
        - 无 stale 缓存时返回 None (掉线即缺失)
        - 有缓存时做 Delta Rebase 重锚定:
              w̃ = W(r_k) + (w_stale − W(r_stale))
          使贡献严格锚定当前基 W(r_k), 不破坏 VCM 前提
        - 之后照常按样本量加权 + 量化 + 加掩码返回 masked int

        Args:
            global_flat: 当前组训练基 W(r_k)
            global_version: 当前基版本 (用于 mask 派生, 未直接使用)
            package_token: 组包 token (mask 派生输入)

        Returns:
            masked_int (int64) 或 None
        """
        if self._p2_stale_flat is None:
            return None
        if self._p2_stale_base_flat is not None:
            # 有全局锚: Delta Rebase 重锚定到当前基 (不破坏 VCM 前提)
            rebased = global_flat.astype(np.float64) + (self._p2_stale_flat - self._p2_stale_base_flat)
        else:
            # offline_train 链式成果 (锚点非全局版本): 按 MAFSA 本义直接提交
            # 完整陈旧模型 (等价于一个陈旧参与者的 FedAvg 贡献, 权重仍由
            # 样本量承担); 断网不断算场景的 SecAgg 干净旁路
            rebased = self._p2_stale_flat.astype(np.float64)
        dim = rebased.shape[0]
        model_int = (rebased * self._p2_sample_size * QUANTIZE_SCALE).astype(np.int64)
        mask_int = PRG.mask_int(self._ta_seed, self.id, package_token, dim)
        return model_int + mask_int

    def p2_train_mix_mask_and_upload(
        self,
        global_flat: np.ndarray,
        global_version: int,
        package_token: str,
        model_meta: list,
        train_loader,
        local_epochs: int,
        lr: float,
        staleness_beta: float = 0.3,
        reuse_rule: str = 'ours',
        rho_max: float = 0.5,
        fedprox_mu: float = 0.0,
        curriculum: str = 'none',
        dpr_q: float = 0.0,
        clip_upd_k: float = 0.0,
    ) -> Optional[np.ndarray]:
        """point2: 训练 → stale reuse 混合 → 掩码 → 上传.

        对应文档 Sec.4.2 + Sec 8.1 消融表:
        - reuse_rule='none': 无 stale reuse
        - reuse_rule='mafsa': MAFSA 原始方向 (α 衰减 fresh, 陈旧结果权重递增)
        - reuse_rule='ours': Ours Decayed-Stale (ρ 衰减 stale, 陈旧结果权重递减), 参数空间混合
        - reuse_rule='ours_delta': Ours Delta Rebase (增量空间混合, 重锚定到当前基 W(r_k))

        Args:
            global_flat: 当前全局模型展平向量 W(t_k)
            global_version: 当前组训练基版本 t_k
            package_token: 组包标识
            model_meta: 模型结构元数据
            train_loader: 训练数据
            local_epochs: 本地训练轮数
            lr: 学习率
            staleness_beta: β_c, stale 权重衰减速度
            reuse_rule: 复用规则 {'none', 'mafsa', 'ours'}
            rho_max: ρ_max, stale 结果最大权重 (仅 ours 生效)

        Returns:
            masked_int (int64) or None on failure
        """
        try:
            # 1. 设置全局模型并训练
            unflatten_parameters(self.model, global_flat, model_meta)
            train_local(self.model, train_loader, local_epochs, lr,
                        prox_mu=fedprox_mu, curriculum=curriculum, dpr_q=dpr_q)

            # 2. 获取训练后新模型 w_i^{new,k}
            new_flat, _ = flatten_parameters(self.model)
            self._p2_last_training_base_version = global_version

            # 3. Stale reuse 混合
            if (reuse_rule != 'none'
                    and self._p2_stale_flat is not None
                    and self._p2_stale_version >= 0):
                E_i = global_version - self._p2_stale_version
                if E_i > 0:
                    import math
                    if reuse_rule == 'ours' or reuse_rule == 'ours_delta':
                        # Ours Decayed-Stale: ρ_i = ρ_max · exp(-β_c · E_i)
                        # w_i = (1-ρ_i)·w_new + ρ_i·w_stale
                        rho_i = rho_max * math.exp(-staleness_beta * E_i)
                        if reuse_rule == 'ours_delta' and self._p2_stale_base_flat is not None:
                            # Delta Rebase (增量空间重锚定):
                            #   w̃ = W(r_k) + (1−ρ)·Δ_new + ρ·Δ_stale
                            #   Δ_new = w_new − W(r_k),  Δ_stale = w_stale − W(r_stale)
                            # 修复参数空间混合把旧锚点漂移 W(r_stale)−W(r_k)
                            # 注入组包、破坏 VCM 前提的缺陷 (见 docs/POINT2_VCM_REUSE_DIAGNOSIS.md)
                            delta_new = new_flat - global_flat
                            delta_stale = self._p2_stale_flat - self._p2_stale_base_flat
                            mixed_flat = (global_flat
                                          + (1.0 - rho_i) * delta_new
                                          + rho_i * delta_stale)
                        else:
                            mixed_flat = (1.0 - rho_i) * new_flat + rho_i * self._p2_stale_flat
                    elif reuse_rule == 'mafsa':
                        # MAFSA Original: α_i = exp(-β_c · E_i)
                        # w_i = α_i·w_new + (1-α_i)·w_stale
                        alpha_i = math.exp(-staleness_beta * E_i)
                        mixed_flat = alpha_i * new_flat + (1.0 - alpha_i) * self._p2_stale_flat
                    else:
                        mixed_flat = new_flat
                else:
                    mixed_flat = new_flat
            else:
                mixed_flat = new_flat

            # 保存本轮新训练的模型 (用于上传失败时缓存 stale)
            self._p2_pending_plain_flat = new_flat.astype(np.float64).copy()
            self._p2_pending_base_flat = global_flat.astype(np.float64).copy()

            # 客户端自剪裁 (arXiv:2606.13287 裁剪-延迟解耦的客户端实现):
            # 上传更新范数超过自身历史 EMA 的 k 倍时截断, 抑制发散更新;
            # EMA 只依赖自己的历史, 无需服务器信息, SecAgg 兼容 (掩码前执行)
            delta_flat = (mixed_flat.astype(np.float64)
                          - global_flat.astype(np.float64))
            upd_norm = float(np.linalg.norm(delta_flat))
            if clip_upd_k > 0:
                if self._p2_upd_norm_ema is None:
                    self._p2_upd_norm_ema = upd_norm
                elif upd_norm > clip_upd_k * self._p2_upd_norm_ema:
                    scale = (clip_upd_k * self._p2_upd_norm_ema
                             / (upd_norm + 1e-12))
                    delta_flat = delta_flat * scale
                    mixed_flat = (global_flat.astype(np.float64) + delta_flat)
                self._p2_upd_norm_ema = (0.9 * self._p2_upd_norm_ema
                                         + 0.1 * float(np.linalg.norm(delta_flat)))

            # 上传明文元数据: 本地更新范数 m̂_i=‖w̃−W(r_k)‖ (DVAFL 式(15) 一阶矩,
            # 供云端矩校准 VCM 使用; 标量泄露量级不泄露方向, 随包以明文上报)
            self._p2_last_upd_norm = float(np.linalg.norm(delta_flat))

            # 4. 样本量加权 + 量化 + 掩码 + 上传
            #    w̄_i^k = s_i^k·w_i^k + m_i^k: 权重只乘明文不乘掩码,
            #    否则 TA 求掩码和时需按样本量加权, 破坏单 TA 掩码链路
            dim = mixed_flat.shape[0]
            model_int = (mixed_flat * self._p2_sample_size * QUANTIZE_SCALE).astype(np.int64)
            mask_int = PRG.mask_int(self._ta_seed, self.id, package_token, dim)
            masked_int = model_int + mask_int

            return masked_int

        except Exception as e:
            logger.error(f"User {self.id}: point2训练/混合/掩码失败: {e}")
            import traceback
            traceback.print_exc()
            return None
