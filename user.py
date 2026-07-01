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

from typing import TYPE_CHECKING

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
        self._p2_cached_flat: np.ndarray | None = None   # w_i^old
        self._p2_last_accepted_version: int = -1          # t_i^last
        self._p2_pending_plain_flat: np.ndarray | None = None

    def p2_set_accepted_cache(self, flat_model: np.ndarray, version: int):
        """Checkpoint 成功后刷新本地缓存."""
        self._p2_cached_flat = flat_model.astype(np.float64).copy()
        self._p2_last_accepted_version = version

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
    ) -> np.ndarray | None:
        """point2: 训练 → stale reuse 混合 → 掩码 → 上传.

        对应文档 Sec.4.2:
        1. 设置全局模型 W(global_version)
        2. 本地训练 → w_i^new
        3. Stale reuse 混合: w_i = α_i * w_new + (1-α_i) * w_old
        4. 掩码上传

        Args:
            global_flat: 当前全局模型展平向量
            global_version: 当前全局版本 v
            package_token: 组包标识
            model_meta: 模型结构元数据
            train_loader: 训练数据
            local_epochs: 本地训练轮数
            lr: 学习率
            staleness_beta: β_c, stale reuse 衰减参数

        Returns:
            uploaded mixed flat (plaintext), or None on failure
        """
        try:
            # 1. 设置全局模型并训练
            unflatten_parameters(self.model, global_flat, model_meta)
            local_train(self.model, train_loader, local_epochs, lr)

            # 2. 获取训练后新模型
            new_flat, _ = flatten_parameters(self.model)

            # 3. Stale reuse 混合
            if self._p2_cached_flat is not None and self._p2_last_accepted_version >= 0:
                E_i = global_version - self._p2_last_accepted_version
                if E_i > 0:
                    import math
                    alpha_i = math.exp(-staleness_beta * E_i)
                    mixed_flat = (alpha_i * new_flat +
                                  (1.0 - alpha_i) * self._p2_cached_flat)
                else:
                    mixed_flat = new_flat
            else:
                mixed_flat = new_flat

            self._p2_pending_plain_flat = mixed_flat.astype(np.float64).copy()

            # 4. 量化 + 掩码 + 上传 (不传 server，只返回 masked 向量)
            dim = mixed_flat.shape[0]
            model_int = (mixed_flat * QUANTIZE_SCALE).astype(np.int64)
            mask_int = PRG.mask_int(self._ta_seed, self.id, package_token, dim)
            masked_int = model_int + mask_int

            return masked_int

        except Exception as e:
            logger.error(f"User {self.id}: point2训练/混合/掩码失败: {e}")
            import traceback
            traceback.print_exc()
            return None
