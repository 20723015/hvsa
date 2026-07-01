import sys
import pickle
import logging
import numpy as np

try:
    from utils import KA, LHH, PRG, SIG
except ImportError:
    print("错误: 无法从 utils.py 导入 KA, LHH, PRG 或 SIG。")
    sys.exit(1)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cloud import CloudServer

logger = logging.getLogger(__name__)


class TrustedAuthority:
    """
    可信机构 (TA)

    职责:
    1. 与用户协商 DH 种子
    2. 生成 LHH 公共参数
    3. PKI: 为 ES 生成 Ed25519 签名密钥对
    4. 每轮根据在线用户列表计算全局掩码 M_global 和 H_mask
    """

    def __init__(self, gradient_dim: int, ta_id: str = "TA-01"):
        if not isinstance(gradient_dim, int):
            raise TypeError(f"TA: gradient_dim 必须是 int，而不是 {type(gradient_dim)}")
        self.id = ta_id
        self.gradient_dim = gradient_dim

        # 1. KA 密钥 (用于 User)
        try:
            self.ka_pub_key, self.__ka_priv_key = KA.gen()
            logger.info(f"TA {self.id}: 成功生成 DH 密钥对。")
        except Exception as e:
            logger.error(f"TA {self.id} 失败：无法使用 KA.gen() - {e}")
            sys.exit(1)

        self._user_secrets = {}

        # 2. LHH 参数
        logger.info(f"TA {self.id}: 正在生成 LHH 公共参数 (基于 NIST P-256, dim={gradient_dim})...")
        self.lhh_pp = LHH.Gen(gradient_dim)

        # 3. PKI 系统
        self.es_public_keys = {}
        self._verification_required_edges = set()

        # 4. 每轮的状态
        self._current_round = -1
        self._all_online_users = set()
        self.h_mask_all = None

        logger.info(f"TA {self.id} 已初始化。")

    # --- User 注册 ---
    def get_public_key_bytes(self) -> bytes:
        return self.ka_pub_key

    def register_user(self, user_id: str, user_public_key_bytes: bytes):
        try:
            secret_seed = KA.agree(self.__ka_priv_key, user_public_key_bytes)
            self._user_secrets[user_id] = secret_seed
            logger.info(f"TA: 成功注册用户 {user_id}")
        except Exception as e:
            logger.error(f"TA: 注册用户 {user_id} 失败: {e}")

    # --- EdgeServer 注册 (PKI) ---
    def register_edge_server(self, edge_id: str, require_verification: bool = True):
        try:
            pub_key, priv_key = SIG.gen()
            self.es_public_keys[edge_id] = pub_key
            if require_verification:
                self._verification_required_edges.add(edge_id)
            else:
                self._verification_required_edges.discard(edge_id)
            logger.info(f"TA: 已为 {edge_id} 生成签名密钥对。")
            return priv_key
        except Exception as e:
            logger.error(f"TA: 注册 EdgeServer 失败: {e}")
            return None

    def get_all_es_public_keys(self):
        return self.es_public_keys

    # --- 验证与 LHH ---
    def get_lhh_pp(self):
        return self.lhh_pp

    def get_verification_info(self) -> tuple:
        return self.lhh_pp, self.h_mask_all

    # --- 掩码计算 ---
    def _generate_mask_for_user_int(self, user_id: str, round_t: int) -> np.ndarray:
        secret_seed = self._user_secrets.get(user_id)
        if secret_seed is None:
            raise ValueError(f"TA 秘密库中未找到用户 {user_id}")
        return PRG.mask_int(secret_seed, user_id, round_t, self.gradient_dim)

    def compute_global_mask(self, online_user_ids: list[str], round_t: int) -> np.ndarray:
        global_mask = np.zeros(self.gradient_dim, dtype=np.int64)
        for user_id in online_user_ids:
            try:
                global_mask += self._generate_mask_for_user_int(user_id, round_t)
            except ValueError as e:
                logger.warning(f"TA 警告: {e} - 跳过。")
        return global_mask

    # --- 轮次逻辑 ---
    def start_new_round(self, round_t: int):
        self._current_round = round_t
        self._all_online_users = set()
        self.h_mask_all = None
        logger.info(f"TA (R{round_t}): 已开始新一轮。")

    def receive_online_list(self, msg_bytes: bytes):
        try:
            data = pickle.loads(msg_bytes)
            if data['round'] != self._current_round:
                return
            self._all_online_users.update(data['online_users'])
            logger.info(f"TA (R{self._current_round}): 收到列表。当前在线总数: {len(self._all_online_users)}")
        except Exception as e:
            logger.error(f"TA Error: {e}")

    def trigger_global_mask_computation(self, cloud_server: 'CloudServer'):
        """
        计算全局掩码 M_global，发送给 CloudServer，
        同时计算 H_mask 供 ES 验证。
        """
        num_online = len(self._all_online_users)

        if num_online == 0:
            logger.warning(f"TA {self.id} (R{self._current_round}): 没有在线用户。")
            global_mask = np.zeros(self.gradient_dim, dtype=np.int64)
        else:
            logger.info(f"TA {self.id} (R{self._current_round}): 计算全局掩码...")
            global_mask = self.compute_global_mask(list(self._all_online_users), self._current_round)

        msg_cs = {
            "ta_id": self.id,
            "round": self._current_round,
            "global_mask": global_mask,
            "num_online_users": num_online
        }
        cloud_server.receive_global_mask(pickle.dumps(msg_cs))
        logger.info(f"TA {self.id} (R{self._current_round}): 全局掩码已发送至 CloudServer。")

        self.h_mask_all = LHH.Hash(self.lhh_pp, global_mask)
        logger.info(f"TA {self.id} (R{self._current_round}): H_mask 计算完成。")

    # ============================================================
    #  研究点2: buffer/checkpoint 半异步方法
    #  不破坏研究点1的同步接口。
    # ============================================================

    def point2_reset(self):
        """初始化/重置 point2 专用状态."""
        self._p2_package_users: dict[str, list[str]] = {}  # package_token -> [user_ids]
        self._p2_checkpoint_h_mask: dict[int, object] = {}  # checkpoint_id -> H_mask
        self._p2_checkpoint_M_Q: dict[int, np.ndarray] = {}  # checkpoint_id -> M_Q

    def p2_receive_package_online_list(self, package_token: str, edge_id: str,
                                        base_version: int, online_users: list[str]):
        """point2: ES 上报组包在线用户.

        对应文档 Sec.4.3: TA 按组包记录参与集合，
        不向 Cloud 单独释放单个组包的 M_k。
        """
        self._p2_package_users[package_token] = list(online_users)
        logger.debug(f"TA {self.id}: p2收到组包 {package_token} "
                     f"edge={edge_id} base_v={base_version} n={len(online_users)}")

    def p2_build_checkpoint_mask(self, selected_packages: list) -> tuple:
        """point2: 仅对 checkpoint 选中的组包集合返回聚合掩码.

        对应文档 Sec.4.3 + Sec.4.5:
        - 只计算 M_Q(v) = Σ M_k(r_k) for Π_k ∈ Q(v)
        - 不提前释放任何单组 M_k

        Args:
            selected_packages: list of GroupPackage

        Returns:
            (M_Q, H_mask_raw): 聚合掩码和及其 LHH 哈希
        """
        M_Q = np.zeros(self.gradient_dim, dtype=np.int64)
        for pkg in selected_packages:
            users = self._p2_package_users.get(pkg.package_token, [])
            if users:
                M_k = self.compute_global_mask(users, pkg.package_token)
                M_Q += M_k

        H_mask_raw = LHH.Hash(self.lhh_pp, M_Q)

        logger.info(f"TA {self.id}: p2 checkpoint mask built, "
                    f"packages={len(selected_packages)} n_users_tracked={len(self._p2_package_users)}")

        return M_Q, H_mask_raw
