import os
import json
import sqlite3
import sys
import pickle
import logging
from datetime import datetime, timezone

import numpy as np

try:
    from utils import KA, LHH, PRG, SIG
except ImportError:
    print("错误: 无法从 utils.py 导入 KA, LHH, PRG 或 SIG。")
    sys.exit(1)

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cloud import CloudServer
    from point2_types import (CheckpointReceipt, CheckpointVote,
                              ProtocolConfig)

logger = logging.getLogger(__name__)


def _load_or_create_receipt_keys(key_dir: Optional[str]) -> tuple:
    """TA receipt 签名密钥: 给定目录时 load-or-create 持久化.

    与持久化 ledger (--ta-state-path) 配对使用: 否则每次进程启动都会重新
    生成密钥, 重启后的 TA 无法再验证此前签发的 receipt (也无法与 ES/Cloud
    共享同一公钥), ledger 记录虽在却不可核对。
    """
    if not key_dir:
        return SIG.gen()
    pub_p = os.path.join(key_dir, 'pub.ed25519')
    priv_p = os.path.join(key_dir, 'priv.ed25519')
    if os.path.exists(pub_p) and os.path.exists(priv_p):
        with open(pub_p, 'rb') as f:
            pub = f.read()
        with open(priv_p, 'rb') as f:
            priv = f.read()
        return pub, priv
    return SIG.gen(key_dir)


class TALedger:
    """TA 权威 ledger: 仅追加的 token 注册/消费与 checkpoint 版本记录.

    SQLite 持久化 (path=None 时退化为 ':memory:', 兼容测试/性能模式):
    - registered_tokens:    ES 封包时注册的合法 token (PRIMARY KEY)
    - prepared_checkpoints: prepare 阶段原子保存的**完整 receipt** (含签名
                            pickle) + token 元数据 (token→edge_id/base_version)
                            + result_digest, 按 checkpoint_version 唯一。
                            finalize 只有核对到对应 prepare 记录且 receipt
                            摘要一致才允许落地 —— 伪造或未经 prepare 签发的
                            receipt 无法写 ledger (P0-1); 记录落在持久化 DB 中,
                            进程崩溃后依然可核对。
    - checkpoints:          已 finalize 的 checkpoint 版本记录 (version
                            PRIMARY KEY, receipt_id UNIQUE)
    - consumed_tokens:      已随某个成功 finalize 的 checkpoint 消费的 token
    写入在单一事务内完成, 依赖主键/唯一约束保证 only-append。
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._conn = sqlite3.connect(path or ':memory:')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS registered_tokens ('
            ' token TEXT PRIMARY KEY, edge_id TEXT NOT NULL,'
            ' base_version INTEGER NOT NULL)')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS consumed_tokens ('
            ' token TEXT PRIMARY KEY, checkpoint_version INTEGER NOT NULL,'
            ' edge_id TEXT NOT NULL, base_version INTEGER NOT NULL)')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS checkpoints ('
            ' version INTEGER PRIMARY KEY, receipt_id TEXT NOT NULL UNIQUE,'
            ' token_digest TEXT NOT NULL, trigger_mode TEXT NOT NULL,'
            ' committed_at TEXT NOT NULL)')
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS prepared_checkpoints ('
            ' checkpoint_version INTEGER PRIMARY KEY,'
            ' receipt_id TEXT NOT NULL UNIQUE,'
            ' receipt_digest TEXT NOT NULL, receipt_pickle BLOB NOT NULL,'
            ' token_digest TEXT NOT NULL, trigger_mode TEXT NOT NULL,'
            ' result_digest TEXT NOT NULL, tokens_json TEXT NOT NULL,'
            ' token_meta_json TEXT NOT NULL, prepared_at TEXT NOT NULL,'
            ' finalized INTEGER NOT NULL DEFAULT 0, finalized_at TEXT)')
        self._conn.commit()

    # --- 查询 ---
    def next_checkpoint_version(self) -> int:
        row = self._conn.execute(
            'SELECT COALESCE(MAX(version) + 1, 0) FROM checkpoints'
        ).fetchone()
        return int(row[0])

    def is_registered(self, token: str) -> bool:
        return self._conn.execute(
            'SELECT 1 FROM registered_tokens WHERE token = ?',
            (token,)).fetchone() is not None

    def is_consumed(self, token: str) -> bool:
        return self._conn.execute(
            'SELECT 1 FROM consumed_tokens WHERE token = ?',
            (token,)).fetchone() is not None

    def receipt_exists(self, receipt_id: str) -> bool:
        return self._conn.execute(
            'SELECT 1 FROM checkpoints WHERE receipt_id = ?',
            (receipt_id,)).fetchone() is not None

    # --- prepare 记录 (P0-1: finalize 的权威依据) ---
    def get_prepared(self, receipt_id: str) -> Optional[dict]:
        row = self._conn.execute(
            'SELECT checkpoint_version, receipt_id, receipt_digest,'
            ' receipt_pickle, token_digest, trigger_mode, result_digest,'
            ' tokens_json, token_meta_json, finalized'
            ' FROM prepared_checkpoints WHERE receipt_id = ?',
            (receipt_id,)).fetchone()
        return self._row_to_prepared(row)

    def get_prepared_by_version(self, version: int) -> Optional[dict]:
        row = self._conn.execute(
            'SELECT checkpoint_version, receipt_id, receipt_digest,'
            ' receipt_pickle, token_digest, trigger_mode, result_digest,'
            ' tokens_json, token_meta_json, finalized'
            ' FROM prepared_checkpoints WHERE checkpoint_version = ?',
            (int(version),)).fetchone()
        return self._row_to_prepared(row)

    @staticmethod
    def _row_to_prepared(row) -> Optional[dict]:
        if row is None:
            return None
        return {
            'checkpoint_version': int(row[0]), 'receipt_id': row[1],
            'receipt_digest': row[2], 'receipt_pickle': row[3],
            'token_digest': row[4], 'trigger_mode': row[5],
            'result_digest': row[6], 'tokens': json.loads(row[7]),
            'token_meta': json.loads(row[8]), 'finalized': bool(row[9]),
        }

    # --- 写入 (仅追加) ---
    def register_token(self, token: str, edge_id: str,
                       base_version: int) -> None:
        try:
            self._conn.execute(
                'INSERT INTO registered_tokens (token, edge_id, base_version)'
                ' VALUES (?, ?, ?)', (token, edge_id, int(base_version)))
            self._conn.commit()
        except sqlite3.IntegrityError:
            pass  # 重复注册幂等

    def prepare_checkpoint(self, version: int, receipt_id: str,
                           receipt_digest: str, receipt_pickle: bytes,
                           token_digest: str, trigger_mode: str,
                           result_digest: str, tokens: list,
                           token_meta: list) -> tuple:
        """原子保存 prepare 记录, 返回 (receipt_pickle, status).

        status ∈ {created, exists, replaced}:
        - created: 新候选的 prepare 记录;
        - exists:  同一 receipt (摘要一致) 重复 prepare → 幂等, 返回已存
                   receipt 字节 (保证调用方拿到与记录逐字节一致的 receipt);
        - replaced: 同版本存在另一候选 (上一次 prepare 的 receipt 未被
                   finalize 即被放弃) → 原子替换, 旧 receipt 随之作废
                   (finalize 会因摘要不符/无记录而拒绝)。

        已 finalize 的版本不可再 prepare (授权阶段的版本连续性检查已先
        拒绝; 此处再兜底抛错, 保证不可回退)。
        """
        params = (int(version), receipt_id, receipt_digest,
                  sqlite3.Binary(receipt_pickle), token_digest, trigger_mode,
                  result_digest, json.dumps(sorted(str(t) for t in tokens)),
                  json.dumps(token_meta),
                  datetime.now(timezone.utc).isoformat())
        try:
            self._conn.execute(
                'INSERT INTO prepared_checkpoints (checkpoint_version,'
                ' receipt_id, receipt_digest, receipt_pickle, token_digest,'
                ' trigger_mode, result_digest, tokens_json, token_meta_json,'
                ' prepared_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                params)
            self._conn.commit()
            return receipt_pickle, 'created'
        except sqlite3.IntegrityError:
            self._conn.rollback()

        existing = self.get_prepared_by_version(version)
        if existing is not None and existing['receipt_digest'] == receipt_digest:
            return existing['receipt_pickle'], 'exists'
        if existing is not None and existing['finalized']:
            raise ValueError(
                f"prepare 拒绝: v{version} 已 finalize, 不可再 prepare")
        if existing is not None and existing['receipt_id'] != receipt_id:
            logger.warning(
                f"TALedger: v{version} 的 prepare 记录被替换 "
                f"({existing['receipt_id']} → {receipt_id}); 旧 receipt 作废")
        try:
            self._conn.execute('BEGIN')
            self._conn.execute(
                'DELETE FROM prepared_checkpoints WHERE checkpoint_version = ?',
                (int(version),))
            self._conn.execute(
                'INSERT INTO prepared_checkpoints (checkpoint_version,'
                ' receipt_id, receipt_digest, receipt_pickle, token_digest,'
                ' trigger_mode, result_digest, tokens_json, token_meta_json,'
                ' prepared_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                params)
            self._conn.commit()
            return receipt_pickle, 'replaced'
        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            raise ValueError(f"prepare 拒绝: {e}") from e

    def commit_prepared(self, receipt_id: str) -> str:
        """finalize 落地: 版本记录 + token 消费 + 标记 prepare 已 finalize.

        单一事务; 已 finalize 的记录直接返回 'already' (幂等重放)。
        """
        row = self.get_prepared(receipt_id)
        if row is None:
            raise ValueError(f"finalize 拒绝: 无 prepare 记录 ({receipt_id})")
        if row['finalized']:
            return 'already'
        try:
            self._conn.execute('BEGIN')
            self._conn.execute(
                'INSERT INTO checkpoints (version, receipt_id, token_digest,'
                ' trigger_mode, committed_at) VALUES (?, ?, ?, ?, ?)',
                (row['checkpoint_version'], receipt_id, row['token_digest'],
                 row['trigger_mode'], datetime.now(timezone.utc).isoformat()))
            for tok, eid, bv in row['token_meta']:
                self._conn.execute(
                    'INSERT INTO consumed_tokens (token, checkpoint_version,'
                    ' edge_id, base_version) VALUES (?, ?, ?, ?)',
                    (tok, row['checkpoint_version'], eid, int(bv)))
            self._conn.execute(
                'UPDATE prepared_checkpoints SET finalized = 1,'
                ' finalized_at = ? WHERE receipt_id = ?',
                (datetime.now(timezone.utc).isoformat(), receipt_id))
            self._conn.commit()
            return 'committed'
        except sqlite3.IntegrityError as e:
            self._conn.rollback()
            raise ValueError(f"TA ledger 拒绝: {e}") from e


class TrustedAuthority:
    """
    可信机构 (TA)

    职责:
    1. 与用户协商 DH 种子
    2. 生成 LHH 公共参数
    3. PKI: 为 ES 生成 Ed25519 签名密钥对
    4. 每轮根据在线用户列表计算全局掩码 M_global 和 H_mask
    """

    def __init__(self, gradient_dim: int, ta_id: str = "TA-01",
                 enable_verification: bool = True,
                 ledger_path: Optional[str] = None):
        if not isinstance(gradient_dim, int):
            raise TypeError(f"TA: gradient_dim 必须是 int，而不是 {type(gradient_dim)}")
        self.id = ta_id
        self.gradient_dim = gradient_dim
        self.enable_verification = enable_verification

        # 1. KA 密钥 (用于 User)
        try:
            self.ka_pub_key, self.__ka_priv_key = KA.gen()
            logger.info(f"TA {self.id}: 成功生成 DH 密钥对。")
        except Exception as e:
            logger.error(f"TA {self.id} 失败：无法使用 KA.gen() - {e}")
            sys.exit(1)

        # 1b. Ed25519 密钥 (签发 CheckpointReceipt; ES 从 get_receipt_public_key
        #     获取公钥用于验签 —— 权威提交授权的签名方)。
        #     ledger 持久化时密钥同目录持久化 (load-or-create): 重启后仍可
        #     验证/核对此前签发的 receipt 与 prepare 记录。
        key_dir = None
        if ledger_path:
            key_dir = os.path.join(
                os.path.dirname(os.path.abspath(ledger_path)),
                os.path.basename(ledger_path) + '.keys')
        self.receipt_pub_key, self._receipt_priv_key = (
            _load_or_create_receipt_keys(key_dir))

        # 1c. 权威 ledger (仅追加: token 注册/消费 + checkpoint 版本)
        self.ledger = TALedger(ledger_path)

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

        if self.enable_verification:
            self.h_mask_all = LHH.Hash(self.lhh_pp, global_mask)
            logger.info(f"TA {self.id} (R{self._current_round}): H_mask 计算完成。")
        else:
            self.h_mask_all = None

    # ============================================================
    #  研究点2: buffer/checkpoint 半异步方法
    #  不破坏研究点1的同步接口。
    # ============================================================

    def point2_reset(self):
        """初始化/重置 point2 专用状态."""
        self._p2_package_users: dict[str, list[str]] = {}  # package_token -> [user_ids]
        self._p2_checkpoint_h_mask: dict[int, object] = {}  # checkpoint_id -> H_mask
        self._p2_checkpoint_M_Q: dict[int, np.ndarray] = {}  # checkpoint_id -> M_Q
        # 策略绑定: TA 本地预置协议配置 + 独立维护的组参与计数与共识版本
        # (由 commit 通知推进), 用于对云端声明权重做独立复算 (声明值只作审计)
        self._p2_policy_config = None
        self._p2_participation_counts: dict[str, int] = {}
        self._p2_committed_version: int = 0

    def p2_set_policy_config(self, policy_config) -> None:
        """预置版本化协议配置 (与各 ES 同一实例, 初始化阶段一次性下发)."""
        self._p2_policy_config = policy_config

    def get_receipt_public_key(self) -> bytes:
        """receipt 验签公钥 (ES/Cloud 校验 CheckpointReceipt 用)."""
        return self.receipt_pub_key

    # --- 权威授权流: token 注册 → vote 聚合 → receipt 签发 (ledger 落地) ---

    def p2_register_package_token(self, package_token: str, edge_id: str,
                                  base_version: int) -> None:
        """ES 封包后立即向 TA 注册 token (合法 token 的权威登记)."""
        self.ledger.register_token(package_token, edge_id, base_version)

    def p2_confirm_receipt(self, receipt) -> bool:
        """核对**完整 receipt** 是否已在权威 ledger 落地 (ES 推进前确认).

        校验: TA 签名有效 + 存在对应 prepare 记录 + 记录摘要与 receipt
        逐字节一致 + 该记录已 finalize。只查 receipt_id 无法发现"同 id
        不同内容"的伪造/replay; 传入 str 时退化为按 id 查询 (兼容旧调用)。
        """
        from point2_types import CheckpointReceipt
        if isinstance(receipt, str):
            row = self.ledger.get_prepared(receipt)
            return bool(row is not None and row['finalized'])
        if not isinstance(receipt, CheckpointReceipt):
            return False
        if not SIG.verify(bytes.fromhex(receipt.payload_digest()),
                          receipt.signature, self.receipt_pub_key):
            return False
        row = self.ledger.get_prepared(receipt.receipt_id)
        return bool(row is not None and row['finalized']
                    and row['receipt_digest'] == receipt.payload_digest())

    def p2_finalize_checkpoint(self, receipt) -> str:
        """finalize 阶段: 核对 prepare 记录后把 checkpoint 原子写入权威 ledger.

        必须由 Cloud 凭该 receipt 成功提交后调用。校验链 (P0-1):
        1. receipt 必须是 TA 签发的 CheckpointReceipt 且签名有效 —— 伪造
           receipt 直接拒绝;
        2. ledger 中必须存在同 receipt_id 的 prepare 记录, 且摘要逐字节
           一致 —— 未经 prepare 签发 (或内容被改) 的 receipt 无法落地;
        3. prepare 记录必须属于当前待落地版本 (版本连续性), token 仍注册
           且未消费。

        幂等: 同一 receipt 重复 finalize 直接返回 'already' (不重复消费,
        不改动任何状态) —— 崩溃恢复/重试可安全重放。不同 receipt 与已有
        记录冲突才拒绝 (抛 ValueError)。返回 'committed' | 'already'。
        """
        from point2_types import CheckpointReceipt
        if not isinstance(receipt, CheckpointReceipt):
            raise ValueError("finalize 拒绝: 需要 CheckpointReceipt 实例")
        if not SIG.verify(bytes.fromhex(receipt.payload_digest()),
                          receipt.signature, self.receipt_pub_key):
            raise ValueError("finalize 拒绝: receipt 签名无效 (非 TA 签发)")

        rdig = receipt.payload_digest()
        row = self.ledger.get_prepared(receipt.receipt_id)
        if row is None:
            other = self.ledger.get_prepared_by_version(
                int(receipt.checkpoint_version))
            if other is not None:
                raise ValueError(
                    f"finalize 拒绝: v{receipt.checkpoint_version} 的 prepare "
                    f"记录对应另一 receipt ({other['receipt_id']}), "
                    f"本 receipt 已作废")
            raise ValueError(
                "finalize 拒绝: 无 prepare 记录 (receipt 未经 prepare 签发)")
        if row['receipt_digest'] != rdig:
            raise ValueError(
                "finalize 拒绝: receipt 内容与 prepare 记录不符 (被篡改)")

        if row['finalized']:
            logger.info(f"TA {self.id}: finalize 幂等重放 "
                        f"{receipt.receipt_id} (已落地, 不重复消费)")
            return 'already'

        expected_v = self.ledger.next_checkpoint_version()
        if int(receipt.checkpoint_version) != expected_v:
            raise ValueError(
                f"finalize 拒绝: 版本错位 (receipt v{receipt.checkpoint_version}, "
                f"ledger 期望 v{expected_v})")
        for tok in receipt.selected_tokens:
            if not self.ledger.is_registered(tok):
                raise ValueError(f"finalize 拒绝: token 未向 TA 注册 ({tok})")
            if self.ledger.is_consumed(tok):
                raise ValueError(f"finalize 拒绝: token 已消费 ({tok})")

        status = self.ledger.commit_prepared(receipt.receipt_id)
        logger.info(f"TA {self.id}: finalize {receipt.receipt_id} "
                    f"(v{receipt.checkpoint_version}, "
                    f"{len(receipt.selected_tokens)} token 消费, "
                    f"status={status})")
        return status

    def p2_authorize_checkpoint(self, selected_packages: list,
                                votes: list,
                                checkpoint_version: int,
                                trigger_mode: str,
                                config: 'ProtocolConfig') -> 'CheckpointReceipt':
        """prepare 阶段: 验证全部 ES vote 并签发 CheckpointReceipt.

        全部检查通过后签发带 TA 签名的 receipt, 并把**完整 receipt + token
        元数据 + result_digest** 原子写入持久化 prepared_checkpoints 记录;
        token 消费与版本记录仍由 `p2_finalize_checkpoint` 在 Cloud 提交成功
        后落地, 使"提交失败不推进权威状态"成立 (P0-2)。finalize 会核对本
        记录, 因此未经 prepare 的伪造 receipt 无法落地 (P0-1)。任何检查
        失败抛 ValueError, 不产生 receipt、不写记录。
        """
        from point2_types import CheckpointReceipt, tokens_digest

        if config is None:
            raise ValueError("TA 未预置 ProtocolConfig")
        tokens = [p.package_token for p in selected_packages]
        tdig = tokens_digest(tokens)

        # 1. vote 完整性: 数量、来源、签名、内容一致、全部通过、结果摘要一致
        if not votes:
            raise ValueError("无任何 ES vote")
        seen_voters = set()
        result_digests = set()
        for v in votes:
            pub = self.es_public_keys.get(v.voter)
            if pub is None:
                raise ValueError(f"vote 来自未知 ES: {v.voter}")
            if v.voter in seen_voters:
                raise ValueError(f"重复 vote: {v.voter}")
            seen_voters.add(v.voter)
            if not SIG.verify(
                    bytes.fromhex(v.payload_digest()), v.signature, pub):
                raise ValueError(f"vote 签名无效: {v.voter}")
            if not v.passed:
                raise ValueError(f"vote 未通过: {v.voter} ({v.reason})")
            if v.checkpoint_version != checkpoint_version:
                raise ValueError(f"vote 版本错位: {v.voter}")
            if v.token_digest != tdig:
                raise ValueError(f"vote token 摘要不符: {v.voter}")
            if v.config_digest != config.digest():
                raise ValueError(f"vote 配置摘要不符: {v.voter}")
            if not v.result_digest:
                raise ValueError(f"vote 缺少结果摘要 (P0-1 绑定): {v.voter}")
            result_digests.add(v.result_digest)
        if seen_voters != set(self.es_public_keys.keys()):
            raise ValueError("vote 未覆盖全部 ES (验证必须全体参与)")
        if len(result_digests) != 1:
            raise ValueError("各 ES vote 的结果摘要不一致 (候选被调包?)")
        result_dig = result_digests.pop()

        # 2. 调度约束 (依据 ProtocolConfig; 调度公平性/漏选仍属
        #    anti-censorship 出界, 这里只检查触发语义与时效)
        n = len(selected_packages)
        if trigger_mode == 'normal':
            if n != config.K:
                raise ValueError(f"常规触发必须选 K={config.K} 个包, 实际 {n}")
        elif trigger_mode == 'timeout':
            if not (1 <= n < config.K):
                raise ValueError(f"超时触发要求 1 ≤ |Q| < K, 实际 {n}")
        else:
            raise ValueError(f"未知触发模式: {trigger_mode}")
        for p in selected_packages:
            age = checkpoint_version - p.base_version
            if not (0 <= age <= config.s_max):
                raise ValueError(
                    f"组包 {p.package_token} 陈旧度 {age} 超出 [0, {config.s_max}]")

        # 3. token 状态: 已注册 (ES 封包登记) 且未消费 (ledger 权威状态)
        for p in selected_packages:
            if not self.ledger.is_registered(p.package_token):
                raise ValueError(
                    f"token 未向 TA 注册: {p.package_token} (未消费校验前置)")
            if self.ledger.is_consumed(p.package_token):
                raise ValueError(f"token 已消费 (重放): {p.package_token}")

        # 4. 版本连续性: 本 checkpoint 必须消费下一个未用版本号
        expected_v = self.ledger.next_checkpoint_version()
        if checkpoint_version != expected_v:
            raise ValueError(
                f"checkpoint 版本错位: 声明 v{checkpoint_version}, "
                f"ledger 期望 v{expected_v}")

        # 5. 签发 receipt + 原子保存 prepare 记录 (完整 receipt 字节 + token
        #    元数据 + result_digest)。token 消费与版本记录仍只在 finalize
        #    落地 —— 提交失败不推进权威状态。同一候选重复 prepare 幂等
        #    (返回记录中逐字节一致的 receipt); 同版本换候选则替换记录,
        #    旧 receipt 随之作废 (finalize 核对失败)。
        receipt_id = f"{self.id}:v{checkpoint_version}:{tdig[:16]}"
        receipt = CheckpointReceipt(
            receipt_id=receipt_id, checkpoint_version=checkpoint_version,
            token_digest=tdig, selected_tokens=sorted(tokens),
            trigger_mode=trigger_mode, result_digest=result_dig)
        receipt = CheckpointReceipt(
            receipt_id=receipt.receipt_id,
            checkpoint_version=receipt.checkpoint_version,
            token_digest=receipt.token_digest,
            selected_tokens=receipt.selected_tokens,
            trigger_mode=receipt.trigger_mode,
            result_digest=receipt.result_digest,
            signature=SIG.sign(bytes.fromhex(receipt.payload_digest()),
                               self._receipt_priv_key))
        stored, status = self.ledger.prepare_checkpoint(
            version=checkpoint_version, receipt_id=receipt_id,
            receipt_digest=receipt.payload_digest(),
            receipt_pickle=pickle.dumps(receipt),
            token_digest=tdig, trigger_mode=trigger_mode,
            result_digest=result_dig,
            tokens=list(receipt.selected_tokens),
            token_meta=[[p.package_token, p.edge_id, int(p.base_version)]
                        for p in selected_packages])
        if status != 'created':
            # 幂等/替换: 以 ledger 中记录为准返回 (摘要与记录一致)
            receipt = pickle.loads(stored)
        logger.info(f"TA {self.id}: 签发 {receipt.receipt_id} "
                    f"({trigger_mode}, {n} 包, 全部 vote 验签通过; "
                    f"prepare={status}, 待 finalize 落地)")
        return receipt

    def p2_note_committed_checkpoint(self, selected_packages: list,
                                     committed_version: int = None) -> None:
        """commit 通知: 仅成功提交的 checkpoint 计入 TA 侧参与历史与版本.

        幂等: 通知版本 ≤ 已记录版本 (崩溃恢复重发通知) 直接跳过, 不重复
        计入参与次数 (参与历史是 V-PDA 权重复算的本地输入)。
        """
        if (committed_version is not None
                and int(committed_version) <= self._p2_committed_version):
            logger.info(f"TA {self.id}: commit 通知 v{committed_version} ≤ "
                        f"已落地 v{self._p2_committed_version}, 幂等跳过")
            return
        for pkg in selected_packages:
            self._p2_participation_counts[pkg.edge_id] = (
                self._p2_participation_counts.get(pkg.edge_id, 0) + 1)
        if committed_version is not None:
            self._p2_committed_version = int(committed_version)

    def _p2_independent_weights(self, selected_packages: list):
        """按预置配置与本地参与历史独立复算掩码域权重.

        返回 None 表示本配置为等权 (none) 或不可复算 (rel2/wait 需要共识
        模型历史, TA 不持有) —— 后者退回云端声明值并记录审计告警, 由 ES
        侧复算兜底。Cloud 声明值只作审计, 不作为可信输入。
        """
        cfg = self._p2_policy_config
        if cfg is None or not self.enable_verification:
            return None
        src = cfg.expected_weight_source
        if src == 'none':
            return None
        if src == 'vpda':
            from utils import vpda_weights
            counts = [self._p2_participation_counts.get(p.edge_id, 0)
                      for p in selected_packages]
            all_counts = [self._p2_participation_counts.get(e, 0)
                          for e in self.es_public_keys]
            _, ws = vpda_weights(counts, kappa=cfg.debias_kappa,
                                 wmin=cfg.debias_min, wmax=cfg.debias_max,
                                 scale=cfg.debias_scale, all_counts=all_counts)
            return ws
        if src == 'staleness':
            if cfg.staleness_metric != 'version':
                logger.warning(
                    f"TA {self.id}: staleness 度量 {cfg.staleness_metric} "
                    f"需共识模型历史, TA 无法独立复算 → 按声明值构建 "
                    f"(ES 侧复算兜底)")
                return None
            from utils import staleness_alpha, staleness_int_weights
            v = self._p2_committed_version
            alphas = [staleness_alpha('version', v - p.base_version,
                                      cfg.staleness_sigma)
                      for p in selected_packages]
            return staleness_int_weights(alphas)
        return None

    def p2_receive_package_online_list(self, package_token: str, edge_id: str,
                                        base_version: int, online_users: list[str]):
        """point2: ES 上报组包在线用户.

        对应文档 Sec.4.3: TA 按组包记录参与集合，
        不向 Cloud 单独释放单个组包的 M_k。
        """
        self._p2_package_users[package_token] = list(online_users)
        logger.debug(f"TA {self.id}: p2收到组包 {package_token} "
                     f"edge={edge_id} base_v={base_version} n={len(online_users)}")

    def p2_build_checkpoint_mask(self, selected_packages: list, alphas=None,
                                 weights=None, audit_only: bool = False) -> tuple:
        """point2: 仅对 checkpoint 选中的组包集合返回聚合掩码.

        对应文档 Sec.4.3 + Sec.4.5:
        - 只计算 M_Q(v) = Σ M_k(r_k) for Π_k ∈ Q(v)
        - 不提前释放任何单组 M_k
        - M_Q 始终等权 (掩码域整数加法): 浮点陈旧度权重不进入掩码域,
          否则破坏 LHH 同态验证 (EC 标量乘仅支持整数)

        Args:
            selected_packages: list of GroupPackage
            alphas: 保留参数但忽略 (M_Q 恒等权)
            weights: sigma-target=agg 时的整数化权重 w_k (σ 权重 × 2^16 取整),
                     给出时 M_Q = Σ w_k·M_k (整数标量乘, LHH 同态性保持:
                     H(Σ w_k·W̄_k) = Σ w_k·H_k 对整数 w_k 仍可验证)
            audit_only: True 时跳过独立复算覆盖, 按声明值构建 —— 仅用于
                        影子/诊断路径, 协议主路径不得开启

        Returns:
            (M_Q, H_mask_raw): 聚合掩码和及其 LHH 哈希
        """
        M_Q = np.zeros(self.gradient_dim, dtype=np.int64)
        # 独立复算优先: Cloud 声明权重只作审计, 与 TA 复算不符时以复算值
        # 为准并记录告警 (TA 可信, 不会把声明错误带入掩码)
        declared = [int(w) for w in weights] if weights is not None else None
        independent = (None if audit_only
                       else self._p2_independent_weights(selected_packages))
        if independent is not None:
            if declared is not None and independent != declared:
                logger.warning(
                    f"TA {self.id}: 云端声明权重与独立复算不符 "
                    f"(declared={declared}, independent={independent}) "
                    f"→ 以复算值为准 (审计事件)")
            weights = independent
        for j, pkg in enumerate(selected_packages):
            users = self._p2_package_users.get(pkg.package_token, [])
            if users:
                M_k = self.compute_global_mask(users, pkg.package_token)
                if weights is not None:
                    M_Q += int(weights[j]) * M_k
                else:
                    M_Q += M_k

        # 性能模式 (--verify 关) 跳过 LHH: H_mask_raw 仅被 ES 验证消费,
        # 攻击实验必须开 --verify, 此时与完整验证行为一致
        H_mask_raw = (LHH.Hash(self.lhh_pp, M_Q)
                      if self.enable_verification else None)

        logger.info(f"TA {self.id}: p2 checkpoint mask built, "
                    f"packages={len(selected_packages)} n_users_tracked={len(self._p2_package_users)}")

        return M_Q, H_mask_raw
