"""
研究点2 类型定义: 组包 + Checkpoint 结果

这些 dataclass 专用于 buffer/checkpoint 半异步协议，
不破坏研究点1的同步 round-based 类型。
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class GroupPackage:
    """组级掩码包 (Π_k).

    对应文档 Sec.4.2: ES 封装的组级掩码聚合包，
    存入 Cloud buffer 时不包含解掩码所需材料。
    """
    edge_id: str
    package_token: str                     # 如 "edge_1:v0:p2"
    base_version: int                      # r_k: 该包基于的全局版本
    base_hash: bytes                       # h_base(r_k): W(base_version) 的哈希
    online_users: List[str]                # U_online^k(r_k)
    n_k: int                               # |U_online^k(r_k)|
    S_k: int                               # Σ s_i^k: 组内总样本量
    masked_sum_int: np.ndarray             # W̄_k(r_k): int64 掩码和
    H_k: object                            # LHH.Hash result (NIST P-256 Point)
    signature: bytes                       # Ed25519 σ_k(r_k)
    created_tick: int = 0                  # 模拟时钟: 组包生成时刻


@dataclass
class CheckpointResult:
    """Checkpoint 聚合结果.

    对应文档 Sec.4.5-4.6: Cloud 在一次 checkpoint 中
    恢复的数据以及验证所需材料。
    """
    version: int                           # v: checkpoint 序号
    selected_packages: List[GroupPackage]  # Q(v)
    N_Q: int                               # 总参与客户端数
    S_Q: int                               # Σ S_k: 总样本量
    raw_sum_int: np.ndarray                # W_sum^raw(v): int64
    raw_sum_float: np.ndarray              # 解量化后的 raw sum (float64)
    M_Q: np.ndarray                        # TA 返回的聚合掩码 (int64)
    H_mask_raw: object                     # H_mask^raw(v)
    corrected_sum_float: np.ndarray        # W_sum^corr(v): 补偿后 (float64)
    final_model: np.ndarray               # W(v+1): 新全局模型 (float64)
    compensation_per_package: dict = field(default_factory=dict)
    verification_passed: bool = False
