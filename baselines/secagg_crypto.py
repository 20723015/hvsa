"""Bonawitz SecAgg user/server encryption workload (masking only, no verification)."""

from __future__ import annotations

import os

import numpy as np

from baselines.crypto_common import (
    AE_CIPHERTEXT_OVERHEAD,
    FLOAT32_ELEM,
    KA_PUBKEY_BYTES,
    OverheadMetrics,
    SS_SHARE_BYTES,
    comm_server_verifl_mb,
    comm_user_payload_kb,
    server_aggregate_sum,
    user_p2p_cost,
)
from baselines.shamir import share
from utils import KA, PRG


def _secagg_p2p(num_peers: int) -> tuple[float, int, int]:
    uplink_per_peer = KA_PUBKEY_BYTES + SS_SHARE_BYTES + AE_CIPHERTEXT_OVERHEAD
    return user_p2p_cost(num_peers, uplink_per_peer=uplink_per_peer)


def measure_user_encryption(d: int, n: int, round_t: int = 1) -> tuple[np.ndarray, dict]:
    """
    User i: (N-1) pairwise PRG masks + individual mask PRG(e_i) + Shamir shares to neighbors.

    Bonawitz-style: Vec_i = x_i + sum_{j<i} PRG(s_ij) - sum_{j>i} PRG(s_ij) + PRG(e_i)
    """
    user_id = "secagg-user-0"
    combined = np.zeros(d, dtype=np.int64)
    neighbor_pub_keys = [KA.gen()[0] for _ in range(max(n - 1, 0))]
    user_priv, _ = KA.gen()

    for j in range(n - 1):
        shared = KA.agree(user_priv, neighbor_pub_keys[j])
        sign = 1 if j % 2 == 0 else -1
        combined += sign * PRG.mask_int(shared, user_id, (round_t, j), d)

    e_seed = os.urandom(32)
    combined += PRG.mask_int(e_seed, user_id, (round_t, "e"), d)
    shares = share(e_seed, threshold=max(1, (n - 1) // 2), num_shares=max(n - 1, 0))

    meta = {"n_neighbors": n - 1, "n_shares": len(shares)}
    return combined, meta


def measure_server_encryption(d: int, n: int) -> np.ndarray:
    """CS sums N masked vectors."""
    vectors = [np.random.randint(-100, 100, size=d, dtype=np.int64) for _ in range(n)]
    return server_aggregate_sum(vectors)


def comm_user_kb(d: int, n: int) -> float:
    num_peers = max(n - 1, 0)
    _, p2p_up, p2p_down = _secagg_p2p(num_peers)
    return comm_user_payload_kb(d) + (p2p_up + p2p_down) / 1024.0


def comm_server_mb(d: int, n: int) -> float:
    return comm_server_verifl_mb(d, n)


def measure_overhead(d: int, n: int, repeats: int = 5) -> OverheadMetrics:
    from baselines.crypto_common import median_ms

    num_peers = max(n - 1, 0)
    p2p_tx_ms, p2p_up, p2p_down = _secagg_p2p(num_peers)
    crypto_ms, result = median_ms(repeats, lambda: measure_user_encryption(d, n))
    meta = result[1] if result else {}
    server_ms, _ = median_ms(repeats, lambda: measure_server_encryption(d, n))
    return OverheadMetrics(
        comp_user_ms=crypto_ms + p2p_tx_ms,
        comp_server_ms=server_ms,
        comm_user_kb=comm_user_kb(d, n),
        comm_server_mb=comm_server_mb(d, n),
        extra={
            **meta,
            "crypto_ms": crypto_ms,
            "p2p_tx_ms": p2p_tx_ms,
            "p2p_bytes": p2p_up + p2p_down,
            "p2p_peers": num_peers,
        },
    )
