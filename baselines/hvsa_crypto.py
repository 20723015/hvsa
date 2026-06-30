"""HVSA user/server encryption workload (single-TA PRG masking via utils)."""

from __future__ import annotations

import pickle

import numpy as np

from baselines.crypto_common import (
    M_EDGE_HVSA,
    OverheadMetrics,
    comm_server_hvsa_mb,
    comm_user_payload_kb,
    server_aggregate_sum,
)
from utils import KA, PRG

NUM_TAS = 1


def _derive_ta_seeds(user_id: str = "hvsa-bench") -> list[bytes]:
    """TA DH seed from registration (amortized setup, outside per-round timer)."""
    seeds: list[bytes] = []
    user_priv, _ = KA.gen()
    for _ in range(NUM_TAS):
        ta_pub, _ = KA.gen()
        seeds.append(KA.agree(user_priv, ta_pub))
    return seeds


def measure_user_encryption(d: int, ta_seeds: list[bytes], round_t: int = 1) -> tuple[np.ndarray, float]:
    user_id = "hvsa-bench"
    gradient = np.random.randint(-100, 100, size=d, dtype=np.int64)

    total_mask = np.zeros(d, dtype=np.int64)
    for seed in ta_seeds:
        total_mask += PRG.mask_int(seed, user_id, round_t, d)
    masked = gradient + total_mask

    payload = pickle.dumps(masked)
    return masked, len(payload) / 1024.0


def measure_server_encryption(d: int, m_edge: int = M_EDGE_HVSA) -> np.ndarray:
    vectors = [np.random.randint(-100, 100, size=d, dtype=np.int64) for _ in range(m_edge)]
    return server_aggregate_sum(vectors)


def comm_user_kb(comm_kb_measured: float) -> float:
    return comm_kb_measured


def comm_server_mb(d: int, m_edge: int = M_EDGE_HVSA) -> float:
    return comm_server_hvsa_mb(d, m_edge)


def measure_overhead(d: int, n: int, repeats: int = 5) -> OverheadMetrics:
    from baselines.crypto_common import median_ms

    ta_seeds = _derive_ta_seeds()

    def run_user():
        measure_user_encryption(d, ta_seeds)
        return None

    user_ms, _ = median_ms(repeats, run_user)
    server_ms, _ = median_ms(repeats, lambda: measure_server_encryption(d))
    comm_kb = comm_user_payload_kb(d)

    return OverheadMetrics(
        comp_user_ms=user_ms,
        comp_server_ms=server_ms,
        comm_user_kb=comm_kb,
        comm_server_mb=comm_server_mb(d),
        extra={"n_independent": True, "num_tas": NUM_TAS, "neighbor_p2p": False},
    )
