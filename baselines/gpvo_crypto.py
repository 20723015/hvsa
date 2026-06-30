"""GPVO-FL user/server encryption workload (Algorithm 2 masking, verification excluded)."""

from __future__ import annotations

import os

import numpy as np

from baselines.crypto_common import (
    AE_CIPHERTEXT_OVERHEAD,
    INT64_ELEM,
    KA_PUBKEY_BYTES,
    M_EDGE_HVSA,
    OverheadMetrics,
    SS_SHARE_BYTES,
    server_aggregate_sum,
    user_p2p_cost,
)
from baselines.shamir import share
from utils import KA, PRG

NUM_PAIRWISE_DH = 3
NUM_RING_NEIGHBORS = 2


def _gpvo_p2p() -> tuple[float, int, int]:
    uplink_per_peer = KA_PUBKEY_BYTES + SS_SHARE_BYTES + AE_CIPHERTEXT_OVERHEAD
    return user_p2p_cost(NUM_RING_NEIGHBORS, uplink_per_peer=uplink_per_peer)


def measure_user_encryption(
    d: int,
    user_priv: bytes,
    peer_pubs: list[bytes],
    round_t: int = 1,
) -> np.ndarray:
    """GPVO-FL user (Alg. 2): 3× DH + 3× PRG + SS to two ring neighbors."""
    user_id = "gpvo-user-0"
    gradient = np.random.randint(-100, 100, size=d, dtype=np.int64)

    shared_left = KA.agree(user_priv, peer_pubs[0])
    shared_right = KA.agree(user_priv, peer_pubs[1])
    shared_ta = KA.agree(user_priv, peer_pubs[2])

    mask = np.zeros(d, dtype=np.int64)
    mask += PRG.mask_int(shared_ta, user_id, (round_t, "ta"), d)
    mask -= PRG.mask_int(shared_left, user_id, (round_t, "left"), d)
    mask += PRG.mask_int(shared_right, user_id, (round_t, "right"), d)

    local_seed = os.urandom(32)
    share(local_seed, threshold=NUM_RING_NEIGHBORS, num_shares=NUM_RING_NEIGHBORS)

    return gradient + mask


def measure_server_encryption(d: int, m_edge: int = M_EDGE_HVSA) -> np.ndarray:
    vectors = [np.random.randint(-100, 100, size=d, dtype=np.int64) for _ in range(m_edge)]
    return server_aggregate_sum(vectors)


def comm_user_kb(d: int, n: int) -> float:
    del n
    _, p2p_up, p2p_down = _gpvo_p2p()
    return (d * INT64_ELEM + p2p_up + p2p_down) / 1024.0


def comm_server_mb(d: int, m_edge: int = M_EDGE_HVSA) -> float:
    return (m_edge * d * INT64_ELEM) / (1024.0 ** 2)


def measure_overhead(d: int, n: int, repeats: int = 5) -> OverheadMetrics:
    from baselines.crypto_common import median_ms

    user_priv, _ = KA.gen()
    peer_pubs = [KA.gen()[0] for _ in range(NUM_PAIRWISE_DH)]
    p2p_tx_ms, p2p_up, p2p_down = _gpvo_p2p()

    crypto_ms, _ = median_ms(
        repeats,
        lambda: measure_user_encryption(d, user_priv, peer_pubs),
    )
    server_ms, _ = median_ms(repeats, lambda: measure_server_encryption(d))

    return OverheadMetrics(
        comp_user_ms=crypto_ms + p2p_tx_ms,
        comp_server_ms=server_ms,
        comm_user_kb=comm_user_kb(d, n),
        comm_server_mb=comm_server_mb(d),
        extra={
            "pairwise_dh": NUM_PAIRWISE_DH,
            "ring_neighbors": NUM_RING_NEIGHBORS,
            "m_edge": M_EDGE_HVSA,
            "crypto_ms": crypto_ms,
            "p2p_tx_ms": p2p_tx_ms,
            "p2p_bytes": p2p_up + p2p_down,
            "p2p_peers": NUM_RING_NEIGHBORS,
        },
    )
