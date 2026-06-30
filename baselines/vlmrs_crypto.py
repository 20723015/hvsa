"""VLMRS user/server encryption workload (Algorithm 1, verification excluded)."""

from __future__ import annotations

import os

import numpy as np

from baselines.crypto_common import (
    AE_CIPHERTEXT_OVERHEAD,
    KA_PUBKEY_BYTES,
    OverheadMetrics,
    SS_SHARE_BYTES,
    comm_server_vlmrs_mb,
    comm_user_payload_kb,
    server_aggregate_sum,
    user_p2p_cost,
)
from baselines.flamingo_graph import (
    DEFAULT_V,
    choose_set,
    epsilon_for_target_degree,
    find_neighbors,
    mean_neighbor_count,
)
from baselines.shamir import share
from utils import AE, KA, PRG

AE_PAYLOAD_BYTES = 36  # shared secret (32) + peer id (4)


def _online_set(n_pool: int, n_online: int, round_t: int) -> list[int]:
    n_t = min(n_online, n_pool)
    return choose_set(DEFAULT_V, round_t, n_t, n_pool)


def _graph_neighbors(n_pool: int, n_online: int, round_t: int, client_id: int) -> tuple[list[int], list[int], float]:
    m_t = _online_set(n_pool, n_online, round_t)
    if client_id not in m_t and m_t:
        client_id = m_t[0]
    eps = epsilon_for_target_degree(len(m_t))
    neighbors = find_neighbors(DEFAULT_V, m_t, client_id, eps) if m_t else []
    return m_t, neighbors, eps


def _vlmrs_p2p(num_peers: int) -> tuple[float, int, int]:
    share_bytes = 2 * (SS_SHARE_BYTES + AE_CIPHERTEXT_OVERHEAD)
    uplink_per_peer = KA_PUBKEY_BYTES + share_bytes + AE_PAYLOAD_BYTES + AE_CIPHERTEXT_OVERHEAD
    return user_p2p_cost(num_peers, uplink_per_peer=uplink_per_peer)


def measure_user_encryption(
    d: int,
    neighbors: list[int],
    peer_pubs: dict[int, bytes],
    user_priv: bytes,
    client_id: int = 1,
    round_t: int = 1,
) -> np.ndarray:
    """Crypto with |neighbors| ~ O(log N) peers only (graph discovery excluded from timer)."""
    user_id = f"vlmrs-user-{client_id}"

    beta_j = os.urandom(32)
    n_nb = len(neighbors)
    if n_nb > 0:
        share(beta_j, threshold=max(1, n_nb // 2), num_shares=n_nb)
        share(user_priv, threshold=max(1, n_nb // 2), num_shares=n_nb)

    ae_key, ae_nonce = AE.gen()
    for nb in neighbors:
        shared = KA.agree(user_priv, peer_pubs[nb])
        payload = shared + nb.to_bytes(4, "big")
        AE.encrypt(ae_key, ae_nonce, payload)

    combined = np.zeros(d, dtype=np.int64)
    combined += PRG.mask_int(beta_j, user_id, round_t, d)

    for nb in neighbors:
        s_jm = KA.agree(user_priv, peer_pubs[nb])
        sign = 1 if nb < client_id else -1
        combined += sign * PRG.mask_int(s_jm, user_id, (round_t, nb), d)

    e_j = os.urandom(32)
    combined += PRG.mask_int(e_j, user_id, (round_t, "e"), d)
    return combined


def measure_server_encryption(d: int, n: int) -> np.ndarray:
    vectors = [np.random.randint(-100, 100, size=d, dtype=np.int64) for _ in range(n)]
    return server_aggregate_sum(vectors)


def comm_user_kb(d: int, n_pool: int, n_online: int | None = None) -> float:
    if n_online is None:
        n_online = n_pool
    _, neighbors, _ = _graph_neighbors(n_pool, n_online, 1, 1)
    _, p2p_up, p2p_down = _vlmrs_p2p(len(neighbors))
    return comm_user_payload_kb(d) + (p2p_up + p2p_down) / 1024.0


def comm_server_mb(d: int, n: int) -> float:
    return comm_server_vlmrs_mb(d, n)


def measure_overhead(d: int, n: int, repeats: int = 5) -> OverheadMetrics:
    from baselines.crypto_common import median_ms

    m_t, neighbors, eps = _graph_neighbors(n, n, 1, 1)
    num_peers = len(neighbors)
    user_priv, _ = KA.gen()
    peer_pubs = {cid: KA.gen()[0] for cid in neighbors}

    crypto_ms, _ = median_ms(
        repeats,
        lambda: measure_user_encryption(d, neighbors, peer_pubs, user_priv, client_id=1),
    )
    p2p_tx_ms, p2p_up, p2p_down = _vlmrs_p2p(num_peers)
    server_ms, _ = median_ms(repeats, lambda: measure_server_encryption(d, n))

    return OverheadMetrics(
        comp_user_ms=crypto_ms + p2p_tx_ms,
        comp_server_ms=server_ms,
        comm_user_kb=comm_user_kb(d, n, n),
        comm_server_mb=comm_server_mb(d, n),
        extra={
            "online": len(m_t),
            "neighbors": num_peers,
            "epsilon": eps,
            "p2p_peers": num_peers,
            "expected_neighbors": mean_neighbor_count(DEFAULT_V, n, n, 1),
            "crypto_ms": crypto_ms,
            "p2p_tx_ms": p2p_tx_ms,
            "p2p_bytes": p2p_up + p2p_down,
        },
    )
