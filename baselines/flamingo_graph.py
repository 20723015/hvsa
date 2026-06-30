"""Flamingo Figure 1: ChooseSet / FindNeighbors (Ma et al., IEEE S&P 2023)."""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from typing import Iterable

# Public randomness v (128-bit seed stand-in for lambda-bit string).
DEFAULT_V = hashlib.sha256(b"HVSA-overhead-flamingo-v1").digest()


def _prf(v: bytes, msg: bytes) -> bytes:
    return hmac.new(v, msg, hashlib.sha256).digest()


def _prg_stream(seed: bytes) -> Iterable[int]:
    counter = 0
    while True:
        block = _prf(seed, struct.pack(">Q", counter))
        counter += 1
        for i in range(0, len(block), 4):
            if i + 4 <= len(block):
                yield int.from_bytes(block[i : i + 4], "big")


def choose_set(v: bytes, t: int, n_t: int, n_pool: int) -> list[int]:
    """ChooseSet(v, t, n_t, N) -> S_t of size n_t, ids in [1, n_pool]."""
    if n_t <= 0:
        return []
    if n_t > n_pool:
        return list(range(1, n_pool + 1))

    bits = max(1, int(math.ceil(math.log2(n_pool))))
    mask = (1 << bits) - 1
    v_star = _prf(v, b"choose-set|" + struct.pack(">Q", t))
    stream = _prg_stream(v_star)

    chosen: set[int] = set()
    while len(chosen) < n_t:
        val = next(stream) & mask
        client_id = (val % n_pool) + 1
        chosen.add(client_id)
    return sorted(chosen)


def _has_edge(v: bytes, i: int, j: int, rho_bits: int) -> bool:
    """Edge (i,j) if the first rho_bits of PRF(v,i,j) are zero (Flamingo bit test)."""
    digest = _prf(v, struct.pack(">QQ", i, j))
    total_bits = len(digest) * 8
    rho_bits = min(max(rho_bits, 1), total_bits)
    val = int.from_bytes(digest, "big")
    return val >> (total_bits - rho_bits) == 0


def find_neighbors(v: bytes, s_t: list[int], client_id: int, epsilon: float) -> list[int]:
    """FindNeighbors(v, S_t, i) — local neighbor discovery without full graph."""
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    if client_id not in s_t:
        return []

    rho_bits = max(1, int(math.ceil(math.log2(1.0 / epsilon))))
    neighbors: set[int] = set()

    for other in s_t:
        if other == client_id:
            continue
        if _has_edge(v, client_id, other, rho_bits) or _has_edge(v, other, client_id, rho_bits):
            neighbors.add(other)
    return sorted(neighbors)


def epsilon_for_target_degree(n_t: int, target_degree: float | None = None) -> float:
    """Pick epsilon so expected degree ~ 4*log2(n_t) (Flamingo Sec. 8.2, bit-level edges)."""
    if n_t <= 1:
        return 0.5
    if target_degree is None:
        target_degree = 4.0 * math.log2(max(n_t, 2))
    ratio = max((n_t - 1) / target_degree, 2.0)
    rho_bits = max(1, int(math.ceil(math.log2(ratio))))
    return min(max(2.0 ** (-rho_bits), 1e-6), 0.5)


def mean_neighbor_count(v: bytes, n_pool: int, n_t: int, round_t: int = 1) -> float:
    """Average |FindNeighbors| over S_t (for workload sizing)."""
    s_t = choose_set(v, round_t, n_t, n_pool)
    eps = epsilon_for_target_degree(len(s_t))
    if not s_t:
        return 0.0
    total = sum(len(find_neighbors(v, s_t, cid, eps)) for cid in s_t)
    return total / len(s_t)
