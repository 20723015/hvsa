"""Shared constants and timing helpers for encryption-overhead benchmarks."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

# Payload sizes (bytes)
INT64_ELEM = 8
FLOAT32_ELEM = 4
SS_SHARE_BYTES = 64
AE_CIPHERTEXT_OVERHEAD = 32  # AES-GCM tag + nonce overhead estimate

M_EDGE_HVSA = 5  # shared M for hierarchical GPVO-FL / HVSA cloud-side benchmarks

# Analytical CS ingress (aligned with exp8.py / Table complexity; masking scope only)
VLMRS_PROOF_BYTES = 512  # per-user verifiable tag (constant w.r.t. d)
CS_FIXED_OVERHEAD_VERIFL_BYTES = 100_000
CS_FIXED_OVERHEAD_VLMRS_BYTES = 50_000
MODEL_PAYLOAD_FACTOR_VERIFL = 1.08
MODEL_PAYLOAD_FACTOR_VLMRS = 1.04

# Analytical link model for user–user P2P (same for all schemes with inter-user interaction).
KA_PUBKEY_BYTES = 32
LINK_BANDWIDTH_MBPS = 10.0  # typical mobile uplink
LINK_RTT_MS = 20.0  # one RTT per distinct peer interaction


def transmission_ms(num_bytes: int, num_rtt: int = 0) -> float:
    """Wire time (ms) + RTT latency for P2P neighbor rounds."""
    wire_ms = (num_bytes * 8) / (LINK_BANDWIDTH_MBPS * 1e6) * 1000.0
    return wire_ms + num_rtt * LINK_RTT_MS


def user_p2p_cost(
    num_peers: int,
    *,
    uplink_per_peer: int,
    downlink_per_peer: int | None = None,
) -> tuple[float, int, int]:
    """
    Analytical user↔peer transmission cost.

    Returns (tx_ms, uplink_bytes, downlink_bytes).
    One RTT is charged per peer (sequential interaction model).
    """
    if num_peers <= 0:
        return 0.0, 0, 0
    if downlink_per_peer is None:
        downlink_per_peer = KA_PUBKEY_BYTES
    uplink = num_peers * uplink_per_peer
    downlink = num_peers * downlink_per_peer
    tx_ms = transmission_ms(uplink + downlink, num_rtt=num_peers)
    return tx_ms, uplink, downlink

T = TypeVar("T")


@dataclass(frozen=True)
class OverheadMetrics:
    comp_user_ms: float
    comp_server_ms: float
    comm_user_kb: float
    comm_server_mb: float
    extra: dict | None = None


def median_ms(repeats: int, fn: Callable[[], T]) -> tuple[float, T | None]:
    """Run fn() `repeats` times; return median wall time (ms) and last result."""
    times: list[float] = []
    last: T | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), last


def vlmrs_expected_degree(n: int) -> float:
    """Flamingo-style expected neighbor count K ~ 3 log2(N) (exp8)."""
    return 3.0 * math.log2(n) if n > 1 else 1.0


def comm_server_verifl_mb(d: int, n: int) -> float:
    """
    VeriFL / Bonawitz CS ingress: N masked models + O(N^2) share-recovery traffic.

    Dropout reconstruction is not timed separately; share term models server-side recovery load.
    """
    n = max(n, 0)
    model_bytes = n * d * FLOAT32_ELEM * MODEL_PAYLOAD_FACTOR_VERIFL
    share_bytes = n * max(n - 1, 0) * SS_SHARE_BYTES
    total = model_bytes + share_bytes + CS_FIXED_OVERHEAD_VERIFL_BYTES
    return total / (1024.0**2)


def comm_server_vlmrs_mb(d: int, n: int) -> float:
    """
    VLMRS CS ingress: N masked models + O(N log N) neighbor share artifacts + N proofs.

    Graph reduces user P2P only; CS still receives one update per online user (O(Nd)).
    """
    n = max(n, 0)
    k = vlmrs_expected_degree(n)
    model_bytes = n * d * FLOAT32_ELEM * MODEL_PAYLOAD_FACTOR_VLMRS
    share_bytes = n * k * SS_SHARE_BYTES
    proof_bytes = n * VLMRS_PROOF_BYTES
    total = model_bytes + share_bytes + proof_bytes + CS_FIXED_OVERHEAD_VLMRS_BYTES
    return total / (1024.0**2)


def model_payload_bytes(d: int, n_vectors: int = 1, *, elem: int = FLOAT32_ELEM) -> int:
    """Wire payload for ``n_vectors`` model updates of dimension ``d`` (default float32)."""
    return n_vectors * d * elem


def comm_user_payload_kb(d: int, *, elem: int = FLOAT32_ELEM) -> float:
    """One user upload: ``d`` model parameters."""
    return model_payload_bytes(d, 1, elem=elem) / 1024.0


def comm_server_hvsa_mb(
    d: int,
    m_edge: int = M_EDGE_HVSA,
    *,
    include_ta_mask: bool = False,
    elem: int = FLOAT32_ELEM,
) -> float:
    """HVSA CS ingress: ``M`` edge vectors (+ optional one TA mask vector)."""
    n_vectors = m_edge + (1 if include_ta_mask else 0)
    return model_payload_bytes(d, n_vectors, elem=elem) / (1024.0**2)


def server_aggregate_sum(vectors: list[np.ndarray]) -> np.ndarray:
    """CS aggregates masked gradient vectors (FedAvg sum)."""
    if not vectors:
        return np.empty(0, dtype=np.int64)
    stacked = np.stack(vectors, axis=0)
    return np.sum(stacked, axis=0, dtype=np.int64)
