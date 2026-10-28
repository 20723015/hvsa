"""Minimal Shamir secret-sharing for microbenchmark workloads."""

from __future__ import annotations

import hashlib
import os
import secrets


def share(secret: bytes | None = None, threshold: int = 1, num_shares: int = 1) -> list[bytes]:
    """
    S.share(s, t, L) -> L shares (simplified: independent keyed hashes).

    Cost scales with num_shares, matching the dominant loop in SecAgg/VLMRS.
    """
    if num_shares <= 0:
        return []
    if secret is None:
        secret = secrets.token_bytes(32)
    shares: list[bytes] = []
    for idx in range(1, num_shares + 1):
        shares.append(hashlib.sha256(secret + idx.to_bytes(4, "big")).digest())
    return shares
