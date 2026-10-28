"""Real Shamir sharing for 32-byte seeds/keys, with threshold reconstruction.

Unlike the historical microbenchmark helper, these shares are polynomial field
evaluations and actually reconstruct the secret. This is not verifiable sharing.
"""
from __future__ import annotations
import secrets

PRIME = 2**521-1
SECRET_BYTES = 32


def _interpolate(points, target):
    total = 0
    for i, (x_i, y_i) in enumerate(points):
        numerator = denominator = 1
        for j, (x_j, _) in enumerate(points):
            if i != j:
                numerator = numerator*(target-x_j) % PRIME
                denominator = denominator*(x_i-x_j) % PRIME
        total = (total+y_i*numerator*pow(denominator, -1, PRIME)) % PRIME
    return total


def share_secret(secret: bytes, threshold: int, count: int):
    if not isinstance(secret, bytes) or len(secret) != SECRET_BYTES:
        raise ValueError("Expected a 32-byte seed/key")
    if not 1 <= threshold <= count < PRIME:
        raise ValueError("Expected 1 <= threshold <= count < field order")
    coefficients = [int.from_bytes(secret, "big")] + [secrets.randbelow(PRIME) for _ in range(threshold-1)]
    shares = []
    for x in range(1, count+1):
        y = 0
        for coefficient in reversed(coefficients):
            y = (y*x+coefficient) % PRIME
        shares.append((x, y))
    return shares


def reconstruct_secret(shares, threshold: int) -> bytes:
    shares = list(shares)
    if threshold < 1 or len(shares) < threshold:
        raise ValueError("Insufficient shares for the threshold")
    if any(not isinstance(v, int) or isinstance(v, bool) for point in shares for v in point):
        raise ValueError("Share coordinates must be integers")
    if any(not 0 < x < PRIME or not 0 <= y < PRIME for x, y in shares):
        raise ValueError("Share outside the field")
    if len({x for x, _ in shares}) != len(shares):
        raise ValueError("Duplicate share coordinates")
    points = sorted(shares)
    basis = points[:threshold]
    for x, y in points[threshold:]:
        if _interpolate(basis, x) != y:
            raise ValueError("Inconsistent additional shares")
    recovered = _interpolate(basis, 0)
    if recovered >= 2**(8*SECRET_BYTES):
        raise ValueError("Reconstructed value is not a 32-byte key")
    return recovered.to_bytes(SECRET_BYTES, "big")
