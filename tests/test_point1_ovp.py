"""Actual field cancellation and full two-service coordinate verification."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tmp"/"point1_pairing_deps"))
pytest.importorskip("py_ecc")
from experiments.point1_compare.adapters.ovp_full import (OVPProtocol, Q, additive_shares,
    field_sum, prng_vector, gt_read, gt_bytes)
from py_ecc.optimized_bls12_381 import G1, G2, pairing, multiply


def test_additive_field_cancellation_not_int64_or_scalar_projection():
    inputs = [[-8, 0, 2, 10**10], [9, -7, 0, -10**10], [2, 1, -5, 13]]
    streams = [[prng_vector(bytes([n, v]), 4) for n in range(3)] for v in range(2)]
    parts = [[], []]
    for n, vector in enumerate(inputs):
        alpha = prng_vector(bytes([n, 5]), 4)
        combined = field_sum([streams[0][n], streams[1][n]], 4)
        parts[0].append([(x-a+m) % Q for x, a, m in zip(vector, alpha, combined)])
        parts[1].append([(a+m) % Q for a, m in zip(alpha, combined)])
    # Complete-set selection excludes the missing user's part and both masks.
    selected = [0, 2]
    corrected = []
    for v in range(2):
        total = field_sum([streams[v][n] for n in selected], 4)
        shares = additive_shares(total, len(selected))
        assert field_sum(shares, 4) == total
        corrected.extend([[(x-2*s) % Q for x, s in zip(parts[v][n], share)]
                          for n, share in zip(selected, shares)])
    expected = [sum(inputs[n][j] for n in selected) % Q for j in range(4)]
    assert field_sum(corrected, 4) == expected


def test_real_two_service_chain_partial_upload_and_coordinated_feedback():
    users = ["u0", "u1", "u2", "u3"]
    vectors = {u: np.array([i-3, 17-2*i], dtype=np.int64) for i, u in enumerate(users)}
    state = OVPProtocol(2, users)
    assert gt_read(gt_bytes(state.base)) == state.base
    # Bilinear precomputation agrees with an actual direct pairing, including
    # the negative exponent domain used by centered signed input encoding.
    assert state.base**((-17*23) % Q) == pairing(multiply(G2, 23), multiply(G1, -17 % Q))
    normal = state.run_round(vectors, [set(users), set(users)], 0)
    assert normal.verified and np.array_equal(normal.raw_sum, sum(vectors.values()))
    partial = state.run_round(vectors, [set(users), set(users[:-1])], 1)
    assert partial.verified and partial.n_uploaded == 3
    assert np.array_equal(partial.raw_sum, sum(vectors[u] for u in users[:-1]))
    only = state.run_round(vectors, [set(users), set(users)], 2, "result_only")
    coordinated = state.run_round(vectors, [set(users), set(users)], 3, "coordinated")
    assert not only.verified and coordinated.verified
    expected = sum(vectors.values()) + np.array([1, 0], dtype=np.int64)
    assert np.array_equal(only.raw_sum, expected) and np.array_equal(coordinated.raw_sum, expected)
    assert coordinated.extra["own_plus_claimed_other_equals_claimed_total"] == [True, True]
    assert set(coordinated.extra["accepted_coordinates"].values()) == {2}
    assert coordinated.extra["attacker_uses_service_betas"] is False
    too_few = state.run_round(vectors, [{"u0", "u1"}, {"u0"}], 4)
    assert not too_few.completed and too_few.raw_sum is None and too_few.verified is None
