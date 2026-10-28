"""Real pairing tests; py-ecc is an isolated optional experiment dependency."""
import numpy as np
import pytest

pytest.importorskip("py_ecc")
from experiments.point1_compare.adapters.vlmrs_verification import VLMRSVerification, coordinated_scale
from experiments.point1_compare.adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol


def test_vlmrs_literal_verification_and_public_root_scaling():
    verifier = VLMRSVerification(4, root_exponent=7)
    vectors = [np.array([1, -2, 0, 4], dtype=np.int64), np.array([-3, 1, 7, 6], dtype=np.int64)]
    users = ["u0", "u1"]
    proof = verifier.aggregate(verifier.create(u, x) for u, x in zip(users, vectors))
    actual = sum(vectors)
    assert verifier.verify(actual, proof, users)[0]
    wrong, forged = coordinated_scale(actual, proof, 2, 7)
    assert not verifier.verify(wrong, proof, users)[0]
    accepted, checks = verifier.verify(wrong, forged, users)
    assert accepted and all(checks.values())
    with pytest.raises(ValueError):
        VLMRSVerification(4, root_exponent=-1)


def test_vlmrs_fullgraph_recovery_and_explicit_reuse_diagnostic():
    vectors = {f"u{i}": np.array([i-3, i*7, -i], dtype=np.int64) for i in range(4)}
    state = VLMRSFullGraphProtocol(3, vectors, threshold=3)
    honest = state.run_round(vectors, set(vectors), 0)
    assert honest.completed and honest.verified
    np.testing.assert_array_equal(honest.raw_sum, sum(vectors.values()))
    online = set(vectors)-{"u0"}
    dropout = state.run_round(vectors, online, 1)
    assert dropout.completed and dropout.verified
    np.testing.assert_array_equal(dropout.raw_sum, sum(vectors[u] for u in sorted(online)))
    # No inputs or verifier keys are supplied to this cloud-only recovery.
    np.testing.assert_array_equal(state.recover_past_upload_with_later_key("u0", 0), vectors["u0"])
    failed = state.run_round(vectors, set(vectors), 2, recovery_clients={"u1", "u2"})
    assert not failed.completed and failed.raw_sum is None and failed.verified is None
    forged = state.run_round(vectors, set(vectors), 3, scale_attack=2)
    assert forged.completed and forged.verified
    np.testing.assert_array_equal(forged.raw_sum, 2*sum(vectors.values()))
