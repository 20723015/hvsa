"""Real decided GPVO chain: cancellation, session freshness and actual recipients."""
import numpy as np
import pytest
from experiments.point1_compare.adapters.gpvo_decided import (
    GPVODecidedProtocol, hash_vector, verify_candidate)
from experiments.point1_compare.adapters.gpvo_full import com_param_gen, lhf_hash


def test_measured_chain_fresh_keys_dropout_and_public_coordinated_change():
    # Small RSA here is a speed-limited functional test, not the production setting.
    params = com_param_gen(1024)
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    vectors = {f"u{i}": np.array([i-3, 0, 100-i, -2*i], dtype=np.int64) for i in range(6)}
    protocol = GPVODecidedProtocol(4, groups, params=params)
    assert hash_vector(params, vectors["u0"]) == lhf_hash(params, vectors["u0"])
    first = protocol.run_round(vectors, set(vectors), 0)
    assert first.completed and first.verified
    np.testing.assert_array_equal(first.raw_sum, sum(vectors.values()))
    online = set(vectors)-{"u0", "u4"}
    second = protocol.run_round(vectors, online, 1)
    assert second.completed and second.verified
    np.testing.assert_array_equal(second.raw_sum, sum(vectors[u] for u in online))
    assert protocol.session_public_fingerprints[0] != protocol.session_public_fingerprints[1]
    after = protocol.run_round(vectors, set(vectors), 2, model_recipients=online)
    np.testing.assert_array_equal(after.raw_sum, first.raw_sum)
    assert after.verified and sorted(m.receiver for m in after.messages if m.payload_kind == "verified_model") == sorted(online)
    assert all(m.receiver != "users" for m in after.messages)
    ctx = protocol.last_public_context
    delta = np.array([1, 0, 0, 0], dtype=np.int64)
    wrong = ctx["candidate"]+delta
    assert not verify_candidate(params, wrong, ctx["commitment"], ctx["mask_hash"], ctx["randomiser"])
    coordinated = ctx["commitment"]*pow(hash_vector(params, delta), params.e, params.n2) % params.n2
    assert verify_candidate(params, wrong, coordinated, ctx["mask_hash"], ctx["randomiser"])
    with pytest.raises(ValueError):
        protocol.run_round(vectors, set(vectors), 2)
