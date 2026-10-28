from dataclasses import replace

import numpy as np
import pytest

from cache_calibration import CacheCalibrationOracle, _proposal_id


def oracle():
    return CacheCalibrationOracle([10.0, 20.0], {"a": 2, "b": 3, "c": 5})


def test_receive_reject_and_expire_do_not_advance_caches():
    state = oracle()
    state.stage("one", 0, {"a": [1, -2]})
    state.prepare(["one"], "calibration")  # rejected candidate remains retryable
    assert state.version == 0 and state.cache_sequences["a"] == 0
    np.testing.assert_array_equal(state.reference, [0, 0])
    assert state.discard("one")
    assert not state.discard("one")
    state.stage("two", 0, {"a": [3, 4]})
    assert state.cache_sequences["a"] == 0


def test_weighted_partial_updates_preserve_invariant_and_replay_is_idempotent():
    state = oracle()
    state.stage("one", 0, {"a": [1, -2], "c": [3, 4]})
    proposal = state.prepare(["one"], "calibration")
    np.testing.assert_allclose(proposal.model, [10 + 17/7, 20 + 16/7])
    np.testing.assert_allclose(proposal.reference, [1.7, 1.6])
    receipt = state.commit(proposal)
    assert state.commit(proposal) is receipt
    assert state.version == 1 and state.cache_sequences["a"] == 1
    state.stage("two", 0, {"b": [2, -1]})  # concurrent old model base is valid
    incremental = state.prepare(["two"], "increment")
    calibrated = state.prepare(["two"], "calibration")
    np.testing.assert_allclose(calibrated.model - incremental.model, [1.7, 1.6])
    state.commit(state.prepare(["two"], "calibration"))
    np.testing.assert_allclose(state.reference, [2.3, 1.3])
    assert state.commit(proposal) is receipt and state.version == 2


def test_full_participation_cancels_calibration_after_nonzero_history():
    state = oracle()
    state.stage("warm", 0, {"a": [1, -2], "c": [3, 4]})
    state.commit(state.prepare(["warm"], "calibration"))
    state.stage("all", 1, {"a": [.8, 2], "b": [-.5, -1], "c": [.6, 3]})
    incremental = state.prepare(["all"], "increment")
    calibrated = state.prepare(["all"], "calibration")
    np.testing.assert_allclose(incremental.model, calibrated.model, atol=1e-14)


def test_joint_state_tamper_rejected_without_partial_advance():
    state = oracle()
    state.stage("one", 0, {"a": [1, -2]})
    good = state.prepare(["one"], "calibration")
    bad = replace(good, reference=good.reference * 5)
    with pytest.raises(ValueError, match="content changed"):
        state.commit(bad)
    assert state.version == 0 and "one" in state.pending
    np.testing.assert_array_equal(state.caches["a"], [0, 0])


def test_wrong_reference_with_self_consistent_digest_still_fails_reconstruction():
    state = oracle()
    state.stage("one", 0, {"a": [1, -2]})
    good = state.prepare(["one"], "calibration")
    # Aggregate and next model remain correct; malicious proposer recomputes
    # its digest after using selected mass instead of total population mass.
    wrong_reference = good.reference * 5
    forged = replace(good, reference=wrong_reference, proposal_id=_proposal_id(
        good.version, good.tokens, good.mode, good.model, wrong_reference))
    with pytest.raises(ValueError, match="joint model/reference"):
        state.commit(forged)
    assert state.version == 0 and "one" in state.pending
    np.testing.assert_array_equal(state.model, [10, 20])
    np.testing.assert_array_equal(state.reference, [0, 0])


def test_conflicting_pending_client_and_stale_candidate_rejected():
    state = oracle()
    state.stage("one", 0, {"a": [1, -2]})
    with pytest.raises(ValueError, match="outstanding"):
        state.stage("duplicate", 0, {"a": [3, 4]})
    state.stage("two", 0, {"b": [2, -1]})
    first = state.prepare(["one"], "calibration")
    stale = state.prepare(["two"], "calibration")
    state.commit(first)
    with pytest.raises(ValueError, match="uncommitted state"):
        state.commit(stale)
    assert state.version == 1 and state.cache_sequences["b"] == 0
