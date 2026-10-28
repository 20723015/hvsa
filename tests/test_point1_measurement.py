"""Measurement regressions: no nested double counting and complete wire fan-out."""
import pickle

import numpy as np


def test_input_digest_binds_users_values_and_normalizes_order():
    from experiments.point1_compare.adapters.common import vector_digest
    vectors = {"u1": np.array([1, -2], dtype=np.int32), "u0": np.array([0, 4], dtype=np.int64)}
    reordered = {"u0": vectors["u0"], "u1": vectors["u1"].astype(np.int64)}
    assert vector_digest(vectors) == vector_digest(reordered)
    assert vector_digest(vectors) != vector_digest({"other": vectors["u0"], "u1": vectors["u1"]})
    assert vector_digest(vectors) != vector_digest({"u0": vectors["u0"]+1, "u1": vectors["u1"]})

from experiments.point1_compare.adapters.common import Stopwatch, WireLedger


def test_nested_receiver_time_is_not_charged_to_sender(monkeypatch):
    ticks = iter([0.0, 0.001, 0.004, 0.010])
    monkeypatch.setattr("experiments.point1_compare.adapters.common.time.perf_counter",
                        lambda: next(ticks))
    sw = Stopwatch()
    sw.time("send", "user", lambda: sw.time("receive", "edge", lambda: None,
                                            actor="edge_0"), actor="u0")
    assert abs(sum(p.elapsed_ms for p in sw.records) - 10.0) < 1e-9
    assert abs(sum(p.elapsed_ms for p in sw.records if p.role == "user") - 7.0) < 1e-9
    assert sw.current_role == ""


def test_wire_bytes_include_each_broadcast_recipient():
    payload = {"round": 1, "sum": np.arange(5, dtype=np.int64), "proof": bytes(64)}
    ledger = WireLedger(Stopwatch())
    for target in ["edge_0", "edge_1", "edge_2"]:
        received = ledger.transfer(payload, link="cloud->edge", phase="verify",
                                   kind="aggregate_and_evidence", sender="CS", receiver=target,
                                   sender_role="cloud", receiver_role="edge")
        assert np.array_equal(received["sum"], payload["sum"])
    assert len(ledger.messages) == 3
    assert sum(m.num_bytes for m in ledger.messages) == 3 * len(pickle.dumps(payload))
    assert {m.receiver for m in ledger.messages} == {"edge_0", "edge_1", "edge_2"}


def test_full_verification_off_does_not_compute_unused_ta_hash(monkeypatch):
    from experiments.point1_compare.adapters.hvsa_point1 import HVSAProtocol
    from utils import LHH
    original = LHH.Hash
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(LHH, "Hash", counted)
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    vectors = {u: np.arange(8, dtype=np.int64)-i for i, u in enumerate(sum(groups, []))}
    state = HVSAProtocol(8, groups, enable_verification=False)
    result = state.run_round(vectors, set(vectors), 0)
    assert result.completed and result.verified is None
    np.testing.assert_array_equal(result.raw_sum, sum(vectors.values()))
    assert not calls and state.ta.h_mask_all is None
