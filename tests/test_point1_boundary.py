"""Reject falsified availability conclusions even after output hashes are rebuilt."""
import numpy as np
import pytest

from experiments.point1_compare.adapters.common import reference_sum, vector_digest
from experiments.point1_compare.artifacts import create_run, digest, finalize_run
from experiments.point1_compare.validate_results import validate


@pytest.fixture
def boundary_evidence(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# artifact-contract fixture; no cryptography is claimed\n", encoding="utf-8")
    groups = [["u0", "u1"], ["u2", "u3"]]
    cases = [
        {"case": "empty_group", "uploaded": ["u2", "u3"], "recipients": ["u2", "u3"],
         "expected_completed": False, "expected_verified": None},
        {"case": "post_upload", "uploaded": ["u0", "u1", "u2", "u3"], "recipients": ["u2", "u3"],
         "expected_completed": True, "expected_verified": True},
        {"case": "no_receivers", "uploaded": ["u0", "u1", "u2", "u3"], "recipients": [],
         "expected_completed": True, "expected_verified": True},
    ]
    manifest = {"run_id": "boundary-fixture", "experiment": "e4_boundary", "command": "fixture",
                "seed": 42, "schemes": ["hvsa"], "git_commit": "fixture", "lib_versions": {},
                "quant": {}, "protocol_mapping": "fixture", "paper_versions": {},
                "concurrency": "serial", "timing_scope": "fixture", "verification": "full",
                "aggregation": "sync", "N": 4, "M": 2, "d": 3, "groups": groups, "cases": cases,
                "source_sha256": {"source.py": digest(source)}}
    run = tmp_path / "run"
    create_run(run, manifest, tmp_path)
    raw, messages, observations = [], [], []
    for index, spec in enumerate(cases):
        vectors = {f"u{i}": np.array([index*100+i, -i, 2], dtype=np.int64) for i in range(4)}
        snapshot = run / (spec["case"] + ".npz")
        np.savez(snapshot, user_ids=np.array(sorted(vectors)), vectors=np.stack(list(vectors.values())))
        completed, verified = spec["expected_completed"], spec["expected_verified"]
        rows = [{"scheme": "hvsa", "run_id": "boundary-fixture", "group": spec["case"],
                 "round": index, "link": "cloud->edge", "direction": "send", "phase": "aggregate",
                 "payload_kind": "proof", "sender": "CS", "receiver": "edge0", "num_bytes": 10, "note": ""}]
        if completed:
            rows += [{**rows[0], "receiver": u, "sender": "edge0", "link": "edge->user",
                      "payload_kind": "verified_model"} for u in spec["recipients"]]
        messages.extend(rows)
        raw.append({"scheme": "hvsa", "run_id": "boundary-fixture", "seed": 42, "group": spec["case"],
                    "repeat": 0, "N_selected": 4, "N_uploaded": len(spec["uploaded"]), "M": 2, "d": 3,
                    "round": index, "phase": "aggregate", "role": "edge", "actor": "edge0", "detail": "",
                    "elapsed_ms": 1., "sent_bytes": None, "success": completed, "verified": verified,
                    "error_reason": "" if completed else "sync_barrier"})
        reference = reference_sum([vectors[u] for u in spec["uploaded"]]).tolist()
        observations.append({"case": spec["case"], "round": index, "uploaded": spec["uploaded"],
                             "recipients": spec["recipients"], "input_sha256": vector_digest(vectors),
                             "snapshot": snapshot.name, "snapshot_sha256": digest(snapshot),
                             "completed": completed, "verified": verified, "exact_match": completed,
                             "matches_expected_boundary": True, "reference_sum": reference,
                             "returned_sum": reference if completed else None,
                             "model_receivers": spec["recipients"] if completed else [],
                             "per_group_uploaded": [len(set(g) & set(spec["uploaded"])) for g in groups],
                             "role_ms": {"edge": 1.}, "sent_bytes": 10*len(rows)})
    corr = {"experiment": "e4_boundary", "observations": observations, "execution_errors": 0}
    summary = [{k: o[k] for k in ["case", "completed", "verified", "matches_expected_boundary"]}
               for o in observations]
    finalize_run(run, manifest, raw, messages, corr, summary, [])
    return run, manifest, raw, messages, corr, summary


def test_expected_barrier_is_not_a_rejection_or_execution_error(boundary_evidence):
    run, *_ = boundary_evidence
    assert validate(run) == []


@pytest.mark.parametrize("corruption", ["group_counts", "reference", "receiver", "barrier_verdict",
                                        "raw_verdict", "duplicate_snapshot_id", "error_count"])
def test_false_boundary_claims_rejected_after_rehash(boundary_evidence, corruption):
    run, manifest, raw, messages, corr, summary = boundary_evidence
    first, second, _ = corr["observations"]
    if corruption == "group_counts":
        first["per_group_uploaded"] = [1, 1]
    elif corruption == "reference":
        second["reference_sum"][0] += 1
    elif corruption == "receiver":
        messages[:] = [m for m in messages if not (m["group"] == "post_upload"
                                                   and m["receiver"] == "u2")]
        second["sent_bytes"] -= 10
    elif corruption == "barrier_verdict":
        first["verified"] = False
        summary[0]["verified"] = False
    elif corruption == "raw_verdict":
        raw[0]["success"] = True
    elif corruption == "duplicate_snapshot_id":
        path = run / first["snapshot"]
        with np.load(path, allow_pickle=False) as stored:
            values = stored["vectors"].copy()
        np.savez(path, user_ids=np.array(["u0", "u0", "u2", "u3"]), vectors=values)
        first["snapshot_sha256"] = digest(path)
    else:
        corr["execution_errors"] = 1
    finalize_run(run, manifest, raw, messages, corr, summary, [])
    assert validate(run), corruption
