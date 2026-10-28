"""Persisted evidence must reject missing records, corruption and false coverage."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.point1_compare.artifacts import create_run, dump_json, finalize_run
from experiments.point1_compare.validate_results import validate


@pytest.fixture
def valid_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    source.write_text("# reproducible fixture\n", encoding="utf-8")
    case = {"group": "small_0000", "size": "small", "trial": 0,
            "ordinal": 0, "seed": 42, "N": 10, "M": 2, "d": 32}
    manifest = {"run_id": "fixture", "experiment": "e5", "command": "fixture", "seed": 42,
                "schemes": ["hvsa"], "git_commit": "fixture", "lib_versions": {},
                "quant": {}, "protocol_mapping": "fixture", "paper_versions": {},
                "concurrency": "serial", "timing_scope": "fixture", "verification": "full",
                "aggregation": "integer", "schedule": [case],
                "tamper_types": {"hvsa": ["test_attack"]},
                "source_sha256": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    run = tmp_path / "run"
    create_run(run, manifest, repo)
    transcript = run / "public.pkl"
    transcript.write_bytes(b"fixture-not-deserialized")
    raw, messages, honest = [], [], []
    for i, context in enumerate(["initial", "next_same_set", "next_changed_set"]):
        raw.append({"scheme": "hvsa", "run_id": "fixture", "seed": 42, "group": case["group"],
                    "repeat": 0, "N_selected": 10, "N_uploaded": 10 if i != 2 else 9,
                    "M": 2, "d": 32, "round": i, "phase": "honest", "role": "edge",
                    "actor": "edge_0", "detail": "", "elapsed_ms": .5, "sent_bytes": None,
                    "success": True, "verified": True, "error_reason": ""})
        messages.append({"scheme": "hvsa", "run_id": "fixture", "group": case["group"],
                         "round": i, "link": "cloud->edge", "direction": "send", "phase": "honest",
                         "payload_kind": "fixture", "sender": "CS", "receiver": "edge_0",
                         "num_bytes": 100, "note": ""})
        honest.append({"scheme": "hvsa", "group": case["group"], "context": context, "round": i,
                       "d": 32, "N": 10, "M": 2, "exact_match": True, "verified": True,
                       "input_sha256": hashlib.sha256(context.encode()).hexdigest()})
    raw.append({**raw[0], "phase": "tamper_verify", "detail": "test_attack", "verified": False})
    trial = {"scheme": "hvsa", "group": case["group"], "size": "small", "trial": 0,
             "tamper_type": "test_attack", "accepted": False, "honest_context_valid": True,
             "verification_ms": .5}
    counts = {"trials": 1, "rejected": 1, "accepted": 0}
    corr = {"experiment": "e5", "schemes": ["hvsa"], "trials": [trial], "honest_rounds": honest,
            "cases": [{"case": case, "scheme": "hvsa", "public_transcript": "public.pkl",
                       "public_transcript_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest()}],
            "types": {"hvsa:small:test_attack": counts}, "execution_errors": 0,
            "all_tested_attacks_rejected": True}
    summary = [{"scheme": "hvsa", "size": "small", "tamper_type": "test_attack", **counts}]
    finalize_run(run, manifest, raw, messages, corr, summary, [])
    assert validate(run) == []
    return run, manifest, raw, messages, corr, summary


@pytest.mark.parametrize("corruption", ["empty_raw", "missing_actor", "negative_time",
                                        "missing_trial", "false_all_rejected", "missing_case"])
def test_reject_incomplete_or_false_evidence(valid_run, corruption):
    run, manifest, raw, messages, corr, summary = valid_run
    if corruption == "empty_raw":
        raw = []
    elif corruption == "missing_actor":
        raw[0]["actor"] = ""
    elif corruption == "negative_time":
        # JSON NaN is itself rejected by the writer; negative finite time is still invalid.
        raw[0]["elapsed_ms"] = -1
    elif corruption == "missing_trial":
        corr["trials"] = []
    elif corruption == "false_all_rejected":
        corr["all_tested_attacks_rejected"] = False
    else:
        corr["cases"] = []
    # Recompute output hashes: semantic validation must still catch false content.
    finalize_run(run, manifest, raw, messages, corr, summary, [])
    assert validate(run)


def test_archived_source_is_checked_independently_of_workspace(valid_run):
    run, *_ = valid_run
    (run.parent / "repo" / "source.py").write_text("# later workspace edit\n", encoding="utf-8")
    assert validate(run) == []
    (run / "sources" / "source.py").write_text("# changed archive\n", encoding="utf-8")
    assert any("Archived source hash mismatch" in problem for problem in validate(run))


def test_output_corruption_and_public_transcript_corruption_are_detected(valid_run):
    run, *_ = valid_run
    (run / "summary.csv").write_text("changed", encoding="utf-8")
    assert validate(run)


def test_public_transcript_checked_even_when_output_hashes_unchanged(valid_run):
    run, *_ = valid_run
    (run / "public.pkl").write_bytes(b"changed-public-evidence")
    assert any("Public transcript hash mismatch" in problem for problem in validate(run))
