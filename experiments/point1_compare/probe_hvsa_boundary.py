"""Actual full-dimension concentrated-dropout boundary; not a random rate scan."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from .adapters.common import RoundResult, reference_sum, vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .artifacts import create_run, digest, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata


def main():
    dimension = 7850
    groups = [[f"u{g*20+i}" for i in range(20)] for g in range(5)]
    users = sorted(u for group in groups for u in group)
    missing = set(groups[0])
    cases = [
        {"case": "one_empty_group_before_upload", "uploaded": sorted(set(users)-missing),
         "recipients": sorted(set(users)-missing), "expected_completed": False, "expected_verified": None},
        {"case": "same_group_offline_after_upload", "uploaded": users,
         "recipients": sorted(set(users)-missing), "expected_completed": True, "expected_verified": True},
        {"case": "all_clients_offline_after_upload", "uploaded": users,
         "recipients": [], "expected_completed": True, "expected_verified": True},
    ]
    run_id = _new_run_id("e4_hvsa_boundary")
    manifest = {"run_id": run_id, "experiment": "e4_boundary", "command": " ".join(sys.argv),
        "seed": 42, "schemes": ["hvsa"], "git_commit": _git_commit(), "lib_versions": _lib_versions(),
        "quant": {"input": "signed int64", "amplitude": 1000},
        "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
        "paper_versions": {"hvsa": "current real point-1 entities"}, "concurrency": "serial",
        "timing_scope": "functional boundary diagnostic, not cost ranking", "verification": "full",
        "aggregation": "point-1 all-registered-edge barrier; no empty-group redesign",
        "N": 100, "M": 5, "d": dimension, "groups": groups, "cases": cases, "execution_status": "running"}
    manifest.update(_review_metadata({"dimension": dimension, "groups": groups, "cases": cases}))
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    (directory/"inputs").mkdir()
    print("RUN_DIR="+str(directory), flush=True)
    state = HVSAProtocol(dimension, groups)
    setup = RoundResult("hvsa", True, None, None, len(users), 0, state.init_phases, state.init_messages)
    raw = round_rows(setup, scheme="hvsa", run_id=run_id, seed=42, group="init", dimension=dimension, m=5, round_t=-1)
    messages = message_rows(setup, scheme="hvsa", run_id=run_id, group="init", round_t=-1)
    rng = np.random.default_rng(42)
    observations, failures = [], []
    for round_t, spec in enumerate(cases):
        vectors = {u: rng.integers(-1000,1001,dimension,dtype=np.int64) for u in users}
        snapshot = directory/"inputs"/(spec["case"]+".npz")
        np.savez(snapshot, user_ids=np.array(users), vectors=np.stack([vectors[u] for u in users]))
        result = state.run_round(vectors, set(spec["uploaded"]), round_t,
                                 model_recipients=set(spec["recipients"]), n_registered=100)
        reference = reference_sum([vectors[u] for u in spec["uploaded"]])
        exact = result.completed and np.array_equal(reference, result.raw_sum)
        publishes = [m for m in result.messages if m.payload_kind == "verified_model"]
        matches = (result.completed is spec["expected_completed"]
                   and result.verified is spec["expected_verified"]
                   and (exact if result.completed else result.raw_sum is None)
                   and sorted(m.receiver for m in publishes) == (spec["recipients"] if result.completed else []))
        observations.append({"case": spec["case"], "round": round_t, "uploaded": spec["uploaded"],
            "recipients": spec["recipients"], "input_sha256": vector_digest(vectors),
            "snapshot": str(snapshot.relative_to(directory)), "snapshot_sha256": digest(snapshot),
            "completed": result.completed, "verified": result.verified, "exact_match": bool(exact),
            "matches_expected_boundary": bool(matches), "reason": result.failure_reason,
            "reference_sum": reference.tolist(), "returned_sum": result.raw_sum.tolist() if result.raw_sum is not None else None,
            "model_receivers": sorted(m.receiver for m in publishes),
            "per_group_uploaded": [len(set(g)&set(spec["uploaded"])) for g in groups],
            "role_ms": {r:result.phase_ms(r) for r in ["user","edge","ta","cloud"]},
            "sent_bytes": sum(m.num_bytes for m in result.messages)})
        raw.extend(round_rows(result, scheme="hvsa", run_id=run_id, seed=42, group=spec["case"], dimension=dimension, m=5, round_t=round_t))
        messages.extend(message_rows(result, scheme="hvsa", run_id=run_id, group=spec["case"], round_t=round_t))
        if not matches:
            failures.append({"case": spec["case"], "reason": "actual boundary differs from expected protocol behavior"})
        print(f"{spec['case']}: completed={result.completed} verified={result.verified} exact={exact} matches_boundary={matches}", flush=True)
    summary = [{"case":o["case"], "completed":o["completed"], "verified":o["verified"],
                "matches_expected_boundary":o["matches_expected_boundary"]} for o in observations]
    finalize_run(directory, manifest, raw, messages, {"experiment":"e4_boundary", "observations":observations,
                 "execution_errors":len(failures)}, summary, failures)
    from .validate_results import validate
    problems = validate(directory)
    if problems:
        raise ValueError("Boundary evidence failed reconciliation: "+"; ".join(problems))
    print("COMPLETED="+str(directory), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
