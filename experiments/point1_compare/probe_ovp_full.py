"""Coordinate-wise OVP Fig.3 admission diagnostic; not the E8 cost matrix.

An optional --after-batch gate waits for the explicitly named foreground
serial experiment batch to finish, preventing concurrent crypto workloads.
This is one execution dependency, not a recurring or scheduled task.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .adapters.common import RoundResult, reference_sum, vector_digest
from .artifacts import create_run, finalize_run, digest
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .artifacts import round_rows, message_rows


def await_terminal_batch(path):
    """Never compute alongside an active batch or silently ignore a failed one."""
    import psutil
    path = Path(path).resolve()
    first = json.loads(path.read_text(encoding="utf-8"))
    if first["kind"] != "one_shot_serial_batch":
        raise ValueError("Expected a single-execution experiment batch")
    expected_pid, started = first["pid"], first["started"]
    print("WAIT_FOR_BATCH="+str(path), flush=True)
    while True:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current["pid"] != expected_pid or current["started"] != started:
            raise ValueError("Batch identity changed while waiting")
        if current["status"] != "running":
            if current["status"] != "completed_hvsa_jobs_only":
                raise ValueError("Prior batch needs review: "+current["status"])
            # The status is written just before normal parent exit. Let it
            # finish before admitting another cryptographic worker.
            try:
                psutil.Process(expected_pid).wait(timeout=30)
            except psutil.NoSuchProcess:
                pass
            return {"path": str(path), "status": current["status"], "sha256": digest(path)}
        if not psutil.pid_exists(expected_pid):
            raise ValueError("Batch journal says running but its process is absent")
        time.sleep(10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=257)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairing-deps", default=str(Path(ROOT)/"tmp"/"point1_pairing_deps"))
    parser.add_argument("--after-batch", help="Compute only after this serial batch completes successfully")
    parser.add_argument("--check-components", action="store_true", help="Run actual field/full-chain regression checks before admission diagnostic")
    args = parser.parse_args(argv)
    if args.dimension < 1 or args.clients < 3:
        parser.error("Positive dimension and at least three users required for the partial-upload case")
    dependency = await_terminal_batch(args.after_batch) if args.after_batch else None
    if args.check_components:
        subprocess.run([sys.executable, "-m", "pytest", str(Path(ROOT)/"tests"/"test_point1_ovp.py"), "-q"],
                       cwd=ROOT, check=True)
    sys.path.insert(0, str(Path(args.pairing_deps).resolve()))
    import py_ecc
    from .adapters.ovp_full import OVPProtocol, SCHEME
    rng = np.random.default_rng(args.seed)
    users = [f"u{i}" for i in range(args.clients)]
    reduced = users[:-max(1, args.clients//5)]
    specifications = [("honest", [users, users], None),
                      ("before_upload", [reduced, reduced], None),
                      ("partial_upload", [users, users[:-1]], None),
                      ("result_only_tamper", [users, users], "result_only"),
                      ("coordinated_tamper", [users, users], "coordinated")]
    cases = [{"case": name, "uploaded": sorted(set(a)&set(b)), "received_sets": [a, b], "attack": attack}
             for name, (a, b), attack in specifications]
    run_id = _new_run_id("ovp_full_diagnostic")
    manifest = {"run_id": run_id, "experiment": "ovp_full_diagnostic", "command": " ".join(sys.argv),
                "seed": args.seed, "schemes": [SCHEME], "git_commit": _git_commit(),
                "lib_versions": {**_lib_versions(), "py_ecc": py_ecc.__version__},
                "quant": {"input": "signed int64", "amplitude": 1000, "decoding": "centered F_q residues"},
                "protocol_mapping": "docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md",
                "paper_versions": {"ovp": "10.1109/TNSE.2025.3543601"}, "concurrency": "serial",
                "timing_scope": "full-chain admission diagnostic; not performance ranking",
                "verification": "each complete uploader verifies every coordinate",
                "aggregation": "actual DH/AE, two VS masks, additive sharing and coordinate GT feedback",
                "N": args.clients, "M": 2, "d": args.dimension, "cases": cases,
                "execution_status": "running", "prior_batch_dependency": dependency}
    manifest.update(_review_metadata(vars(args)))
    manifest["performance_qualified"] = False
    manifest["measurement_limitations"] = [
        "Coordinate-wise vector lifting follows the O(M) verification analysis, not author code.",
        "BLS12-381 DH and AES-GCM instantiate abstract KA/AE; paper experiments state CBC/1024 bits.",
        "Public pairing base precomputed once; full GT powers and bytes are used for every coordinate.",
        "DH keys and shared-key PRNG streams reused; no claim of qualified multi-round privacy.",
        "No additional authenticated cross-sum binding beyond Fig.3 feedback equations.",
        "This is not the E8 N20,d257/7850 3-warmup/5-observation performance matrix."]
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    print("RUN_DIR="+str(directory), flush=True)
    state = OVPProtocol(args.dimension, users)
    setup = RoundResult(SCHEME, True, None, None, args.clients, 0, state.init_phases, state.init_messages)
    raw = round_rows(setup, scheme=SCHEME, run_id=run_id, seed=args.seed, group="init",
                     dimension=args.dimension, m=2, round_t=-1)
    messages = message_rows(setup, scheme=SCHEME, run_id=run_id, group="init", round_t=-1)
    observations, failures = [], []
    for round_t, spec in enumerate(cases):
        vectors = {u: rng.integers(-1000, 1001, args.dimension, dtype=np.int64) for u in users}
        result = state.run_round(vectors, [set(s) for s in spec["received_sets"]], round_t, spec["attack"])
        reference = reference_sum([vectors[u] for u in spec["uploaded"]])
        exact = bool(result.completed and np.array_equal(reference, result.raw_sum))
        observations.append({"case": spec["case"], "round": round_t, "input_sha256": vector_digest(vectors),
            "uploaded": spec["uploaded"], "completed": result.completed, "verified": result.verified,
            "exact_match": exact, "reason": result.failure_reason, "reference_sum": reference.tolist(),
            "returned_sum": result.raw_sum.tolist() if result.raw_sum is not None else None,
            "role_ms": {r: result.phase_ms(r) for r in ["user", "cloud", "ta", "edge"]},
            "sent_bytes": sum(m.num_bytes for m in result.messages), "details": result.extra})
        raw.extend(round_rows(result, scheme=SCHEME, run_id=run_id, seed=args.seed, group=spec["case"],
                              dimension=args.dimension, m=2, round_t=round_t))
        messages.extend(message_rows(result, scheme=SCHEME, run_id=run_id, group=spec["case"], round_t=round_t))
        if spec["attack"] is None and not (exact and result.verified):
            failures.append({"case": spec["case"], "reason": "honest chain did not complete correctly and verify"})
        print(f"{spec['case']}: exact={exact} verified={result.verified}", flush=True)
    summary = [{"case": o["case"], "completed": o["completed"], "exact_match": o["exact_match"],
                "verified": o["verified"], "sent_bytes": o["sent_bytes"]} for o in observations]
    correctness = {"experiment": "ovp_full_diagnostic", "observations": observations,
                   "execution_errors": len(failures), "paper_reproduction_qualified": False}
    finalize_run(directory, manifest, raw, messages, correctness, summary, failures)
    from .validate_results import validate
    problems = validate(directory)
    if problems:
        raise ValueError("Diagnostic artifact failed reconciliation: "+"; ".join(problems))
    print("COMPLETED="+str(directory), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
