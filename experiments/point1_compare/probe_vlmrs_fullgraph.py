"""Complete all-neighbour VLMRS diagnostic with explicit instantiation choices.

Not a qualified paper performance run: public d/HF/graph/key-lifecycle choices
remain different from an author-verified implementation. All algorithms are
executed; failures and accepted coordinated modifications remain in evidence.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .artifacts import create_run, finalize_run, round_rows, message_rows
from .adapters.common import RoundResult, vector_digest, reference_sum
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-deps", default=str(Path(ROOT)/"tmp"/"point1_pairing_deps"))
    parser.add_argument("--dimension", type=int, default=257)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.dimension < 1:
        parser.error("Positive dimension required")
    sys.path.insert(0, str(Path(args.pairing_deps).resolve()))
    import py_ecc
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    users, threshold = [f"u{i}" for i in range(6)], 4
    all_users = set(users)
    specs = [
        {"case": "honest", "uploaded": users, "recovery": users, "scale": None},
        {"case": "before_upload", "uploaded": users[1:], "recovery": users[1:], "scale": None},
        {"case": "after_upload", "uploaded": users, "recovery": users[1:], "scale": None},
        {"case": "below_threshold", "uploaded": users, "recovery": users[3:], "scale": None},
        {"case": "coordinated_scaling", "uploaded": users, "recovery": users, "scale": 2}]
    run_id = _new_run_id("vlmrs_fullgraph_diagnostic")
    scheme = "vlmrs_fullgraph_public_d"
    manifest = {"run_id": run_id, "experiment": "vlmrs_fullgraph_diagnostic", "command": " ".join(sys.argv),
                "seed": args.seed, "schemes": [scheme], "git_commit": _git_commit(),
                "lib_versions": {**_lib_versions(), "py_ecc": py_ecc.__version__},
                "quant": {"input": "signed int64", "amplitude": 1000},
                "protocol_mapping": "docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md",
                "paper_versions": {"vlmrs": "10.1109/TDSC.2025.3606413"},
                "concurrency": "serial", "timing_scope": "diagnostic work only; not performance ranking",
                "verification": "all remaining clients execute real BLS12-381 equations 4-10",
                "aggregation": "double-masked integer sum with genuine Shamir reconstruction",
                "users": users, "threshold": threshold, "d": args.dimension,
                "cases": specs, "execution_status": "running"}
    manifest.update(_review_metadata(vars(args)))
    manifest["performance_qualified"] = False
    manifest["measurement_limitations"] = [
        "All-neighbour graph epsilon=1; not frozen paper graph settings.",
        "Keyed linear vector HF and shared verification keys are explicit choices.",
        "Public root exponent d=7; long-term masking/encryption keys reused.",
        "Accepted modifications and later-key recovery are conditional diagnostics, not a paper-wide vulnerability verdict."]
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    print("RUN_DIR="+str(directory), flush=True)
    state = VLMRSFullGraphProtocol(args.dimension, users, threshold=threshold)
    setup = RoundResult(scheme, True, None, None, len(users), 0, state.init_phases, state.init_messages)
    raw = round_rows(setup, scheme=scheme, run_id=run_id, seed=args.seed, group="init", dimension=args.dimension, m=0, round_t=-1)
    messages = message_rows(setup, scheme=scheme, run_id=run_id, group="init", round_t=-1)
    observations, failures = [], []
    rng = np.random.default_rng(args.seed)
    past_reference = None
    for round_t, spec in enumerate(specs):
        vectors = {u: rng.integers(-1000, 1000, args.dimension, dtype=np.int64) for u in users}
        if round_t == 0:
            past_reference = vectors["u0"].copy()
        result = state.run_round(vectors, set(spec["uploaded"]), round_t,
                                 recovery_clients=set(spec["recovery"]), scale_attack=spec["scale"])
        expected = reference_sum([vectors[u] for u in spec["uploaded"]])
        exact = bool(result.completed and np.array_equal(result.raw_sum, expected))
        observation = {"case": spec["case"], "round": round_t, "input_sha256": vector_digest(vectors),
                       "uploaded": spec["uploaded"], "recovery": spec["recovery"],
                       "completed": result.completed, "verified": result.verified,
                       "exact_match": exact, "reason": result.failure_reason,
                       "reference_sum": expected.tolist(),
                       "returned_sum": result.raw_sum.tolist() if result.raw_sum is not None else None,
                       "role_ms": {r: result.phase_ms(r) for r in ["user", "cloud", "ta", "edge"]},
                       "sent_bytes": sum(m.num_bytes for m in result.messages), "details": result.extra}
        observations.append(observation)
        raw.extend(round_rows(result, scheme=scheme, run_id=run_id, seed=args.seed, group=spec["case"],
                              dimension=args.dimension, m=0, round_t=round_t))
        messages.extend(message_rows(result, scheme=scheme, run_id=run_id, group=spec["case"], round_t=round_t))
        if spec["case"] in ("honest", "before_upload", "after_upload") and not (exact and result.verified):
            failures.append({"case": spec["case"], "reason": "honest correctness or verification failure"})
        if spec["case"] == "below_threshold" and result.completed:
            failures.append({"case": spec["case"], "reason": "completed without threshold"})
        print(f"{spec['case']}: completed={result.completed} exact={exact} verified={result.verified}", flush=True)
    recovered = state.recover_past_upload_with_later_key("u0", 0)
    reuse = {"condition": "same masking DH key across rounds; later dropout reconstructs it",
             "past_round": 0, "reconstruction_round": 1, "user": "u0",
             "attacker_uses_reference_vectors_or_verification_keys": False,
             "reference": past_reference.tolist(), "cloud_recovered": recovered.tolist(),
             "exact_match": bool(np.array_equal(recovered, past_reference))}
    correctness = {"experiment": manifest["experiment"], "schemes": [scheme],
                   "observations": observations, "later_key_recovery": reuse,
                   "paper_reproduction_qualified": False, "execution_errors": len(failures)}
    summary = [{"scheme": scheme, "case": o["case"], "completed": o["completed"],
                "exact_match": o["exact_match"], "verified": o["verified"]} for o in observations]
    finalize_run(directory, manifest, raw, messages, correctness, summary, failures)
    print("COMPLETED="+str(directory)+" failures="+str(len(failures)), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
