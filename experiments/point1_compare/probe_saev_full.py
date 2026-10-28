"""SAEV full-chain E8 admission diagnostic, not the formal cost matrix.

Uses the explicit CRT/field interpretation in adapters/saev_full.py. The
3-warmup/5-observation E8 performance matrix remains gated on correctness,
authentication and paper-interface qualification, regardless of exit status.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from .adapters.common import RoundResult, vector_digest, reference_sum, partition_users
from .adapters.saev_full import SAEVProtocol
from .artifacts import create_run, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=257)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--edges", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.dimension < 1 or args.clients < 2*args.edges or args.edges < 1:
        parser.error("Positive dimensions/edges and at least two users per edge required")
    import gmpy2
    rng = np.random.default_rng(args.seed)
    groups = [[f"u{i}" for i in group] for group in partition_users(rng, args.clients, args.edges, 2)]
    users = sorted(u for group in groups for u in group)
    online = sorted(set(users)-set(rng.choice(users, size=max(1, args.clients//5), replace=False)))
    specs = [{"case": "honest", "uploaded": users, "attack": None},
             {"case": "before_upload", "uploaded": online, "attack": None},
             {"case": "result_only_tamper", "uploaded": users, "attack": "result_only"},
             {"case": "coordinated_tamper", "uploaded": users, "attack": "coordinated"}]
    run_id = _new_run_id("saev_full_diagnostic")
    scheme = "saev_ffdhe3072_diagnostic"
    manifest = {"run_id": run_id, "experiment": "saev_full_diagnostic", "command": " ".join(sys.argv),
                "seed": args.seed, "schemes": [scheme], "git_commit": _git_commit(),
                "lib_versions": {**_lib_versions(), "gmpy2": gmpy2.version()},
                "quant": {"input": "signed int64", "amplitude": 1000, "decoding": "centered residues"},
                "protocol_mapping": "docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md",
                "paper_versions": {"saev": "10.1109/TIV.2025.3599909"}, "concurrency": "serial",
                "timing_scope": "admission diagnostic; not performance ranking", "verification": "all uploaded clients",
                "aggregation": "full CRT ciphertext sum; per-client decrypt and HH verification",
                "N": args.clients, "M": args.edges, "d": args.dimension, "groups": groups,
                "cases": specs, "field": "RFC7919 ffdhe3072 order-q subgroup, about 125 bits per RFC",
                "execution_status": "running"}
    manifest.update(_review_metadata(vars(args)))
    manifest["performance_qualified"] = False
    manifest["measurement_limitations"] = [
        "Explicit finite-field interpretation of HH/positive blinders, not author code.",
        "Uses the same 3072-bit p for the CRT interval and HH field; preserves full-width ciphertexts.",
        "Centered signed decoding and refreshed random mask seeds are implementation choices.",
        "Conditional diagnostic is not the E8 3-warmup/5-observation performance matrix."]
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    print("RUN_DIR="+str(directory), flush=True)
    state = SAEVProtocol(args.dimension, groups)
    setup = RoundResult(scheme, True, None, None, args.clients, 0, state.init_phases, state.init_messages)
    raw = round_rows(setup, scheme=scheme, run_id=run_id, seed=args.seed, group="init",
                     dimension=args.dimension, m=args.edges, round_t=-1)
    messages = message_rows(setup, scheme=scheme, run_id=run_id, group="init", round_t=-1)
    observations, failures = [], []
    for round_t, spec in enumerate(specs):
        vectors = {u: rng.integers(-1000, 1001, args.dimension, dtype=np.int64) for u in users}
        result = state.run_round(vectors, set(spec["uploaded"]), round_t, attack=spec["attack"])
        expected = reference_sum([vectors[u] for u in spec["uploaded"]])
        exact = bool(result.completed and np.array_equal(result.raw_sum, expected))
        observation = {"case": spec["case"], "round": round_t, "input_sha256": vector_digest(vectors),
                       "uploaded": spec["uploaded"], "completed": result.completed,
                       "verified": result.verified, "exact_match": exact, "reason": result.failure_reason,
                       "reference_sum": expected.tolist(), "returned_sum": result.raw_sum.tolist(),
                       "role_ms": {r: result.phase_ms(r) for r in ["user", "cloud", "ta", "edge"]},
                       "sent_bytes": sum(m.num_bytes for m in result.messages), "details": result.extra}
        observations.append(observation)
        raw.extend(round_rows(result, scheme=scheme, run_id=run_id, seed=args.seed, group=spec["case"],
                              dimension=args.dimension, m=args.edges, round_t=round_t))
        messages.extend(message_rows(result, scheme=scheme, run_id=run_id, group=spec["case"], round_t=round_t))
        if spec["attack"] is None and not (exact and result.verified and result.extra["all_recovered_vectors_equal"]):
            failures.append({"case": spec["case"], "reason": "honest correctness or verification failed"})
        print(f"{spec['case']}: exact={exact} verified={result.verified} "
              f"ciphertext_bits={result.extra['ciphertext_coordinate_max_bits']}", flush=True)
    correctness = {"experiment": manifest["experiment"], "schemes": [scheme], "observations": observations,
                   "paper_reproduction_qualified": False, "execution_errors": len(failures)}
    summary = [{"scheme": scheme, "case": o["case"], "completed": o["completed"],
                "exact_match": o["exact_match"], "verified": o["verified"]} for o in observations]
    finalize_run(directory, manifest, raw, messages, correctness, summary, failures)
    print("COMPLETED="+str(directory)+" failures="+str(len(failures)), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
