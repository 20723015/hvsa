"""E5 individual trials, independent inputs, full-size checks and public transcripts.

GPVO names below identify diagnostic compatibility variants, never a validated
reproduction of the paper. Passing execution is distinct from rejecting attacks.
Run from the repository root: python -m experiments.point1_compare.run_integrity
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from .adapters.common import gen_client_vectors, partition_users, reference_sum, vector_digest
from .adapters.gpvo_full import GPVOSession, com_param_gen, lhf_hash, run_gpvo_round
from .adapters.hvsa_point1 import HVSAProtocol
from .artifacts import create_run, digest, dump_json, finalize_run, message_rows, round_rows
from .run_compare import (ROOT, _git_commit, _gpvo_verify_only, _lib_versions,
                          _new_run_id, _review_metadata)
from utils import LHH

COMMON_TYPES = ["modify_raw_sum_single", "modify_raw_sum_multi", "modify_raw_sum_scale",
                "modify_evidence_hash", "drop_group", "duplicate_group",
                "coordinated_sum_and_hash", "coordinated_drop_group",
                "coordinated_duplicate_group", "replay_same_set", "replay_changed_set"]
HVSA_TYPES = COMMON_TYPES + ["modify_evidence_signature", "modify_context_round",
                             "modify_context_base_hash", "modify_context_online_set",
                             "modify_raw_sum_order_alias", "modify_raw_sum_float_fraction",
                             "modify_raw_sum_wrong_shape"]
GPVO_TYPES = COMMON_TYPES
SCHEMES = ["hvsa", "gpvo_compat_fixed", "gpvo_compat_fresh"]
MIX = {"uniform": .8, "zero": .1, "boundary": .1}


def mutate_sum(kind, original, rng):
    result = original.copy()
    if kind.endswith("single"):
        result[int(rng.integers(result.size))] += int(rng.integers(1, 1000000))
    elif kind.endswith("multi"):
        indices = rng.choice(result.size, max(1, result.size // 10), replace=False)
        result[indices] += rng.integers(1, 1000000, len(indices), dtype=np.int64)
    else:
        result = (result.astype(np.float64) * 1.01).astype(np.int64)
        if np.array_equal(result, original):
            result[0] += 1
    return result


def hvsa_mutation(kind, original, evidence, first_sum, pp, rng, round_t):
    """Attacker sees cloud uploads and public parameters, not signing keys."""
    x, proof = original.copy(), copy.deepcopy(evidence)
    delta = np.zeros_like(x)
    delta[0] = 7
    if kind == "modify_raw_sum_order_alias":
        x = x.astype(object)
        x[0] += LHH.CURVE.order
    elif kind == "modify_raw_sum_float_fraction":
        x = x.astype(np.float64)
        x[0] += .5 if x[0] >= 0 else -.5
    elif kind == "modify_raw_sum_wrong_shape":
        x = x.reshape(1, -1)
    elif kind.startswith("modify_raw_sum"):
        x = mutate_sum(kind, x, rng)
    elif kind in ("modify_evidence_hash", "coordinated_sum_and_hash"):
        proof[0]["lhh_hash"] = LHH.Mul(proof[0]["lhh_hash"], LHH.Hash(pp, delta))
        if kind == "coordinated_sum_and_hash":
            x += delta
    elif kind == "modify_evidence_signature":
        sig = proof[0]["signature"]
        proof[0]["signature"] = bytes([sig[0] ^ 1]) + sig[1:]
    elif kind == "modify_context_round":
        proof[0]["round"] = round_t + 1
    elif kind == "modify_context_base_hash":
        value = proof[0]["base_hash"]
        proof[0]["base_hash"] = bytes([value[0] ^ 1]) + value[1:]
    elif kind == "modify_context_online_set":
        proof[0]["online_users"] = list(proof[0]["online_users"]) + ["ghost-user"]
    elif kind in ("drop_group", "coordinated_drop_group"):
        proof = proof[1:]
        if kind == "coordinated_drop_group":
            x -= first_sum
    elif kind in ("duplicate_group", "coordinated_duplicate_group"):
        proof.append(copy.deepcopy(proof[0]))
        if kind == "coordinated_duplicate_group":
            x += first_sum
    else:
        raise ValueError(kind)
    return x, proof


def gpvo_mutation(kind, original, art, session, rng):
    x, c = original.copy(), art["cloud"]["c_agg"]
    edge = next(iter(art["edge"].values()))
    if kind.startswith("modify_raw_sum"):
        x = mutate_sum(kind, x, rng)
    elif kind == "modify_evidence_hash":
        c ^= 1
    elif kind == "coordinated_sum_and_hash":
        delta = np.zeros_like(x)
        delta[0] = 7
        x += delta
        c = c * pow(lhf_hash(session.com, delta), session.com.e, session.com.n2) % session.com.n2
    elif kind in ("drop_group", "coordinated_drop_group"):
        c = c * pow(edge["c_k"], -1, session.com.n2) % session.com.n2
        if kind == "coordinated_drop_group":
            x -= edge["x_agg_k"]
    elif kind in ("duplicate_group", "coordinated_duplicate_group"):
        c = c * edge["c_k"] % session.com.n2
        if kind == "coordinated_duplicate_group":
            x += edge["x_agg_k"]
    else:
        raise ValueError(kind)
    return x, c


def execute_case(scheme, case, directory, manifest, params):
    seed, n, m, d = (case[k] for k in ["seed", "N", "M", "d"])
    rng = np.random.default_rng(seed)
    groups = [[f"u{i}" for i in group] for group in partition_users(rng, n, m, 2)]
    online = {u for group in groups for u in group}
    # Remove one user from a >=2-person group; no empty-group abort is counted as rejection.
    changed_online = online - {groups[0][-1]}
    raw, messages, honest, attacks, failures, transcripts = [], [], [], [], [], []
    start = time.perf_counter()
    state = None
    captured = []
    old_x = old_art = old_proof = first_sum = None

    def record_attack(kind, accepted, elapsed, round_t, payload_digest):
        attacks.append({"scheme": scheme, "group": case["group"], "size": case["size"],
                        "trial": case["trial"], "tamper_type": kind, "accepted": bool(accepted),
                        "honest_context_valid": True, "verification_ms": elapsed,
                        "payload_sha256": payload_digest, "round": round_t})
        raw.append({"scheme": scheme, "run_id": manifest["run_id"], "seed": seed,
                    "group": case["group"], "repeat": 0, "N_selected": n,
                    "N_uploaded": len(changed_online) if kind == "replay_changed_set" else n,
                    "M": m, "d": d, "round": round_t, "phase": "tamper_verify",
                    "role": "edge", "actor": "edge_0", "detail": kind,
                    "elapsed_ms": elapsed, "sent_bytes": None, "success": True,
                    "verified": bool(accepted), "error_reason": ""})

    for context_index, context in enumerate(["initial", "next_same_set", "next_changed_set"]):
        if state is None or scheme == "gpvo_compat_fresh":
            if scheme == "hvsa":
                state = HVSAProtocol(d, groups)
                original_receive = state.cs.receive_pre_aggregation

                def capture(data):
                    captured.append(pickle.loads(data))
                    return original_receive(data)

                state.cs.receive_pre_aggregation = capture
            else:
                state = GPVOSession(d, groups, com_params=params)
            # The legacy GPVO initializer does not have exclusive setup records.
            if scheme == "hvsa":
                from .adapters.common import RoundResult
                setup = RoundResult(scheme, True, None, None, n, 0,
                                    state.init_phases, state.init_messages)
                raw.extend(round_rows(setup, scheme=scheme, run_id=manifest["run_id"],
                                      seed=seed, group=case["group"], dimension=d, m=m,
                                      round_t=-1))
                messages.extend(message_rows(setup, scheme=scheme, run_id=manifest["run_id"],
                                             group=case["group"], round_t=-1))
        vectors = gen_client_vectors(rng, n, d, MIX, 1000000000)
        current_online = changed_online if context_index == 2 else online
        round_t = 3 * case["ordinal"] + context_index
        captured.clear()
        result = (state.run_round(vectors, current_online, round_t) if scheme == "hvsa"
                  else run_gpvo_round(state, vectors, current_online, round_t))
        exact = bool(result.completed and result.raw_sum is not None and np.array_equal(
            result.raw_sum, reference_sum([vectors[u] for u in sorted(current_online)])))
        honest.append({"scheme": scheme, "group": case["group"], "context": context,
                       "round": round_t, "exact_match": exact, "verified": bool(result.verified),
                       "input_sha256": vector_digest(vectors), "n_online": len(current_online),
                       "d": d, "N": n, "M": m})
        raw.extend(round_rows(result, scheme=scheme, run_id=manifest["run_id"], seed=seed,
                              group=case["group"], dimension=d, m=m, round_t=round_t))
        messages.extend(message_rows(result, scheme=scheme, run_id=manifest["run_id"],
                                     group=case["group"], round_t=round_t))
        if not (exact and result.verified):
            failures.append({"scheme": scheme, "group": case["group"], "context": context,
                             "reason": result.failure_reason or "honest correctness/verification failed"})
            break
        if scheme == "hvsa":
            proof = copy.deepcopy(state.cs._collected_hash_data)
            transcripts.append({"round": round_t, "raw_sum": result.raw_sum,
                                "evidence": proof, "cloud_received_group_packets": copy.deepcopy(captured),
                                "ta_verification_aid": state.ta.get_verification_info(),
                                "edge_public_keys": copy.deepcopy(state.ta.get_all_es_public_keys()),
                                "base_hash": state.edges[0]._p1_base_hash})
        else:
            art = copy.deepcopy(result.extra["artifacts"])
            transcripts.append({"round": round_t, "artifacts": art, "public_parameters": state.com})

        if context_index == 0:
            old_x = result.raw_sum.copy()
            old_art = copy.deepcopy(result.extra.get("artifacts"))
            old_proof = copy.deepcopy(proof) if scheme == "hvsa" else None
            first_sum = captured[0]["pre_aggregated_gradient"].copy() if scheme == "hvsa" else None
            types = HVSA_TYPES if scheme == "hvsa" else GPVO_TYPES
            for kind in types:
                if kind.startswith("replay_"):
                    continue
                if scheme == "hvsa":
                    tx, tp = hvsa_mutation(kind, old_x, old_proof, first_sum,
                                           state.ta.get_lhh_pp(), rng, round_t)
                    check = lambda: state.edges[0].verify_global_result(tx, tp)
                else:
                    tx, tp = gpvo_mutation(kind, old_x, old_art, state, rng)
                    aid = old_art["ta"]
                    check = lambda: _gpvo_verify_only(state, tx, tp, aid["h_ta"], aid["r"])
                payload_digest = hashlib.sha256(pickle.dumps((tx, tp))).hexdigest()
                tick = time.perf_counter()
                accepted = check()
                record_attack(kind, accepted, (time.perf_counter()-tick)*1000, round_t, payload_digest)
        else:
            kind = "replay_same_set" if context_index == 1 else "replay_changed_set"
            tick = time.perf_counter()
            if scheme == "hvsa":
                accepted = state.edges[0].verify_global_result(old_x, old_proof)
                payload = old_proof
            else:
                aid = art["ta"]  # Current randomness, including fresh-session r.
                accepted = _gpvo_verify_only(state, old_x, old_art["cloud"]["c_agg"],
                                             aid["h_ta"], aid["r"])
                payload = old_art["cloud"]["c_agg"]
            elapsed = (time.perf_counter()-tick)*1000
            record_attack(kind, accepted, elapsed, round_t,
                          hashlib.sha256(pickle.dumps((old_x, payload))).hexdigest())

    case_dir = directory / "checkpoints" / scheme
    case_dir.mkdir(parents=True, exist_ok=True)
    public_path = case_dir / (case["group"] + "_public.pkl")
    public_path.write_bytes(pickle.dumps(transcripts))
    checkpoint = {"case": case, "scheme": scheme, "groups": groups, "honest_rounds": honest,
                  "trials": attacks, "raw": raw, "messages": messages, "failures": failures,
                  "public_transcript": public_path.relative_to(directory).as_posix(),
                  "public_transcript_sha256": digest(public_path),
                  "elapsed_seconds": time.perf_counter()-start}
    dump_json(case_dir / (case["group"] + ".json"), checkpoint)
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", default="hvsa")
    parser.add_argument("--small-trials", type=int, default=100)
    parser.add_argument("--full-trials", type=int, default=10)
    parser.add_argument("--full-dim", type=int, default=7850)
    parser.add_argument("--small-dim", type=int, default=32)
    parser.add_argument("--rsa-bits", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.small_trials < 0 or args.full_trials < 0 or args.small_trials+args.full_trials == 0:
        parser.error("At least one trial is required, with non-negative counts")
    schemes = args.schemes.split(",")
    if len(set(schemes)) != len(schemes) or not set(schemes) <= set(SCHEMES):
        parser.error("Use unique schemes from " + ",".join(SCHEMES))
    logging.disable(logging.CRITICAL)
    run_id = _new_run_id("e5_v2")
    schedule = []
    for size, count, n, m, d in [("small", args.small_trials, 10, 2, args.small_dim),
                                  ("full", args.full_trials, 20, 5, args.full_dim)]:
        for trial in range(count):
            ordinal = len(schedule)
            schedule.append({"group": f"{size}_{trial:04d}", "size": size, "trial": trial,
                             "ordinal": ordinal, "seed": args.seed+777+ordinal,
                             "N": n, "M": m, "d": d})
    cfg = vars(args).copy()
    manifest = {"run_id": run_id, "experiment": "e5", "command": " ".join(sys.argv),
                "seed": args.seed, "schemes": schemes, "git_commit": _git_commit(),
                "lib_versions": _lib_versions(), "quant": {"scale": 1000000},
                "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
                "paper_versions": {"gpvo": "TNSM 2025; compatibility variants only"},
                "concurrency": "single-process serial replay", "timing_scope": "local replay",
                "verification": "full", "aggregation": "integer sum then deterministic edge average",
                "execution_status": "running", "schedule": schedule,
                "tamper_types": {s: HVSA_TYPES if s == "hvsa" else GPVO_TYPES for s in schemes},
                "synthetic_inputs": True}
    manifest.update(_review_metadata(cfg))
    manifest["measurement_limitations"] = [
        "HVSA bytes are serialized protocol envelopes, not measured TCP/TLS traffic.",
        "GPVO compatibility variants have unresolved paper fidelity and measurement alignment.",
        "Independent synthetic integer updates; not training snapshots or a security proof."]
    directory = Path(ROOT) / "results" / "point1_compare" / run_id
    create_run(directory, manifest, Path(ROOT))
    print(f"RUN_DIR={directory}", flush=True)
    params = com_param_gen(args.rsa_bits) if any(s != "hvsa" for s in schemes) else None
    all_checkpoints = []
    for scheme in schemes:
        for case in schedule:
            checkpoint = execute_case(scheme, case, directory, manifest, params)
            all_checkpoints.append(checkpoint)
            accepted = sum(t["accepted"] for t in checkpoint["trials"])
            print(f"{scheme} {case['group']} d={case['d']} accepted={accepted}/"
                  f"{len(checkpoint['trials'])} failures={len(checkpoint['failures'])} "
                  f"seconds={checkpoint['elapsed_seconds']:.2f}", flush=True)
    raw = [r for c in all_checkpoints for r in c["raw"]]
    messages = [r for c in all_checkpoints for r in c["messages"]]
    failures = [r for c in all_checkpoints for r in c["failures"]]
    trials = [r for c in all_checkpoints for r in c["trials"]]
    honest = [r for c in all_checkpoints for r in c["honest_rounds"]]
    types, summary = {}, []
    for scheme in schemes:
        for size in sorted({c["size"] for c in schedule}):
            for kind in manifest["tamper_types"][scheme]:
                records = [t for t in trials if t["scheme"] == scheme and t["size"] == size
                           and t["tamper_type"] == kind]
                counts = {"trials": len(records), "rejected": sum(not t["accepted"] for t in records),
                          "accepted": sum(t["accepted"] for t in records)}
                types[f"{scheme}:{size}:{kind}"] = counts
                summary.append({"scheme": scheme, "size": size, "tamper_type": kind, **counts})
    case_refs = [{k: c[k] for k in ["case", "scheme", "public_transcript", "public_transcript_sha256"]}
                 for c in all_checkpoints]
    correctness = {"experiment": "e5", "schemes": schemes, "types": types, "trials": trials,
                   "honest_rounds": honest, "cases": case_refs,
                   "all_tested_attacks_rejected": bool(trials) and not any(t["accepted"] for t in trials),
                   "execution_errors": len(failures)}
    finalize_run(directory, manifest, raw, messages, correctness, summary, failures)
    print(f"COMPLETED={directory} attacks={len(trials)} execution_errors={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
