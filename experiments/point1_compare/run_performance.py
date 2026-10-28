"""Measured HVSA E2/E3/E4/E6; no incomplete baseline is silently substituted.

Defaults follow configs/full.json. All rounds execute the real point-1 entities.
The 3-mask control is an E6 ablation, not GPVO. Wall time is serial local replay,
serialized bytes are logical envelopes, and verification-off is not secure.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .adapters.common import RoundResult, gen_client_vectors, partition_users, reference_sum
from .adapters.hvsa_point1 import HVSAProtocol
from .artifacts import create_run, dump_json, finalize_run, message_rows, round_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .run_integrity import vector_digest
from .watchdog import RoundDeadline
from utils import LHH


def specifications(cfg, args):
    """Unique actual workpoints, with explicitly labelled interventions."""
    exp = args.experiment
    if args.point:
        n, m, d = map(int, args.point.split(","))
        if exp not in ("e2", "e6"):
            raise ValueError("--point applies only to E2/E6")
        values = [(n, m, d)]
    elif exp == "e2":
        points = set()
        for axis in ("n", "d", "m") if args.scan == "all" else [args.scan]:
            scan = cfg[axis + "_scan"]
            for value in scan["values"]:
                points.add(tuple(value if key == axis else scan[key] for key in ("n", "m", "d")))
        values = sorted(points)
    elif exp == "e6":
        scan = cfg["ablation"]["mask_layers"]
        values = [(scan["n"], scan["m"], d) for d in scan["d"]]
    elif exp == "e3":
        scan = cfg["round_scan"]
        values = [(scan["n"], scan["m"], scan["d"])]
    else:
        scan = cfg["dropout"]
        values = [(scan["n_selected"], scan["m"], scan["d"])]
    specs = []
    for n, m, d in values:
        if n < 2*m or d <= 0:
            raise ValueError("Each point requires N>=2*M and d>0")
        base = {"N": n, "M": m, "d": d, "mask_layers": 1, "verification_enabled": True,
                "round_timeout_seconds": getattr(args, "round_timeout", 1800)}
        if exp in ("e2", "e6"):
            for variant, layers, enabled in ([("hvsa", 1, True)] if exp == "e2" else
                                            [("hvsa", 1, True), ("hvsa_three_mask_control", 3, True),
                                             ("hvsa_verification_off", 1, False)]):
                if exp == "e6" and getattr(args, "variants", None) and variant not in args.variants.split(","):
                    continue
                specs.append({**base, "variant": variant, "mask_layers": layers,
                              "verification_enabled": enabled, "warmup": args.warmup,
                              "rounds": args.repeats, "participation": "fixed"})
        elif exp == "e3":
            for participation in ["fixed", "changing"]:
                specs.append({**base, "variant": "hvsa", "warmup": 0,
                              "rounds": max(cfg["round_scan"]["values"]),
                              "prefix_cutoffs": cfg["round_scan"]["values"],
                              "participation": participation})
        else:
            for stage in cfg["dropout"]["phases"]:
                for rate in cfg["dropout"]["rates"]:
                    specs.append({**base, "variant": "hvsa", "warmup": 0,
                                  "rounds": len(cfg["dropout"]["trajectory_seeds"]),
                                  "participation": "dropout", "dropout_stage": stage,
                                  "dropout_rate": rate,
                                  "trajectory_seeds": cfg["dropout"]["trajectory_seeds"]})
    return [{"setting": f"setting_{i:03d}", **s} for i, s in enumerate(specs)]


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(values)), "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)), "count": int(values.size)}


def iter_setting(spec, manifest, directory):
    n, m, d = (spec[k] for k in ["N", "M", "d"])
    rng = np.random.default_rng(manifest["seed"])
    groups = [[f"u{i}" for i in g] for g in partition_users(rng, n, m, 2)]
    all_users = {u for group in groups for u in group}
    # Give all E6 variants the same pseudorandom input stream and group assignment.
    LHH._generator_for.cache_clear()
    state = HVSAProtocol(d, groups, enable_verification=spec["verification_enabled"],
                         mask_layers=spec["mask_layers"])
    raw, messages, observations, failures = [], [], [], []
    scheme = spec["variant"]
    setup = RoundResult(scheme, True, None, None, n, 0, state.init_phases, state.init_messages)
    raw.extend(round_rows(setup, scheme=scheme, run_id=manifest["run_id"], seed=manifest["seed"],
                          group=spec["setting"], dimension=d, m=m, round_t=-1))
    messages.extend(message_rows(setup, scheme=scheme, run_id=manifest["run_id"],
                                 group=spec["setting"], round_t=-1))
    for index in range(spec["warmup"] + spec["rounds"]):
        warmup = index < spec["warmup"]
        repeat = index-spec["warmup"]
        online, recipients, unavailable = set(all_users), set(all_users), set()
        if spec["participation"] in ("changing", "dropout"):
            trace_seed = (spec["trajectory_seeds"][index] if spec["participation"] == "dropout"
                          else manifest["seed"]+10000+index)
            trace_rng = np.random.default_rng(trace_seed)
            rate = spec.get("dropout_rate", .2)
            unavailable = set(trace_rng.choice(sorted(all_users), size=int(n*rate), replace=False))
            recipients -= unavailable
            if spec.get("dropout_stage", "before_upload") == "before_upload":
                online -= unavailable
        vectors = gen_client_vectors(rng, n, d, {"uniform": 1., "zero": 0., "boundary": 0.}, 1000000000)
        def save_timeout():
            # Lists below contain only already completed rounds. The current
            # round's partial entity work must not be called a valid result.
            record = {"setting": spec["setting"], "round": index, "seed": manifest["seed"],
                      "round_timeout_seconds": spec["round_timeout_seconds"],
                      "input_sha256": vector_digest(vectors), "uploaded": sorted(online),
                      "recipients": sorted(recipients), "completed": False,
                      "reason": "round deadline exceeded; process terminated, later rounds not executed"}
            dump_json(directory/"timeout.json", record)
            dump_json(directory/"checkpoints"/"timeout_completed_prefix.json",
                      {"raw": raw, "messages": messages, "observations": observations, "failures": failures})
            manifest["execution_status"] = "timed_out"
            manifest["termination"] = record
            dump_json(directory/"manifest.json", manifest)
            print(f"TIMEOUT setting={spec['setting']} round={index}; incomplete run preserved", flush=True)
        tick = time.perf_counter()
        with RoundDeadline(spec["round_timeout_seconds"], save_timeout):
            result = state.run_round(vectors, online, index, model_recipients=recipients)
        wall_ms = (time.perf_counter()-tick)*1000
        exact = bool(result.completed and result.raw_sum is not None and np.array_equal(
            result.raw_sum, reference_sum([vectors[u] for u in sorted(online)])))
        valid = exact and (result.verified is True if spec["verification_enabled"] else result.verified is None)
        group = f"{spec['setting']}:{index}"
        raw.extend(round_rows(result, scheme=scheme, run_id=manifest["run_id"], seed=manifest["seed"],
                              group=group, dimension=d, m=m, round_t=index, repeat=repeat))
        messages.extend(message_rows(result, scheme=scheme, run_id=manifest["run_id"], group=group, round_t=index))
        actor_ms = defaultdict(float)
        for record in result.phases:
            actor_ms[f"{record.role}:{record.actor}"] += record.elapsed_ms
        links = defaultdict(int)
        for record in result.messages:
            links[record.link] += record.num_bytes
        observation = {"scheme": scheme, "setting": spec["setting"], "group": group,
                       "round": index, "repeat": repeat, "warmup": warmup,
                       "N": n, "M": m, "d": d, "n_uploaded": len(online),
                       "unavailable": sorted(unavailable), "input_sha256": vector_digest(vectors),
                       "completed": result.completed, "exact_match": exact, "verified": result.verified,
                       "valid": valid, "wall_ms": wall_ms, "role_ms": {
                           r: result.phase_ms(r) for r in ["user", "edge", "ta", "cloud"]},
                       "actor_ms": dict(actor_ms), "link_bytes": dict(links),
                       "total_sent_bytes": sum(links.values()), "failure_reason": result.failure_reason}
        observations.append(observation)
        if not valid:
            failures.append({"scheme": scheme, "setting": spec["setting"], "round": index,
                             "reason": result.failure_reason or "correctness/verification failure"})
        checkpoint = {"specification": spec, "init_ms": state.init_ms,
                      "init_sent_bytes": sum(x.num_bytes for x in state.init_messages),
                      "raw": raw, "messages": messages, "observations": observations, "failures": failures,
                      "cache": {"maxsize": 4096, "policy": "global cache cleared at each state's initialization; E6 state initializations interleave during warmup; no per-round clearing"}}
        dump_json(directory / "checkpoints" / (spec["setting"]+".json"), checkpoint)
        print(f"{spec['setting']} {scheme} d={d} round={index} warmup={warmup} valid={valid} "
              f"wall_ms={wall_ms:.1f}", flush=True)
        yield checkpoint


def execute_setting(spec, manifest, directory):
    latest = None
    for latest in iter_setting(spec, manifest, directory):
        pass
    return latest


def execute_e6_interleaved(specs, manifest, directory):
    """Same per-case input stream, real persistent states, rotating run order."""
    points = {}
    for spec in specs:
        points.setdefault((spec["N"], spec["M"], spec["d"]), []).append(spec)
    latest = {}
    for point_number, point_specs in enumerate(points.values()):
        if len({(s["warmup"], s["rounds"]) for s in point_specs}) != 1:
            raise ValueError("Interleaved E6 needs equal repeat schedules")
        iterators = [iter_setting(s, manifest, directory) for s in point_specs]
        for index in range(point_specs[0]["warmup"]+point_specs[0]["rounds"]):
            offset = (index+point_number) % len(iterators)
            order = list(range(len(iterators)))[offset:] + list(range(len(iterators)))[:offset]
            for which in order:
                latest[point_specs[which]["setting"]] = next(iterators[which])
    return [latest[s["setting"]] for s in specs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["e2", "e3", "e4", "e6"])
    parser.add_argument("--scan", choices=["all", "n", "d", "m"], default="all")
    parser.add_argument("--point", help="Explicit N,M,d workpoint for E2/E6")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "full.json"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-timeout", type=float, default=1800,
                        help="Hard seconds per round, shared by all settings; preserve prefix and stop on timeout")
    parser.add_argument("--variants", help="E6 only: comma-separated subset of the three named controls")
    args = parser.parse_args(argv)
    if args.variants:
        names = args.variants.split(",")
        allowed = {"hvsa", "hvsa_three_mask_control", "hvsa_verification_off"}
        if args.experiment != "e6" or len(names) != len(set(names)) or not set(names) <= allowed:
            parser.error("--variants requires unique E6 control names")
    if args.warmup < 0 or args.repeats <= 0 or not np.isfinite(args.round_timeout) or args.round_timeout <= 0:
        parser.error("Nonnegative warmup and positive repeats required")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    specs = specifications(cfg, args)
    logging.disable(logging.CRITICAL)
    run_id = _new_run_id(args.experiment+"_hvsa_v2")
    manifest = {"run_id": run_id, "experiment": args.experiment, "command": " ".join(sys.argv),
                "seed": args.seed, "schemes": sorted({s["variant"] for s in specs}),
                "git_commit": _git_commit(), "lib_versions": _lib_versions(), "quant": {"scale": 1000000},
                "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
                "paper_versions": {"hvsa": "local point-1 implementation"},
                "concurrency": "single-process serial replay", "timing_scope": "local protocol replay",
                "verification": "explicit per setting; off variant provides no integrity",
                "aggregation": "integer sum then deterministic edge average", "settings": specs,
                "execution_status": "running", "timeout_policy": "hard per-round process termination; incomplete prefix preserved",
                "setting_execution": "E6 rotates persistent variants each round at a fixed workpoint; others execute settings serially"}
    if args.experiment == "e6" and any(not s["verification_enabled"] for s in specs):
        manifest["verification_off_scope"] = "full_per_round_including_ta_mask_hash"
    manifest.update(_review_metadata({"cli": vars(args), "full_config": cfg}))
    manifest["measurement_limitations"] = ["HVSA and its controls only; no cross-paper performance ranking.",
        "Real serialized protocol envelopes, excluding transport headers and network latency.",
        "Dropout stages are replayed availability traces, not a deployed network experiment."]
    directory = Path(ROOT) / "results" / "point1_compare" / run_id
    create_run(directory, manifest, Path(ROOT))
    (directory / "checkpoints").mkdir()
    print(f"RUN_DIR={directory}", flush=True)
    checkpoints = (execute_e6_interleaved(specs, manifest, directory) if args.experiment == "e6"
                   else [execute_setting(s, manifest, directory) for s in specs])
    observations = [o for c in checkpoints for o in c["observations"]]
    summary = []
    for c in checkpoints:
        measured = [o for o in c["observations"] if not o["warmup"]]
        metrics = {"wall_ms": [o["wall_ms"] for o in measured],
                   "total_sent_bytes": [o["total_sent_bytes"] for o in measured]}
        metrics.update({r+"_ms": [o["role_ms"][r] for o in measured]
                        for r in ["user", "edge", "ta", "cloud"]})
        for metric, values in metrics.items():
            summary.append({"setting": c["specification"]["setting"],
                            "scheme": c["specification"]["variant"], "metric": metric, **stats(values)})
    corr = {"experiment": args.experiment, "schemes": manifest["schemes"],
            "observations": observations, "initialization": [
                {k: c[k] for k in ["specification", "init_ms", "init_sent_bytes", "cache"]} for c in checkpoints]}
    failures = [r for c in checkpoints for r in c["failures"]]
    finalize_run(directory, manifest, [r for c in checkpoints for r in c["raw"]],
                 [r for c in checkpoints for r in c["messages"]], corr, summary, failures)
    print(f"COMPLETED={directory} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
