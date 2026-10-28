"""Paired-input comparison of explicitly decided, non-author-certified baselines.

--pilot executes honest/before/after availability cases once; it is not a
performance distribution. Otherwise each scheme executes 3+10 honest rounds.
No claim of equal security, exact paper instantiation or SOTA is made.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import logging
import math
from pathlib import Path
import sys
import time
import gmpy2
import numpy as np
from .adapters.common import RoundResult, partition_users, reference_sum, vector_digest
from .adapters.hvsa_point1 import HVSAProtocol
from .adapters.gpvo_decided import GPVODecidedProtocol, SCHEME as GPVO
from .artifacts import create_run, digest, dump_json, finalize_run, round_rows, message_rows
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata
from .run_performance import stats
from .validate_results import validate
from .watchdog import RoundDeadline
from utils import LHH

VLMRS = "vlmrs_fullgraph_public_d"
SCHEMES = ["hvsa", GPVO, VLMRS]
DECISIONS = Path(ROOT)/"experiments/point1_compare/configs/baseline_decisions_20261019.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", default="20,5,257", help="N,M,d; M applies only to hierarchical schemes")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    n,m,d = map(int, args.point.split(","))
    if n < 2*m or m < 1 or d < 1 or args.warmup < 0 or args.repeats < 1 or not math.isfinite(args.round_timeout) or args.round_timeout <= 0:
        parser.error("Positive point/deadline and nonnegative warmup required; N>=2*M")
    sys.path.insert(0, str(Path(ROOT)/"tmp/point1_pairing_deps"))
    import py_ecc
    from .adapters.vlmrs_fullgraph import VLMRSFullGraphProtocol
    logging.disable(logging.CRITICAL)
    decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))
    groups = [[f"u{i}" for i in group] for group in partition_users(np.random.default_rng(args.seed), n, m, 2)]
    users = sorted(u for group in groups for u in group)
    interleaved_users = [group[i] for i in range(max(map(len,groups))) for group in groups if i<len(group)]
    unavailable = set(interleaved_users[:n//5])
    cases = (["honest", "before_upload", "after_upload"] if args.pilot else ["honest"])
    warmup, repeats = (0,1) if args.pilot else (args.warmup,args.repeats)
    specs = [{"setting": f"setting_{i:03d}", "variant": scheme, "case": case,
              "N": n, "M": None if scheme==VLMRS else m, "d": d, "warmup": warmup,
              "rounds": repeats, "verification_enabled": True}
             for i,(case,scheme) in enumerate((c,s) for c in cases for s in SCHEMES)]
    run_id = _new_run_id("decided_baseline_pilot" if args.pilot else "decided_baseline_cost")
    manifest = {"run_id": run_id, "experiment": "decided_comparison", "command": " ".join(sys.argv),
        "seed": args.seed, "schemes": SCHEMES, "git_commit": _git_commit(),
        "lib_versions": {**_lib_versions(), "gmpy2": gmpy2.version(), "py_ecc": py_ecc.__version__},
        "quant": {"input": "signed int64", "amplitude": 1000000000, "scale": 1000000},
        "protocol_mapping": "experiments/point1_compare/protocol_mapping.md",
        "paper_versions": {"gpvo": "declared Z_N2/fresh-session instantiation", "vlmrs": "declared fullgraph/public-d/reuse instantiation"},
        "concurrency": "serial, scheme order rotates by round", "timing_scope": "local full-chain replay",
        "verification": "all protocol-designated verifiers execute all checks",
        "aggregation": "same uploaded-vector integer reference, protocol-specific output encoding",
        "settings": specs, "groups": groups, "users": users, "pilot": args.pilot,
        "round_timeout_seconds": args.round_timeout, "execution_status": "running",
        "decisions": decisions, "decisions_sha256": digest(DECISIONS),
        "source_comparison": "recollected HVSA current version; old N-scan observations not pooled"}
    manifest.update(_review_metadata({"cli":vars(args), "decisions":decisions}))
    manifest["implementation_variants"] = {s:s for s in SCHEMES}
    manifest["performance_qualified"] = False
    manifest["scope_review"] = "docs/POINT1_BASELINE_DECISIONS_20261019.md"
    manifest["measurement_limitations"] = [
        "Declared implementations are cost-comparable under explicit choices, not proven equally secure or author-certified.",
        "GPVO uniform-unit g has no maximal-order certificate; integer-group LHF differs from stated P-256.",
        "GPVO gmpy2 is a native backend; VLMRS pairings and HVSA curve work use their recorded Python libraries.",
        "VLMRS public-root scaling/history recovery remain; costs do not constitute security acceptance.",
        "Pilot availability cases are individual observations, not a 3+10 performance experiment."]
    directory = Path(ROOT)/"results/point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    (directory/"inputs").mkdir(); (directory/"checkpoints").mkdir()
    print("RUN_DIR="+str(directory), flush=True)
    raw, messages, observations, failures, initialization = [],[],[],[],[]

    def timed_out(context):
        dump_json(directory/"checkpoints/completed_prefix.json", {"raw":raw,"messages":messages,"observations":observations,"failures":failures})
        manifest["execution_status"] = "timed_out"
        manifest["termination"] = context
        dump_json(directory/"manifest.json", manifest)
        print("TIMED_OUT="+repr(context), flush=True)
    states = {}
    LHH._generator_for.cache_clear()
    for scheme in SCHEMES:
        with RoundDeadline(args.round_timeout, lambda s=scheme: timed_out({"phase":"initialization","scheme":s})):
            state = (HVSAProtocol(d,groups) if scheme=="hvsa" else GPVODecidedProtocol(d,groups)
                     if scheme==GPVO else VLMRSFullGraphProtocol(d,users,threshold=math.ceil(2*n/3),root_exponent=7))
        states[scheme] = state
        setup = RoundResult(scheme,True,None,None,n,0,state.init_phases,state.init_messages)
        raw += round_rows(setup,scheme=scheme,run_id=run_id,seed=args.seed,group="init:"+scheme,dimension=d,m=None if scheme==VLMRS else m,round_t=-1)
        messages += message_rows(setup,scheme=scheme,run_id=run_id,group="init:"+scheme,round_t=-1)
        initialization.append({"scheme":scheme,"role_ms":{r:setup.phase_ms(r) for r in ["user","edge","ta","cloud"]},
                               "sent_bytes":sum(x.num_bytes for x in setup.messages)})
        print("INITIALIZED="+scheme,flush=True)
    params = states[GPVO].params
    manifest["gpvo_public_parameters"] = {"N":params.n_mod,"g":params.g,"e":params.e,
        "modulus_bits":params.n_mod.bit_length(),"e_bits":params.e.bit_length(),
        "e_is_prime":bool(gmpy2.is_prime(params.e)),"g_is_unit":math.gcd(params.g,params.n_mod)==1,
        "maximal_order_certified":False}
    dump_json(directory/"manifest.json",manifest)
    for case_index,case in enumerate(cases):
        uploaded = set(users)-unavailable if case=="before_upload" else set(users)
        recipients = set(users)-unavailable if case in ["before_upload","after_upload"] else set(users)
        for index in range(warmup+repeats):
            rng = np.random.default_rng(args.seed+case_index*100000+index)
            vectors = {u:rng.integers(-10**9,10**9,d,dtype=np.int64) for u in users}
            snapshot = directory/"inputs"/(case+"_"+str(index)+".npz")
            np.savez(snapshot,user_ids=np.array(users),vectors=np.stack([vectors[u] for u in users]))
            reference = reference_sum([vectors[u] for u in sorted(uploaded)])
            offset = (index+case_index)%len(SCHEMES)
            for scheme in SCHEMES[offset:]+SCHEMES[:offset]:
                spec = next(s for s in specs if s["variant"]==scheme and s["case"]==case)
                protocol_round = case_index*(warmup+repeats)+index
                context = {"setting":spec["setting"],"round":index,"protocol_round":protocol_round,
                           "input_sha256":vector_digest(vectors),"snapshot":str(snapshot.relative_to(directory)),
                           "uploaded":sorted(uploaded),"recipients":sorted(recipients),"completed":False}
                tick = time.perf_counter()
                with RoundDeadline(args.round_timeout,lambda c=context:timed_out(c)):
                    result = (states[scheme].run_round(vectors,uploaded,protocol_round,recovery_clients=recipients)
                              if scheme==VLMRS else states[scheme].run_round(vectors,uploaded,protocol_round,model_recipients=recipients))
                wall_ms = (time.perf_counter()-tick)*1000
                exact = bool(result.completed and result.raw_sum is not None and np.array_equal(result.raw_sum,reference))
                valid = exact and result.verified is True
                group = spec["setting"]+":"+str(index)
                raw += round_rows(result,scheme=scheme,run_id=run_id,seed=args.seed,group=group,dimension=d,m=spec["M"],round_t=protocol_round,repeat=index-warmup)
                messages += message_rows(result,scheme=scheme,run_id=run_id,group=group,round_t=protocol_round)
                actors,links = defaultdict(float),defaultdict(int)
                for p in result.phases: actors[p.role+":"+p.actor] += p.elapsed_ms
                for msg in result.messages: links[msg.link] += msg.num_bytes
                observations.append({"scheme":scheme,"setting":spec["setting"],"group":group,"case":case,
                    "round":index,"protocol_round":protocol_round,"repeat":index-warmup,"warmup":index<warmup,
                    "N":n,"M":spec["M"],"d":d,"n_uploaded":len(uploaded),"uploaded":sorted(uploaded),"recipients":sorted(recipients),
                    "input_sha256":vector_digest(vectors),"snapshot":str(snapshot.relative_to(directory)),"snapshot_sha256":digest(snapshot),
                    "reference_sum":reference.tolist(),"returned_sum":result.raw_sum.tolist() if result.raw_sum is not None else None,
                    "completed":result.completed,"exact_match":exact,"verified":result.verified,"valid":valid,
                    "wall_ms":wall_ms,"role_ms":{r:result.phase_ms(r) for r in ["user","edge","ta","cloud"]},
                    "actor_ms":dict(actors),"link_bytes":dict(links),"total_sent_bytes":sum(links.values()),
                    "failure_reason":result.failure_reason})
                if not valid: failures.append({"scheme":scheme,"setting":spec["setting"],"round":index,"reason":result.failure_reason or "honest-chain mismatch"})
                dump_json(directory/"checkpoints/completed_prefix.json",{"raw":raw,"messages":messages,"observations":observations,"failures":failures})
                print(f"{scheme} {case} round={index} valid={valid} wall_ms={wall_ms:.2f}",flush=True)
    summary = []
    for spec in specs:
        obs = [o for o in observations if o["setting"]==spec["setting"] and not o["warmup"]]
        for metric in ["wall_ms","total_sent_bytes","user_ms","edge_ms","ta_ms","cloud_ms"]:
            values = [o["role_ms"][metric[:-3]] if metric.endswith("_ms") and metric!="wall_ms" else o[metric] for o in obs]
            summary.append({"setting":spec["setting"],"scheme":spec["variant"],"metric":metric,**stats(values)})
    corr = {"experiment":"decided_comparison","observations":observations,"initialization":initialization,
            "execution_errors":len(failures),"paper_reproduction_qualified":False}
    finalize_run(directory,manifest,raw,messages,corr,summary,failures)
    problems = validate(directory)
    if problems: raise ValueError("Evidence reconciliation failed: "+"; ".join(problems))
    print("COMPLETED="+str(directory),flush=True)
    return int(bool(failures))


if __name__=="__main__":
    raise SystemExit(main())
