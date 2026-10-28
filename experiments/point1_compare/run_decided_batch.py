"""One-shot serial execution of the four authorised paired cost workpoints."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from .artifacts import digest, dump_json
from .run_compare import ROOT, _new_run_id
from .validate_results import validate

POINTS = [(20,5,1000),(20,5,7850),(50,5,1000),(50,5,7850)]
SCHEMES = {"hvsa","gpvo_zn2_fresh_prime_e","vlmrs_fullgraph_public_d"}
DECISIONS = "experiments/point1_compare/configs/baseline_decisions_20261019.json"
PILOT = "results/point1_compare/20261019-182131-848006_decided_baseline_pilot"
CRITICAL = ["user.py","edge.py","ta.py","cloud.py","utils.py",DECISIONS,
    "experiments/point1_compare/adapters/common.py",
    "experiments/point1_compare/adapters/hvsa_point1.py",
    "experiments/point1_compare/adapters/gpvo_full.py",
    "experiments/point1_compare/adapters/gpvo_decided.py",
    "experiments/point1_compare/adapters/vlmrs_fullgraph.py",
    "experiments/point1_compare/adapters/vlmrs_verification.py",
    "experiments/point1_compare/adapters/sharing.py",
    "experiments/point1_compare/artifacts.py",
    "experiments/point1_compare/run_decided_comparison.py",
    "experiments/point1_compare/run_decided_batch.py",
    "experiments/point1_compare/run_compare.py",
    "experiments/point1_compare/run_performance.py",
    "experiments/point1_compare/validate_results.py",
    "experiments/point1_compare/watchdog.py"]


def cost_scope_problems(manifest, corr, point):
    """Reject a pilot, partial schedule or another point masquerading as this job."""
    n,m,d=point
    problems=[]
    specs=manifest.get("settings",[])
    obs=corr.get("observations",[])
    if (manifest.get("experiment")!="decided_comparison" or manifest.get("pilot") is not False
            or set(manifest.get("schemes",[]))!=SCHEMES or len(specs)!=3):
        problems.append("Not the authorised three-scheme non-pilot cost run")
    for s in specs:
        if (s.get("N")!=n or s.get("d")!=d or s.get("M")!=(None if s.get("variant")=="vlmrs_fullgraph_public_d" else m)
                or s.get("warmup")!=3 or s.get("rounds")!=10 or s.get("case")!="honest"):
            problems.append("Cost point/repeat specification differs")
    if (len(obs)!=39 or sum(o.get("warmup") is True for o in obs)!=9
            or any(not o.get("valid") for o in obs)):
        problems.append("Cost run does not contain 39 valid actual rounds, including 9 warmups")
    return problems


def main():
    repository=Path(ROOT)
    pilot=repository/PILOT
    problems=validate(pilot)
    if problems: raise ValueError("Pilot archive invalid: "+repr(problems))
    pm=json.loads((pilot/"manifest.json").read_text(encoding="utf-8"))
    pc=json.loads((pilot/"correctness.json").read_text(encoding="utf-8"))
    if (not pm["pilot"] or pm["decisions_sha256"]!=digest(repository/DECISIONS)
            or len(pc["observations"])!=9 or not all(o["valid"] for o in pc["observations"])):
        raise ValueError("Decisions or pilot coverage do not match")
    import psutil
    for process in psutil.process_iter(["pid","name","cmdline"]):
        if process.pid==os.getpid(): continue
        try:
            args=process.info.get("cmdline") or []
            if (process.info.get("name") or "").lower() in ("python.exe","python") and any(
                    item in ("experiments.point1_compare.run_decided_comparison","experiments.point1_compare.run_decided_batch") for item in args):
                raise ValueError("Another decided-matrix worker is active: "+str(process.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    directory=repository/"results/point1_compare"/_new_run_id("decided_minimum_batch")
    directory.mkdir(parents=True,exist_ok=False)
    journal={"kind":"one_shot_decided_cost_batch","pid":os.getpid(),"started":datetime.now().astimezone().isoformat(),
             "status":"running","scope":"four explicitly decided implementation workpoints; not an equal-security ranking",
             "pilot":{"path":str(pilot),"manifest_sha256":digest(pilot/"manifest.json")},
             "decisions_sha256":digest(repository/DECISIONS),
             "critical_source_sha256":{p:digest(repository/p) for p in CRITICAL},
             "jobs":[{"name":f"n{n}_d{d}","point":[n,m,d],"status":"pending"} for n,m,d in POINTS]}
    dump_json(directory/"batch.json",journal)
    print("BATCH_DIR="+str(directory),flush=True)
    for job in journal["jobs"]:
        changed=[p for p,h in journal["critical_source_sha256"].items() if digest(repository/p)!=h]
        if changed:
            journal.update(status="stopped_source_changed",changed=changed)
            dump_json(directory/"batch.json",journal)
            return 2
        point=",".join(map(str,job["point"]))
        command=[sys.executable,"-u","-m","experiments.point1_compare.run_decided_comparison",
                 "--point",point,"--warmup","3","--repeats","10","--seed","42","--round-timeout","1800"]
        job.update(status="running",command=command,started=datetime.now().astimezone().isoformat())
        dump_json(directory/"batch.json",journal)
        env=os.environ.copy(); env["PYTHONIOENCODING"]="utf-8"
        run=None
        print("START="+job["name"],flush=True)
        with (directory/(job["name"]+".log")).open("w",encoding="utf-8") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding="utf-8",errors="replace",env=env)
            job["pid"]=child.pid
            dump_json(directory/"batch.json",journal)
            for line in child.stdout:
                log.write(line); log.flush()
                print(job["name"]+": "+line.rstrip(),flush=True)
                if line.startswith("RUN_DIR="):
                    run=Path(line.strip().split("=",1)[1])
                    job["run_dir"]=str(run)
                    dump_json(directory/"batch.json",journal)
            code=child.wait()
        job.update(returncode=code,finished=datetime.now().astimezone().isoformat())
        problems=validate(run) if run else ["Child did not report its run directory"]
        if not problems:
            manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            corr=json.loads((run/"correctness.json").read_text(encoding="utf-8"))
            problems+=cost_scope_problems(manifest,corr,job["point"])
            if manifest["decisions_sha256"]!=journal["decisions_sha256"]:
                problems.append("Child decision version differs from batch")
            job["manifest_sha256"]=digest(run/"manifest.json")
            job["observations"]=len(corr["observations"])
            job["invalid_observations"]=sum(not o["valid"] for o in corr["observations"])
        changed=[p for p,h in journal["critical_source_sha256"].items() if digest(repository/p)!=h]
        problems += ["Frozen source changed: "+p for p in changed]
        job.update(validation_problems=problems,status="completed" if not problems and code==0 else "incomplete_or_invalid")
        dump_json(directory/"batch.json",journal)
        if job["status"]!="completed":
            journal["status"]="stopped_needs_review"
            dump_json(directory/"batch.json",journal)
            return code or 2
        print("VALIDATED="+job["name"],flush=True)
    journal.update(status="completed_decided_minimum_matrix",finished=datetime.now().astimezone().isoformat(),
                   executed_rounds=156,warmup_rounds=36,measured_rounds=120)
    dump_json(directory/"batch.json",journal)
    print("BATCH_COMPLETED="+str(directory),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
