"""Continue a paused cost matrix, preserving complete points and frozen protocols."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from .artifacts import digest,dump_json
from .run_compare import ROOT,_new_run_id
from .run_decided_batch import POINTS,cost_scope_problems
from .validate_results import validate


def continuation_plan(parent_path,repository,validator=validate):
    parent=json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("kind")!="one_shot_decided_cost_batch" or parent.get("status") not in (
            "paused_by_user","stopped_needs_review"):
        raise ValueError("Only a verified terminal cost batch can be continued")
    if [tuple(j["point"]) for j in parent["jobs"]]!=POINTS:
        raise ValueError("Parent workpoint schedule differs")
    changed=[p for p,h in parent["critical_source_sha256"].items() if digest(repository/p)!=h]
    if changed:
        raise ValueError("Frozen sources changed: "+repr(changed))
    jobs=[]
    for old in parent["jobs"]:
        if old["status"]=="running":
            raise ValueError("Parent contains a running job")
        if old["status"]=="completed":
            run=Path(old["run_dir"])
            if digest(run/"manifest.json")!=old["manifest_sha256"]:
                raise ValueError("Completed manifest changed: "+old["name"])
            problems=validator(run)
            manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            corr=json.loads((run/"correctness.json").read_text(encoding="utf-8"))
            problems+=cost_scope_problems(manifest,corr,old["point"])
            if manifest["decisions_sha256"]!=parent["decisions_sha256"]:
                problems.append("Completed point used another decision version")
            if problems:
                raise ValueError("Preserved point invalid: "+repr(problems))
            job=copy.deepcopy(old)
            job["preserved_from"]={"path":str(parent_path),"sha256":digest(parent_path)}
        else:
            job={"name":old["name"],"point":old["point"],"status":"pending"}
            if old.get("run_dir"):
                run=Path(old["run_dir"])
                if old.get("interrupted_manifest_sha256") and digest(run/"manifest.json")!=old["interrupted_manifest_sha256"]:
                    raise ValueError("Interrupted manifest changed")
                job["excluded_interrupted_attempt"]={"path":str(run),"status":old["status"],
                    "manifest_sha256":digest(run/"manifest.json"),
                    "completed_prefix_observations":old.get("completed_prefix_observations"),
                    "reason":"restart incomplete workpoint from fresh state; no paused timing is pooled"}
        jobs.append(job)
    return parent,jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent",help="Paused batch.json; its completed points are never rerun")
    args=parser.parse_args()
    repository=Path(ROOT)
    parent_path=Path(args.parent).resolve()
    parent,jobs=continuation_plan(parent_path,repository)
    import psutil
    modules={"experiments.point1_compare.run_decided_batch","experiments.point1_compare.continue_decided_batch",
             "experiments.point1_compare.run_decided_comparison"}
    for p in psutil.process_iter(["pid","name","cmdline"]):
        if p.pid==os.getpid(): continue
        try:
            if ((p.info.get("name") or "").lower() in ("python.exe","python")
                    and modules.intersection(p.info.get("cmdline") or [])):
                raise ValueError("Another matrix worker is active: "+str(p.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    directory=repository/"results/point1_compare"/_new_run_id("decided_minimum_continuation")
    directory.mkdir(parents=True,exist_ok=False)
    (directory/"parent_batch_snapshot.json").write_bytes(parent_path.read_bytes())
    (directory/"continuation_source.py").write_bytes(Path(__file__).read_bytes())
    journal={"kind":parent["kind"],"pid":os.getpid(),"started":datetime.now().astimezone().isoformat(),
        "status":"running","scope":parent["scope"],"pilot":parent["pilot"],
        "decisions_sha256":parent["decisions_sha256"],"critical_source_sha256":parent["critical_source_sha256"],
        "continued_from":{"path":str(parent_path),"sha256":digest(parent_path)},
        "continuation_source_sha256":digest(directory/"continuation_source.py"),"jobs":jobs,
        "resume_policy":"reuse complete N20 points; restart incomplete N50 point; interrupted prefix excluded"}
    dump_json(directory/"batch.json",journal)
    print("BATCH_DIR="+str(directory),flush=True)
    for job in jobs:
        if job["status"]=="completed":
            print("PRESERVED="+job["name"],flush=True)
            continue
        changed=[p for p,h in journal["critical_source_sha256"].items() if digest(repository/p)!=h]
        if changed:
            journal.update(status="stopped_source_changed",changed=changed)
            dump_json(directory/"batch.json",journal)
            return 2
        command=[sys.executable,"-u","-m","experiments.point1_compare.run_decided_comparison",
                 "--point",",".join(map(str,job["point"])),"--warmup","3","--repeats","10","--seed","42","--round-timeout","1800"]
        job.update(status="running",command=command,started=datetime.now().astimezone().isoformat())
        dump_json(directory/"batch.json",journal)
        env=os.environ.copy();env["PYTHONIOENCODING"]="utf-8"
        run=None
        print("START="+job["name"],flush=True)
        with (directory/(job["name"]+".log")).open("w",encoding="utf-8") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding="utf-8",errors="replace",env=env)
            job["pid"]=child.pid
            dump_json(directory/"batch.json",journal)
            for line in child.stdout:
                log.write(line);log.flush()
                print(job["name"]+": "+line.rstrip(),flush=True)
                if line.startswith("RUN_DIR="):
                    run=Path(line.strip().split("=",1)[1]);job["run_dir"]=str(run)
                    dump_json(directory/"batch.json",journal)
            code=child.wait()
        job.update(returncode=code,finished=datetime.now().astimezone().isoformat())
        problems=validate(run) if run else ["Child did not report its run directory"]
        if not problems:
            manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
            corr=json.loads((run/"correctness.json").read_text(encoding="utf-8"))
            problems+=cost_scope_problems(manifest,corr,job["point"])
            if manifest["decisions_sha256"]!=journal["decisions_sha256"]:
                problems.append("Decision version differs")
            job.update(manifest_sha256=digest(run/"manifest.json"),observations=len(corr["observations"]),
                       invalid_observations=sum(not o["valid"] for o in corr["observations"]))
        problems += ["Frozen source changed: "+p for p,h in journal["critical_source_sha256"].items() if digest(repository/p)!=h]
        job.update(validation_problems=problems,status="completed" if not problems and code==0 else "incomplete_or_invalid")
        dump_json(directory/"batch.json",journal)
        if job["status"]!="completed":
            journal["status"]="stopped_needs_review"
            dump_json(directory/"batch.json",journal)
            return code or 2
        print("VALIDATED="+job["name"],flush=True)
    journal.update(status="completed_decided_minimum_matrix",finished=datetime.now().astimezone().isoformat(),
                   executed_rounds=156,warmup_rounds=36,measured_rounds=120,
                   preserved_rounds=sum(j["observations"] for j in jobs if "preserved_from" in j),
                   newly_executed_rounds=sum(j["observations"] for j in jobs if "preserved_from" not in j))
    dump_json(directory/"batch.json",journal)
    print("BATCH_COMPLETED="+str(directory),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
