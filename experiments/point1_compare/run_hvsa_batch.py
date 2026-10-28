"""One-shot serial continuation of the remaining HVSA matrix.

This is an ordinary foreground experiment driver, not a scheduled automation.
Existing N-scan results are preserved, repeated E2 workpoints are omitted, and
unqualified cross-paper baselines are not silently included. Every child run
archives its actual source and is validated before the next run starts.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from .artifacts import digest, dump_json
from .run_compare import ROOT, _new_run_id
from .validate_results import validate

CRITICAL = ["user.py", "edge.py", "ta.py", "cloud.py", "utils.py",
            "experiments/point1_compare/adapters/common.py",
            "experiments/point1_compare/adapters/hvsa_point1.py",
            "experiments/point1_compare/run_performance.py",
            "experiments/point1_compare/watchdog.py", "experiments/point1_compare/configs/full.json"]


def jobs():
    queue = [("e6_formal", ["e6"]), ("e3_50_rounds", ["e3"]), ("e4_dropouts", ["e4"])]
    queue += [(f"e2_d{d}", ["e2", "--point", f"50,5,{d}"]) for d in [1000, 10000, 30000]]
    queue += [(f"e2_m{m}", ["e2", "--point", f"100,{m},7850"]) for m in [2, 10]]
    queue += [("e2_d100k_pilot", ["e2", "--point", "50,5,100000", "--warmup", "0", "--repeats", "1"]),
              ("e2_d100k_formal", ["e2", "--point", "50,5,100000"])]
    return queue


def continuation_jobs(parent):
    if parent["status"] == "running" or any(j["status"] == "running" for j in parent["jobs"]):
        raise ValueError("Prior batch must be verified terminal before continuing; do not duplicate a live worker")
    completed = {"completed", "completed_with_protocol_failures"}
    queue = [(j["name"], j["arguments"]) for j in parent["jobs"] if j["status"] not in completed]
    if any(name in ("e6_formal", "e6_full_off_control") for name, _ in queue):
        return queue
    done_e6 = [j for j in parent["jobs"] if j["name"] in ("e6_formal", "e6_full_off_control")
               and j["status"] in completed]
    full_off_done = False
    for job in done_e6:
        path = Path(job["run_dir"])/"manifest.json"
        if digest(path) != job["manifest_sha256"]:
            raise ValueError("Completed E6 manifest changed; review before reusing its observations")
        recorded = json.loads(path.read_text(encoding="utf-8"))
        off_points = {(s["N"], s["M"], s["d"]) for s in recorded.get("settings", [])
                      if s["variant"] == "hvsa_verification_off" and not s["verification_enabled"]}
        full_off_done |= (recorded.get("verification_off_scope") == "full_per_round_including_ta_mask_hash"
                          and {(50, 5, 7850), (50, 5, 30000)} <= off_points)
    if done_e6 and not full_off_done:
        queue.insert(0, ("e6_full_off_control", ["e6", "--variants", "hvsa_verification_off"]))
    return queue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-batch", help="Terminal prior batch.json; preserve completed jobs and continue pending work")
    args = parser.parse_args(argv)
    parent = None
    if args.resume_batch:
        parent_path = Path(args.resume_batch).resolve()
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        completed = {"completed", "completed_with_protocol_failures"}
        queue = continuation_jobs(parent)
    else:
        queue = jobs()
    directory = Path(ROOT)/"results"/"point1_compare"/_new_run_id("hvsa_batch")
    directory.mkdir(parents=True, exist_ok=False)
    journal = {"kind": "one_shot_serial_batch", "pid": os.getpid(), "started": datetime.now().isoformat(),
               "status": "running", "critical_source_sha256": {p: digest(Path(ROOT)/p) for p in CRITICAL},
               "preserved_e2_n_scan": "20261018-195929-325740_e2_hvsa_v2",
               "source_version_note": "Earlier N scan predates canonical-output guard; preserve its provenance and disclose the version boundary.",
               "jobs": [{"name": name, "arguments": options, "status": "pending"} for name, options in queue]}
    if parent is not None:
        journal["continued_from"] = {"path": str(parent_path), "sha256": digest(parent_path),
                                     "completed_jobs": [j for j in parent["jobs"] if j["status"] in completed]}
    dump_json(directory/"batch.json", journal)
    print("BATCH_DIR="+str(directory), flush=True)
    pilot_valid = bool(parent and any(j["name"] == "e2_d100k_pilot" and j["status"] == "completed"
                                     and j.get("invalid_observations") == 0 for j in parent["jobs"]))
    for job in journal["jobs"]:
        changed = [p for p, expected in journal["critical_source_sha256"].items() if digest(Path(ROOT)/p) != expected]
        if changed:
            journal.update(status="stopped_source_changed", changed=changed)
            dump_json(directory/"batch.json", journal)
            return 2
        if job["name"] == "e2_d100k_formal" and not pilot_valid:
            job["status"] = "skipped_pilot_not_qualified"
            continue
        command = [sys.executable, "-u", "-m", "experiments.point1_compare.run_performance", *job["arguments"]]
        job.update(status="running", command=command, started=datetime.now().isoformat())
        dump_json(directory/"batch.json", journal)
        print("START="+job["name"], flush=True)
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        run_dir = None
        with (directory/(job["name"]+".log")).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", env=environment)
            job["pid"] = child.pid
            dump_json(directory/"batch.json", journal)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(job["name"]+": "+line.rstrip(), flush=True)
                if line.startswith("RUN_DIR="):
                    run_dir = Path(line.strip().split("=", 1)[1])
                    job["run_dir"] = str(run_dir)
                    dump_json(directory/"batch.json", journal)
            code = child.wait()
        job.update(returncode=code, finished=datetime.now().isoformat())
        problems = validate(run_dir) if run_dir else ["Child did not report run directory"]
        job["validation_problems"] = problems
        if not problems:
            manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
            corr = json.loads((run_dir/"correctness.json").read_text(encoding="utf-8"))
            job["manifest_sha256"] = digest(run_dir/"manifest.json")
            job["invalid_observations"] = sum(not o["valid"] for o in corr["observations"])
            job["status"] = "completed" if not job["invalid_observations"] else "completed_with_protocol_failures"
            if job["name"] == "e2_d100k_pilot":
                pilot_valid = code == 0 and job["invalid_observations"] == 0
        else:
            job["status"] = "incomplete_or_invalid"
        dump_json(directory/"batch.json", journal)
        # E4 may lawfully preserve completion failures for its success-rate
        # analysis. Other correctness failures need review before more costs.
        if problems or (code != 0 and job["name"] != "e4_dropouts"):
            journal["status"] = "stopped_for_review"
            dump_json(directory/"batch.json", journal)
            return 1
    journal.update(status="completed_hvsa_jobs_only", finished=datetime.now().isoformat())
    dump_json(directory/"batch.json", journal)
    print("BATCH_COMPLETED="+str(directory), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
