"""Serial, resumable 6-pair development screen; no hyperparameter search."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_cache_calibration/screen_20261008")
    parser.add_argument("--datasets", nargs="+", default=["FashionMNIST", "CIFAR10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-checkpoints", type=int, default=20)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent
    root = Path(args.root).resolve()
    jobs = [(dataset, seed) for dataset in args.datasets for seed in args.seeds]
    commands = []
    for dataset, seed in jobs:
        directory = root / f"{dataset}_NI_s{seed}"
        command = [sys.executable, str(workspace / "plain_cache_bench.py"),
                   "--dataset", dataset, "--seed", str(seed), "--alpha", "0.5",
                   "--max-checkpoints", str(args.max_checkpoints),
                   "--output-dir", str(directory)]
        commands.append((directory, command))
    if args.plan:
        print(json.dumps([command for _, command in commands], indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        plan = {"phase": "development_screen", "datasets": args.datasets,
                "seeds": args.seeds, "max_checkpoints": args.max_checkpoints,
                "alpha": .5, "arms": ["increment", "calibration"],
                "primary_budget": "shared simulation time and actual local training work",
                "holdout_seeds_reserved": [52, 53, 54, 55, 56],
                "no_parameter_search": True,
                "screen_gate": "target cell mean final gain >=0.5pp, >=2/3 wins; inspect last5 and worst seed",
                "commands": [command for _, command in commands]}
        plan_path = root / "screen_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("existing screen plan differs")
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        manifest = root / "manifest.jsonl"
        for number, (directory, command) in enumerate(commands, 1):
            directory.mkdir(parents=True, exist_ok=True)
            print(f"[{number}/{len(commands)}] {directory.name} started", flush=True)
            start = time.perf_counter()
            with (directory / "pair.log").open("a", encoding="utf-8") as stream:
                code = subprocess.call(command, cwd=workspace, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONUTF8="1",
                                                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                                                CUBLAS_WORKSPACE_CONFIG=":4096:8"))
            entry = {"pair": directory.name, "command": command, "exit_code": code,
                     "wall_seconds": time.perf_counter() - start}
            if code == 0:
                metadata = json.loads((directory / "pair_meta.json").read_text(encoding="utf-8"))
                entry.update(config_hash=metadata["config_hash"], summary=metadata["summary"])
                gain = (metadata["summary"]["calibration"]["final_accuracy"] -
                        metadata["summary"]["increment"]["final_accuracy"])
                print(f"[{number}/{len(commands)}] {directory.name} complete delta={gain:+.2f}pp", flush=True)
            else:
                print(f"[{number}/{len(commands)}] {directory.name} failed exit={code}", flush=True)
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if code != 0:
                raise SystemExit(code)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
