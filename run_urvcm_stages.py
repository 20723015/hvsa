"""UR-VCM 分阶段实验驱动 (更新相对可信域 VCM).

UR-VCM (vcm-mode=trust): R=原始聚合均值, B=所选组基模型样本量加权均值,
Δtrain=R−B, C_avg=原始补偿均值, η=min(1, ρ·‖Δtrain‖/(‖C_avg‖+ε)), W=R+ηC_avg。

阶段 1 (MNIST 筛选): G5, UR05, UR10, UR20 × seeds 42-44  (12 runs)
阶段 2 (泛化): 胜出 ρ 在 FashionMNIST / CIFAR10 Non-IID × 42-44 × {G5, UR}
              (12 runs), 同 seed 独立配对。

用法:
    python run_urvcm_stages.py --stage 1 --workers 4
    python run_urvcm_stages.py --stage 2 --rho 1.0 --workers 4
    python run_urvcm_stages.py --stage 1 --dry-run
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = "results_urvcm"
BASE = ("--dataset {ds} --max-checkpoints 20 --max-ticks 300 "
        "-K 3 --s-max 3 --noniid --alpha 0.5 --reuse-rule none "
        "--vcm --gamma-vcm 1.0 --staleness-sigma 0.5")

SEEDS = [42, 43, 44]
S1_GROUPS = [("G5", "--vcm-mode full"),
             ("UR05", "--vcm-mode trust --vcm-trust-ratio 0.5"),
             ("UR10", "--vcm-mode trust --vcm-trust-ratio 1.0"),
             ("UR20", "--vcm-mode trust --vcm-trust-ratio 2.0")]


def build_jobs(stage, args):
    jobs = []
    if stage == 1:
        base = BASE.format(ds="MNIST")
        for tag, extra in S1_GROUPS:
            for s in SEEDS:
                jobs.append((f"s1_{tag}_s{s}", f"{base} --seed {s} {extra}"))
    elif stage == 2:
        if args.rho is None:
            raise SystemExit("stage 2 需要 --rho (阶段 1 胜出的可信域比率)")
        ur = f"--vcm-mode trust --vcm-trust-ratio {args.rho}"
        for ds in ("FashionMNIST", "CIFAR10"):
            base = BASE.format(ds=ds)
            for s in SEEDS:
                jobs.append((f"s2_{ds}_G5_s{s}",
                             f"{base} --seed {s} --vcm-mode full"))
                jobs.append((f"s2_{ds}_UR_s{s}", f"{base} --seed {s} {ur}"))
    else:
        raise SystemExit(f"stage {stage} 未定义 (可用: 1, 2)")
    return jobs


def run_job(job, sdir, workers_env):
    tag, extra = job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(sdir, f"point2_log_{tag}_{ts}.txt")
    cmd = [sys.executable, os.path.abspath("main_point2.py"),
           "--tag", tag] + extra.split()
    env = dict(os.environ, **workers_env)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=env, cwd=sdir)
    return tag, log_path, ret, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑: 跳过该阶段已有 npz 的任务 (按 tag 匹配)")
    args = ap.parse_args()

    jobs = build_jobs(args.stage, args)
    sdir = os.path.join(ROOT, f"s{args.stage}_screen" if args.stage == 1
                        else "s2_generalize")
    if args.dry_run:
        for tag, extra in jobs:
            print(f"[{tag}] python main_point2.py {extra}")
        print(f"stage {args.stage}: total {len(jobs)}")
        return

    os.makedirs(sdir, exist_ok=True)
    if args.resume:
        done_tags = set()
        import glob as _glob
        for p in _glob.glob(os.path.join(sdir, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(p))
            if m:
                done_tags.add(m.group(1))
        before = len(jobs)
        jobs = [j for j in jobs if j[0] not in done_tags]
        print(f"resume: 跳过 {before - len(jobs)} 个已完成任务, 剩余 {len(jobs)}", flush=True)
    workers_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"stage {args.stage}: {len(jobs)} jobs -> {sdir} "
          f"(workers={args.workers}, 单线程确定性模式)", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, sdir, workers_env) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, log_path, ret, elapsed = fut.result()
            done += 1
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"[{done}/{len(jobs)}] {status} {tag}  {elapsed:.0f}s "
                  f"-> {os.path.basename(log_path)}", flush=True)


if __name__ == "__main__":
    main()
