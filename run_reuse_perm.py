"""固定永久离线子集下的复用实验 (真实部署语义).

与 s3 的区别: 离线设备在建模开始即确定 (--permanent-dropout-frac), 之后不再变化;
不使用 --dropout-rate/--dropout-burst 的"每轮重掷 + 永久化"衰减语义。

网格: (v4=2ES×2U/K2, v6=2ES×3U/K2, v20=5ES×4U/K3) × f ∈ {0.25, 0.50}
      × 臂 {skip, bypass} × 10 seeds (42-51)
      (v20 只跑 f=0.25)

用法: python run_reuse_perm.py --workers 6 [--resume] [--dry-run]
"""

import argparse
import concurrent.futures as cf
import glob as g
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = "results_reuse/s4_perm_fixed"
SEEDS = list(range(42, 52))
CELLS = [("v4", 2, 2, 2), ("v6", 2, 3, 2), ("v20", 5, 4, 3)]
BYPASS = ("--dropout-mode offline_train --dropout-reuse rebase --reuse-rule none")


def build_jobs():
    jobs = []
    for cell, E, U, K in CELLS:
        for f in (0.25, 0.50):
            if cell == "v20" and f == 0.50:
                continue
            for arm in ("skip", "bypass"):
                extra = BYPASS if arm == "bypass" else ""
                for s in SEEDS:
                    base = (f"--dataset MNIST --max-checkpoints 20 --max-ticks 400 "
                            f"-K {K} --s-max 3 --noniid --alpha 0.5 --vcm "
                            f"--gamma-vcm 1.0 --staleness-sigma 0.5 "
                            f"--num-edge-servers {E} --users-per-edge {U} "
                            f"--delta-timeout 15 --permanent-dropout-frac {f} "
                            f"--seed {s} {extra}")
                    jobs.append((f"{cell}_f{int(f*100)}_{arm}_s{s}", base.strip()))
    return jobs


def run_job(job, env_extra):
    tag, extra = job
    os.makedirs(ROOT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = os.path.join(ROOT, f"point2_log_{tag}_{ts}.txt")
    cmd = [sys.executable, os.path.abspath("main_point2.py"), "--tag", tag] + extra.split()
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, **env_extra), cwd=ROOT)
    return tag, ret, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = build_jobs()
    if args.dry_run:
        print(f"total {len(jobs)}")
        return
    os.makedirs(ROOT, exist_ok=True)
    if args.resume:
        done = set()
        for p in g.glob(os.path.join(ROOT, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(p))
            if m:
                done.add(m.group(1))
        jobs = [j for j in jobs if j[0] not in done]
        # 清理未完成的残留日志
        print(f"resume: 剩余 {len(jobs)}", flush=True)
    env_extra = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"{len(jobs)} jobs -> {ROOT} (workers={args.workers})", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, env_extra) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, ret, el = fut.result()
            n += 1
            if n % 10 == 0 or ret != 0:
                print(f"[{n}/{len(jobs)}] {'OK' if ret == 0 else f'FAIL({ret})'} {tag} {el:.0f}s",
                      flush=True)


if __name__ == "__main__":
    main()
