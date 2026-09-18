"""暂时断连(替换失效)场景: 并发离线组数使 checkpoint 替换机制失效.

上一轮发现: K-of-E 调度会吸收暂时断连 —— 2/5 条 edge 离线时另 3 条自动补足
K=3, N_Q 恒为 12, 断连不掉点, 复用纯负收益。

本实验让**并发离线组数越过替换边界** (可用组 < K), 这时聚合被迫在偏置子集上
进行, 才是暂时断连真正伤精度的机制:
  p3of5: 3/5 条 edge 同时离线 (可用 2 < K=3)
  p4of5: 4/5 条 edge 同时离线 (可用 1 < K=3)
两者都在区间结束后恢复 (暂时断连, 非永久)。

用法: python run_reuse_temp2.py --workers 6 [--resume] [--dry-run]
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

ROOT = "results_reuse/s6_temp_break"
SEEDS = [42, 43, 44, 45, 46, 47]
BASE = ("--dataset MNIST --max-checkpoints 20 --max-ticks 300 -K 3 --s-max 3 "
        "--noniid --alpha {a} --vcm --gamma-vcm 1.0 --staleness-sigma 0.5 "
        "--num-edge-servers 5 --users-per-edge 4 --dropout-rate 0 "
        "--delta-timeout 15")
BYPASS = "--dropout-mode offline_train --dropout-reuse rebase --reuse-rule none"
PATTERNS = {
    "p3of5": "0:20-55,1:20-55,3:25-60",     # 可用 2 组 (< K=3)
    "p4of5": "0:20-60,1:20-60,3:20-60,4:25-65",  # 可用 1 组 (< K)
}


def build_jobs():
    jobs = []
    for a in (0.1, 0.5):
        for pat, spans in PATTERNS.items():
            for arm in ("none", "skip", "bypass"):
                if arm == "none" and pat != "p3of5":
                    continue          # none 参照只需一份
                extra = "" if arm == "none" else f"--edge-offline-spans {spans}"
                if arm == "bypass":
                    extra += f" {BYPASS}"
                for s in SEEDS:
                    jobs.append((f"{pat}_a{int(a*100)}_{arm}_s{s}",
                                 f"{BASE.format(a=a)} --seed {s} {extra}".strip()))
    return jobs


def run_job(job):
    tag, extra = job
    os.makedirs(ROOT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = os.path.join(ROOT, f"point2_log_{tag}_{ts}.txt")
    cmd = [sys.executable, os.path.abspath("main_point2.py"), "--tag", tag] + extra.split()
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=dict(os.environ, OMP_NUM_THREADS="1",
                                       MKL_NUM_THREADS="1"), cwd=ROOT)
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
        for t, e in jobs[:2]:
            print(f"  [{t}] {e}")
        return
    os.makedirs(ROOT, exist_ok=True)
    if args.resume:
        done = set()
        for p in g.glob(os.path.join(ROOT, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(p))
            if m:
                done.add(m.group(1))
        jobs = [j for j in jobs if j[0] not in done]
        print(f"resume: 剩余 {len(jobs)}", flush=True)
    print(f"{len(jobs)} jobs -> {ROOT} (workers={args.workers})", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, ret, el = fut.result()
            n += 1
            if n % 10 == 0 or ret != 0:
                print(f"[{n}/{len(jobs)}] {'OK' if ret == 0 else f'FAIL({ret})'} {tag} {el:.0f}s",
                      flush=True)


if __name__ == "__main__":
    main()
