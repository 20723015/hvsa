"""复用可用性边界实验: 退化率(训练不起来) vs 冗余度 × 永久掉线率.

目的 (接 docs/REUSE_DIRECTION_NOTES.md §2.5):
  P1 发现 4 用户 skip 臂 2/7 直接 0 checkpoint (联邦功能性崩溃)。本实验把
  "退化率" 做成可预测的协议可用性边界, 而非单个平均 Δ。

解析预测:
  每 edge U 用户, 每用户以 p 永久离线 → 该 edge 能封组 (在线≥n_min=2) 的概率
      P_seal(U,p) = 1 − [p^U + U(1−p)p^(U−1)]
  E 个 edge 下至少一个能封组 (配合 --delta-timeout 允许保守 checkpoint) 的概率
      P_viable = 1 − [1 − P_seal]^E
  退化率预测 = 1 − P_viable

网格: 2ES×{2,3,4}U (4/6/8 用户) × p∈{0.15,0.25,0.40}; 另含 12(3×4)/20(5×4) 单点。
  臂: skip (退化率主证据) + bypass (对退化最重的格子验证恢复)。
  每格 10 seeds (42-51)。

用法:
  python run_reuse_viability.py --workers 6
  python run_reuse_viability.py --resume
  python run_reuse_viability.py --dry-run
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = "results_reuse"
SEEDS = list(range(42, 52))
BASE = ("--dataset MNIST --max-checkpoints 20 --max-ticks 400 -K {K} --s-max 3 "
        "--noniid --alpha 0.5 --vcm --gamma-vcm 1.0 --staleness-sigma 0.5 "
        "--num-edge-servers {E} --users-per-edge {U} --delta-timeout 15")
SKIP = "--dropout-rate {p} --dropout-burst 999"
BYPASS = ("--dropout-rate {p} --dropout-burst 999 --dropout-mode offline_train "
          "--dropout-reuse rebase --reuse-rule none")

# (cell, E, U, K)
CELLS = [("v4", 2, 2, 2), ("v6", 2, 3, 2), ("v8", 2, 4, 2),
         ("v12", 3, 4, 3), ("v20", 5, 4, 3)]
RATES = [0.15, 0.25, 0.40]
# bypass 只跑退化最重的格子 (U=2 的 4 用户, 及 p=0.40 的 6 用户)
BYPASS_CELLS = {("v4", 0.25), ("v4", 0.40), ("v6", 0.40)}


def cell_p(E, U, K, p):
    return BASE.format(K=K, E=E, U=U) + " --dropout-rate %.2f --dropout-burst 999" % p


def build_jobs():
    jobs = []
    for cell, E, U, K in CELLS:
        rates = RATES if cell in ("v4", "v6") else [0.25]
        for p in rates:
            for s in SEEDS:
                jobs.append((f"{cell}_p{int(p*100)}_{s}", cell_p(E, U, K, p) + f" --seed {s}",
                             p, cell, U, E, "skip"))
                if (cell, p) in BYPASS_CELLS:
                    jobs.append((f"{cell}_p{int(p*100)}_byp_{s}",
                                 cell_p(E, U, K, p) + f" --seed {s} "
                                 + BYPASS.format(p=p), p, cell, U, E, "bypass"))
    return jobs


def run_job(job, sdir, env_extra):
    tag, extra, _, _, _, _, _ = job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log = os.path.join(sdir, f"point2_log_{tag}_{ts}.txt")
    cmd = [sys.executable, os.path.abspath("main_point2.py"), "--tag", tag] + extra.split()
    env = dict(os.environ, **env_extra)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=sdir)
    return tag, log, ret, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = build_jobs()
    sdir = os.path.join(ROOT, "s3_viability")
    if args.dry_run:
        print(f"total {len(jobs)}")
        for t, e, *_ in jobs[:3]:
            print(f"  [{t}] {e}")
        return
    os.makedirs(sdir, exist_ok=True)
    if args.resume:
        done = set()
        import glob as _g
        for f in _g.glob(os.path.join(sdir, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(f))
            if m:
                done.add(m.group(1))
        jobs = [j for j in jobs if j[0] not in done]
        print(f"resume: 剩余 {len(jobs)}", flush=True)
    env_extra = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"{len(jobs)} jobs -> {sdir} (workers={args.workers})", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, sdir, env_extra) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, log, ret, el = fut.result()
            n += 1
            if n % 10 == 0 or ret != 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {'OK' if ret == 0 else f'FAIL({ret})'} "
                      f"{tag} {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
