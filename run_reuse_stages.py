"""P1: 小联邦(冗余不足)下 stale reuse 的决定性实验.

假设 (见 docs/REUSE_DIRECTION_NOTES.md §P1): 复用收益 ≈ f(冗余度)。
20 用户下丢 20% 上传精度中性 (其余用户冗余覆盖) → 没有损失可回收, 只剩偏置成本。
本实验把联邦缩到 4~6 用户 (每个数据区域仅 1~3 个用户), 检验复用是否开始有正收益。

网格 (MNIST, VCM+σ0.5 底座, 20 checkpoints, 同 seed 配对):
  A t6_burst   : 2ES×3U=6 用户, K=2, α=0.5, 突发离线 rate=0.15 B=5
                 臂 = skip / bypass / bypass_mix
  B t6_perm    : 2ES×3U=6 用户, K=2, α=0.5, 永久离线 rate=0.17 (burst=999)
                 臂 = skip / bypass
  C t4_perm    : 2ES×2U=4 用户, K=2, α=0.5, 永久离线 rate=0.25 (单组饥饿→delta_timeout)
                 臂 = skip / bypass
  D t6_perm_a01: 2ES×3U=6 用户, K=2, α=0.1 (近不相交), 永久离线 rate=0.17
                 臂 = skip / bypass

臂语义:
  skip        : 掉线即跳过 (无回收)
  bypass      : offline_train + dropout-reuse rebase, 但 reuse-rule=none
                (离线算力经旁路回收, 不把 stale 混入新鲜上传)
  bypass_mix  : offline_train + dropout-reuse rebase + reuse-rule=ours_delta
                (既旁路回收, 又对返回用户的 fresh 上传做无锚点 stale 混合)

用法:
  python run_reuse_stages.py --workers 6
  python run_reuse_stages.py --resume
  python run_reuse_stages.py --dry-run
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
SEEDS = [42, 43, 44]
BASE = ("--dataset MNIST --max-checkpoints 20 --max-ticks 400 -K 2 --s-max 3 "
        "--noniid --alpha {alpha} --vcm --gamma-vcm 1.0 --staleness-sigma 0.5 "
        "--num-edge-servers 2 --users-per-edge {u} --delta-timeout 15")
SKIP = "--dropout-rate {rate} --dropout-burst {burst}"
BYPASS = ("--dropout-rate {rate} --dropout-burst {burst} "
          "--dropout-mode offline_train --dropout-reuse rebase --reuse-rule none")
BYPASS_MIX = ("--dropout-rate {rate} --dropout-burst {burst} "
              "--dropout-mode offline_train --dropout-reuse rebase "
              "--reuse-rule ours_delta --rho-max 0.5")


def build_jobs():
    jobs = []
    cells = [
        # (tag, alpha, users_per_edge, rate, burst, arms)
        ("t6_burst",    0.5, 3, 0.15, 5,   ["skip", "bypass", "bypass_mix"]),
        ("t6_perm",     0.5, 3, 0.17, 999, ["skip", "bypass"]),
        ("t4_perm",     0.5, 2, 0.25, 999, ["skip", "bypass"]),
        ("t6_perm_a01", 0.1, 3, 0.17, 999, ["skip", "bypass"]),
    ]
    for cell, alpha, u, rate, burst, arms in cells:
        base = BASE.format(alpha=alpha, u=u)
        for arm in arms:
            extra = {"skip": SKIP, "bypass": BYPASS, "bypass_mix": BYPASS_MIX}[arm]
            extra = extra.format(rate=rate, burst=burst)
            for s in SEEDS:
                jobs.append((f"{cell}_{arm}_s{s}", f"{base} --seed {s} {extra}"))
    return jobs


def run_job(job, sdir, env_extra):
    tag, extra = job
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
    sdir = os.path.join(ROOT, "s2_tiny_fed")
    if args.dry_run:
        for tag, extra in jobs:
            print(f"[{tag}] python main_point2.py {extra}")
        print(f"total {len(jobs)}")
        return
    os.makedirs(sdir, exist_ok=True)
    if args.resume:
        done = set()
        import glob as _g
        for p in _g.glob(os.path.join(sdir, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(p))
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
            print(f"[{n}/{len(jobs)}] {'OK' if ret == 0 else f'FAIL({ret})'} "
                  f"{tag} {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
