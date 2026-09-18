"""Aσ-VCM 独立复核批次 (单一代码版本, 消除跨批次基线不一致).

发现: 文档主表的 CIFAR 格使用 Sep-8 基线, 与 Sep-12 策略臂在同配置下轨迹不同
(疑为旧代码版本); 且 MNIST/Fashion IID 的 42-44 用的是 BOOST(强制放大)消融臂,
不是三档策略真身。本批次用当前代码补齐:

 A. 当前代码 CIFAR10 G5 基线 (σ=0.5), IID/NI × seeds 42-46   -> 10 runs
 B. 当前代码三档策略 (--sigma-policy alignment, 默认阈值) 在
    MNIST/FashionMNIST IID × seeds 42-44                     -> 6 runs

用法: python run_asig_verify.py --workers 8 [--resume] [--dry-run]
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

ROOT = "results_asig_verify"
COMMON = ("--vcm --gamma-vcm 1.0 --staleness-sigma 0.5 --reuse-rule none "
          "--num-edge-servers 5 --users-per-edge 4 -K 3 --s-max 3 "
          "--max-checkpoints 20 --max-ticks 300")


def build_jobs(cli=None):
    jobs = []
    for dist, extra in (("IID", ""), ("NI", "--noniid")):
        for s in range(42, 47):
            jobs.append((f"A_CIFAR10_{dist}_G5_s{s}",
                         f"--dataset CIFAR10 {extra} {COMMON} --seed {s}"))
    for ds in ("MNIST", "FashionMNIST"):
        for s in (42, 43, 44):
            jobs.append((f"B_{ds}_IID_ALIGN_s{s}",
                         f"--dataset {ds} {COMMON} --sigma-policy alignment --seed {s}"))
    # 单批次全量复核: 6 格 × 5 种子 × {G5, ALIGN}, 标签与 A_* 复用同一命名
    for ds in ("MNIST", "FashionMNIST", "CIFAR10"):
        for dist, extra in (("IID", ""), ("NI", "--noniid")):
            for s in range(42, 47):
                jobs.append((f"A_{ds}_{dist}_G5_s{s}",
                             f"--dataset {ds} {extra} {COMMON} --seed {s}"))
                jobs.append((f"A_{ds}_{dist}_ALIGN_s{s}",
                             f"--dataset {ds} {extra} {COMMON} "
                             f"--sigma-policy alignment --seed {s}"))
    # CIFAR 确定性重跑 (GPU 非确定性已被 --deterministic 消除)
    if cli is not None and getattr(cli, 'cifar_det', False):
        jobs = []
        for dist, extra in (("IID", ""), ("NI", "--noniid")):
            for s in range(42, 47):
                jobs.append((f"DET_CIFAR10_{dist}_G5_s{s}",
                             f"--dataset CIFAR10 {extra} {COMMON} --deterministic --seed {s}"))
                jobs.append((f"DET_CIFAR10_{dist}_ALIGN_s{s}",
                             f"--dataset CIFAR10 {extra} {COMMON} --deterministic "
                             f"--sigma-policy alignment --seed {s}"))
        return jobs
    # CIFAR 基线已由首批完成 -> 去重
    seen = set()
    uniq = []
    for tag, e in jobs:
        if tag in seen:
            continue
        seen.add(tag)
        uniq.append((tag, e))
    return uniq


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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cifar-det", action="store_true",
                    help="仅跑 CIFAR 确定性复核 (20 runs)")
    a = ap.parse_args()
    jobs = build_jobs(a)
    if a.dry_run:
        print(f"total {len(jobs)}")
        for t, e in jobs:
            print(f"  [{t}] {e}")
        return
    os.makedirs(ROOT, exist_ok=True)
    if a.resume:
        done = set()
        for p in g.glob(os.path.join(ROOT, "point2_data_*.npz")):
            m = re.search(r"point2_data_(.+?)_\d{8}-\d{6}\.npz$", os.path.basename(p))
            if m:
                done.add(m.group(1))
        jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} jobs -> {ROOT} (workers={a.workers})", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, ret, el = fut.result()
            n += 1
            print(f"[{n}/{len(jobs)}] {'OK' if ret == 0 else f'FAIL({ret})'} {tag} {el:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
