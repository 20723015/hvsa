"""V-PDA 分阶段实验驱动 (研究点 2 新创新点).

阶段 2: 三种子快速筛选 (MNIST Non-IID α=0.5, seeds 42-44, 20 checkpoints).

实验组 (12 runs):
  G5      VCM+σ0.5, 无 V-PDA (当前最佳基线)
  PDA05   VCM+σ0.5 + V-PDA κ=0.5
  PDA10   VCM+σ0.5 + V-PDA κ=1.0
  PDA-only 无 VCM/σ + V-PDA κ=0.5

统一确定性: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, 单线程。

用法:
    python run_vpda_stages.py --stage 2 --workers 4
    python run_vpda_stages.py --stage 2 --dry-run
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = "results_vpda"
BASE = ("--dataset MNIST --max-checkpoints 20 --max-ticks 300 "
        "-K 3 --s-max 3 --noniid --alpha 0.5 --reuse-rule none "
        "--vcm --gamma-vcm 1.0 --staleness-sigma 0.5")
BASE_NO_VCM = ("--dataset MNIST --max-checkpoints 20 --max-ticks 300 "
               "-K 3 --s-max 3 --noniid --alpha 0.5 --reuse-rule none "
               "--staleness-sigma 0")

S2_SEEDS = [42, 43, 44]
S2_GROUPS = [
    ("G5",      BASE,        ""),
    ("PDA05",   BASE,        "--participation-debias ipw --debias-kappa 0.5"),
    ("PDA10",   BASE,        "--participation-debias ipw --debias-kappa 1.0"),
    ("PDAONLY", BASE_NO_VCM, "--participation-debias ipw --debias-kappa 0.5"),
]


def build_jobs(stage, args):
    jobs = []
    if stage == 2:
        for tag, base, extra in S2_GROUPS:
            for s in S2_SEEDS:
                jobs.append((f"s2_{tag}_s{s}",
                             f"{base} --seed {s} {extra}".strip()))
    elif stage == 3:
        # 修正证据链后的重跑 (全组均值权重 + 累计 TV/权重质量), 独立目录
        for tag, base, extra in S2_GROUPS:
            for s in S2_SEEDS:
                jobs.append((f"s3_{tag}_s{s}",
                             f"{base} --seed {s} {extra}".strip()))
    else:
        raise SystemExit(f"stage {stage} 未定义 (当前可用: 2, 3)")
    return jobs


def run_job(job, sdir, workers_env):
    tag, extra = job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(sdir, f"point2_log_{tag}_MNIST_{ts}.txt")
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
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑: 跳过该阶段已有 npz 的任务 (按 tag 匹配)")
    args = ap.parse_args()

    jobs = build_jobs(args.stage, args)
    sdir = os.path.join(ROOT, f"s{args.stage}_screen" if args.stage == 2
                        else "s2b_corrected")
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