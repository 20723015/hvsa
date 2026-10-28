"""RA-CMA 分阶段实验驱动 (研究点 2 第三个创新).

统一底座: G3+VCM+σ0.5, K=3, s_max=3, 20 checkpoints, reuse=none。
每阶段独立结果目录 (results_cma/sN_*), 禁止混入旧 EMA 日志。
单线程确定性模式: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 (每进程)。

阶段:
  1 固定 EMA 基准重跑        raw + fixed β={0.4,0.5,0.6} × s42-44           (12)
  2 RA-CMA 初筛              STL/CNF/RISK25/50/75/RISK47 × s42-44           (18)
  3 八种子确认与消融          best-fixed + 2×risk + STL + CNF × s12,45-48   (25)
  4 多数据集泛化             3 数据集 × IID/Non-IID × s42-46 × 2 方法       (60)
  5 机制敏感性               α={0.1,1.0} + max_delay={3,8} × s42-46        (20)
  6 安全与开销               Tiny × verify on/off × 4 攻击 × raw/cma × s42-44 (60)

用法:
    python run_cma_stages.py --stage 1 --workers 4
    python run_cma_stages.py --stage 2 --workers 4
    python run_cma_stages.py --stage 3 --best-fix 0.5 --risk-a RISK50 --risk-b RISK75
    python run_cma_stages.py --stage 4 --best-fix 0.5 --risk-final RISK50
    python run_cma_stages.py --stage 5 --risk-final RISK50
    python run_cma_stages.py --stage 6 --risk-final RISK50 --workers 4
    python run_cma_stages.py --stage 3 --dry-run
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = "results_cma"
BASE = ("--dataset {ds} --max-checkpoints 20 --max-ticks 300 "
        "-K 3 --s-max 3 --reuse-rule none --vcm --gamma-vcm 1.0")
NI = "--noniid --alpha {alpha} --staleness-sigma 0.5"   # Non-IID: σ=0.5
IID = "--staleness-sigma 0"                             # IID: 主协议 σ=0

S1_SEEDS = [42, 43, 44]
S3_EXTRA_SEEDS = [12, 45, 46, 47, 48]
S4_SEEDS = [42, 43, 44, 45, 46]
S5_SEEDS = [42, 43, 44, 45, 46]
S6_SEEDS = [42, 43, 44]

FIX_BETAS = [0.4, 0.5, 0.6]
SCREEN = [  # (tag, extra)
    ("STL",    "--cma-mode stale"),
    ("CNF",    "--cma-mode conflict"),
    ("RISK25", "--cma-mode risk --cma-lambda-stale 0.25"),
    ("RISK50", "--cma-mode risk --cma-lambda-stale 0.5"),
    ("RISK75", "--cma-mode risk --cma-lambda-stale 0.75"),
    ("RISK47", "--cma-mode risk --cma-lambda-stale 0.5 --cma-beta-min 0.4 --cma-beta-max 0.7"),
]
ATTACKS = [("TG", "tamper_global"), ("TM", "tamper_manifest"),
           ("DR", "drop_replace_pkg"), ("RP", "replay")]


def stage_dir(stage):
    d = {1: "s1_fixed_ema", 2: "s2_screen", 3: "s3_confirm", 4: "s4_generalize",
         5: "s5_sensitivity", 6: "s6_security"}[stage]
    return os.path.join(ROOT, d)


def build_jobs(stage, args):
    jobs = []
    if stage == 1:
        base = f"{BASE.format(ds='MNIST')} {NI.format(alpha=0.5)}"
        for s in S1_SEEDS:
            jobs.append((f"s1_RAW_s{s}", f"{base} --seed {s}"))
        for b in FIX_BETAS:
            for s in S1_SEEDS:
                jobs.append((f"s1_FIX{b:g}_s{s}",
                             f"{base} --seed {s} --cma-mode fixed --cma-beta {b}"))
    elif stage == 2:
        base = f"{BASE.format(ds='MNIST')} {NI.format(alpha=0.5)}"
        for tag, extra in SCREEN:
            for s in S1_SEEDS:
                jobs.append((f"s2_{tag}_s{s}", f"{base} --seed {s} {extra}"))
    elif stage == 3:
        if args.best_fix is None or not args.risk_a or not args.risk_b:
            raise SystemExit("stage 3 需要 --best-fix β --risk-a TAG --risk-b TAG "
                             "(由阶段 2 分析 top2 决定)")
        base = f"{BASE.format(ds='MNIST')} {NI.format(alpha=0.5)}"
        rmap = {t: e for t, e in SCREEN}
        for tag, extra in [("FIX", f"--cma-mode fixed --cma-beta {args.best_fix}"),
                           ("STL", rmap["STL"]), ("CNF", rmap["CNF"]),
                           (args.risk_a, rmap[args.risk_a]),
                           (args.risk_b, rmap[args.risk_b])]:
            for s in S3_EXTRA_SEEDS:
                jobs.append((f"s3_{tag}_s{s}", f"{base} --seed {s} {extra}"))
    elif stage == 4:
        if args.best_fix is None or not args.risk_final:
            raise SystemExit("stage 4 需要 --best-fix β --risk-final TAG")
        rmap = {t: e for t, e in SCREEN}
        for ds in ("MNIST", "FashionMNIST", "CIFAR10"):
            for ni, dist_args in ((False, IID), (True, NI.format(alpha=0.5))):
                base = f"{BASE.format(ds=ds)} {dist_args}"
                dist = "NI" if ni else "IID"
                for s in S4_SEEDS:
                    jobs.append((f"s4_{ds}_{dist}_FIX_s{s}",
                                 f"{base} --seed {s} --cma-mode fixed --cma-beta {args.best_fix}"))
                    jobs.append((f"s4_{ds}_{dist}_{args.risk_final}_s{s}",
                                 f"{base} --seed {s} {rmap[args.risk_final]}"))
    elif stage == 5:
        if not args.risk_final:
            raise SystemExit("stage 5 需要 --risk-final TAG")
        rmap = {t: e for t, e in SCREEN}
        extra = rmap[args.risk_final]
        for alpha in (0.1, 1.0):
            base = f"{BASE.format(ds='MNIST')} {NI.format(alpha=alpha)}"
            for s in S5_SEEDS:
                jobs.append((f"s5_A{alpha:g}_s{s}", f"{base} --seed {s} {extra}"))
        for delay in (3, 8):
            base = (f"{BASE.format(ds='MNIST')} {NI.format(alpha=0.5)} "
                    f"--max-delay {delay}")
            for s in S5_SEEDS:
                jobs.append((f"s5_D{delay}_s{s}", f"{base} --seed {s} {extra}"))
    elif stage == 6:
        if not args.risk_final:
            raise SystemExit("stage 6 需要 --risk-final TAG")
        rmap = {t: e for t, e in SCREEN}
        cma = rmap[args.risk_final]
        sec = f"{BASE.format(ds='MNIST')} {NI.format(alpha=0.5)} --model Tiny"
        for s in S6_SEEDS:
            jobs.append((f"s6_NONE_n_raw_s{s}", f"{sec} --seed {s}"))
            jobs.append((f"s6_NONE_n_cma_s{s}", f"{sec} --seed {s} {cma}"))
            jobs.append((f"s6_NONE_v_raw_s{s}", f"{sec} --seed {s} --verify"))
            jobs.append((f"s6_NONE_v_cma_s{s}", f"{sec} --seed {s} --verify {cma}"))
        for pre, atk in ATTACKS:
            for s in S6_SEEDS:
                # 仅 verify-on: main_point2 fail-fast 拒绝 --attack 无 --verify
                # (无验证的攻击无安全语义, 旧 verify-off 变体任务已移除)
                for v, verify, ticks in (("v", "--verify",
                                          60 if pre in ("TG", "TM", "DR") else 300),):
                    for m, extra in (("raw", ""), ("cma", cma)):
                        jobs.append((f"s6_{pre}{v}_{m}_s{s}",
                                     f"{sec} --seed {s} {verify} --attack {atk} "
                                     f"--max-ticks {ticks} {extra}".strip()))
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
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--best-fix", type=float, default=None)
    ap.add_argument("--risk-a", default=None)
    ap.add_argument("--risk-b", default=None)
    ap.add_argument("--risk-final", default=None)
    args = ap.parse_args()

    jobs = build_jobs(args.stage, args)
    sdir = stage_dir(args.stage)
    if args.dry_run:
        for tag, extra in jobs:
            print(f"[{tag}] python main_point2.py {extra}")
        print(f"stage {args.stage}: total {len(jobs)}")
        return

    os.makedirs(sdir, exist_ok=True)
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
            print(f"[{done}/{len(jobs)}] {status} {tag}  {elapsed:.0f}s -> {os.path.basename(log_path)}",
                  flush=True)


if __name__ == "__main__":
    main()