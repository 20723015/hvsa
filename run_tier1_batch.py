"""Tier-1 机制批量实验运行器 (服务器 EMA / FedProx / RA-CMA 模块筛选).

基线族: G3+VCM+σ0.5 (noniid α=0.5, K=3, s_max=3, reuse=none, 其余默认)。
EMA 为纯后处理: raw 轨迹应与基线逐值一致 (自动兼作确定性校验)。
RA-CMA (第三个创新) 的模块筛选任务: stale-only / conflict-only / 完整 risk
(λs={0.25,0.5,0.75}) 与范围敏感性 (β∈[0.4,0.7], λs=0.5), 种子 42-44。
fixed 模式正确性回归由既有 EMA 任务覆盖 (--ema-decay 现已映射到修正后
fixed 影子平均, 无偏差校正; 日志打印 "最终准确率(影子 fixed)").

用法:
    python run_tier1_batch.py                 # 全部任务 (默认 4 并行)
    python run_tier1_batch.py --dry-run       # 仅打印任务
    python run_tier1_batch.py --only ema0.8   # 只跑某一组
    python run_tier1_batch.py --only risk50   # 只跑 RA-CMA 筛选组
    python run_tier1_batch.py --workers 2
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from datetime import datetime

SEEDS = [12, 42, 43, 44, 45, 46, 47, 48]
BASE_ARGS = ("--dataset MNIST --max-checkpoints 20 --max-ticks 300 "
             "-K 3 --s-max 3 --noniid --alpha 0.5 --reuse-rule none "
             "--vcm --gamma-vcm 1.0 --staleness-sigma 0.5")

EMA_DECAYS = [0.4, 0.5, 0.6, 0.8, 0.9]
FEDPROX_MUS = [0.01, 0.1]
# Tier-2 (2025 新论文模块): FedBSS 课程化 (AAAI 2025) + DPR 参数重置 (NeurIPS 2025 适配)
BSS_CURRICULA = [('BSSasc', 'bias_asc'), ('BSSdesc', 'bias_desc')]
DPR_QS = [0.9, 0.75]
# Tier-4: RA-CMA (第三个创新) 模块筛选 —— 设计路线第 2 步。
# 默认 β∈[0.3,0.65]; RISK47 = 范围敏感性 (β∈[0.4,0.7], λs=0.5)。
# 晋级条件: 相对 raw ≥0.8pp 平均提升、≥2/3 种子为正、不低于最佳固定 EMA 0.1pp。
CMA_SCREEN_SEEDS = [42, 43, 44]
CMA_SCREEN = [
    ("STL",    "--cma-mode stale"),
    ("CNF",    "--cma-mode conflict"),
    ("RISK25", "--cma-mode risk --cma-lambda-stale 0.25"),
    ("RISK50", "--cma-mode risk --cma-lambda-stale 0.5"),
    ("RISK75", "--cma-mode risk --cma-lambda-stale 0.75"),
    ("RISK47", "--cma-mode risk --cma-lambda-stale 0.5 --cma-beta-min 0.4 --cma-beta-max 0.7"),
]


def build_jobs():
    jobs = []  # (tag, extra_args)
    for a in EMA_DECAYS:
        tag = f"EMA{a}"
        for s in SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} --ema-decay {a}"))
    for m in FEDPROX_MUS:
        tag = f"PX{m}"
        for s in SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} --fedprox-mu {m}"))
    for tag, cur in BSS_CURRICULA:
        for s in SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} --curriculum {cur}"))
    for q in DPR_QS:
        tag = f"DPR{q}"
        for s in SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} --dpr-q {q}"))
    # Tier-3: σ 陈旧度度量变体 (arXiv:2603.08211 / FedBuff-fair-weighting)
    for tag, extra in [("REL2_5", "--staleness-metric rel2 --staleness-sigma 5"),
                       ("REL2_15", "--staleness-metric rel2 --staleness-sigma 15"),
                       ("WAIT0.5", "--staleness-metric wait --staleness-sigma 0.5"),
                       ("WAIT1.0", "--staleness-metric wait --staleness-sigma 1.0")]:
        for s in SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} {extra}"))
    # Tier-4: RA-CMA 模块筛选 (设计路线第 2 步, 种子 42-44)
    for tag, extra in CMA_SCREEN:
        for s in CMA_SCREEN_SEEDS:
            jobs.append((f"{tag}_s{s}", f"{BASE_ARGS} --seed {s} {extra}"))
    # 安全线: 全密码学 (LHH+Ed25519) 端到端, Tiny 模型 (dim=7850) 使 verify 可承受
    SEC_BASE = f"{BASE_ARGS} --model Tiny"
    ATTK = {"TG": "tamper_global", "TM": "tamper_manifest",
            "DR": "drop_replace_pkg", "RP": "replay"}
    for s in SEEDS:
        jobs.append((f"TINY_s{s}", f"{SEC_BASE} --seed {s}"))
        jobs.append((f"VF_s{s}", f"{SEC_BASE} --seed {s} --verify"))
    for pre, atk in ATTK.items():
        for s in SEEDS:  # verify-on 才能承载攻击语义 (fail-closed 死锁类攻击限 60 ticks)
            ticks = 60 if pre in ("TG", "TM", "DR") else 300
            jobs.append((f"{pre}v_s{s}",
                         f"{SEC_BASE} --seed {s} --verify --attack {atk} --max-ticks {ticks}"))
        # verify-off 的攻击任务已被 main_point2 的 fail-fast 拒绝
        # ("--attack requires --verify"): 无验证的攻击无安全语义, 故移除。
    return jobs


def build_command(tag, extra):
    """构造唯一标记产物的子进程命令。"""
    return [sys.executable, "main_point2.py"] + extra.split() + ["--tag", tag]


def run_job(job, workers_env):
    tag, extra = job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"point2_log_{tag}_MNIST_{ts}.txt"
    # tag 同时进入 main_point2，使其内部日志、NPZ 和 PNG 均按任务隔离；
    # 否则并行任务在同一秒启动时会写入相同文件名。
    cmd = build_command(tag, extra)
    env = dict(os.environ, **workers_env)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        ret = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return tag, log_path, ret, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="仅跑 tag 前缀匹配的任务, 如 ema0.8 / px0.01")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.only:
        key = args.only.lower().replace("ema", "EMA").replace("px", "PX")
        jobs = [j for j in jobs if j[0].lower().startswith(args.only.lower())]

    if args.dry_run:
        for tag, extra in jobs:
            cmd = build_command(tag, extra)
            print(f"[{tag}] " + " ".join(cmd))
        print(f"total: {len(jobs)}")
        return

    # 单线程 = 比特级确定性 (多线程浮点非确定性会被 20 轮训练混沌放大,
    # 实测 OMP=1 可逐值复现 08-28/08-29 基线), 跨批次配对对比的前提。
    workers_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    print(f"{len(jobs)} jobs, workers={args.workers}, threads/worker=1 (确定性模式)")

    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j, workers_env) for j in jobs]
        for fut in cf.as_completed(futs):
            tag, log_path, ret, elapsed = fut.result()
            done += 1
            status = "OK" if ret == 0 else f"FAIL({ret})"
            print(f"[{done}/{len(jobs)}] {status} {tag}  {elapsed:.0f}s  -> {log_path}",
                  flush=True)


if __name__ == "__main__":
    main()
