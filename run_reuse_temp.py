"""暂时断连场景下的复用实验 (边级成段离线, 会恢复).

背景: 逐用户 i.i.d. 掉线在本模拟器里精度中性 (聚合按 S_Q 重归一化 + 回归设备
重新同步), 所以"没有损失可回收"。要让**暂时**断连产生可回收损失, 必须让它有结构:
某条 edge 链路成段不可用 → 连续若干 checkpoint 缺失一整个数据区域 (强异质下不可替代)。

臂:
  none   无断连 (损伤参照)
  skip   断连期间该组不贡献
  bypass 断连期间该组本地继续训练并经旁路(当前 package token 新掩码)持续贡献
  skip_c 同 skip, 但两条 edge 并发离线 (缺席更集中)

设定: MNIST, 20 用户 (5ES×4U), K=3, VCM+σ0.5, --dropout-rate 0 (隔离归因),
      α ∈ {0.1, 0.5}, spans tick 口径, 6 seeds。

用法: python run_reuse_temp.py --workers 6 [--resume] [--dry-run]
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

ROOT = "results_reuse/s5_temporary"
SEEDS = [42, 43, 44, 45, 46, 47]
SPANS_STAGGER = "0:20-55,2:35-70"
SPANS_CONC = "0:20-55,1:20-55"
BASE = ("--dataset MNIST --max-checkpoints 20 --max-ticks 300 -K 3 --s-max 3 "
        "--noniid --alpha {a} --vcm --gamma-vcm 1.0 --staleness-sigma 0.5 "
        "--num-edge-servers 5 --users-per-edge 4 --dropout-rate 0 "
        "--delta-timeout 15")
BYPASS = "--dropout-mode offline_train --dropout-reuse rebase --reuse-rule none"
ARMS = [("none", ""), ("skip", f"--edge-offline-spans {SPANS_STAGGER}"),
        ("bypass", f"--edge-offline-spans {SPANS_STAGGER} {BYPASS}"),
        ("skipc", f"--edge-offline-spans {SPANS_CONC}"),
        ("bypassc", f"--edge-offline-spans {SPANS_CONC} {BYPASS}")]


def build_jobs():
    jobs = []
    for a in (0.1, 0.5):
        for arm, extra in ARMS:
            if arm in ("skipc", "bypassc") and a != 0.1:
                continue
            for s in SEEDS:
                jobs.append((f"t{a}_{arm}_s{s}",
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
        for t, e in jobs[:3]:
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
