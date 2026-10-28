"""三点创新实验矩阵批量运行器.

阶梯 (对应初稿三个创新点, 其余机制关闭, reuse=none, verify=off 性能模式):
  G1   FedAvg 同步基线          fedavg_point2.py --rounds 20
  G2   组间同步 (无机制)        main_point2.py -K 5 --s-max 3 --reuse-rule none
  G3   ①检查点式组级半异步      -K 3 --s-max 3 --reuse-rule none
  G3s  ①+σ 幂衰减加权          G3  + --staleness-sigma 0.5
  G3v  ①+Direct-VCM 端补偿     G3  + --vcm --gamma-vcm 1.0
  G5   ①+VCM+σ 完整协议        G3  + --vcm --gamma-vcm 1.0 --staleness-sigma 0.5

矩阵: {MNIST, FashionMNIST, CIFAR10} x {IID, NI(α=0.5)} x seeds {42,43,44}.

产物: point2_data_{ds}_{PART}_{arm}_s{seed}_*.npz / 同名 log,
汇总追加到 point2_matrix_summary.jsonl (断点续跑: 已有 npz 的任务自动跳过).

用法:
    python run_point_matrix.py                  # 全部 108 runs (4 并行)
    python run_point_matrix.py --workers 2
    python run_point_matrix.py --datasets CIFAR10
    python run_point_matrix.py --arms G1 G5
"""

import argparse
import concurrent.futures as cf
import glob
import json
import subprocess
import sys
import time
from datetime import datetime

SEEDS = [42, 43, 44]
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
PARTS = [("IID", []), ("NI", ["--noniid", "--alpha", "0.5"])]

# arm -> (脚本, 额外参数)
ARMS = {
    "G1":  ("fedavg_point2.py", ["--rounds", "20"]),
    "G2":  ("main_point2.py", ["-K", "5", "--s-max", "3", "--reuse-rule", "none"]),
    "G3":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none"]),
    "G3s": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--staleness-sigma", "0.5"]),
    "G3v": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--vcm", "--gamma-vcm", "1.0"]),
    "G5":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--vcm", "--gamma-vcm", "1.0",
                               "--staleness-sigma", "0.5"]),
    # 第三点试点: 矩校准 VCM (DVAFL TMC2026) 与朴素服务器动量 (arXiv:2402.09247)
    "G3vm1": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--vcm", "--gamma-vcm", "1.0",
                                 "--vcm-mode", "moment1"]),
    "G3vm1g": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--vcm", "--gamma-vcm", "0.8",
                                  "--vcm-mode", "moment1"]),
    # 归因实验: moment1 的收益来自比值放大(≈γ调大)还是逐包调制?
    "G3v15":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--vcm", "--gamma-vcm", "1.5",
                                  "--vcm-mode", "full"]),
    "G3vm1c": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--vcm", "--gamma-vcm", "1.0",
                                  "--vcm-mode", "moment1c"]),
    "G3vm1n": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--vcm", "--gamma-vcm", "1.0",
                                  "--vcm-mode", "moment1n"]),
    "G3v12":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--vcm", "--gamma-vcm", "1.2",
                                  "--vcm-mode", "full"]),
    # 聚合端 σ: 整数化权重进入掩码域的真加权安全聚合 (文献空白, LHH 整数同态)
    "G5a":   ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--staleness-sigma", "0.5",
                                 "--sigma-target", "agg"]),
    # 第三轮: 客户端陈旧度自适应 lr (经典 AsyncSGD 锚点) + FedLAW 聚合收缩
    "G3sl":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--staleness-lr-lambda", "0.3"]),
    "G3wd":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--agg-shrink", "0.995"]),
    # 第四轮: FADAS delay-adaptive lr 门控 (ICML 2024) + FedAdam (ICLR 2021)
    "G3gate": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--server-opt", "gate"]),
    "G3adam": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--server-opt", "adam"]),
    # 第五轮: checkpoint 选择策略 (①框架内子机制, 零实现成本)
    "G3ff":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--policy", "fresh_first"]),
    "G3fair": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--policy", "fair_fresh_first", "--D-fair", "2"]),
    # 第六轮: 客户端自剪裁 (arXiv:2606.13287, 对症 Fashion NI 发散)
    "G3clip": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--clip-upd-k", "2.5"]),
    # 第七轮: 突发离线下的 reuse 复活实验 (burst 造成 NI 分布代表性损失)
    "Bskip": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--dropout-rate", "0.15", "--dropout-burst", "5",
                                 "--dropout-mode", "skip"]),
    "Brec":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "ours_delta",
                                 "--dropout-rate", "0.15", "--dropout-burst", "5",
                                 "--dropout-mode", "offline_train",
                                 "--dropout-reuse", "rebase"]),
    # 第八轮: 服务器端聚合更新信赖域 (对症 Fashion NI 轨迹发散)
    "G3trc": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--server-opt", "trc", "--trc-k", "2.0"]),
    "G3vc":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--vcm", "--gamma-vcm", "1.0", "--vcm-mode", "clip"]),
    # 第九轮: 曲率感知 VCM (方向级修正, 对症 NI 轨迹转弯失准)
    "G3cur": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--vcm", "--gamma-vcm", "1.0", "--vcm-mode", "curve",
                                 "--vcm-curve-lambda", "0.5"]),
    # 极限探针: α=0.1 极端 Non-IID + burst=8 + rate=0.2 (饥饿最重场景)
    "BskipX": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                  "--dropout-rate", "0.2", "--dropout-burst", "8",
                                  "--dropout-mode", "skip", "--alpha", "0.1"]),
    "BrecX":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "ours_delta",
                                  "--dropout-rate", "0.2", "--dropout-burst", "8",
                                  "--dropout-mode", "offline_train",
                                  "--dropout-reuse", "rebase", "--alpha", "0.1"]),
    "G3vm2": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--vcm", "--gamma-vcm", "1.0",
                                 "--vcm-mode", "moment2"]),
    "G3vmo": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                                 "--vcm", "--gamma-vcm", "1.0",
                                 "--momentum", "0.6"]),
}

COMMON = ["--max-checkpoints", "20", "--max-ticks", "300"]
SUMMARY = "point2_matrix_summary.jsonl"


def build_jobs(args):
    parts = [p for p in PARTS if not args.parts or p[0] in args.parts]
    jobs = []
    for ds in args.datasets:
        for part, part_args in parts:
            for arm in args.arms:
                for seed in args.seeds:
                    tag = f"{ds}_{part}_{arm}_s{seed}"
                    if glob.glob(f"point2_data_{tag}_*.npz"):
                        continue  # 断点续跑: 已完成
                    script, arm_args = ARMS[arm]
                    # COMMON 仅适用于 main_point2.py (fedavg 用 --rounds 控制轮数)
                    common = COMMON if script == "main_point2.py" else []
                    cmd = [sys.executable, script,
                           "--dataset", ds, "--seed", str(seed),
                           "--tag", tag] + common + part_args + arm_args
                    jobs.append((tag, cmd))
    return jobs


def parse_npz(tag):
    import numpy as np
    hits = glob.glob(f"point2_data_{tag}_*.npz")
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True)

    def g(k, default=0.0):
        v = d.get(k)
        try:
            return float(v) if v is not None else default
        except Exception:
            return default

    rec = {"npz": hits[0], "final_acc": round(g("final_acc"), 2),
           "wall_clock_s": round(g("wall_clock"), 1)}
    if d.get("avg_wait") is not None:  # main_point2 产物
        ticks = d.get("checkpoint_ticks")
        rec.update({
            "avg_wait": round(g("avg_wait"), 2),
            "max_wait": round(g("max_wait"), 2),
            "final_tick": int(ticks[-1]) if ticks is not None and len(ticks) else None,
            "checkpoints": int(len(d.get("acc_history", []))),
            "comp_gain": round(g("comp_gain"), 4),
        })
        ema = d.get("final_acc_ema")
        if ema is not None and not np.isnan(ema):
            rec["final_acc_ema"] = round(float(ema), 2)
    else:  # fedavg 产物
        acc_h = d.get("acc_history")
        rec.update({"checkpoints": int(len(acc_h)) if acc_h is not None else 0,
                    "avg_wait": None, "final_tick": None})
    return rec


def run_job(job):
    tag, cmd = job
    t0 = time.time()
    out_path = f"point2_run_{tag}.out"
    try:
        with open(out_path, "w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=7200)
        rc = proc.returncode
    except Exception as e:  # noqa: BLE001
        rc = -1
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"\nRUNNER ERROR: {e!r}\n")
    rec = {"tag": tag, "rc": rc, "runner_wall_s": round(time.time() - t0, 1),
           "finished_at": datetime.now().strftime("%m-%d %H:%M:%S")}
    # tag 约定: {dataset}_{PART}_{arm}_s{seed}, 数据集名不含下划线
    try:
        ds, part, arm, s_seed = tag.split("_")
        rec.update({"dataset": ds, "part": part, "arm": arm,
                    "seed": int(s_seed[1:])})
    except ValueError:
        pass
    parsed = parse_npz(tag)
    if parsed:
        rec.update(parsed)
    with open(SUMMARY, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    status = (f"acc={rec.get('final_acc')}% wall={rec.get('wall_clock_s')}s"
              if parsed else f"NO-NPZ rc={rc}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE {tag}: {status}",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--parts", nargs="+", default=None,
                    help="过滤 partition (IID / NI), 默认全部")
    args = ap.parse_args()

    jobs = build_jobs(args)
    print(f"待运行 {len(jobs)} 个任务 (workers={args.workers})", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run_job, jobs))
    print(f"全部完成, 总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
