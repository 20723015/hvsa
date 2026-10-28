"""Tier-2 结果配对分析: 2025 新论文模块 (FedBSS 课程化 / DPR 参数重置).

基线 = SIG_NI_0.5 (s12/42/43/44/45) + SIG2_NI_0.5 (s46/47/48)。
两机制均不新增 DataLoader 迭代抽取 (RNG 流与基线逐值一致) → 跨批次配对精确。

用法: python analyze_tier2.py
"""

import glob
import re
import statistics as st

SEEDS = [12, 42, 43, 44, 45, 46, 47, 48]
SERIES = ["BSSasc", "BSSdesc", "DPR0.9", "DPR0.75",
          "REL2_5", "REL2_15", "WAIT0.5", "WAIT1.0"]


def find_log(prefix):
    hits = sorted(glob.glob(f"point2_log_{prefix}_MNIST_*.txt"))
    return hits[-1] if hits else None


def parse_log(path):
    try:
        txt = open(path, encoding="utf-8", errors="strict").read()
    except UnicodeDecodeError:
        txt = open(path, encoding="gbk", errors="replace").read()
    hist = eval(re.search(r"Accuracy history: (\[.*?\])", txt, re.S).group(1))
    acc = float(re.search(r"最终准确率: ([0-9.]+)%", txt).group(1))
    return acc, [b for _, b in hist]


def main():
    base_acc, base_hist = {}, {}
    for s in SEEDS:
        p = (find_log(f"SIG_NI_0.5_s{s}") or find_log(f"SIG2_NI_0.5_s{s}"))
        if not p:
            raise FileNotFoundError(f"no baseline log for seed {s}")
        base_acc[s], base_hist[s] = parse_log(p)

    for tag in SERIES:
        rows = {}
        for s in SEEDS:
            p = find_log(f"{tag}_s{s}")
            if not p:
                print(f"[缺日志] {tag}_s{s}")
                continue
            rows[s] = parse_log(p)[0]
        diffs = [rows[s] - base_acc[s] for s in SEEDS if s in rows]
        npos = sum(1 for d in diffs if d > 0)
        print(f"\n== {tag} vs 基线 (精确配对) ==")
        print("seed     " + "".join(f"{s:>7}" for s in SEEDS if s in rows))
        print("baseline " + "".join(f"{base_acc[s]:7.2f}" for s in SEEDS if s in rows))
        print(f"{tag:8s} " + "".join(f"{rows[s]:7.2f}" for s in SEEDS if s in rows))
        print("paired Δ " + "".join(f"{d:+7.2f}" for d in diffs))
        vals = [rows[s] for s in SEEDS if s in rows]
        print(f"mean: {st.mean(vals):.2f}±{st.stdev(vals):.2f}  "
              f"Δmean: {st.mean(diffs):+.2f}±{st.stdev(diffs):.2f}  "
              f"正/负: {npos}/{len(diffs)-npos}")


if __name__ == "__main__":
    main()
