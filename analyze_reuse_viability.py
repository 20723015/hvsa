"""复用可用性边界分析: 实证退化率 vs 解析预测 P_viable.

用法: python analyze_reuse_viability.py --dir results_reuse/s3_viability
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

N_MIN = 2


def p_seal(U, p):
    """单个 edge 能封组 (在线≥n_min) 的概率: 在线数 ~ Binomial(U, 1−p)."""
    q = 1.0 - p
    p0 = p ** U
    p1 = U * q * (p ** (U - 1)) if U >= 1 else 0.0
    return max(0.0, 1.0 - p0 - p1)


def p_viable(E, U, p):
    """E 个 edge 至少一个能封组 (delta_timeout 保守 checkpoint) 的概率."""
    return 1.0 - (1.0 - p_seal(U, p)) ** E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_reuse/s3_viability")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "point2_data_*.npz"))):
        m = re.match(r"point2_data_(\w+?)_p(\d+?)(_byp)?_(\d+)_\d{8}-\d{6}\.npz$",
                     os.path.basename(f))
        if not m:
            continue
        cell, pr, byp, seed = m.group(1), int(m.group(2)) / 100.0, bool(m.group(3)), int(m.group(4))
        z = np.load(f, allow_pickle=True)
        acc = np.asarray(z["acc_history"], dtype=float)
        nqs = np.asarray(z["checkpoint_Ns"], dtype=float) if len(z["checkpoint_Ns"]) else np.array([])
        rows.append(dict(
            cell=cell, p=pr, arm="bypass" if byp else "skip", seed=seed,
            ckps=len(acc), mean_nq=float(nqs.mean()) if nqs.size else 0.0,
            min_nq=float(nqs.min()) if nqs.size else 0.0,
            n_invalid=int((nqs < N_MIN).sum()) if nqs.size else 0,
            final=float(z["final_acc"]),
            # 退化 = 一个 checkpoint 都提交不了 (联邦训练不起来)
            degenerate=(len(acc) == 0),
        ))

    print(f"{'cell':<6}{'p':>6}{'arm':<8}{'n':>4}{'退化率':>9}{'均值N_Q':>9}"
          f"{'最小N_Q':>9}{'终值':>9}{'解析退化率':>11}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["p"], r["arm"])].append(r)

    out = {}
    for (cell, p, arm), rs in sorted(groups.items()):
        U = {"v4": 2, "v6": 3, "v8": 4, "v12": 4, "v20": 4}[cell]
        E = {"v4": 2, "v6": 2, "v8": 2, "v12": 3, "v20": 5}[cell]
        deg = np.mean([r["degenerate"] for r in rs])
        pred = 1.0 - p_viable(E, U, p)
        nq = np.mean([r["mean_nq"] for r in rs])
        fin = np.mean([r["final"] for r in rs if not r["degenerate"]]) if any(
            not r["degenerate"] for r in rs) else float("nan")
        print(f"{cell:<6}{p:>6.2f}{arm:<8}{len(rs):>4}{deg:>9.1%}{nq:>9.2f}"
              f"{min(r['min_nq'] for r in rs):>9.1f}{fin:>9.2f}{pred:>11.1%}")
        out[f"{cell}|p{p}|{arm}"] = dict(n=len(rs), degen_rate=float(deg),
                                         pred_degen=float(pred), mean_nq=float(nq),
                                         mean_final=float(fin) if not math.isnan(fin) else None)

    # 汇总: 退化率 vs (U, p) 与解析对照
    print("\n== 实证 vs 解析 (skip 臂, 每 edge 用户数 U × 永久掉线率 p) ==")
    print(f"{'U':>3}{'E':>3}{'p':>6}{'实证退化率':>11}{'解析退化率':>11}{'差':>8}")
    for cell, E, U in (("v4", 2, 2), ("v6", 2, 3), ("v8", 2, 4)):
        for p in (0.15, 0.25, 0.40):
            rs = groups.get((cell, p, "skip"), [])
            if not rs:
                continue
            emp = float(np.mean([r["degenerate"] for r in rs]))
            pred = 1.0 - p_viable(E, U, p)
            print(f"{U:>3}{E:>3}{p:>6.2f}{emp:>11.1%}{pred:>11.1%}{emp-pred:>+8.1%}")

    # bypass 恢复效果 (退化格子)
    print("\n== bypass 对退化格子的恢复 ==")
    for (cell, p) in sorted({(r["cell"], r["p"]) for r in rows if r["arm"] == "bypass"}):
        sk = groups.get((cell, p, "skip"), [])
        by = groups.get((cell, p, "bypass"), [])
        if not sk or not by:
            continue
        d_sk = float(np.mean([r["degenerate"] for r in sk]))
        d_by = float(np.mean([r["degenerate"] for r in by]))
        f_sk = float(np.mean([r["final"] for r in sk]))
        f_by = float(np.mean([r["final"] for r in by]))
        print(f"  {cell} p={p:.2f}: 退化率 {d_sk:.0%} -> {d_by:.0%};  "
              f"终值 {f_sk:.1f} -> {f_by:.1f} (Δ{f_by-f_sk:+.1f});  "
              f"N_Q {np.mean([r['mean_nq'] for r in sk]):.1f} -> {np.mean([r['mean_nq'] for r in by]):.1f}")

    outp = os.path.join(args.dir, "reuse_viability_analysis.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n导出: {outp}")


if __name__ == "__main__":
    main()
