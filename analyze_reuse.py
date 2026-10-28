"""P1 小联邦复用实验分析器: 同 cell 同 seed 与 skip 臂配对.

用法: python analyze_reuse.py --dir results_reuse/s2_tiny_fed
"""

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np


def load(dirpath):
    rows = []
    for p in sorted(glob.glob(os.path.join(dirpath, "point2_data_*.npz"))):
        fn = os.path.basename(p)
        m = re.match(r"point2_data_(.+)_(\d{8}-\d{6})\.npz$", fn)
        if not m:
            continue
        tag = m.group(1)
        # tag = {cell}_{arm}_s{seed}; cell 自身含下划线
        mm = re.match(r"(.+)_(bypass_mix|bypass|skip)_s(\d+)$", tag)
        if not mm:
            continue
        cell, arm, seed = mm.group(1), mm.group(2), int(mm.group(3))
        z = np.load(p, allow_pickle=True)
        acc = np.asarray(z["acc_history"], dtype=float)
        rows.append(dict(
            cell=cell, arm=arm, seed=seed, tag=tag,
            final=float(z["final_acc"]),
            last5=float(np.mean(acc[-5:, 1])) if len(acc) >= 5 else float("nan"),
            ckps=len(acc),
            ticks=int(z["checkpoint_ticks"][-1]) if len(z["checkpoint_ticks"]) else -1,
            stale_reuse=int(z["total_stale_reuses"]) if "total_stale_reuses" in z.files else 0,
            stale_in_ckp=int(z["stale_in_checkpoint"]) if "stale_in_checkpoint" in z.files else 0,
            nq=float(np.mean(z["checkpoint_Ns"])) if len(z["checkpoint_Ns"]) else 0.0,
            sq=float(np.mean(z["checkpoint_Ss"])) if len(z["checkpoint_Ss"]) else 0.0,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_reuse/s2_tiny_fed")
    args = ap.parse_args()
    rows = load(args.dir)

    by = defaultdict(dict)
    for r in rows:
        by[(r["cell"], r["seed"])][r["arm"]] = r

    print(f"{'cell':<14}{'seed':>5}{'arm':<12}{'acc':>8}{'last5':>8}{'Δskip':>8}"
          f"{'recov':>7}{'staleCkp':>9}{'ticks':>6}{'N_Q':>6}")
    cells = sorted({r["cell"] for r in rows})
    pairs = defaultdict(list)
    for cell in cells:
        for seed in sorted({k[1] for k in by if k[0] == cell}):
            arms = by[(cell, seed)]
            base = arms.get("skip")
            for arm in ("skip", "bypass", "bypass_mix"):
                r = arms.get(arm)
                if r is None:
                    continue
                d = (r["final"] - base["final"]) if base else float("nan")
                if arm != "skip" and base:
                    pairs[(cell, arm)].append((seed, d, r["last5"] - base["last5"]))
                print(f"{cell:<14}{seed:>5}{arm:<12}{r['final']:>8.2f}"
                      f"{r['last5']:>8.2f}"
                      f"{(f'{d:+.2f}' if not math.isnan(d) and arm != 'skip' else '—'):>8}"
                      f"{r['stale_reuse']:>7}{r['stale_in_ckp']:>9}{r['ticks']:>6}"
                      f"{r['nq']:>6.1f}")

    print("\n== 同 seed 配对 (相对 skip 臂) ==")
    print(f"{'cell':<14}{'arm':<12}{'Δmean':>9}{'±std':>8}{'正/负':>8}{'last5Δ':>9}  逐seed")
    summary = {}
    for (cell, arm), vals in sorted(pairs.items()):
        ds = [v[1] for v in vals]
        l5 = [v[2] for v in vals]
        pos = sum(1 for x in ds if x > 0)
        neg = sum(1 for x in ds if x < 0)
        print(f"{cell:<14}{arm:<12}{st.mean(ds):>+9.2f}"
              f"{(st.stdev(ds) if len(ds) > 1 else 0):>8.2f}{f'{pos}/{neg}':>8}"
              f"{st.mean(l5):>+9.2f}  "
              + ", ".join(f"s{v[0]}:{v[1]:+.2f}" for v in vals))
        summary[f"{cell}|{arm}"] = dict(mean=st.mean(ds),
                                        std=st.stdev(ds) if len(ds) > 1 else 0.0,
                                        n_pos=pos, n=len(ds),
                                        last5=st.mean(l5))
    out = os.path.join(args.dir, "reuse_p1_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n导出: {out}")

    # 判定: 每个 cell 内是否有任一复用臂 2/3 正且 Δmean>0
    print("\n== 判定 ==")
    for cell in cells:
        arms = [a for (c, a) in pairs if c == cell]
        if not arms:
            continue
        best = None
        for a in arms:
            s = summary[f"{cell}|{a}"]
            if best is None or s["mean"] > best[1]["mean"]:
                best = (a, s)
        a, s = best
        verdict = "复用有利" if (s["mean"] > 0 and s["n_pos"] >= math.ceil(2 * s["n"] / 3)) \
            else ("持平" if abs(s["mean"]) < 0.5 else "复用仍不利")
        print(f"  {cell:<14} 最佳臂={a:<11} Δ={s['mean']:+.2f} 正={s['n_pos']}/{s['n']} -> {verdict}")


if __name__ == "__main__":
    main()
