"""三点创新实验矩阵分析: 汇总表 + 配对差分 + 图.

输入: point2_matrix_summary.jsonl (run_point_matrix.py 产出)
输出: 终端汇总表 / point2_matrix_report.md / point2_matrix_bars.png

阶梯归属:
  ① 检查点式组级半异步: G3 vs G2 vs G1  (acc + sim ticks/吞吐)
  ③ Direct-VCM 端补偿:  G3v - G3
  σ 幂衰减加权:         G5 - G3v (叠加 VCM) 与 G3s - G3 (单独)
"""

import json
import sys
from collections import defaultdict

import numpy as np

ARMS = ["G1", "G2", "G3", "G3s", "G3v", "G5"]
ARM_NAMES = {
    "G1": "FedAvg", "G2": "组间同步(K5)", "G3": "半异步①",
    "G3s": "半异步+σ", "G3v": "半异步+VCM③", "G5": "完整(VCM+σ)",
}


def load(path="point2_matrix_summary.jsonl"):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("rc") == 0 and r.get("final_acc") is not None:
            rows.append(r)
    return rows


def key(r):
    return (r["dataset"], r["part"])


def cell_stats(rows, ds, part, arm):
    accs = [r["final_acc"] for r in rows
            if r["dataset"] == ds and r["part"] == part and r["arm"] == arm]
    if not accs:
        return None
    walls = [r.get("wall_clock_s") or 0 for r in rows
             if r["dataset"] == ds and r["part"] == part and r["arm"] == arm]
    ticks = [r["final_tick"] for r in rows
             if r["dataset"] == ds and r["part"] == part and r["arm"] == arm
             and r.get("final_tick")]
    return {
        "n": len(accs), "mean": np.mean(accs), "std": np.std(accs),
        "accs": accs, "wall": np.mean(walls),
        "tick": int(np.mean(ticks)) if ticks else None,
    }


def paired_diff(rows, ds, part, arm_a, arm_b):
    """同 seed 配对差 arm_b - arm_a."""
    by = {(r["arm"], r["seed"]): r["final_acc"] for r in rows
          if r["dataset"] == ds and r["part"] == part}
    ds_ = []
    for seed in sorted({s for (a, s) in by if a == arm_a}):
        if (arm_a, seed) in by and (arm_b, seed) in by:
            ds_.append(by[(arm_b, seed)] - by[(arm_a, seed)])
    return ds_


def fmt_ci(deltas):
    if not deltas:
        return ""
    m, s = np.mean(deltas), np.std(deltas)
    n = len(deltas)
    # 精确 t(0.975, df=n-1) 单样本 CI
    t975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365}.get(n, 2.0)
    ci = t975 * s / np.sqrt(n) if n >= 2 else 0
    wins = sum(1 for d in deltas if d > 0)
    return f"{m:+.2f} ± {s:.2f} (CI95±{ci:.2f}, {wins}/{n} wins)"


def main(path="point2_matrix_summary.jsonl", out_md="point2_matrix_report.md"):
    rows = load(path)
    if not rows:
        print("无可用结果")
        return
    cells = sorted({key(r) for r in rows})

    lines = ["# 三点创新实验矩阵报告 (IID/Non-IID × 3 数据集)", ""]
    lines.append("> 阶梯: G1 FedAvg → G2 组间同步(K5) → G3 半异步① → "
                 "G3+σ / G3+VCM③ → G5=VCM+σ 完整。"
                 "20 checkpoints, 5ES×4U, K=3, s_max=3, reuse=none, "
                 "verify off, NI=Dirichlet α=0.5。seeds: MNIST/Fashion {42–46} (n=5), "
                 "CIFAR10 {42–44} (n=3)。")
    lines.append(">")
    lines.append("> 注: σ 幂衰减权重在实现中只缩放 VCM 补偿项 C_k (掩码域须恒等权"
                 "整数加法以保 LHH 同态验证, cloud.py), 故 G3s (σ 无 VCM) 与 G3 恒等"
                 "(MNIST 上逐位一致已验证; 其 CIFAR 微小差为 GPU conv 非确定性噪声"
                 "≈±0.3pp)。σ 的效果检验 = G5−G3v。")
    lines.append("")

    for ds, part in cells:
        lines.append(f"## {ds} — {'Non-IID α=0.5' if part == 'NI' else 'IID'}")
        lines.append("")
        lines.append("| 方法 | 最终 acc% (mean±std, n) | 墙钟s | sim ticks |")
        lines.append("|---|---|---|---|")
        for arm in ARMS:
            st = cell_stats(rows, ds, part, arm)
            if not st:
                continue
            tick = st["tick"] if st["tick"] else "-"
            lines.append(f"| {ARM_NAMES[arm]} ({arm}) | "
                         f"{st['mean']:.2f} ± {st['std']:.2f} (n={st['n']}) | "
                         f"{st['wall']:.0f} | {tick} |")
        lines.append("")
        # 配对差分
        for arm_a, arm_b, label in [
                ("G2", "G3", "① 半异步 vs 组间同步"),
                ("G3", "G3v", "③ +VCM"),
                ("G3v", "G5", "σ 叠加 VCM"),
                ("G3", "G3s", "σ 单独")]:
            d = paired_diff(rows, ds, part, arm_a, arm_b)
            if d:
                lines.append(f"- {label} ({arm_b}−{arm_a}, 同 seed 配对): "
                             f"{fmt_ci(d)}  (逐 seed: {[round(x, 2) for x in d]})")
        lines.append("")

    md = "\n".join(lines)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)

    # --- 柱状图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({ds for ds, _ in cells})
    parts = sorted({p for _, p in cells})
    fig, axes = plt.subplots(len(datasets), len(parts),
                             figsize=(5.2 * len(parts), 3.6 * len(datasets)),
                             squeeze=False)
    for i, ds in enumerate(datasets):
        for j, part in enumerate(parts):
            ax = axes[i][j]
            xs, ys, es, labels = [], [], [], []
            for k, arm in enumerate(ARMS):
                st = cell_stats(rows, ds, part, arm)
                if st:
                    xs.append(k)
                    ys.append(st["mean"])
                    es.append(st["std"])
                    labels.append(arm)
            bars = ax.bar(xs, ys, yerr=es, capsize=4,
                          color=["#999", "#7aa6c2", "#2a6f97", "#5fa8d3",
                                 "#e8985e", "#c2452d"][:len(xs)])
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=9)
            base = cell_stats(rows, ds, part, "G3")
            top = max(ys) + max(es) + 2 if ys else 10
            ax.set_ylim(min(40, min(ys) - 8) if ys else 0, top)
            yl = ax.get_ylim()
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2,
                        yl[0] + 0.03 * (yl[1] - yl[0]), f"{y:.1f}",
                        ha="center", fontsize=8, color="black", weight="bold")
            ax.set_title(f"{ds} — {'Non-IID' if part == 'NI' else 'IID'}",
                         fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            if j == 0:
                ax.set_ylabel("Final Test Acc (%)")
    plt.tight_layout()
    plt.savefig("point2_matrix_bars.png", dpi=150)
    print(f"\n图已保存: point2_matrix_bars.png, 报告: {out_md}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "point2_matrix_summary.jsonl"
    main(p)
