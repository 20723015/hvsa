"""研究点2: 对比图生成器 (实验二 G2 vs FedAvg 等任意 npz 曲线).

用法:
  python plot_point2_compare.py --npz point2_data_MNIST_xxx.npz --label "Grouped+Sync (G2)" \
      --npz point2_data_fedavg_MNIST_xxx.npz --label "FedAvg" \
      --xlabel "Global Round / Checkpoint" --ylabel "Test Accuracy (%)" \
      --out exp2_vs_fedavg
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="point2 曲线对比生成器")
    parser.add_argument("--npz", action="append", required=True,
                        help="npz 文件 (可多次指定, 与 --label 一一对应)")
    parser.add_argument("--label", action="append", required=True,
                        help="每条曲线的标签")
    parser.add_argument("--xlabel", default="Global Round / Checkpoint")
    parser.add_argument("--ylabel", default="Test Accuracy (%)")
    parser.add_argument("--out", default="point2_compare")
    args = parser.parse_args()
    assert len(args.npz) == len(args.label)

    plt.figure(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v"]
    for i, (path, label) in enumerate(zip(args.npz, args.label)):
        d = np.load(path, allow_pickle=True)
        acc = d["acc_history"]
        if np.ndim(acc) == 2 and acc.shape[1] == 2:
            xs, ys = acc[:, 0], acc[:, 1]  # (ckpt, acc)
        else:
            xs, ys = np.arange(1, len(acc) + 1), np.asarray(acc, dtype=float)
        plt.plot(xs, ys, marker=markers[i % len(markers)], markevery=2,
                 linewidth=1.8, label=label, alpha=0.9)

    plt.xlabel(args.xlabel, fontsize=11)
    plt.ylabel(args.ylabel, fontsize=11)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{args.out}.png", dpi=200)
    plt.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"saved: {args.out}.png / {args.out}.pdf")


if __name__ == "__main__":
    main()