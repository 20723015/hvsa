#!/usr/bin/env python3
"""
Encryption/masking overhead comparison (VeriFL-SA, VLMRS, HVSA).

Frozen spec: docs/OVERHEAD_EXPERIMENT.md
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines.crypto_common import LINK_BANDWIDTH_MBPS, LINK_RTT_MS
from baselines.hvsa_crypto import measure_overhead as hvsa_overhead
from baselines.secagg_crypto import measure_overhead as verifl_sa_overhead
from baselines.vlmrs_crypto import measure_overhead as vlmrs_overhead

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SCHEMES = ("VeriFL-SA", "VLMRS", "HVSA")
# Paper-facing legend (matches exp8.py / exp_comparison_final.pdf)
PLOT_LABELS = {"VeriFL-SA": "VeriFL", "VLMRS": "VLMRS", "HVSA": "HVSA"}
PLOT_STYLES = {
    "VeriFL-SA": {"linestyle": ":", "color": "#888888", "linewidth": 4.0},
    "VLMRS": {"linestyle": "-.", "color": "#2CA02C", "linewidth": 4.0},
    "HVSA": {"linestyle": "-", "color": "#E15759", "linewidth": 6.0},
}

# Sweep ranges (user-requested)
EXP_A_DIM = 10_000
EXP_A_N = [500, 750, 1000, 1250, 1500, 1750, 2000]
EXP_B_N = 1000  # fixed N when sweeping d (midpoint of 500–2000)
EXP_B_DIMS = [10_000, 30_000, 50_000, 70_000, 100_000]

MEASURERS = {
    "VeriFL-SA": verifl_sa_overhead,
    "VLMRS": vlmrs_overhead,
    "HVSA": hvsa_overhead,
}


def run_sweep(x_values: list[int], fixed_n: int | None, fixed_d: int, repeats: int, vary: str) -> dict:
    results = {s: {"user_ms": [], "server_ms": [], "user_kb": [], "server_mb": []} for s in SCHEMES}
    meta_log: list[dict] = []

    for x in x_values:
        if vary == "N":
            n, d = x, fixed_d
        else:
            n, d = fixed_n, x

        logger.info(f"--- vary {vary}: n={n}, d={d} ---")
        point_meta = {"n": n, "d": d, "schemes": {}}

        for scheme in SCHEMES:
            m = MEASURERS[scheme](d, n, repeats=repeats)
            results[scheme]["user_ms"].append(float(m.comp_user_ms))
            results[scheme]["server_ms"].append(float(m.comp_server_ms))
            results[scheme]["user_kb"].append(float(m.comm_user_kb))
            results[scheme]["server_mb"].append(float(m.comm_server_mb))
            point_meta["schemes"][scheme] = {
                "user_ms": float(m.comp_user_ms),
                "server_ms": float(m.comp_server_ms),
                "user_kb": float(m.comm_user_kb),
                "server_mb": float(m.comm_server_mb),
                "extra": m.extra,
            }
            logger.info(
                f"  {scheme:8s} user={m.comp_user_ms:8.2f} ms  "
                f"server={m.comp_server_ms:8.2f} ms  "
                f"user_comm={m.comm_user_kb:7.2f} KB"
            )

        meta_log.append(point_meta)

    return {"x": x_values, "vary": vary, "results": results, "meta": meta_log}


def _apply_paper_rcparams() -> None:
    """Match exp8.py / exp_comparison_final.pdf typography."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 24,
        "axes.labelsize": 28,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    })


def _plot_sub(
    ax,
    x_data,
    y_by_scheme: dict[str, list[float]],
    xlab: str,
    ylab: str,
    panel_label: str,
    title_text: str,
    *,
    logy: bool = False,
) -> None:
    for scheme in SCHEMES:
        st = PLOT_STYLES[scheme]
        ax.plot(
            x_data,
            y_by_scheme[scheme],
            linestyle=st["linestyle"],
            color=st["color"],
            linewidth=st["linewidth"],
            label=PLOT_LABELS[scheme],
        )
    ax.set_title(title_text, fontsize=28, fontweight="bold", pad=15)
    ax.set_xlabel(xlab, fontweight="bold", labelpad=18)
    ax.set_ylabel(ylab, fontweight="bold")
    ax.legend(fontsize=18, frameon=True, edgecolor="black", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    if logy:
        ax.set_yscale("log")
    # Panel tag sits below x-axis label (extra bottom margin in plot_eight_panel).
    ax.text(
        0.5,
        -0.34,
        panel_label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
    )


def plot_eight_panel(exp_a: dict, exp_b: dict, out_path: Path) -> None:
    _apply_paper_rcparams()
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))

    def series(exp: dict, field: str) -> dict[str, list[float]]:
        return {s: exp["results"][s][field] for s in SCHEMES}

    x_n = exp_a["x"]
    x_d = exp_b["x"]

    # Row 1: vary N (fixed d)
    _plot_sub(
        axes[0, 0],
        x_n,
        series(exp_a, "user_ms"),
        "Number of Users",
        "Latency (ms)",
        "(a)",
        "User Computation Latency",
        logy=True,
    )
    _plot_sub(
        axes[0, 1],
        x_n,
        series(exp_a, "user_kb"),
        "Number of Users",
        "Overhead (KB)",
        "(b)",
        "User Communication Overhead",
    )
    _plot_sub(
        axes[0, 2],
        x_n,
        series(exp_a, "server_ms"),
        "Number of Users",
        "Latency (ms)",
        "(c)",
        "CS Computation Latency",
    )
    _plot_sub(
        axes[0, 3],
        x_n,
        series(exp_a, "server_mb"),
        "Number of Users",
        "Overhead (MB)",
        "(d)",
        "CS Communication Overhead",
        logy=True,
    )

    # Row 2: vary d (fixed N)
    _plot_sub(
        axes[1, 0],
        x_d,
        series(exp_b, "user_ms"),
        "Model Dimension",
        "Latency (ms)",
        "(e)",
        "User Computation Latency",
        logy=True,
    )
    _plot_sub(
        axes[1, 1],
        x_d,
        series(exp_b, "user_kb"),
        "Model Dimension",
        "Overhead (KB)",
        "(f)",
        "User Communication Overhead",
    )
    _plot_sub(
        axes[1, 2],
        x_d,
        series(exp_b, "server_ms"),
        "Model Dimension",
        "Latency (ms)",
        "(g)",
        "CS Computation Latency",
    )
    _plot_sub(
        axes[1, 3],
        x_d,
        series(exp_b, "server_mb"),
        "Model Dimension",
        "Overhead (MB)",
        "(h)",
        "CS Communication Overhead",
    )

    plt.tight_layout(pad=2.5, h_pad=4.5, w_pad=2.5)
    # Room for (a)–(h) below x-axis labels; gap between rows for top-row tags vs bottom-row titles.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.16, hspace=0.62, wspace=0.28)
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.35)
    logger.info(f"Saved figure: {out_path}")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Encryption overhead benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Smaller sweeps for smoke test")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig", type=Path, default=Path("figures/exp_crypto_overhead.pdf"))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figure from results/exp_crypto_overhead.json",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig.parent.mkdir(parents=True, exist_ok=True)

    n_vals = [500, 1000] if args.quick else EXP_A_N
    d_vals = [10_000, 50_000] if args.quick else EXP_B_DIMS

    env = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repeats": args.repeats,
        "spec": "docs/OVERHEAD_EXPERIMENT.md",
        "note": (
            "VeriFL-SA=Bonawitz SecAgg masking; VLMRS=Flamingo neighbors; HVSA=M=5 edges. "
            "User: measured crypto + analytical P2P (10 Mbps, 20 ms RTT/peer). "
            "CS comp: measured sum; CS comm: analytical (VeriFL Nd+N^2 shares, "
            "VLMRS Nd+N log N shares+proof, HVSA Md)."
        ),
    }

    json_path = args.out_dir / "exp_crypto_overhead.json"
    if args.plot_only:
        if not json_path.is_file():
            logger.error(f"Missing {json_path}; run benchmark first.")
            return 1
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
        exp_a = {
            "x": payload["exp_a"]["x"],
            "results": payload["exp_a"]["results"],
        }
        exp_b = {
            "x": payload["exp_b"]["x"],
            "results": payload["exp_b"]["results"],
        }
        logger.info(f"Plot-only from {json_path}")
    else:
        logger.info("========== Exp-A: vary N ==========")
        exp_a = run_sweep(n_vals, None, EXP_A_DIM, args.repeats, vary="N")

        logger.info("\n========== Exp-B: vary d ==========")
        exp_b = run_sweep(d_vals, EXP_B_N, EXP_B_DIMS[0], args.repeats, vary="d")

        payload = json_safe({
            "env": env,
            "config": {
                "exp_a": {"fixed_d": EXP_A_DIM, "n_values": n_vals},
                "exp_b": {"fixed_n": EXP_B_N, "d_values": d_vals},
            },
            "exp_a": {"x": exp_a["x"], "vary": exp_a["vary"], "results": exp_a["results"]},
            "exp_b": {"x": exp_b["x"], "vary": exp_b["vary"], "results": exp_b["results"]},
        })
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")

    def save_figure(path: Path) -> Path:
        try:
            plot_eight_panel(exp_a, exp_b, path)
            return path
        except PermissionError:
            for i in range(2, 10):
                alt = path.with_name(f"{path.stem}_v{i}{path.suffix}")
                try:
                    plot_eight_panel(exp_a, exp_b, alt)
                    logger.warning(f"Cannot write {path} (file locked); saved to {alt}")
                    return alt
                except PermissionError:
                    continue
            raise

    out_pdf = save_figure(args.fig)
    save_figure(out_png := args.fig.with_suffix(".png"))
    logger.info(f"Saved PNG: {out_png}")
    logger.info(f"Saved PDF: {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
