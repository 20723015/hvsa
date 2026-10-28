"""E4/N3 开销对比: 掩码层跨方案基准 (论文 3.5.6(a)(b)).

按 docs/OVERHEAD_EXPERIMENT.md 的冻结规格实现驱动: 三个对照方案 + 本方案 HVSA,
两组扫描 (Exp-A 固定 d 变 N; Exp-B 固定 N 变 d)。

方案实现均取自仓库内 `baselines/` (已在版本控制中, 未改动):
  VeriFL  -> baselines/secagg_crypto.py   (Bonawitz SecAgg 掩码层, 用户间 P2P 全对)
  VLMRS   -> baselines/vlmrs_crypto.py    (Flamingo 图邻居, 用户 P2P 仅邻居)
  GPVO-FL -> baselines/gpvo_crypto.py     (分组 + 验证外包, 云侧 M 个组向量)
  HVSA    -> baselines/hvsa_crypto.py     (单 TA, 1× PRG 掩码, 无用户 P2P, 云侧 M=5)

口径边界 (规格原文, 正文引用时必须一并标注):
  1. 本基准范围 = **掩码与云侧入流量**, 不含 LHH/签名/双线性配对验证;
  2. VeriFL 曲线是**安全聚合掩码路径**, 不是完整 VeriFL 验证;
  3. VLMRS 云侧仍为 O(Nd) 主导 (图只省用户 P2P), **不是** O(log N);
  4. HVSA 云侧为 O(Md), 与 N 无关。

用法:
  python exp_overhead_crypto.py                 # 全量两组扫描
  python exp_overhead_crypto.py --repeats 3
  python exp_overhead_crypto.py --plot-only     # 从已有 JSON 出图
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OUT_JSON = os.path.join(ROOT, "results", "exp_crypto_overhead.json")
OUT_PDF = os.path.join(ROOT, "figures", "exp_crypto_overhead.pdf")

EXP_A_N = [500, 750, 1000, 1250, 1500, 1750, 2000]
EXP_A_D = 10_000
EXP_B_D = [10_000, 30_000, 50_000, 70_000, 100_000]
EXP_B_N = 1000

SCHEMES = [
    ("VeriFL", "baselines.secagg_crypto"),
    ("VLMRS", "baselines.vlmrs_crypto"),
    ("GPVO-FL", "baselines.gpvo_crypto"),
    ("HVSA", "baselines.hvsa_crypto"),
]


def load_schemes():
    import importlib
    mods = []
    for label, path in SCHEMES:
        mod = importlib.import_module(path)
        if not hasattr(mod, "measure_overhead"):
            print("  跳过 %s: 无 measure_overhead" % label)
            continue
        mods.append((label, mod))
    return mods


def run_sweep(mods, repeats):
    out = {"exp_a": {"d": EXP_A_D, "n_values": EXP_A_N, "rows": []},
           "exp_b": {"n": EXP_B_N, "d_values": EXP_B_D, "rows": []}}

    print("=" * 96)
    print("Exp-A: d=%d, N 扫描" % EXP_A_D)
    print("=" * 96)
    print("%-10s %6s %14s %16s %14s %14s"
          % ("scheme", "N", "user_comp(ms)", "server_comp(ms)", "user_kb", "server_mb"))
    for n in EXP_A_N:
        for label, mod in mods:
            m = mod.measure_overhead(EXP_A_D, n, repeats)
            out["exp_a"]["rows"].append({
                "scheme": label, "d": EXP_A_D, "n": n,
                "comp_user_ms": m.comp_user_ms,
                "comp_server_ms": m.comp_server_ms,
                "comm_user_kb": m.comm_user_kb,
                "comm_server_mb": m.comm_server_mb,
                "extra": m.extra,
            })
            print("%-10s %6d %14.2f %16.2f %14.2f %14.3f"
                  % (label, n, m.comp_user_ms, m.comp_server_ms,
                     m.comm_user_kb, m.comm_server_mb))
        print()

    print("=" * 96)
    print("Exp-B: N=%d, d 扫描" % EXP_B_N)
    print("=" * 96)
    print("%-10s %8s %14s %16s %14s %14s"
          % ("scheme", "d", "user_comp(ms)", "server_comp(ms)", "user_kb", "server_mb"))
    for d in EXP_B_D:
        for label, mod in mods:
            m = mod.measure_overhead(d, EXP_B_N, repeats)
            out["exp_b"]["rows"].append({
                "scheme": label, "d": d, "n": EXP_B_N,
                "comp_user_ms": m.comp_user_ms,
                "comp_server_ms": m.comp_server_ms,
                "comm_user_kb": m.comm_user_kb,
                "comm_server_mb": m.comm_server_mb,
                "extra": m.extra,
            })
            print("%-10s %8d %14.2f %16.2f %14.2f %14.3f"
                  % (label, d, m.comp_user_ms, m.comp_server_ms,
                     m.comm_user_kb, m.comm_server_mb))
        print()

    out["meta"] = {
        "repeats": repeats,
        "scope": "masking + cloud-side ingress only; excludes LHH/signatures/pairing",
        "caveats": [
            "VeriFL curve is the SA masking path, not full VeriFL verification.",
            "VLMRS cloud ingress is O(Nd) dominant, not O(log N) at cloud.",
            "HVSA cloud ingress is O(Md), independent of N.",
        ],
    }
    return out


def plot(data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("  绘图跳过 (matplotlib 不可用):", e)
        return
    labels = [s for s, _ in SCHEMES]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, key, xkey, xlabel, title in [
        (axes[0][0], "exp_a", "n", "N (users)", "Exp-A: cloud ingress vs N (d=10k)"),
        (axes[0][1], "exp_b", "d", "d (dim)", "Exp-B: cloud ingress vs d (N=1000)"),
    ]:
        rows = data[key]["rows"]
        for lb in labels:
            pts = sorted([(r[xkey], r["comm_server_mb"]) for r in rows if r["scheme"] == lb])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lb)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cloud ingress (MB)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax, key, xkey, xlabel, title in [
        (axes[1][0], "exp_a", "n", "N (users)", "Exp-A: user computation vs N"),
        (axes[1][1], "exp_b", "d", "d (dim)", "Exp-B: user computation vs d"),
    ]:
        rows = data[key]["rows"]
        for lb in labels:
            pts = sorted([(r[xkey], r["comp_user_ms"]) for r in rows if r["scheme"] == lb])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", label=lb)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("user comp (ms)")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Masking-layer overhead: HVSA vs VeriFL / VLMRS / GPVO-FL\n"
                 "(scope: masking + cloud ingress; excludes LHH/signature/pairing)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, dpi=150)
    print("  图已保存:", OUT_PDF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.plot_only:
        if not os.path.exists(OUT_JSON):
            print("无 %s, 先跑一次全量" % OUT_JSON)
            return 1
        with open(OUT_JSON, encoding="utf-8") as f:
            plot(json.load(f))
        return 0

    mods = load_schemes()
    if not mods:
        print("无可用方案")
        return 1
    data = run_sweep(mods, args.repeats)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("JSON 已保存:", OUT_JSON)
    plot(data)
    print()
    print("口径边界 (引用时必须标注):")
    for c in data["meta"]["caveats"]:
        print("  -", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
