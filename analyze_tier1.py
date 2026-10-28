"""Tier-1 配对分析：固定 EMA、FedProx 与 RA-CMA 模块筛选。

RA-CMA 与固定 EMA 都按同一次运行的影子模型减 raw 模型计算净增益。
筛选门槛：平均增益 >= 0.8pp、至少 2/3 种子为正，并且相对相同筛选
种子上的最佳固定 EMA 平均增益落后不超过 0.1pp。
"""

import ast
import glob
import re
import statistics as st


SEEDS = [12, 42, 43, 44, 45, 46, 47, 48]
EMA_DECAYS = [0.4, 0.5, 0.6, 0.8, 0.9]
FEDPROX_MUS = [0.01, 0.1]
CMA_SCREEN_SEEDS = [42, 43, 44]
CMA_SCREEN = [
    ("STL", "stale-only"),
    ("CNF", "conflict-only"),
    ("RISK25", "risk lambda_s=0.25"),
    ("RISK50", "risk lambda_s=0.50"),
    ("RISK75", "risk lambda_s=0.75"),
    ("RISK47", "risk beta=[0.4,0.7]"),
]


def find_log(prefix):
    """优先读取 runner 捕获日志，兼容直接运行 main_point2 的 tag 日志。"""
    hits = sorted(glob.glob(f"point2_log_{prefix}_MNIST_*.txt"))
    if not hits:
        hits = sorted(glob.glob(f"point2_log_{prefix}_*.txt"))
    return hits[-1] if hits else None


def parse_log(path):
    try:
        with open(path, encoding="utf-8", errors="strict") as handle:
            txt = handle.read()
    except UnicodeDecodeError:  # 旧批次日志为 GBK
        with open(path, encoding="gbk", errors="replace") as handle:
            txt = handle.read()

    hist_match = re.search(r"Accuracy history: (\[.*?\])", txt, re.S)
    acc_match = re.search(r"最终准确率: ([0-9.]+)%", txt)
    if not hist_match or not acc_match:
        raise ValueError(f"incomplete point2 log: {path}")

    hist = ast.literal_eval(hist_match.group(1))
    acc = float(acc_match.group(1))
    shadow = re.search(r"最终准确率\(影子 ([^)]+)\): ([0-9.]+)%", txt)
    legacy_ema = re.search(r"最终准确率\(EMA a=[0-9.]+\): ([0-9.]+)%", txt)
    cma_mode = shadow.group(1) if shadow else ("fixed" if legacy_ema else None)
    cma_acc = (float(shadow.group(2)) if shadow else
               float(legacy_ema.group(1)) if legacy_ema else None)
    return {
        "acc": acc,
        "hist": [value for _, value in hist],
        "ema_acc": cma_acc if cma_mode == "fixed" else None,
        "cma_acc": cma_acc,
        "cma_mode": cma_mode,
    }


def baseline_seed(seed):
    for prefix in (f"SIG_NI_0.5_s{seed}", f"SIG2_NI_0.5_s{seed}"):
        path = find_log(prefix)
        if path:
            return parse_log(path)
    raise FileNotFoundError(f"no baseline log for seed {seed}")


def _std(values):
    return st.stdev(values) if len(values) > 1 else 0.0


def summarize_diffs(diffs):
    if not diffs:
        raise ValueError("at least one paired difference is required")
    return {
        "mean": st.mean(diffs),
        "std": _std(diffs),
        "positive": sum(value > 0 for value in diffs),
        "count": len(diffs),
    }


def promotion_result(diffs, best_fixed_gain):
    """执行 RA-CMA 晋级契约，便于真实分析和单元测试复用。"""
    summary = summarize_diffs(diffs)
    gain_ok = summary["mean"] >= 0.8
    positive_ok = summary["positive"] / summary["count"] >= 2.0 / 3.0
    fixed_ok = (best_fixed_gain is not None and
                summary["mean"] >= best_fixed_gain - 0.1)
    return {
        **summary,
        "gain_ok": gain_ok,
        "positive_ok": positive_ok,
        "fixed_ok": fixed_ok,
        "promote": gain_ok and positive_ok and fixed_ok,
    }


def stats_line(diffs):
    summary = summarize_diffs(diffs)
    negative = summary["count"] - summary["positive"]
    return (f"Δmean: {summary['mean']:+.2f}±{summary['std']:.2f}  "
            f"正/非正: {summary['positive']}/{negative}")


def load_tagged_runs(tag, seeds, metric):
    runs = {}
    for seed in seeds:
        path = find_log(f"{tag}_s{seed}")
        if not path:
            continue
        row = parse_log(path)
        if row[metric] is not None:
            runs[seed] = row
    return runs


def main():
    base_acc = {}
    for seed in SEEDS:
        base_acc[seed] = baseline_seed(seed)["acc"]

    print("== Fixed EMA 效果（EMA - 同 run raw）==")
    fixed_runs = {}
    for decay in EMA_DECAYS:
        tag = f"EMA{decay}"
        rows = load_tagged_runs(tag, SEEDS, "ema_acc")
        fixed_runs[decay] = rows
        if len(rows) != len(SEEDS):
            missing = [seed for seed in SEEDS if seed not in rows]
            print(f"  {tag}: 缺少种子 {missing}，跳过完整统计")
            continue
        diffs = [rows[seed]["ema_acc"] - rows[seed]["acc"]
                 for seed in SEEDS]
        print(f"  {tag}: {stats_line(diffs)}")

    # 必须在与 CMA 一致的 42-44 种子口径上选择最佳 fixed EMA。
    fixed_screen_gains = {}
    for decay, rows in fixed_runs.items():
        if all(seed in rows for seed in CMA_SCREEN_SEEDS):
            fixed_screen_gains[decay] = st.mean(
                rows[seed]["ema_acc"] - rows[seed]["acc"]
                for seed in CMA_SCREEN_SEEDS
            )
    if fixed_screen_gains:
        best_decay = max(fixed_screen_gains, key=fixed_screen_gains.get)
        best_fixed_gain = fixed_screen_gains[best_decay]
        print(f"\n筛选种子上的最佳 fixed EMA: beta={best_decay}, "
              f"Δmean={best_fixed_gain:+.2f}pp")
    else:
        best_fixed_gain = None
        print("\n筛选种子上的 fixed EMA 结果不完整，无法执行最终晋级判定。")

    print("\n== RA-CMA 模块筛选（CMA - 同 run raw）==")
    promoted = []
    for tag, label in CMA_SCREEN:
        rows = load_tagged_runs(tag, CMA_SCREEN_SEEDS, "cma_acc")
        if len(rows) != len(CMA_SCREEN_SEEDS):
            missing = [seed for seed in CMA_SCREEN_SEEDS if seed not in rows]
            print(f"  {tag:<7} {label:<24} 缺少种子 {missing}，不判定")
            continue
        diffs = [rows[seed]["cma_acc"] - rows[seed]["acc"]
                 for seed in CMA_SCREEN_SEEDS]
        result = promotion_result(diffs, best_fixed_gain)
        print(
            f"  {tag:<7} {label:<24} {stats_line(diffs)}  "
            f"raw门槛={'PASS' if result['gain_ok'] else 'FAIL'} "
            f"正种子={'PASS' if result['positive_ok'] else 'FAIL'} "
            f"fixed门槛={'PASS' if result['fixed_ok'] else 'FAIL'} => "
            f"{'晋级' if result['promote'] else '不晋级'}"
        )
        if result["promote"]:
            promoted.append((tag, result["mean"]))

    if promoted:
        promoted.sort(key=lambda item: item[1], reverse=True)
        print("晋级排序: " + ", ".join(
            f"{tag}({gain:+.2f}pp)" for tag, gain in promoted))
    else:
        print("当前没有满足全部条件的 RA-CMA 配置。")

    print("\n== FedProx（相对 SIG 基线）==")
    for mu in FEDPROX_MUS:
        rows = load_tagged_runs(f"PX{mu}", SEEDS, "acc")
        if len(rows) != len(SEEDS):
            missing = [seed for seed in SEEDS if seed not in rows]
            print(f"  PX{mu}: 缺少种子 {missing}，跳过完整统计")
            continue
        diffs = [rows[seed]["acc"] - base_acc[seed] for seed in SEEDS]
        print(f"  PX{mu}: {stats_line(diffs)}")


if __name__ == "__main__":
    main()
