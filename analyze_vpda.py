"""V-PDA 阶段 2 分析器: 配对增益 / TV 下降 / 权重贴边 / 晋级判定.

口径:
- 主指标: 同 run 配对 Δ = PDA_acc − G5 影子 acc (final_acc_g5_shadow,
  同一包序列/参考点, 无 RNG 差异); 辅指标: 跨 run Δ_cross vs G5 组同 seed。
- last-5 平均: 影子/主轨迹最后 5 个 checkpoint 均值差。
- TV 下降: run 内加权选中 TV (label_tv) vs 等权选中 TV (label_tv_eq),
  下降率 = (tv_eq − tv_w)/tv_eq。
- 权重贴边: q_k 序列贴近 [0.5, 2.0] 边界的比例 (>50% 判长期贴边)。
- ticks: PDA run 的完成 tick 与 G5 同 seed 一致 (V-PDA 不改变调度)。

用法: python analyze_vpda.py --dir results_vpda/s2_screen
"""

import argparse
import glob
import json
import math
import os
import statistics as st

import numpy as np

try:
    from collect_point2_results import _parse_namespace_str
except ImportError:
    _parse_namespace_str = None

BOUND_EPS = 0.01


def parse_config(d):
    c = d.get("config")
    if c is None:
        return {}
    if isinstance(c, np.ndarray) and c.ndim == 0:
        c = c.item()
    if isinstance(c, str) and _parse_namespace_str:
        return _parse_namespace_str(c)
    if hasattr(c, "__dict__"):
        return vars(c)
    return {}


def load_runs(dirs):
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "point2_data_*.npz"))):
            try:
                runs.append(np.load(p, allow_pickle=True))
            except Exception as e:
                print(f"  WARN skip {p}: {e}")
    return runs


def group_of(z, c):
    if c.get("participation_debias") != "ipw":
        return "G5"
    kappa = float(c.get("debias_kappa", 0.5))
    if not c.get("vcm"):
        return "PDAONLY"
    return f"PDA{int(kappa * 10):02d}"


def analyze(runs):
    rows = []
    g5_by_seed = {}
    for z in runs:
        c = parse_config(z)
        g = group_of(z, c)
        seed = int(c.get("seed", -1))
        final_acc = float(z["final_acc"])
        ticks = int(z["checkpoint_ticks"][-1]) if len(z["checkpoint_ticks"]) else -1
        if g == "G5":
            g5_by_seed[seed] = dict(acc=final_acc, ticks=ticks)
        rows.append(dict(group=g, seed=seed, z=z, final_acc=final_acc,
                         ticks=ticks))

    out = []
    for r in rows:
        z = r["z"]
        g5s = g5_by_seed.get(r["seed"])
        g5_shadow = float(z["final_acc_g5_shadow"]) \
            if not math.isnan(float(z["final_acc_g5_shadow"])) else None
        acc_hist = np.asarray(z["acc_history"], dtype=float)
        last5_raw = float(np.mean(acc_hist[-5:, 1])) if len(acc_hist) >= 5 else None

        row = dict(group=r["group"], seed=r["seed"],
                   final_acc=r["final_acc"],
                   # 主指标: 同 seed 独立 G5 配对 (性能结论口径)
                   delta_cross=(r["final_acc"] - g5s["acc"]) if g5s else None,
                   # 同 run 影子配对: 仅作轨迹诊断, 不作晋级依据
                   delta_pair=(r["final_acc"] - g5_shadow)
                   if g5_shadow is not None else None,
                   g5_independent_acc=g5s["acc"] if g5s else None,
                   ticks=r["ticks"], g5_ticks=g5s["ticks"] if g5s else None,
                   g5_acc=g5s["acc"] if g5s else None)

        if r["group"] != "G5":
            sh_hist = np.asarray(z["g5_shadow_acc_history"], dtype=float)
            last5_shadow = (float(np.mean(sh_hist[-5:, 1]))
                            if len(sh_hist) >= 5 else None)
            row["last5_delta"] = (last5_raw - last5_shadow
                                  if last5_shadow is not None else None)
            tv_w = np.asarray(z["label_tv"], dtype=float)
            tv_eq = np.asarray(z["label_tv_eq"], dtype=float)
            row["tv_w"] = float(np.mean(tv_w)) if tv_w.size else float("nan")
            row["tv_eq"] = float(np.mean(tv_eq)) if tv_eq.size else float("nan")
            # 累计 TV (实际参与用户原始标签计数, 跨 checkpoint 累计) — 主口径
            tv_c = np.asarray(z["label_tv_cum"], dtype=float)                 if "label_tv_cum" in z.files else np.array([])
            tv_ce = np.asarray(z["label_tv_cum_eq"], dtype=float)                 if "label_tv_cum_eq" in z.files else np.array([])
            row["tv_cum"] = float(tv_c[-1]) if tv_c.size else float("nan")
            row["tv_cum_eq"] = float(tv_ce[-1]) if tv_ce.size else float("nan")
            row["tv_cum_drop"] = ((row["tv_cum_eq"] - row["tv_cum"])
                                  / max(1e-12, row["tv_cum_eq"]))                 if tv_ce.size and row["tv_cum_eq"] > 1e-9 else None
            cq = np.asarray(z["cum_weight_quality"], dtype=float)                 if "cum_weight_quality" in z.files else np.array([])
            row["cum_weight_quality"] = float(cq[-1]) if cq.size else float("nan")
            row["tv_drop"] = ((float(np.mean(tv_eq)) - float(np.mean(tv_w)))
                              / max(1e-12, float(np.mean(tv_eq)))) \
                if tv_eq.size and float(np.mean(tv_eq)) > 1e-9 else None
            qs = np.asarray(z["debias_weights_float"], dtype=float)
            if qs.size:
                near = (np.abs(qs - 0.5) < BOUND_EPS) | (np.abs(qs - 2.0) < BOUND_EPS)
                row["bound_frac"] = float(near.mean())
            else:
                row["bound_frac"] = float("nan")
        out.append(row)
    return out


def print_rows(rows):
    print(f"{'组':<9}{'seed':>5}{'acc':>8}{'ΔG5':>8}{'Δpair':>8}"
          f"{'last5Δ':>8}{'TVcum':>8}{'TVcumEq':>8}{'降%':>7}{'贴边%':>7}{'ticks':>6}")
    for r in sorted(rows, key=lambda x: (x["group"], x["seed"])):
        fmt = (f"{r['group']:<9}{r['seed']:>5}{r['final_acc']:>8.2f}"
               + (f"{r['delta_cross']:>+8.2f}" if r["delta_cross"] is not None else f"{'':>8}")
               + (f"{r['delta_pair']:>+8.2f}" if r["delta_pair"] is not None else f"{'':>8}"))
        if "last5_delta" in r:
            fmt += (f"{r['last5_delta']:>+8.2f}" if r["last5_delta"] is not None else f"{'':>8}")
            _tc = r.get("tv_cum", float("nan"))
            _tce = r.get("tv_cum_eq", float("nan"))
            _tcd = r.get("tv_cum_drop")
            fmt += (f"{_tc:>8.4f}{_tce:>8.4f}"
                    + (f"{100*_tcd:>7.0f}" if _tcd is not None else f"{'':>7}")
                    + (f"{100*r['bound_frac']:>7.0f}" if not math.isnan(r["bound_frac"]) else f"{'':>7}"))
        fmt += f"{r['ticks']:>6}"
        print(fmt)


def promote_check(rows):
    """PDA05/PDA10 晋级判定 (相对同 run G5 影子配对 Δ)."""
    print("\n== 晋级判定 (PDA05/PDA10, 主指标=同 seed 独立 G5 配对 Δ) ==")
    verdicts = []
    for g in ("PDA05", "PDA10"):
        rs = [r for r in rows if r["group"] == g and r["delta_cross"] is not None]
        if not rs:
            print(f"  {g}: 数据缺失")
            continue
        ds = [r["delta_cross"] for r in rs]
        mean_d = st.mean(ds)
        n_pos = sum(1 for x in ds if x > 0)
        worst = min(ds)
        last5 = [r["last5_delta"] for r in rs if r.get("last5_delta") is not None]
        tv_drops = [r["tv_cum_drop"] for r in rs
                    if r.get("tv_cum_drop") is not None]
        bounds = [r["bound_frac"] for r in rs if not math.isnan(r.get("bound_frac", float("nan")))]
        ticks_ok = all(r["ticks"] == r["g5_ticks"] for r in rs if r.get("g5_ticks"))

        ok = {
            "mean_ge_0.8": mean_d >= 0.8,
            "pos_ge_2of3": n_pos >= math.ceil(2 * len(ds) / 3),
            "worst_ge_-0.5": worst >= -0.5,
            "last5_pos": bool(last5) and st.mean(last5) > 0,
            "tv_drop_ge_50": bool(tv_drops) and st.mean(tv_drops) >= 0.5,
            "no_bound_stick": bool(bounds) and st.mean(bounds) < 0.5,
            "ticks_unchanged": ticks_ok,
        }
        passed = all(ok.values())
        verdicts.append((g, passed, ok, dict(mean=mean_d, n_pos=n_pos, n=len(ds),
                                             worst=worst, last5=st.mean(last5) if last5 else None,
                                             tv_drop=st.mean(tv_drops) if tv_drops else None,
                                             bound=st.mean(bounds) if bounds else None)))
        print(f"  {g}: {'晋级' if passed else '淘汰'}  Δmean={mean_d:+.3f} "
              f"正={n_pos}/{len(ds)} 最差={worst:+.2f}")
        print(f"       [Δ≥0.8:{ok['mean_ge_0.8']} 正≥2/3:{ok['pos_ge_2of3']} "
              f"最差≥−0.5:{ok['worst_ge_-0.5']} last5>0:{ok['last5_pos']} "
              f"TV↓≥50%:{ok['tv_drop_ge_50']} 不贴边:{ok['no_bound_stick']} "
              f"ticks不变:{ok['ticks_unchanged']}]")
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    args = ap.parse_args()

    runs = load_runs(args.dir)
    rows = analyze(runs)
    print_rows(rows)
    verdicts = promote_check(rows)

    report = dict(rows=rows, verdicts=[{ "group": v[0], "passed": v[1], "ok": v[2],
                                         "stats": v[3]} for v in verdicts])
    out_json = os.path.join(args.dir[0], "vpda_s2_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n分析导出: {out_json}")
    if not verdicts:
        print("→ 无有效 V-PDA 数据 (运行缺失?), 无法判定")
    elif all(v[1] for v in verdicts):
        print("→ 全部 V-PDA 配置晋级, 可进入阶段 3 (机制消融)")
    elif any(v[1] for v in verdicts):
        print("→ 至少一个配置晋级, 对晋级配置进入阶段 3")
    else:
        print("→ 全部淘汰: 按计划停止, 不继续调大量参数")


if __name__ == "__main__":
    main()