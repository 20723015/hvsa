"""RA-CMA 实验分析器: 同 run 配对增益 / bootstrap CI / 晋级判定 / 机制相关性.

口径 (与实验计划一致):
- 主指标: checkpoint 20 最终准确率; 辅助: 最后 5 checkpoint 平均、跨种子 std、正收益比例。
- 所有增益为同 run 配对: Δ = Acc_CMA − Acc_raw (同一 run 内的影子 vs 原始轨迹)。
- bootstrap 95% CI 作用于配对 Δ 均值 (B=10000, 固定 RNG, 不筛种子)。

用法:
    python analyze_cma.py --dir results_cma/s1_fixed_ema --stage 1
    python analyze_cma.py --dir results_cma/s2_screen --stage 2 \
        --ref-dir results_cma/s1_fixed_ema --best-fix 0.5
    python analyze_cma.py --dir results_cma/s4_generalize --stage 4 --best-fix 0.5
    python analyze_cma.py --dir results_cma/s5_sensitivity --stage 5

输出: 终端表格 + {dir}/cma_analysis.md / .json
"""

import argparse
import glob
import json
import math
import os
import re
import statistics as st

import numpy as np

try:  # 复用注册器的配置解析
    from collect_point2_results import _parse_namespace_str
except ImportError:
    _parse_namespace_str = None

B = 10000
BOOT_SEED = 20260910
LAST5 = 5


# ============================================================
#  数据加载
# ============================================================

def parse_config(d: dict) -> dict:
    c = d.get("config")
    if c is None:
        return {}
    if isinstance(c, np.ndarray) and c.ndim == 0:
        c = c.item()
    if isinstance(c, str):
        if _parse_namespace_str:
            return _parse_namespace_str(c)
        m = re.search(r"Namespace\((.*)\)", c, re.S)
        if not m:
            return {}
        out = {}
        for k, v in re.findall(r"(\w+)=([^,]+)(?:,|$)", m.group(1)):
            try:
                out[k] = eval(v)
            except Exception:
                out[k] = v
        return out
    if hasattr(c, "__dict__"):
        return vars(c)
    return {}

def load_runs(dirs) -> list:
    runs = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "point2_data_*.npz"))):
            try:
                z = np.load(p, allow_pickle=True)
                runs.append((p, z))
            except Exception as e:
                print(f"  WARN skip {p}: {e}")
    return runs


def acc_series(z):
    h = z["acc_history"]
    if h is None or h.size == 0:
        return None
    return np.asarray(h, dtype=float)


def cma_series(z, key):
    h = z[key]
    if h is None or h.size == 0:
        return None
    return np.asarray(h, dtype=float)


def last5_mean(series):
    return float(np.mean(series[-LAST5:, 1])) if series is not None and len(series) >= 1 else float("nan")


def run_metrics(z, c: dict) -> dict:
    acc = acc_series(z)
    cma_acc = cma_series(z, "cma_acc_history")
    final_raw = float(z["final_acc"]) if acc is not None else float("nan")
    final_cma = float(z["final_acc_cma"]) if cma_acc is not None and not math.isnan(float(z["final_acc_cma"])) else None
    return {
        "seed": int(c.get("seed", -1)),
        "dataset": c.get("dataset", ""),
        "noniid": bool(c.get("noniid", False)),
        "alpha": float(c.get("alpha", 0.5)),
        "max_delay": int(c.get("max_delay", 5)),
        "mode": c.get("cma_mode", "none"),
        "beta": float(c.get("cma_beta", 0.5)),
        "beta_min": float(c.get("cma_beta_min", 0.3)),
        "beta_max": float(c.get("cma_beta_max", 0.65)),
        "lam": float(c.get("cma_lambda_stale", 0.5)),
        "final_raw": final_raw,
        "final_cma": final_cma,
        "delta": (final_cma - final_raw) if final_cma is not None else None,
        "last5_raw": last5_mean(acc),
        "last5_cma": last5_mean(cma_acc) if cma_acc is not None else None,
        "acc_hist": acc,
        "cma_hist": cma_acc,
        "risk_hist": cma_series(z, "cma_risk_history"),
        "stale_hist": cma_series(z, "cma_stale_risk_history"),
        "conf_hist": cma_series(z, "cma_conflict_history"),
        "beta_hist": cma_series(z, "cma_beta_history"),
    }


def label(m: dict) -> str:
    if m["mode"] == "none":
        return "raw"
    if m["mode"] == "fixed":
        return f"fixed(β={m['beta']:g})"
    rng = f"β∈[{m['beta_min']:g},{m['beta_max']:g}]"
    if m["mode"] in ("stale", "conflict"):
        return f"{m['mode']}({rng})"
    return f"risk(λs={m['lam']:g},{rng})"


def config_key(m: dict) -> tuple:
    return (m["dataset"], m["noniid"], m["alpha"], m["max_delay"], m["mode"],
            m["beta"], m["beta_min"], m["beta_max"], m["lam"])


# ============================================================
#  统计
# ============================================================

def bootstrap_ci(deltas, alpha=0.05):
    """配对 Δ 均值的 bootstrap 95% CI (有放回重采样同种子数, B 次)."""
    d = np.asarray([x for x in deltas if x is not None], dtype=float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(B)
    n = len(d)
    for i in range(B):
        means[i] = np.mean(rng.choice(d, size=n, replace=True))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def group_report(cfg_runs, verbose=True):
    """按 (配置) 汇总: 每种子 Δ + 均值±std + 正比例 + bootstrap CI + last5."""
    rows = []
    for key, runs in sorted(cfg_runs.items(), key=lambda kv: label(kv[1][0])):
        runs = [r for r in runs if r["delta"] is not None]
        if not runs:
            continue
        deltas = [r["delta"] for r in runs]
        lo, hi = bootstrap_ci(deltas)
        last5_d = [r["last5_cma"] - r["last5_raw"] for r in runs]
        rows.append({
            "key": key, "mode": runs[0]["mode"], "label": label(runs[0]),
            "n_seeds": len(runs),
            "seeds": sorted(r["seed"] for r in runs),
            "final_mean": st.mean([r["final_cma"] for r in runs if r["final_cma"] is not None]),
            "d_mean": st.mean(deltas), "d_std": st.stdev(deltas) if len(deltas) > 1 else float("nan"),
            "n_pos": sum(1 for x in deltas if x > 0), "n_neg": sum(1 for x in deltas if x < 0),
            "ci_lo": lo, "ci_hi": hi,
            "last5_d_mean": st.mean(last5_d),
            "per_seed": {r["seed"]: r["delta"] for r in runs},
        })
    rows.sort(key=lambda r: -r["d_mean"])
    return rows


def print_report(rows, title):
    print(f"\n== {title} ==")
    print(f"{'配置':<24}{'种子数':>5}{'Δ均值':>9}{'±std':>8}{'正/负':>7}{'CI下界':>9}{'CI上界':>9}{'last5Δ':>9}")
    for r in rows:
        print(f"{r['label']:<24}{r['n_seeds']:>5}{r['d_mean']:>9.3f}"
              f"{r['d_std']:>8.3f}{r['n_pos']}/{r['n_neg']:>4}"
              f"{r['ci_lo']:>10.3f}{r['ci_hi']:>9.3f}{r['last5_d_mean']:>9.3f}")
    print("种子明细:")
    for r in rows:
        sd = ", ".join(f"s{s}:{v:+.2f}" for s, v in sorted(r["per_seed"].items()))
        print(f"  {r['label']:<24} {sd}")


def determinism_check(ms):
    """确定性校验: 每个影子 run 的 raw 轨迹必须与同 seed 的 raw-only run 逐值一致."""
    refs = {m["seed"]: m["acc_hist"] for m in ms if m["mode"] == "none"}
    bad = []
    for m in ms:
        if m["mode"] == "none" or m["acc_hist"] is None:
            continue
        ref = refs.get(m["seed"])
        if ref is None:
            bad.append((label(m), m["seed"], "无 raw 参照"))
        elif len(ref) != len(m["acc_hist"]) or not np.array_equal(ref, m["acc_hist"]):
            bad.append((label(m), m["seed"], "raw 轨迹不一致"))
    print(f"\n== 确定性校验 (影子 run raw 曲线 vs raw-only, 应逐 checkpoint 一致) ==")
    if not bad:
        print("  全部一致 OK")
    else:
        for lab, s, why in bad:
            print(f"  *** {lab} s{s}: {why}")
    return bad


# ============================================================
#  阶段判定
# ============================================================

def stage1_pick_best_fixed(rows, top_n=1):
    """选 Δmean 最高的 fixed 配置 (计划: 三组种子上平均增益最高)."""
    fixed = [r for r in rows if r["mode"] == "fixed"]
    fixed.sort(key=lambda r: -r["d_mean"])
    if not fixed:
        return None
    return fixed[0]


def stage2_screen(rows, best_fixed_mean):
    """初筛晋级: Δmean≥0.8pp, 正≥2/3, 且 ≥ best fixed Δmean − 0.1pp."""
    verdicts = []
    for r in rows:
        if r["mode"] not in ("stale", "conflict", "risk"):
            continue
        n = r["n_seeds"]
        ok_mean = r["d_mean"] >= 0.8
        ok_pos = r["n_pos"] >= math.ceil(2 * n / 3)
        ok_fixed = r["d_mean"] >= best_fixed_mean - 0.1
        verdicts.append((r, ok_mean, ok_pos, ok_fixed,
                         ok_mean and ok_pos and ok_fixed))
    return verdicts


def stage3_confirm(rows, risk_keys, best_fixed_mean, stl_key, cnf_key):
    """八种子确认: Δmean≥1.0pp, 正≥6/8, CI 下界>0, ≥best fixed+0.3pp, 且优于 STL/CNF."""
    by_key = {r["key"]: r for r in rows}
    out = []
    for k in risk_keys:
        r = by_key.get(k)
        if r is None:
            out.append((k, None, False, "配置缺失"))
            continue
        ok_mean = r["d_mean"] >= 1.0
        ok_pos = r["n_pos"] >= 6
        ok_ci = r["ci_lo"] > 0
        ok_fixed = r["d_mean"] >= best_fixed_mean + 0.3
        ok_vs_stl = stl_key in by_key and r["d_mean"] > by_key[stl_key]["d_mean"]
        ok_vs_cnf = cnf_key in by_key and r["d_mean"] > by_key[cnf_key]["d_mean"]
        ok = all([ok_mean, ok_pos, ok_ci, ok_fixed, ok_vs_stl, ok_vs_cnf])
        out.append((k, r, ok, dict(mean=ok_mean, pos=ok_pos, ci=ok_ci,
                                   fixed=ok_fixed, vs_stl=ok_vs_stl, vs_cnf=ok_vs_cnf)))
    return out


def stage4_generalize(rows, best_fixed_mean):
    """多数据集: Non-IID 宏平均≥0.5pp; 配对正≥70%; 单数据集≥−0.3; IID 宏平均≥−0.2; ≥2 数据集优于 fixed."""
    # rows 需含 final_risk 与 fixed 两组; 按 (dataset, noniid) 聚合 Δ
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["noniid"], r["label"]), []).append(r)
    # risk 标签统一为 mode=='risk' 的最终配置 (调用方只传入该配置的 runs)
    risk_runs = [r for r in rows if r["delta"] is not None and r["mode"] == "risk"]
    fixed_runs = [r for r in rows if r["delta"] is not None and r["mode"] == "fixed"]
    if not risk_runs:
        return None
    per_ds = {}
    for ds in sorted({r["dataset"] for r in risk_runs}):
        ni = [r["delta"] for r in risk_runs if r["dataset"] == ds and r["noniid"]]
        iid = [r["delta"] for r in risk_runs if r["dataset"] == ds and not r["noniid"]]
        f_ni = [r["delta"] for r in fixed_runs if r["dataset"] == ds and r["noniid"]]
        per_ds[ds] = dict(ni=ni, iid=iid, f_ni=f_ni)
    all_ni = [x for ds in per_ds.values() for x in ds["ni"]]
    all_iid = [x for ds in per_ds.values() for x in ds["iid"]]
    macro_ni = st.mean([st.mean(ds["ni"]) for ds in per_ds.values()])
    macro_iid = st.mean([st.mean(ds["iid"]) for ds in per_ds.values()])
    n_pos_total = sum(1 for x in all_ni if x > 0)
    ratio = n_pos_total / len(all_ni)
    ds_min = min(st.mean(ds["ni"]) for ds in per_ds.values())
    better_fixed = sum(1 for ds in per_ds.values()
                       if st.mean(ds["ni"]) > st.mean(ds.get("f_ni") or [float("nan")])) if per_ds else 0
    ok = dict(
        macro_ni=macro_ni >= 0.5,
        ratio=ratio >= 0.7,
        ds_min=ds_min >= -0.3,
        macro_iid=macro_iid >= -0.2,
        better_fixed=better_fixed >= 2,
    )
    res = dict(macro_ni=macro_ni, macro_iid=macro_iid, ratio=ratio,
               ds_min=ds_min, better_fixed=better_fixed, per_ds=per_ds, ok=ok)
    return res


# ============================================================
#  机制相关性: r_t / q_s / q_d / β_t vs 逐 checkpoint (CMA−raw)
# ============================================================

def corr_analysis(runs, mode="risk"):
    rows = []
    for r in runs:
        if r["mode"] != mode or r["cma_hist"] is None or r["risk_hist"] is None:
            continue
        n = min(len(r["acc_hist"]), len(r["cma_hist"]))
        if n < 3:
            continue
        d_ckp = r["cma_hist"][:n, 1] - r["acc_hist"][:n, 1]
        out = {"seed": r["seed"], "n": n}
        for name, hist in (("r", r["risk_hist"]), ("q_s", r["stale_hist"]),
                           ("q_d", r["conf_hist"]), ("beta", r["beta_hist"])):
            if hist is None or len(hist) < n:
                out[name] = float("nan")
                continue
            x = hist[:n, 1]
            sx, sy = np.std(x), np.std(d_ckp)
            if sx < 1e-12 or sy < 1e-12:
                out[name] = float("nan")
            else:
                out[name] = float(np.corrcoef(x, d_ckp)[0, 1])
        rows.append(out)
    if not rows:
        return None
    print(f"\n== 机制相关性 (逐 checkpoint CMA−raw 差 vs 风险/β 序列, N={len(rows)} runs) ==")
    print(f"{'seed':>6}{'n':>5}{'r(r_t)':>9}{'r(q_s)':>9}{'r(q_d)':>9}{'r(β_t)':>9}")
    for o in rows:
        print(f"{o['seed']:>6}{o['n']:>5}"
              + "".join(f"{o[k]:>9.3f}" for k in ("r", "q_s", "q_d", "beta")))
    for k in ("r", "q_s", "q_d", "beta"):
        vals = [o[k] for o in rows if not math.isnan(o[k])]
        if vals:
            print(f"  {k}: 均值 {st.mean(vals):+.3f}, 正比例 "
                  f"{sum(1 for v in vals if v > 0)}/{len(vals)}")
    return rows


# ============================================================
#  main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--stage", type=int, default=2,
                    help="1=fixed 基准选择, 2=初筛, 3=八种子确认, 4=泛化, 5=敏感性")
    ap.add_argument("--ref-dir", action="append", default=[],
                    help="参考目录 (如 s1/s2 复用种子), 与 --dir 合并分析")
    ap.add_argument("--best-fix", type=float, default=None,
                    help="阶段 2/3/4 的最佳 fixed β (从阶段 1 分析得到)")
    ap.add_argument("--risk-a", default=None, help="阶段 3: 初筛候选 key (如 'risk(λs=0.5,β∈[0.3,0.65])')")
    ap.add_argument("--risk-b", default=None)
    args = ap.parse_args()

    runs = load_runs(args.dir + args.ref_dir)
    ms = [run_metrics(z, parse_config(z)) for _, z in runs]
    ms = [m for m in ms if m["dataset"]]

    groups = {}
    for m in ms:
        groups.setdefault(config_key(m), []).append(m)

    rows = group_report(groups)
    print_report(rows, f"配对增益 (raw 同 run, 主指标=ckp20 最终准确率)")

    # 阶段判定
    report = {"stage": args.stage, "rows": []}
    if args.stage == 1:
        deter_bad = determinism_check(ms)
        report["determinism_ok"] = not deter_bad
        best = stage1_pick_best_fixed(rows)
        if best:
            print(f"\n[阶段1] 最佳 fixed EMA: {best['label']}  Δmean={best['d_mean']:+.3f}pp")
            report["best_fixed"] = best["label"]
        if deter_bad:
            print("\n*** 确定性校验失败: 按计划停止实验, 排查确定性后再继续。")
            return

    elif args.stage == 2:
        bf = args.best_fix
        if bf is None:
            print("\n[阶段2] 需要 --best-fix (阶段 1 选择的最佳 fixed β)")
        else:
            bfr = next((r for r in rows if r["mode"] == "fixed" and abs(r["beta"] - bf) < 1e-9), None)
            bf_mean = bfr["d_mean"] if bfr else 0.0
            print(f"\n[阶段2] 晋级判定 (基准 best fixed Δmean={bf_mean:+.3f}pp, 阈值 0.8pp/2-3 正/−0.1pp):")
            verdicts = stage2_screen(rows, bf_mean)
            for r, a, b, c, ok in verdicts:
                flag = "晋级" if ok else "淘汰"
                print(f"  {r['label']:<26} Δ={r['d_mean']:+6.3f} 正={r['n_pos']}/{r['n_seeds']} "
                      f"vsF={r['d_mean']-bf_mean:+6.3f} -> {flag}  "
                      f"[Δ≥0.8:{a} 正≥2/3:{b} vsF≥−0.1:{c}]")
            passed = [r for r, a, b, c, ok in verdicts if ok and r["mode"] == "risk"]
            if not passed:
                print("\n[阶段2] 所有完整 risk 配置未晋级 —— 按计划停止, RA-CMA 不作创新点。")
                report["passed"] = []
            else:
                passed.sort(key=lambda r: -r["d_mean"])
                print(f"\n[阶段2] 晋级完整 risk 配置 (按 Δ 排序): "
                      + ", ".join(f"{r['label']}({r['d_mean']:+.2f})" for r in passed))
                top = passed[:2]
                print(f"进入阶段 3 的两个配置: {top[0]['label']}, {top[1]['label']}")
                report["passed"] = [r["label"] for r in passed]
                report["top2"] = [r["label"] for r in top]

    elif args.stage == 3:
        if not args.risk_a or not args.risk_b or args.best_fix is None:
            print("[阶段3] 需要 --risk-a --risk-b --best-fix")
        else:
            bfr = next((r for r in rows if r["mode"] == "fixed" and abs(r["beta"] - args.best_fix) < 1e-9), None)
            bf_mean = bfr["d_mean"] if bfr else 0.0
            stl = next((r for r in rows if r["mode"] == "stale"), None)
            cnf = next((r for r in rows if r["mode"] == "conflict"), None)
            stl_key = stl["key"] if stl else None
            cnf_key = cnf["key"] if cnf else None
            ra = next((r for r in rows if r["label"] == args.risk_a), None)
            rb = next((r for r in rows if r["label"] == args.risk_b), None)
            print(f"\n[阶段3] 八种子确认 (基准 fixed Δ={bf_mean:+.3f}pp, "
                  f"STL Δ={stl['d_mean']:+.3f}pp, CNF Δ={cnf['d_mean']:+.3f}pp):")
            for r, name in ((ra, args.risk_a), (rb, args.risk_b)):
                if r is None:
                    print(f"  {name}: 配置缺失!")
                    continue
                v = stage3_confirm([r], [r["key"]], bf_mean, stl_key, cnf_key)[0]
                lab, ok, det = v[0], v[2], v[3]
                if det is False:
                    print(f"  {lab}: 配置缺失 (判定依赖 STL/CNF/best fixed, 缺一不可)")
                else:
                    print(f"  {lab:<26} Δ={r['d_mean']:+6.3f} 正={r['n_pos']}/{r['n_seeds']} "
                          f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                          f"vsF={r['d_mean']-bf_mean:+6.3f} -> {'满足' if ok else '不满足'}")
                    print(f"       [Δ≥1.0:{det['mean']} 正≥6/8:{det['pos']} CI>0:{det['ci']} "
                          f"vsF≥0.3:{det['fixed']} >STL:{det['vs_stl']} >CNF:{det['vs_cnf']}]")
            report["best_fixed_mean"] = bf_mean

    elif args.stage == 4:
        res = stage4_generalize(rows, args.best_fix or float("nan"))
        if res is None:
            print("\n[阶段4] 缺少 risk 配置 runs")
        else:
            o = res["ok"]
            print("\n[阶段4] 通过标准判定:")
            for ds, v in res["per_ds"].items():
                print(f"  {ds:<14} Non-IID Δ={st.mean(v['ni']):+.3f} (n={len(v['ni'])})  "
                      f"IID Δ={st.mean(v['iid']):+.3f}  fixedΔ={st.mean(v['f_ni'] or [0]):+.3f}")
            print(f"  Non-IID 宏平均: {res['macro_ni']:+.3f}pp (≥0.5)  {'OK' if o['macro_ni'] else 'FAIL'}")
            print(f"  配对种子正比例: {res['ratio']:.1%} (≥70%)    {'OK' if o['ratio'] else 'FAIL'}")
            print(f"  任一数据集下限: {res['ds_min']:+.3f}pp (≥−0.3) {'OK' if o['ds_min'] else 'FAIL'}")
            print(f"  IID 宏平均: {res['macro_iid']:+.3f}pp (≥−0.2)   {'OK' if o['macro_iid'] else 'FAIL'}")
            print(f"  优于 fixed 的数据集数: {res['better_fixed']}/3 (≥2)  "
                  f"{'OK' if o['better_fixed'] else 'FAIL'}")
            allok = all(o.values())
            print(f"  => 阶段4 {'全部通过' if allok else '未全部通过'}")
            report["stage4"] = {k: bool(v) for k, v in o.items()}
            report["macro_ni"] = res["macro_ni"]

    elif args.stage == 5:
        corr_analysis([m for m in ms if m["mode"] == "risk"])
        # 敏感性: α/delay 分组汇总 (raw 配对同 run)
        print("\n== 敏感性分组 (最终 risk 配置 Δ 均值) ==")
        for key, grp in sorted(groups.items()):
            m0 = grp[0]
            if m0["mode"] != "risk":
                continue
            ds = [r["delta"] for r in grp if r["delta"] is not None]
            if not ds:
                continue
            try:
                ci = bootstrap_ci(ds)
            except Exception:
                ci = (float("nan"), float("nan"))
            print(f"  α={m0['alpha']:g} max_delay={m0['max_delay']}: "
                  f"Δ={st.mean(ds):+.3f}±{st.stdev(ds):.3f} 正={sum(1 for x in ds if x>0)}/{len(ds)} "
                  f"CI=[{ci[0]:+.3f},{ci[1]:+.3f}]")
        report["sensitivity"] = True

    # 导出
    out_json = os.path.join(args.dir[0], "cma_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n分析导出: {out_json}")


if __name__ == "__main__":
    main()