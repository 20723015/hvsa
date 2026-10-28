"""UR-VCM 阶段1/阶段2 分析器: 同 seed 独立配对 + 晋级判定 + 门控诊断.

口径 (与实验计划一致):
- 主指标: checkpoint 20 最终准确率, 同 seed 独立 G5 配对 Δ。
- 辅助: last-5 平均增益、跨种子样本 std、正收益比例。
- 晋级: Δmean≥0.8pp, 正≥2/3, 最差≥−0.5pp, last5>0, ticks 不增且无验证失败。
- 并列时选较小 ρ。
- 阶段2 泛化: 两数据集平均损失均 ≥−0.5pp; 至少一个外部数据集 ≥+0.5pp 且 2/3 正。

用法: python analyze_urvcm.py --dir results_urvcm/s1_screen --stage 1
      python analyze_urvcm.py --dir results_urvcm/s2_generalize --stage 2
"""

import argparse
import glob
import json
import math
import os
import re
import statistics as st

import numpy as np

try:
    from collect_point2_results import _parse_namespace_str
except ImportError:
    _parse_namespace_str = None


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


def group_of(c):
    if c.get("vcm_mode") == "trust":
        rho = float(c.get("vcm_trust_ratio", 1.0))
        return f"UR{int(round(rho * 10)):02d}"
    return "G5"


def run_metrics(z, c):
    acc = np.asarray(z["acc_history"], dtype=float)
    last5 = float(np.mean(acc[-5:, 1])) if len(acc) >= 5 else float("nan")
    eta = (np.asarray(z["vcm_trust_eta"], dtype=float)
           if "vcm_trust_eta" in z.files else np.array([]))
    trn = (np.asarray(z["vcm_train_norm"], dtype=float)
           if "vcm_train_norm" in z.files else np.array([]))
    cmp_n = (np.asarray(z["vcm_comp_norm"], dtype=float)
             if "vcm_comp_norm" in z.files else np.array([]))
    ticks = int(z["checkpoint_ticks"][-1]) if len(z["checkpoint_ticks"]) else -1
    verify_total = int(z["verify_total_count"]) if "verify_total_count" in z.files else 0
    verify_rej = (int(z["verification_rejection_count"])
                  if "verification_rejection_count" in z.files else 0)
    return dict(
        group=group_of(c), rho=float(c.get("vcm_trust_ratio", 1.0)),
        dataset=c.get("dataset", ""), seed=int(c.get("seed", -1)),
        final_acc=float(z["final_acc"]), last5=last5, ticks=ticks,
        gate_rate=float((eta < 1.0).mean()) if eta.size else float("nan"),
        eta_mean=float(eta.mean()) if eta.size else float("nan"),
        train_norm=float(trn.mean()) if trn.size else float("nan"),
        comp_norm=float(cmp_n.mean()) if cmp_n.size else float("nan"),
        verify_failed=(verify_rej > 0),
    )


def paired(rows):
    """按 (dataset, group) 与 G5 同 seed 配对."""
    g5 = {(r["dataset"], r["seed"]): r for r in rows if r["group"] == "G5"}
    out = {}
    for r in rows:
        if r["group"] == "G5":
            continue
        base = g5.get((r["dataset"], r["seed"]))
        if base is None:
            continue
        d = dict(r)
        d["delta"] = r["final_acc"] - base["final_acc"]
        d["last5_delta"] = r["last5"] - base["last5"]
        d["ticks_ok"] = (r["ticks"] == base["ticks"])
        out.setdefault((r["dataset"], r["group"]), []).append(d)
    return out


def stage1_verdict(pairs, min_mean=0.8, max_worst=-0.5):
    print(f"\n== 阶段1 晋级判定 (同 seed 独立 G5 配对; 阈值 Δmean≥{min_mean}pp) ==")
    verdicts = []
    for (ds, g), rs in sorted(pairs.items()):
        deltas = [r["delta"] for r in rs]
        mean_d = st.mean(deltas)
        std_d = st.stdev(deltas) if len(deltas) > 1 else float("nan")
        n_pos = sum(1 for x in deltas if x > 0)
        worst = min(deltas)
        last5 = [r["last5_delta"] for r in rs]
        ticks_ok = all(r["ticks_ok"] for r in rs)
        no_vfail = not any(r["verify_failed"] for r in rs)
        ok = dict(mean=mean_d >= min_mean,
                  pos=n_pos >= math.ceil(2 * len(deltas) / 3),
                  worst=worst >= max_worst,
                  last5=st.mean(last5) > 0,
                  ticks=ticks_ok, no_verify_fail=no_vfail)
        passed = all(ok.values())
        verdicts.append((g, passed, ok, mean_d, std_d, n_pos, len(deltas), worst,
                         rs[0]["rho"], st.mean(last5)))
        print(f"  {g:<6} Δmean={mean_d:+.3f}±{std_d:.3f} 正={n_pos}/{len(deltas)} "
              f"最差={worst:+.2f} last5={st.mean(last5):+.3f} "
              f"→ {'晋级' if passed else '淘汰'}")
        print(f"         [Δ≥{min_mean}:{ok['mean']} 正≥2/3:{ok['pos']} "
              f"最差≥{max_worst}:{ok['worst']} last5>0:{ok['last5']} "
              f"ticks不变:{ok['ticks']} 无验证失败:{ok['no_verify_fail']}]")
    # 选优: 晋级者中平均增益最高; 并列取较小 ρ
    cand = [v for v in verdicts if v[1]]
    if cand:
        cand.sort(key=lambda v: (-round(v[3], 6), v[8]))
        print(f"\n胜出配置: {cand[0][0]} (ρ={cand[0][8]:g}, Δmean={cand[0][3]:+.3f})")
    else:
        print("\n无配置晋级 → 记录为负结果, 不进入阶段2")
    return verdicts, (cand[0][0] if cand else None), (cand[0][8] if cand else None)


def stage2_verdict(pairs):
    print("\n== 阶段2 泛化判定 (非 MNIST 数据集, 同 seed 独立 G5 配对) ==")
    per_ds = {}
    for (ds, g), rs in sorted(pairs.items()):
        deltas = [r["delta"] for r in rs]
        per_ds[(ds, g)] = dict(mean=st.mean(deltas),
                               n_pos=sum(1 for x in deltas if x > 0),
                               n=len(deltas), std=st.stdev(deltas) if len(deltas) > 1 else float("nan"))
        print(f"  {ds:<14}{g:<6} Δmean={per_ds[(ds,g)]['mean']:+.3f}"
              f"±{per_ds[(ds,g)]['std']:.3f} 正={per_ds[(ds,g)]['n_pos']}/{per_ds[(ds,g)]['n']}")
    losses_ok = all(v["mean"] >= -0.5 for v in per_ds.values()) if per_ds else False
    gain_ok = any(v["mean"] >= 0.5 and v["n_pos"] >= math.ceil(2 * v["n"] / 3)
                  for v in per_ds.values()) if per_ds else False
    print(f"\n  两数据集平均损失均 ≥−0.5pp: {losses_ok}")
    print(f"  至少一个数据集 ≥+0.5pp 且 2/3 正: {gain_ok}")
    print(f"  → 阶段2 {'通过' if (losses_ok and gain_ok) else '未通过 (记录为负结果)'}")
    return per_ds, bool(losses_ok and gain_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()

    runs = load_runs(args.dir)
    rows = [run_metrics(z, parse_config(z)) for z in runs]
    print(f"{'组':<8}{'ds':<14}{'seed':>5}{'acc':>8}{'last5':>8}{'η均':>7}"
          f"{'门控%':>7}{'‖Δtrain‖':>10}{'‖C‖':>8}{'ticks':>6}")
    for r in sorted(rows, key=lambda x: (x["dataset"], x["group"], x["seed"])):
        print(f"{r['group']:<8}{r['dataset']:<14}{r['seed']:>5}{r['final_acc']:>8.2f}"
              f"{r['last5']:>8.2f}{r['eta_mean']:>7.3f}"
              f"{100*r['gate_rate']:>7.1f}{r['train_norm']:>10.4f}"
              f"{r['comp_norm']:>8.4f}{r['ticks']:>6}")

    pairs = paired(rows)
    report = {"stage": args.stage, "rows": rows}
    if args.stage == 1:
        v, winner, rho = stage1_verdict(pairs)
        report["verdicts"] = [dict(group=x[0], passed=x[1], ok=x[2],
                                   mean=x[3], std=x[4], n_pos=x[5], n=x[6],
                                   worst=x[7], rho=x[8], last5=x[9]) for x in v]
        report["winner"] = winner
        report["winner_rho"] = rho
    else:
        per_ds, passed = stage2_verdict(pairs)
        report["per_dataset"] = {f"{k[0]}|{k[1]}": v for k, v in per_ds.items()}
        report["passed"] = passed

    out = os.path.join(args.dir[0], f"urvcm_s{args.stage}_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n分析导出: {out}")


if __name__ == "__main__":
    main()
