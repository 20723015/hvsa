"""明文基线竞争表统计 (盲审补全执行计划 阶段 6.1).

克隆 thesis_stats.py 的配对统计机制 (mean±sd / 95% CI / Cohen's dz /
正种子数 / 稳定提升判定), 新增:
  1. **等客户端更新量**比较模式: x 轴 = npz checkpoint_Ns 累积曲线
     (与等时间 tick 模式并列, 二者都是计划要求的验收口径);
  2. 跨 root 汇入: results_plain_bench 的三臂 + results_thesis_core 的
     G3v (→ ours_plain) 与 G1 (→ g1_ref) 复用臂 (阶段 0.2 对拍通过后)。

公平性口径 (PLAIN_BENCH_ADAPTATIONS.md §4): ours_plain 为冻结仿真器组级
轨迹, 三基线为逐客户端轨迹; G1 的更新轴由"每轮 20 客户端"解析合成
(其 npz 无 checkpoint_Ns, 轮语义下每轮全量聚合 20 用户)。

用法:
  python plain_bench_stats.py --allow-partial      # 批跑完成后去掉开关
  python plain_bench_stats.py --roots results_plain_bench results_thesis_core
输出: <out>/plain_bench_stats.{md,json}
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import thesis_stats as ts

ARMS = ["fedbuff", "mafsa", "dvafl", "ours_plain", "g1_ref"]
ARM_LABELS = {
    "fedbuff": "FedBuff(逐客户端缓冲)",
    "mafsa": "MAFSA(掉线缓存+复用)",
    "dvafl": "DVAFL(版本控制补偿)",
    "ours_plain": "Ours-Plain(G3v verify-off)",
    "g1_ref": "G1 FedAvg(同步参考)",
}
# 竞争比较: 主方法 vs 各基线 (基准臂 = fedbuff, 与 thesis_stats 的 G2 角色
# 对应 —— 规范异步基线); G1 只进主表与等更新量表 (无 tick 语义)。
COMPARISONS = [
    ("ours_plain", "fedbuff", "Ours−FedBuff"),
    ("ours_plain", "mafsa", "Ours−MAFSA"),
    ("ours_plain", "dvafl", "Ours−DVAFL"),
    ("dvafl", "fedbuff", "DVAFL−FedBuff"),
]
DATASETS = ts.DATASETS
PARTS = ts.PARTS
ALIAS = {"G3v": "ours_plain", "G1": "g1_ref"}   # thesis_core root 臂名映射
TOTAL_USERS = 20


def load_runs_multi(roots):
    runs = []
    for root in roots:
        for r in ts.load_runs(root):
            r = dict(r)
            r['arm'] = ALIAS.get(r['arm'], r['arm'])
            r['root'] = os.path.basename(root)
            runs.append(r)
    # 附加等更新量轴: checkpoint_Ns 累积 (G1 缺失时按每轮 20 用户合成)
    for r in runs:
        if 'npz_path' not in r or not r.get('npz_path'):
            continue
        try:
            d = np.load(r['npz_path'], allow_pickle=True)
            ns = np.asarray(d.get('checkpoint_Ns'), dtype=np.int64)
        except Exception:  # noqa: BLE001
            ns = np.array([], dtype=np.int64)
        if len(ns) != len(r['acc']):
            # G1 (fedavg_point2): 无 checkpoint_Ns, 每轮聚合全部 20 用户
            if r['arm'] == 'g1_ref':
                ns = np.full(len(r['acc']), TOTAL_USERS, dtype=np.int64)
            else:
                ns = np.ones(len(r['acc']), dtype=np.int64)
        r['Ns'] = ns
        r['N_cum'] = np.cumsum(ns)
        r['updates'] = int(ns.sum())
    return runs


def acc_at_budget_updates(run, budget_updates):
    """等更新量口径: 累计采纳用户更新 ≤ budget 的最后精度 (LOCF); 截断标记."""
    ncum, acc = run['N_cum'], run['acc']
    mask = ncum <= budget_updates
    if not mask.any():
        return float(acc[0]), True
    idx = int(np.nonzero(mask)[0][-1])
    censored = bool(ncum[-1] > budget_updates)
    return float(acc[idx]), censored


def equal_compute_stats(by, ds, part, arms=ARMS, budget_arm='ours_plain',
                        base_arm='fedbuff'):
    """以同 seed 的 budget_arm 累积更新数为预算, 比较各臂 vs 基准臂."""
    seeds = sorted(by.get((budget_arm, ds, part), {}))
    if not seeds:
        return {}, {}
    out, budget, values, censored = {}, {}, {}, {}
    for s in seeds:
        budget[s] = int(by[(budget_arm, ds, part)][s]['N_cum'][-1])
    for arm in arms:
        if (arm, ds, part) not in by:
            continue
        vals, cens_seeds = {}, []
        for s in seeds:
            if s not in by[(arm, ds, part)]:
                continue
            v, cens = acc_at_budget_updates(by[(arm, ds, part)][s], budget[s])
            vals[s] = v
            if cens:
                cens_seeds.append(s)
        values[arm] = vals
        censored[arm] = cens_seeds
    base_vals = values.get(base_arm, {})
    for arm in arms:
        if arm == base_arm or arm not in values:
            continue
        st = ts.paired_stats(values[arm], base_vals)
        if st:
            common = st['seeds']
            st['mean_at_budget'] = float(np.mean([values[arm][s]
                                                  for s in common]))
            st['base_mean_at_budget'] = float(np.mean([base_vals[s]
                                                       for s in common]))
            st['censored_seeds'] = censored[arm]
            out[arm] = st
    return out, budget


def build_report(by, runs, allow_partial=False):
    lines = ["# 明文基线竞争表统计 (plain-bench)", ""]
    roots = sorted({r.get('root') for r in runs if r.get('root')})
    lines.append(f"- 纳入运行: {len(runs)} 次 (roots: {', '.join(roots)})")
    det = {bool(r.get('deterministic')) for r in runs}
    lines.append(f"- 严格确定性: {'全部开启' if det == {True} else f'异常: {det}'}")
    lines.append(f"- 更新轴: checkpoint_Ns 累积 (g1_ref 按 每轮{TOTAL_USERS}用户 合成)")
    lines.append("")
    json_out = {'runs': len(runs), 'main_table': {}, 'comparisons': {},
                'equal_time': {}, 'equal_compute': {},
                'equal_time_budget': {}, 'equal_compute_budget': {}}

    # 1. 竞争主表
    lines.append("## 1. 竞争主表 (最终准确率 %, mean ± sample std)")
    lines.append("")
    lines.append("| 数据集 | 分布 | " +
                 " | ".join(ARM_LABELS.get(a, a) for a in ARMS) + " |")
    lines.append("|---|---|" + "---|" * len(ARMS))
    for ds in DATASETS:
        for part in PARTS:
            cells = []
            for arm in ARMS:
                st = ts.arm_cell_stats(by, arm, ds, part)
                if st is None:
                    cells.append('—')
                else:
                    flag = '' if (st['n'] >= 5 or allow_partial) else f" ⚠n={st['n']}"
                    cells.append(f"{st['mean']:.2f} ± {st['sd']:.2f}{flag}")
                    json_out['main_table'][f"{arm}|{ds}|{part}"] = st
            lines.append(f"| {ds} | {part} | " + " | ".join(cells) + " |")
    lines.append("")

    # 2. 同 seed 配对比较 (最终精度口径)
    lines.append("## 2. 竞争配对比较 (同 seed; 95% CI / dz / 正种子数)")
    lines.append("")
    lines.append("| 比较 | 数据集 | 分布 | Δmean | 95% CI | dz | 正种子 | n | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for hi, lo, short in COMPARISONS:
        for ds in DATASETS:
            for part in PARTS:
                st = ts.paired_stats(
                    {s: float(by[(hi, ds, part)][s]['acc'][-1])
                     for s in by.get((hi, ds, part), {})},
                    {s: float(by[(lo, ds, part)][s]['acc'][-1])
                     for s in by.get((lo, ds, part), {})})
                lines.append(
                    f"| {short} | {ds} | {part} | "
                    + ("— | — | — | — | — | 无数据 |" if st is None else
                       f"{st['mean_diff']:+.2f} | "
                       f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | "
                       f"{st['dz']:+.2f} | {st['positives']}/{st['n']} | "
                       f"{st['n']} | {ts._verdict(st)} |"))
                json_out['comparisons'][f"{hi}-{lo}|{ds}|{part}"] = st
    lines.append("")

    # 3. 等更新量比较 (计划 6.1 新模式)
    lines.append("## 3. 等客户端更新量比较 (预算 = 同 seed ours_plain 的累计采纳更新; 基准 = fedbuff)")
    lines.append("")
    lines.append("| 数据集 | 分布 | 臂 | 臂@N | fedbuff@N | Δmean vs fedbuff | 95% CI | dz | 正种子 | 截断 seed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for part in PARTS:
            eq, budget = equal_compute_stats(by, ds, part)
            json_out['equal_compute_budget'][f"{ds}|{part}"] = budget
            for arm, st in sorted(eq.items()):
                lines.append(
                    f"| {ds} | {part} | {arm} | {st['mean_at_budget']:.2f} | "
                    f"{st['base_mean_at_budget']:.2f} | {st['mean_diff']:+.2f} | "
                    f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | "
                    f"{st['dz']:+.2f} | {st['positives']}/{st['n']} | "
                    f"{len(st['censored_seeds'])} |")
                json_out['equal_compute'][f"{arm}|{ds}|{part}"] = st
    lines.append("")

    # 4. 等时间比较 (预算 = 同 seed ours_plain 末次 tick; g1_ref 无 tick 语义不进)
    lines.append("## 4. 等时间比较 (预算 = 同 seed ours_plain 末次 tick; 基准 = fedbuff)")
    lines.append("")
    lines.append("| 数据集 | 分布 | 臂 | 臂@T | fedbuff@T | Δmean vs fedbuff | 95% CI | dz | 正种子 | 截断 seed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for part in PARTS:
            eq, budget = ts.equal_time_stats(
                by, ds, part, arms=[a for a in ARMS if a != 'g1_ref'],
                budget_arm='ours_plain', base_arm='fedbuff')
            json_out['equal_time_budget'][f"{ds}|{part}"] = budget
            for arm, st in sorted(eq.items()):
                lines.append(
                    f"| {ds} | {part} | {arm} | {st['mean_at_budget']:.2f} | "
                    f"{st['base_mean_at_budget']:.2f} | {st['mean_diff']:+.2f} | "
                    f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | "
                    f"{st['dz']:+.2f} | {st['positives']}/{st['n']} | "
                    f"{len(st['censored_seeds'])} |")
                json_out['equal_time'][f"{arm}|{ds}|{part}"] = st
    lines.append("")

    # 5. 公平性口径 (逐 run 更新量对照)
    lines.append("## 5. 更新量对照 (每 run 采纳用户更新总数, mean±sd)")
    lines.append("")
    lines.append("| 数据集 | 分布 | " +
                 " | ".join(ARM_LABELS.get(a, a) for a in ARMS) + " |")
    lines.append("|---|---|" + "---|" * len(ARMS))
    for ds in DATASETS:
        for part in PARTS:
            cells = []
            for arm in ARMS:
                seeds = by.get((arm, ds, part), {})
                if not seeds:
                    cells.append('—')
                    continue
                ups = [seeds[s]['updates'] for s in sorted(seeds)]
                cells.append(f"{np.mean(ups):.0f} ± {np.std(ups, ddof=1):.0f}")
            lines.append(f"| {ds} | {part} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> 口径: ours_plain 为冻结仿真器组级轨迹 (掩码域量化), 三基线为"
                 "逐客户端明文轨迹; 两者到达-聚合节奏由各自调度器生成 (被比较对象),"
                 "等更新量/等时间两轴分别对齐计算与调度预算。详见 "
                 "docs/PLAIN_BENCH_ADAPTATIONS.md。")
    return "\n".join(lines), json_out


def main():
    ap = argparse.ArgumentParser(description="明文基线竞争表统计")
    ap.add_argument("--roots", nargs="+",
                    default=["results_plain_bench", "results_thesis_core"])
    ap.add_argument("--out", default="results_plain_bench/stats")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_runs_multi(args.roots)
    have_plain = {r['arm'] for r in runs}
    missing = [a for a in ('fedbuff', 'mafsa', 'dvafl') if a not in have_plain]
    if missing:
        print(f"明文基线臂缺失 {missing}; 先运行 python run_plain_bench.py run")
        sys.exit(1)
    by = ts.index_runs(runs)
    issues = []
    for arm in ARMS:
        for ds in DATASETS:
            for part in PARTS:
                got = set(by.get((arm, ds, part), {}))
                if got != set(ts.__dict__.get('_', ())) and got != {42, 43, 44, 45, 46}:
                    issues.append(f"{arm}/{ds}/{part}: seeds={sorted(got)}")
    if issues and not args.allow_partial:
        print("矩阵不完整 (可用 --allow-partial 跳过):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(2)
    report, json_out = build_report(by, runs, args.allow_partial)
    md = os.path.join(args.out, "plain_bench_stats_report.md")
    with open(md, 'w', encoding='utf-8') as f:
        f.write(report)
    js = os.path.join(args.out, "plain_bench_stats.json")
    with open(js, 'w', encoding='utf-8') as f:
        json.dump(ts._jsonable(json_out), f, ensure_ascii=False, indent=2,
                  allow_nan=False)
    print(f"报告: {md}\n统计: {js}")


if __name__ == "__main__":
    main()
