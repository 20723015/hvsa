"""论文主表统计与图表 (四周补强计划第 3 周).

输入: results_thesis_core/ (run_thesis_core.py 产物; 也支持 --root 指向
      任意含 <tag>/point2_meta_*.json 的目录树)
输出 (默认写到 <root>/stats/):
  - thesis_stats_report.md   主结果表 / 配对比较表 / 等时间比较表 / 消融表
  - thesis_stats.json        机器可读的同一份统计
  - thesis_main_curves.png   六格 accuracy–ticks 曲线 (mean±std 带)
  - thesis_eff_acc.png       精度—效率图 (末次 tick / 本机时间 vs 最终精度)

统计口径 (计划第 3 周):
  - 全部按**同 seed 配对**; 报告 mean±sample std (ddof=1)、配对差均值、
    95% CI (t 分布)、Cohen's dz (配对标准化效应量) 与正种子数;
  - 三个比较是算法机制阶梯: G3−G2 (半异步), G3v−G3 (后置补偿),
    G5−G3v (可选 σ 组件); 第三创新“策略绑定”由攻击证据表支撑;
  - 等时间比较: 以每个 seed 的 G3 完成 tick 为预算, 取各臂在该 tick 之前的
    最后精度 (ticks = 模拟调度时间; 本机墙钟只作计算成本, 非分布式加速);
  - "稳定提升" 判定 (计划第 4 周口径): 配对均值 > 0 且 ≥4/5 seed 为正
    且 95% CI 不跨 0; 其余一律写作"趋势"。

用法:
  python thesis_stats.py                      # 默认 root=results_thesis_core
  python thesis_stats.py --root results_x --out stats_x
  python thesis_stats.py --allow-partial      # 允许 <5 seeds 的部分结果
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

MAIN_ARMS = ["G2", "G3", "G3v", "G5"]
EXTRA_ARMS = ["G1", "GFB", "G3s"]
# (被减臂, 基准臂, 机制简称, 说明)。这里只是算法阶梯消融；论文第三创新
# “策略绑定与可验证执行”由 run_attack_evidence.py 的安全证据支撑。
COMPARISONS = [
    ("G3", "G2", "①半异步", "检查点式组级半异步 vs 组间同步"),
    ("G3v", "G3", "②后置补偿", "Direct-VCM 组级后置补偿 vs 无补偿"),
    ("G5", "G3v", "可选σ组件", "σ 幂衰减陈旧度加权 vs 无 σ"),
]
ARM_LABELS = {
    "G1": "G1 FedAvg(同步参考)", "G2": "G2 组间同步(K=5)",
    "G3": "G3 半异步①", "G3v": "G3v +Direct-VCM(主方法)",
    "G5": "G5 +VCM+σ", "G3s": "G3s 仅σ(消融)",
    "GFB": "GFB Group-buffer K=1消融",
}
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
PARTS = ["IID", "NI"]

# 双尾 95% t 临界值 (df=1..10); df>10 用 1.96 (计划的 seeds=5 → df=4 → 2.776)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t_crit(n):
    if n <= 0:
        return float('nan')
    try:
        from scipy import stats as _s  # 有 scipy 时用精确分位
        return float(_s.t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001
        df = n - 1
        return _T95.get(df, 1.96)


def _acc_column(acc_raw):
    """main_point2 的 acc_history 是 (checkpoint_index, accuracy) 两列; 归一为精度一维."""
    if acc_raw is None:
        return None
    a = np.asarray(acc_raw, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] >= 2:
        return a[:, 1]
    return a.reshape(-1)


def load_runs(root):
    """读取一次运行的记录 (manifest 优先, 否则扫描 <tag>/point2_meta_*.json)."""
    manifest = os.path.join(root, "thesis_core_manifest.jsonl")
    rows = []
    if os.path.exists(manifest):
        with open(manifest, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        for meta_path in sorted(glob.glob(
                os.path.join(root, "*", "point2_meta_*.json"))):
            try:
                with open(meta_path, encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            tag = os.path.basename(os.path.dirname(meta_path))
            parts = tag.split('_')
            # tag 约定: {arm}_{dataset}_{part}_s{seed}[_rep]
            if len(parts) < 4 or not parts[3].startswith('s'):
                continue
            rows.append({
                'tag': tag, 'arm': parts[0], 'dataset': parts[1],
                'part': parts[2], 'seed': int(parts[3][1:]),
                'rep': tag.endswith('_rep'), 'rc': 0,
                'run_dir': os.path.dirname(meta_path),
                'config_hash': meta.get('config_hash'),
                'final_acc': meta.get('final_acc'),
                'wall_clock': meta.get('wall_clock'),
                'deterministic': meta.get('deterministic'),
                'adopted_packages': meta.get('adopted_packages'),
                'adopted_user_updates': meta.get('adopted_user_updates'),
                'group_packages_sealed': meta.get('group_packages_sealed'),
                'verify_seconds_total': meta.get('verify_seconds_total'),
                'comm_total_bytes': meta.get('comm_total_bytes'),
            })

    runs = []
    for r in rows:
        if r.get('rc') not in (0, None) or r.get('rep'):
            continue
        run_dir = r.get('run_dir') or os.path.join(root, r['tag'])
        npz_hits = glob.glob(os.path.join(
            run_dir, f"point2_data_{r['tag']}.npz"))
        if not npz_hits:
            continue
        try:
            d = np.load(npz_hits[0], allow_pickle=True)
        except Exception:  # noqa: BLE001
            continue
        acc = d.get('acc_history')
        ticks = d.get('checkpoint_ticks')
        acc = _acc_column(acc)
        if acc is None or len(acc) == 0:
            continue
        r = dict(r)
        r['acc'] = acc
        r['ticks'] = (np.asarray(ticks, dtype=np.int64)
                      if ticks is not None and len(ticks) == len(acc)
                      else np.arange(1, len(acc) + 1))
        r['npz_path'] = npz_hits[0]
        runs.append(r)
    return runs


def index_runs(runs):
    """(arm, dataset, part) -> {seed: run}"""
    by = defaultdict(dict)
    for r in runs:
        by[(r['arm'], r['dataset'], r['part'])][r['seed']] = r
    return by


def paired_stats(a, b):
    """配对差 a−b 的统计量 (dict); 两者 seed 集合需一致."""
    keys = sorted(set(a) & set(b))
    diffs = np.array([a[k] - b[k] for k in keys], dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return None
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tc = _t_crit(n)
    ci = (mean - tc * se, mean + tc * se) if n > 1 else (mean, mean)
    return {
        'n': n, 'seeds': keys, 'mean_diff': mean, 'sd_diff': sd,
        'ci95': [float(ci[0]), float(ci[1])],
        'dz': (mean / sd if sd > 0 else float('nan')),
        'positives': int((diffs > 0).sum()),
        'diffs': [float(x) for x in diffs],
    }


def _verdict(st, n_expected=5):
    """计划第 4 周口径: 稳定提升 / 趋势."""
    if st is None:
        return '无数据'
    if st['n'] != n_expected:
        return f"样本不足(n={st['n']}, 仅趋势)"
    stable = (st['mean_diff'] > 0 and st['positives'] >= max(1, n_expected - 1)
              and st['ci95'][0] > 0)
    if stable:
        return '稳定提升'
    if st['mean_diff'] > 0:
        return '趋势(正)'
    return '无提升/负'


def arm_cell_stats(by, arm, ds, part):
    seeds = by.get((arm, ds, part), {})
    if not seeds:
        return None
    accs = np.array([seeds[s]['acc'][-1] for s in sorted(seeds)],
                    dtype=np.float64)
    ticks = np.array([int(seeds[s]['ticks'][-1]) for s in sorted(seeds)],
                     dtype=np.float64)
    walls = np.array([float(seeds[s].get('wall_clock') or 0.0)
                      for s in sorted(seeds)], dtype=np.float64)
    return {'n': len(accs), 'mean': float(accs.mean()),
            'sd': float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            'accs': [float(a) for a in accs],
            'tick_mean': float(ticks.mean()),
            'wall_mean': float(walls.mean())}


def final_acc_by_seed(by, arm, ds, part):
    return {s: float(by[(arm, ds, part)][s]['acc'][-1])
            for s in by.get((arm, ds, part), {})}


def acc_at_budget(run, budget_tick):
    """等时间口径: 该臂在预算 tick 之前的最后精度; 未达预算则用自身末值."""
    ticks, acc = run['ticks'], run['acc']
    mask = ticks <= budget_tick
    if not mask.any():
        return float(acc[0]), True
    idx = int(np.nonzero(mask)[0][-1])
    censored = bool(ticks[-1] > budget_tick)
    return float(acc[idx]), censored


def equal_time_stats(by, ds, part, arms=MAIN_ARMS, budget_arm='G3',
                     base_arm='G2'):
    """以 G3 末次 tick 为预算，直接比较各臂与同步 G2 的配对差。"""
    seeds = sorted(by.get((budget_arm, ds, part), {}))
    out = {}
    budget = {}
    for s in seeds:
        budget[s] = int(by[(budget_arm, ds, part)][s]['ticks'][-1])

    values, censored = {}, {}
    for arm in arms:
        if (arm, ds, part) not in by:
            continue
        vals = {}
        censored_seeds = []
        for s in seeds:
            if s not in by[(arm, ds, part)]:
                continue
            v, cens = acc_at_budget(by[(arm, ds, part)][s], budget[s])
            vals[s] = v
            if cens:
                censored_seeds.append(s)
        values[arm] = vals
        censored[arm] = censored_seeds

    base_vals = values.get(base_arm, {})
    for arm in arms:
        if arm == base_arm or arm not in values:
            continue
        st = paired_stats(values[arm], base_vals)
        if st:
            common = st['seeds']
            st['mean_at_budget'] = float(np.mean([values[arm][s]
                                                   for s in common]))
            st['base_mean_at_budget'] = float(np.mean([base_vals[s]
                                                        for s in common]))
            st['censored_seeds'] = censored[arm]
            out[arm] = st
    return out, budget


def completeness_issues(by, expected_seeds=(42, 43, 44, 45, 46)):
    """正式主矩阵必须精确具备 4 臂×6 格×5 seeds。"""
    expected = set(expected_seeds)
    issues = []
    for arm in MAIN_ARMS:
        for ds in DATASETS:
            for part in PARTS:
                got = set(by.get((arm, ds, part), {}))
                if got != expected:
                    issues.append(
                        f"{arm}/{ds}/{part}: got={sorted(got)}, "
                        f"expected={sorted(expected)}")
    return issues


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 中文字形: 优先系统 CJK 字体, 避免图内标签变成方框
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
                 'Source Han Sans SC', 'DejaVu Sans'):
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            break
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _jsonable(obj):
    """NaN/Inf → None (保证 JSON 合法, 便于下游脚本解析)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return _jsonable(float(obj))
    return obj


def plot_curves(by, out_png, arms=MAIN_ARMS):
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=False)
    for j, part in enumerate(PARTS):
        for i, ds in enumerate(DATASETS):
            ax = axes[j][i]
            for arm in arms:
                seeds = sorted(by.get((arm, ds, part), {}))
                if not seeds:
                    continue
                # 共同 tick 网格上作阶梯式 last-observation-carried-forward；
                # checkpoint 之间模型不变化，不能用线性插值制造虚假精度轨迹。
                grid = np.unique(np.concatenate(
                    [by[(arm, ds, part)][s]['ticks'] for s in seeds]))
                curves = []
                for s in seeds:
                    r = by[(arm, ds, part)][s]
                    idx = np.searchsorted(r['ticks'], grid, side='right') - 1
                    curve = np.full(len(grid), np.nan, dtype=np.float64)
                    valid = idx >= 0
                    curve[valid] = r['acc'][idx[valid]]
                    curves.append(curve)
                curves = np.array(curves, dtype=np.float64)
                mean = np.nanmean(curves, axis=0)
                sd = (np.nanstd(curves, axis=0, ddof=1)
                      if len(seeds) > 1 else np.zeros_like(mean))
                ax.plot(grid, mean, label=ARM_LABELS.get(arm, arm), linewidth=1.6)
                if len(seeds) > 1:
                    ax.fill_between(grid, mean - sd, mean + sd, alpha=0.15)
            # 同步参考 (无 tick 语义): 画最终精度的水平虚线作参照
            g1 = by.get(("G1", ds, part), {})
            if g1:
                g1_acc = float(np.mean([g1[s]['acc'][-1] for s in g1]))
                ax.axhline(g1_acc, color='gray', linestyle='--', linewidth=1.0,
                           label='G1 FedAvg(参考,末次精度)')
            ax.set_title(f"{ds} / {part}")
            ax.set_xlabel('模拟调度时间 (ticks)')
            ax.set_ylabel('测试准确率 (%)')
            ax.grid(True, linestyle='--', alpha=0.5)
    axes[0][0].legend(fontsize=8)
    fig.suptitle('主表 accuracy–ticks 曲线 (mean±sample std, 同 seed 配对)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"图表已保存: {out_png}")


def plot_eff_acc(by, out_png, arms=MAIN_ARMS + ["G1", "GFB"]):
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    markers = {'MNIST': 'o', 'FashionMNIST': 's', 'CIFAR10': '^'}
    colors = {'IID': 'tab:blue', 'NI': 'tab:red'}
    for ax, xkey, xlabel in ((axes[0], 'tick_mean', '平均最终 tick (模拟调度时间)'),
                             (axes[1], 'wall_mean', '平均本机运行时间 (s, 非分布式墙钟)')):
        for arm in arms:
            if xkey == 'tick_mean' and arm == 'G1':
                # FedAvg 的 round 索引不是本模拟器 tick，禁止画在同一效率坐标。
                continue
            for ds in DATASETS:
                for part in PARTS:
                    st = arm_cell_stats(by, arm, ds, part)
                    if not st:
                        continue
                    ax.errorbar(st[xkey], st['mean'], yerr=st['sd'], fmt=markers[ds],
                                color=colors[part], alpha=0.85, markersize=6,
                                capsize=3, linestyle='none')
                    ax.annotate(arm, (st[xkey], st['mean']), fontsize=6,
                                xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('最终准确率 (%)')
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[0].set_title('精度—效率 (按模拟 tick)')
    axes[1].set_title('精度—计算成本 (本机时间, 不作加速声明)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"图表已保存: {out_png}")


def build_report(by, runs, allow_partial=False):
    lines = ["# 研究点2 主表统计 (thesis-core)", ""]
    n_runs = len(runs)
    det = {bool(r.get('deterministic')) for r in runs}
    hashes = {r.get('config_hash') for r in runs if r.get('config_hash')}
    lines.append(f"- 纳入运行: {n_runs} 次 (已排除 repeat 与失败运行)")
    lines.append(f"- 严格确定性 (--deterministic): "
                 f"{'全部开启' if det == {True} else f'存在未开启: {det}'}")
    lines.append(f"- 配置摘要 (config_hash) 唯一值: {len(hashes)}")
    lines.append("")

    # 1. 主结果表
    lines.append("## 1. 主结果表 (最终准确率 %, mean ± sample std)")
    lines.append("")
    lines.append("| 数据集 | 分布 | " +
                 " | ".join(ARM_LABELS.get(a, a) for a in MAIN_ARMS + EXTRA_ARMS) + " |")
    lines.append("|---|---|" + "---|" * len(MAIN_ARMS + EXTRA_ARMS))
    for ds in DATASETS:
        for part in PARTS:
            cells = []
            for arm in MAIN_ARMS + EXTRA_ARMS:
                st = arm_cell_stats(by, arm, ds, part)
                if st is None:
                    cells.append('—')
                else:
                    flag = '' if (st['n'] >= 5 or allow_partial) else f" ⚠n={st['n']}"
                    cells.append(f"{st['mean']:.2f} ± {st['sd']:.2f}{flag}")
            lines.append(f"| {ds} | {part} | " + " | ".join(cells) + " |")
    lines.append("")

    # 2. 配对比较 (三个贡献)
    lines.append("## 2. 算法机制阶梯的同 seed 配对比较 (95% CI / Cohen's dz / 正种子数)")
    lines.append("")
    lines.append("| 比较 | 数据集 | 分布 | Δmean | 95% CI | dz | 正种子 | n | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    json_out = {'runs': n_runs, 'main_table': {}, 'comparisons': {},
                'equal_time': {}, 'ablation': {}, 'equal_time_budget': {}}
    for hi, lo, short, desc in COMPARISONS:
        for ds in DATASETS:
            for part in PARTS:
                st = paired_stats(final_acc_by_seed(by, hi, ds, part),
                                  final_acc_by_seed(by, lo, ds, part))
                lines.append(
                    f"| {short}: {hi}−{lo} | {ds} | {part} | "
                    + ("— | — | — | — | — | 无数据 |" if st is None else
                       f"{st['mean_diff']:+.2f} | "
                       f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | "
                       f"{st['dz']:+.2f} | {st['positives']}/{st['n']} | {st['n']} | "
                       f"{_verdict(st)} |"))
                json_out['comparisons'][f"{hi}-{lo}|{ds}|{part}"] = st
    lines.append("")

    # 3. 等时间比较 (预算 = G3 末次 tick)
    lines.append("## 3. 等时间比较 (预算 = 同 seed 的 G3 末次 tick; 基准 = 同步 G2)")
    lines.append("")
    lines.append("| 数据集 | 分布 | 臂 | 臂@T | G2@T | Δmean vs G2 | 95% CI | dz | 正种子 | 截断 seed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for part in PARTS:
            eq, budget = equal_time_stats(by, ds, part)
            json_out['equal_time_budget'][f"{ds}|{part}"] = budget
            if not eq:
                lines.append(f"| {ds} | {part} | — | — | — | — | — | — |")
                continue
            for arm, st in sorted(eq.items()):
                lines.append(
                    f"| {ds} | {part} | {arm} | {st['mean_at_budget']:.2f} | "
                    f"{st['base_mean_at_budget']:.2f} | {st['mean_diff']:+.2f} | "
                    f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | "
                    f"{st['dz']:+.2f} | {st['positives']}/{st['n']} | "
                    f"{len(st['censored_seeds'])} |")
                json_out['equal_time'][f"{arm}|{ds}|{part}"] = st
    lines.append("")

    # 4. 算法消融表；不得把可选 σ 误写成论文第三创新。
    lines.append("## 4. 算法机制消融表 (取自第 2 节配对差)")
    lines.append("")
    lines.append("| 贡献 | 比较 | 平均 Δ (全格均值) | 正号格数 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for hi, lo, short, desc in COMPARISONS:
        vals = []
        for ds in DATASETS:
            for part in PARTS:
                st = paired_stats(final_acc_by_seed(by, hi, ds, part),
                                  final_acc_by_seed(by, lo, ds, part))
                if st:
                    vals.append(st['mean_diff'])
        if vals:
            lines.append(f"| {short} | {hi}−{lo} | "
                         f"{np.mean(vals):+.2f} | {sum(1 for v in vals if v > 0)}/{len(vals)} | "
                         f"{desc} |")
        else:
            lines.append(f"| {short} | {hi}−{lo} | — | — | 无数据 |")
    lines.append("")

    # 5. 开销摘要 (执行/采纳更新数、验证耗时、通信量)
    lines.append("## 5. 开销摘要 (执行/采纳更新、验证耗时、通信量)")
    lines.append("")
    lines.append("> 注: 主矩阵各臂为 verify-off (性能口径), 其验证耗时列为 ~0 属预期, "
                 "不得解读为\"验证零开销\"; 验证开销结论以 verify-on 的 "
                 "run_attack_evidence.py 证据批次 (含 ES 验证+vote+TA 授权+提交) 为准。")
    lines.append("")
    lines.append("| 数据集 | 分布 | 臂 | 执行组包 | 采纳组包 | 采纳/执行 | 验证耗时/ckp (s) | 通信量 (MiB/ckp) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for part in PARTS:
            for arm in MAIN_ARMS + ['GFB']:
                seeds = by.get((arm, ds, part), {})
                if not seeds:
                    continue
                sealed = np.mean([r.get('group_packages_sealed') or 0
                                  for r in seeds.values()])
                adopted = np.mean([r.get('adopted_packages') or 0
                                   for r in seeds.values()])
                vsec = np.mean([r.get('verify_seconds_total') or 0
                                for r in seeds.values()])
                nckp = np.mean([len(r['acc']) for r in seeds.values()])
                cb = np.mean([r.get('comm_total_bytes') or 0
                              for r in seeds.values()])
                ratio = (adopted / sealed) if sealed else 0.0
                lines.append(
                    f"| {ds} | {part} | {arm} | {sealed:.0f} | {adopted:.0f} | "
                    f"{ratio:.2f} | {(vsec / max(1, nckp)):.3f} | "
                    f"{(cb / max(1, nckp)) / 1048576:.2f} |")
    lines.append("")
    return "\n".join(lines), json_out


def main():
    ap = argparse.ArgumentParser(description="thesis-core 统计与图表")
    ap.add_argument("--root", default="results_thesis_core")
    ap.add_argument("--out", default=None, help="输出目录 (默认 <root>/stats)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="允许 <5 seeds 的部分结果 (默认仍输出, 但主表标注 ⚠)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.root, "stats")
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(args.root)
    if not runs:
        print(f"没有找到可用运行 (root={args.root}); 先运行 "
              f"python run_thesis_core.py run")
        sys.exit(1)
    by = index_runs(runs)
    issues = completeness_issues(by)
    if issues and not args.allow_partial:
        print("正式主矩阵不完整，拒绝生成可用于论文的统计报告：")
        for issue in issues:
            print(f"  - {issue}")
        print("仅调试部分结果时可显式加 --allow-partial；该模式不会产生“稳定提升”判定。")
        sys.exit(2)
    report, json_out = build_report(by, runs, args.allow_partial)
    md_path = os.path.join(out_dir, "thesis_stats_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    json_path = os.path.join(out_dir, "thesis_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(json_out), f, ensure_ascii=False, indent=2,
                  allow_nan=False)
    print(f"报告已保存: {md_path}")
    print(f"统计已保存: {json_path}")

    present = {r['arm'] for r in runs}
    plot_arms = [a for a in MAIN_ARMS if a in present] or MAIN_ARMS
    try:
        plot_curves(by, os.path.join(out_dir, "thesis_main_curves.png"), plot_arms)
        plot_eff_acc(by, os.path.join(out_dir, "thesis_eff_acc.png"),
                     plot_arms + [a for a in ("G1", "GFB") if a in present])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 绘图失败 (统计报告仍已生成): {e}")


if __name__ == "__main__":
    main()
