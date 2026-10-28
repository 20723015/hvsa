"""论文表格生成: 统计/证据 JSON → booktabs LaTeX (盲审补全执行计划 阶段 6.2).

单一数据源纪律: 论文表格一律由本脚本从 JSON 产物生成, 禁止手抄 ——
表格数字与 stats JSON / manifest 的三方对账由此按构造成立 (阶段 6.3
的 reconcile 脚本复核)。

表格 → 输入 JSON:
  competitive   plain_bench_stats.json        (5 方法 × 6 单元主表)
  comparisons   plain_bench_stats.json        (竞争配对比较)
  equalcompute  plain_bench_stats.json        (等更新量比较)
  equaltime     plain_bench_stats.json        (等时间比较)
  alphabound    <alpha_root>/stats/thesis_stats.json (α 边界 G3v−G3)
  dimscan       security_overhead_dimension_scan.json (维度扫描)
  memory        memory_overhead_report.json   (per-role 内存)
  commrole      comm_per_role.json            (per-role 通信)
  attacks       attack_evidence JSON 根列表   (攻击证据, 老 root + v2)

用法:
  python thesis_tables.py --out-dir chapters/gen
  python thesis_tables.py --attacks results_thesis_core_attack/attack_evidence.json \\
      results_thesis_core_attack_v2/attack_evidence_v2.json
每个表格一个 .tex 片段 (tabular 环境正文), 论文以 \\input 或粘贴使用。
"""

import argparse
import json
import os

ARM_ORDER = ["fedbuff", "mafsa", "dvafl", "ours_plain", "g1_ref"]
ARM_TEX = {
    "fedbuff": r"FedBuff",
    "mafsa": r"MAFSA",
    "dvafl": r"DVAFL",
    "ours_plain": r"\textbf{Ours (G3v)}",
    "g1_ref": r"FedAvg 同步",
}
DS_TEX = {"MNIST": "MNIST", "FashionMNIST": "FashionMNIST",
          "CIFAR10": "CIFAR-10"}
PART_TEX = {"IID": "IID", "NI": r"Non-IID ($\alpha{=}0.5$)",
            "NI01": r"Non-IID ($\alpha{=}0.1$)",
            "NI10": r"Non-IID ($\alpha{=}1.0$)"}
UNITS = [("MNIST", "IID"), ("MNIST", "NI"), ("FashionMNIST", "IID"),
         ("FashionMNIST", "NI"), ("CIFAR10", "IID"), ("CIFAR10", "NI")]


def _fmt(x, nd=2):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def _cell_pm(st, nd=2):
    if not st:
        return "--"
    return f"{st['mean']:.{nd}f}$\\pm${st['sd']:.{nd}f}"


def competitive_table(stats, nd=2):
    """5 方法 × 6 单元; 单元 = 数据集 × 分布; 行=单元, 列=方法."""
    rows = []
    for ds, part in UNITS:
        cells = []
        for arm in ARM_ORDER:
            st = stats["main_table"].get(f"{arm}|{ds}|{part}")
            cells.append(_cell_pm(st, nd))
        rows.append(f"    {DS_TEX[ds]}/{PART_TEX[part]} & " +
                    " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return ("\n".join([
        r"\begin{tabular}{l" + "c" * len(ARM_ORDER) + "}",
        r"\toprule",
        r"单元 & " + " & ".join(ARM_TEX[a] for a in ARM_ORDER) + r" \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
    ]))


def comparisons_table(stats):
    rows = []
    for key, st in stats.get("comparisons", {}).items():
        hi_lo, ds, part = key.split("|")
        short = hi_lo.replace("-", r"$-$").replace("ours_plain", "Ours")
        short = (short.replace("fedbuff", "FedBuff")
                 .replace("mafsa", "MAFSA").replace("dvafl", "DVAFL"))
        if not st:
            rows.append(f"    {short} & {DS_TEX[ds]}/{PART_TEX[part]} & "
                        r"-- & -- & -- & -- & -- \\")
            continue
        verdict = ("稳定" if st["mean_diff"] > 0 and st["positives"] >= 4
                   and st["ci95"][0] > 0 else
                   ("趋势" if st["mean_diff"] > 0 else "无"))
        rows.append(
            f"    {short} & {DS_TEX[ds]}/{PART_TEX[part]} & "
            f"{st['mean_diff']:+.2f} & "
            f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] & "
            f"{st['dz']:+.2f} & {st['positives']}/{st['n']} & {verdict} \\\\")
    return "\n".join([
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"比较 & 单元 & $\Delta$均值 & 95\% CI & Cohen's $d_z$ & 正种子 & 判定 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def _equal_table(stats, key_stats, budget_key):
    rows = []
    for key, st in sorted(stats.get(key_stats, {}).items()):
        arm, ds, part = key.split("|")
        rows.append(
            f"    {DS_TEX[ds]}/{PART_TEX[part]} & {ARM_TEX.get(arm, arm)} & "
            f"{st['mean_at_budget']:.2f} & {st['base_mean_at_budget']:.2f} & "
            f"{st['mean_diff']:+.2f} & "
            f"[{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] & "
            f"{st['positives']}/{st['n']} & {len(st['censored_seeds'])} \\\\")
    if not rows:
        return "% (无数据)"
    return "\n".join([
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"单元 & 方法 & 方法@预算 & FedBuff@预算 & $\Delta$均值 & 95\% CI & 正种子 & 截断 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def dimscan_table(scan):
    rows = []
    for cell in scan.get("cells", []):
        roles = cell.get("roles", {})
        ev = roles.get("es_verify", {})
        ta = roles.get("ta_mask_build", {})
        rows.append(
            f"    {cell['arm']} & {cell['dim']:,} & "
            f"{ev.get('median_s', 0):.2f} & {ev.get('peak_traced_mb', 0):.1f} & "
            f"{ta.get('median_s', 0):.2f} & "
            f"{cell.get('checkpoint_total_median_s', 0):.2f} "
            f"(n={cell.get('reps', '?')}) \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"臂 & $d$ & ES 验证 (s) & ES 内存 (MB) & TA 掩码 (s) & 合计 (s) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def memory_table(report):
    rows = []
    for role, s in report.get("roles", {}).items():
        rows.append(
            f"    \\texttt{{{role}}} & {s['n_calls']} & "
            f"{s['tracemalloc_peak_mb_mean']:.2f} & "
            f"{s['tracemalloc_peak_mb_max']:.2f} & "
            f"{s['wall_ms_mean']:.1f} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"角色 & 调用数 & 峰值内存均值 (MB) & 峰值内存最大 (MB) & 单次均值 (ms) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def commrole_table(comm):
    t = comm.get("totals_bytes_by_type", {})
    role = comm.get("totals_bytes", {})
    n = max(1, comm.get("n_runs", 1))
    rows = [
        f"    用户 (upload) & {t.get('upload', 0) / n / 1048576:.2f} & "
        f"随 $d$ 线性 \\\\",
        f"    边缘 ES (package+vote) & "
        f"{(t.get('package', 0) + t.get('vote', 0)) / n / 1048576:.2f} & "
        f"package 随 $d$ 线性; vote 定长 \\\\",
        f"    云端 (manifest) & {t.get('manifest', 0) / n / 1048576:.2f} & "
        f"随 $d$ 线性 \\\\",
        f"    TA (receipt+notify) & "
        f"{(t.get('receipt', 0) + t.get('notify', 0)) / n / 1024:.1f} KiB & "
        f"**与 $d$ 无关** \\\\",
    ]
    return "\n".join([
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"角色 (消息) & 每 checkpoint 字节 & 随 $d$ 规律 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def attack_table(evidence_list, units_wanted=100):
    """合并多个 attack_evidence JSON (老 root + v2) → 类别级 booktabs."""
    rows = []
    clean_fr = []
    for ev in evidence_list:
        for c in ev.get("classes", []):
            rate = c.get("rejection_rate")
            stages = ", ".join(f"{k}:{v}" for k, v in
                               sorted((c.get("detection_stages") or {}).items()))
            rows.append(
                f"    {c['class']} & {c.get('attempted', 0)} & "
                f"{c.get('detected', 0)} & {c.get('accepted_attack', 0)} & "
                f"{(rate * 100 if rate is not None else float('nan')):.0f}\\% & "
                f"{stages or '--'} \\\\")
        c = ev.get("clean")
        if c and c.get("false_reject_rate") is not None:
            clean_fr.append((c.get("attempted", 0), c.get("detected", 0)))
    cl_att = sum(a for a, _ in clean_fr)
    cl_det = sum(d for _, d in clean_fr)
    if clean_fr:
        rows.append(f"    clean control & {cl_att} & {cl_det} & 0 & "
                    f"{(1 - cl_det / max(1, cl_att)) * 100:.0f}\\% & 误拒 "
                    f"{cl_det}/{cl_att} \\\\")
    return "\n".join([
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"攻击类 & 注入单元 & 检出 & 被采纳 & 拒绝率 & 检出阶段 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])


def main():
    ap = argparse.ArgumentParser(description="统计/证据 JSON → booktabs 表格")
    ap.add_argument("--out-dir", default="chapters/gen")
    ap.add_argument("--plain-stats", default="results_plain_bench/stats/"
                                             "plain_bench_stats.json")
    ap.add_argument("--alpha-stats", nargs="*", default=[
        "results_n1_alpha01/stats/thesis_stats.json",
        "results_n1_alpha10/stats/thesis_stats.json"])
    ap.add_argument("--dimscan", default="security_overhead_dimension_scan.json")
    ap.add_argument("--memory", default="memory_overhead_report.json")
    ap.add_argument("--commrole", default="comm_per_role.json")
    ap.add_argument("--attacks", nargs="*", default=[
        "results_thesis_core_attack/attack_evidence.json",
        "results_thesis_core_attack_v2/attack_evidence_v2.json"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, body):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body + "\n")
        print(f"  {name}")

    ps = json.load(open(args.plain_stats, encoding="utf-8")) \
        if os.path.exists(args.plain_stats) else None
    if ps:
        write("tab_competitive.tex", competitive_table(ps))
        write("tab_comparisons.tex", comparisons_table(ps))
        write("tab_equalcompute.tex", _equal_table(ps, "equal_compute", "N"))
        write("tab_equaltime.tex", _equal_table(ps, "equal_time", "T"))
    else:
        print(f"[skip] {args.plain_stats} 不存在 (批跑完成后生成)")

    for p in args.alpha_stats:
        if os.path.exists(p):
            root = os.path.basename(os.path.dirname(os.path.dirname(p)))
            st = json.load(open(p, encoding="utf-8"))
            # α 边界: G3v−G3 配对 (Direct-VCM 在该 α 下的增量)
            rows = []
            for key, s in st.get("comparisons", {}).items():
                if not key.startswith("G3v-G3|"):
                    continue
                _, ds, part = key.split("|")
                if not s:
                    continue
                rows.append(
                    f"    {DS_TEX.get(ds, ds)}/{PART_TEX.get(part, part)} & "
                    f"{s['mean_diff']:+.2f} & "
                    f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] & "
                    f"{s['positives']}/{s['n']} \\\\")
            if rows:
                write(f"tab_alphabound_{root}.tex", "\n".join([
                    r"\begin{tabular}{lccc}",
                    r"\toprule",
                    r"单元 & $\Delta$均值 & 95\% CI & 正种子 \\",
                    r"\midrule", *rows,
                    r"\bottomrule", r"\end{tabular}"]))
            else:
                print(f"[skip] {p}: 无 G3v-G3 比较 (需先跑该 root 的 stats)")

    if os.path.exists(args.dimscan):
        write("tab_dimscan.tex", dimscan_table(
            json.load(open(args.dimscan, encoding="utf-8"))))
    if os.path.exists(args.memory):
        write("tab_memory.tex", memory_table(
            json.load(open(args.memory, encoding="utf-8"))))
    if os.path.exists(args.commrole):
        write("tab_commrole.tex", commrole_table(
            json.load(open(args.commrole, encoding="utf-8"))))
    evs = [json.load(open(p, encoding="utf-8"))
           for p in args.attacks if os.path.exists(p)]
    if evs:
        write("tab_attacks.tex", attack_table(evs))
    print(f"表格输出目录: {args.out_dir}")


if __name__ == "__main__":
    main()
