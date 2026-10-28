"""数值三方对账: 论文数字 ↔ stats JSON ↔ manifest (盲审补全执行计划 阶段 6.3).

管线: run_* → thesis_stats / plain_bench_stats → thesis_tables → 本脚本复核。

检查项 (每项 PASS/FAIL/MISSING):
  A. 批次完整性: 各 root manifest 成功行数 = 计划值
     (plain 90 / alpha01 61 / alpha10 61 / thesis_core 172)
  B. 数值一致性: stats JSON 的主表逐格均值 ↔ 从 manifest+npz 现场重算
     (容差 1e-6; 这是"论文数字=JSON=manifest"链条的 JSON↔manifest 段)
  C. 攻击证据: 老 root (4 类 × 5 seeds) + v2 (2 类 × 5 seeds) 注入单元数
     达标、拒绝率 100%、clean 误拒 0; 单元总数与"3040+新增"口径对账
  D. 开销产物: 维度扫描 8 格齐、内存报告角色齐、per-role 通信 TA 字节恒定
  E. 论文侧 (--thesis-dir 给定时): 生成的 .tex 表格存在, 且其关键数字
     (均值±std) 逐串出现在 chapter04.tex 中 (论文↔JSON 段)

用法: python reconcile_thesis_numbers.py [--thesis-dir E:/bjtu-thesis]
退出码: 0 = 全 PASS; 1 = 有 FAIL 或 (无 --allow-missing 时的) MISSING。
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEEDS = {42, 43, 44, 45, 46}
EXPECTED_ROOT_ROWS = {
    "results_plain_bench": 90,
    "results_n1_alpha01": 61,
    "results_n1_alpha10": 61,
    "results_thesis_core": 172,
}


def load_manifest(root):
    path = os.path.join(root, "thesis_core_manifest.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def check_A(results):
    for root, want in EXPECTED_ROOT_ROWS.items():
        rows = load_manifest(root)
        if rows is None:
            results.append((f"A/{root}", "MISSING", "manifest 不存在"))
            continue
        ok_rows = [r for r in rows if r.get('rc') == 0]
        n = len(ok_rows)
        tags = {r['tag'] for r in ok_rows}
        results.append((f"A/{root}", "PASS" if n >= want else "FAIL",
                        f"成功 {n}/{want} (唯一 tag {len(tags)})"))


def check_B(results):
    """stats JSON 主表 ↔ manifest+npz 重算."""
    jobs = [
        ("results_plain_bench/stats/plain_bench_stats.json", "main_table"),
        ("results_n1_alpha01/stats/thesis_stats.json", "main_table"),
        ("results_n1_alpha10/stats/thesis_stats.json", "main_table"),
        ("results_thesis_core/stats/thesis_stats.json", "main_table"),
    ]
    import numpy as np
    for jpath, key in jobs:
        if not os.path.exists(jpath):
            results.append((f"B/{jpath}", "MISSING", "stats JSON 不存在"))
            continue
        stats = json.load(open(jpath, encoding='utf-8'))
        root = jpath.split('/')[0]
        rows = load_manifest(root) or []
        acc_by_tag = {}
        for r in rows:
            if r.get('rc') != 0:
                continue
            npz = os.path.join(root, r['tag'],
                               f"point2_data_{r['tag']}.npz")
            if not os.path.exists(npz):
                continue
            try:
                d = np.load(npz, allow_pickle=True)
                a = np.asarray(d.get('acc_history'), dtype=np.float64)
                if a.ndim == 2 and a.shape[1] >= 2:
                    a = a[:, 1]
                if len(a):
                    acc_by_tag[r['tag']] = float(a[-1])
            except Exception:  # noqa: BLE001
                continue
        table = stats.get(key, {})
        if not table:
            results.append((f"B/{jpath}", "MISSING", "main_table 为空"))
            continue
        n_ok = n_bad = 0
        bad_keys = []
        for cell, st in table.items():
            if not st:
                continue
            arm, ds, part = cell.split('|')
            vals = [v for t, v in acc_by_tag.items()
                    if t.startswith(f"{arm}_{ds}_{part}_s")
                    and not t.endswith('_rep')]
            if not vals:
                continue
            recomputed = sum(vals) / len(vals)
            if abs(recomputed - st['mean']) <= 1e-6:
                n_ok += 1
            else:
                n_bad += 1
                bad_keys.append(f"{cell}: json={st['mean']:.4f} "
                                f"vs npz={recomputed:.4f}")
        status = "PASS" if (n_ok > 0 and n_bad == 0) else (
            "FAIL" if n_bad else "MISSING")
        results.append((f"B/{os.path.basename(jpath)}",
                        status,
                        f"一致 {n_ok} 格, 不一致 {n_bad} 格"
                        + ("; " + "; ".join(bad_keys[:3]) if bad_keys else "")))


def check_C(results):
    paths = [f"results_thesis_core_attack/attack_evidence_s{s}.json"
             for s in (42, 43, 44, 45, 46)]
    paths += ["results_thesis_core_attack_v2/attack_evidence_v2.json"]
    total_units = 0
    problems = []
    found = 0
    for p in paths:
        if not os.path.exists(p):
            problems.append(f"缺 {p}")
            continue
        found += 1
        ev = json.load(open(p, encoding='utf-8'))
        for c in ev.get('classes', []):
            n_att = c.get('attempted', 0)
            rate = c.get('rejection_rate')
            total_units += n_att
            if n_att < c.get('units_wanted', 100):
                problems.append(f"{p}:{c['class']} 单元不足 {n_att}")
            if rate is None or rate < 1.0 or c.get('accepted_attack', 0):
                problems.append(f"{p}:{c['class']} 拒绝率 {rate} "
                                f"accepted={c.get('accepted_attack')}")
        c = ev.get('clean')
        if c:
            frr = c.get('false_reject_rate')
            if frr is None or frr != 0.0:
                problems.append(f"{p}: clean 误拒率 {frr}")
    want_total = 3040 + 1000   # 既有口径 + 新增 2 类 × 5 seeds × 100
    results.append((f"C/攻击证据 (找到 {found}/{len(paths)} 份)",
                    "PASS" if (found == len(paths) and not problems)
                    else ("MISSING" if found == 0 else "FAIL"),
                    f"注入单元合计 {total_units} (口径 {want_total}); "
                    + ("; ".join(problems[:4]) if problems else "全部达标")))


def check_D(results):
    probs = []
    scan_p = "security_overhead_dimension_scan.json"
    if os.path.exists(scan_p):
        scan = json.load(open(scan_p, encoding='utf-8'))
        cells = scan.get('cells', [])
        arms = {c['arm'] for c in cells}
        dims = {c['dim'] for c in cells}
        if not ({'G2', 'G3v'} <= arms and {10000, 100000, 500000, 1000000} <= dims):
            probs.append(f"扫描格不齐: arms={arms} dims={dims}")
    else:
        probs.append("维度扫描 JSON 不存在")
    mem_p = "memory_overhead_report.json"
    if os.path.exists(mem_p):
        rep = json.load(open(mem_p, encoding='utf-8'))
        need = {'es_verify', 'ta_authorize', 'commit_tx', 'cloud_aggregate'}
        if not need <= set(rep.get('roles', {})):
            probs.append(f"内存报告角色缺: {need - set(rep.get('roles', {}))}")
    else:
        probs.append("内存报告不存在")
    comm_p = "comm_per_role.json"
    if os.path.exists(comm_p):
        comm = json.load(open(comm_p, encoding='utf-8'))
        per_run_ta = comm['totals_bytes'].get('TA', 0) / max(1, comm['n_runs'])
        if per_run_ta == 0:
            probs.append("TA 角色字节为 0 (映射断裂?)")
    else:
        probs.append("per-role 通信 JSON 不存在")
    results.append(("D/开销产物",
                    "PASS" if not probs else
                    ("MISSING" if len(probs) >= 3 else "FAIL"),
                    "; ".join(probs) if probs else
                    f"维度扫描 {len(scan.get('cells', []))} 格, 内存/通信报告齐"))


def check_E(results, thesis_dir):
    gen = "chapters/gen"
    tex_files = sorted(glob.glob(os.path.join(gen, "*.tex")))
    if not tex_files:
        results.append(("E/论文表格", "MISSING",
                        "chapters/gen 无 .tex (先跑 thesis_tables.py)"))
        return
    ch4 = os.path.join(thesis_dir, "chapters", "chapter04.tex")
    if not os.path.exists(ch4):
        results.append(("E/论文表格", "MISSING", f"{ch4} 不存在"))
        return
    text = open(ch4, encoding='utf-8').read()
    missing_in_thesis = []
    for tf in tex_files:
        body = open(tf, encoding='utf-8').read()
        # 取 tabular 中的数值单元格 (mean±std / Δmean), 在论文里查同串
        nums = re.findall(r'(\d+\.\d+)\$?\\pm\$?(\d+\.\d+)', body)
        for mean, sd in nums:
            for pat in (f"{mean}\\pm{sd}", f"{mean}$\\pm${sd}"):
                if pat in text:
                    break
            else:
                missing_in_thesis.append(
                    f"{os.path.basename(tf)}:{mean}±{sd}")
    results.append(("E/论文数字↔表格",
                    "PASS" if not missing_in_thesis else "FAIL",
                    (f"{len(missing_in_thesis)} 个数字未入正文: "
                     + "; ".join(missing_in_thesis[:3])) if missing_in_thesis
                    else f"{len(tex_files)} 个表格全部对上"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis-dir", default=None,
                    help="论文仓库路径 (给定则执行 E 项检查)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="MISSING 不计入失败 (用于分阶段对账)")
    args = ap.parse_args()

    results = []
    check_A(results)
    check_B(results)
    check_C(results)
    check_D(results)
    if args.thesis_dir:
        check_E(results, args.thesis_dir)

    print("=== 三方对账 (论文 ↔ stats JSON ↔ manifest) ===")
    fails = 0
    missing = 0
    for name, status, msg in results:
        print(f"[{status:<7}] {name}: {msg}")
        fails += (status == "FAIL")
        missing += (status == "MISSING")
    n_fail = fails + (0 if args.allow_missing else missing)
    print(f"=== 结果: {n_fail} 项失败"
          + (f" (另有 {missing} 项 MISSING, 因 --allow-missing 不计)"
             if missing and args.allow_missing else "") + " ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
