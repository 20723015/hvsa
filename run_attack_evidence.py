"""策略绑定/结果绑定的最低安全证据批次 (四周补强计划第 3 周).

四类攻击 × 100 个**适用攻击单元** + 同规模 clean control; 单元 = 一次
checkpoint 尝试 (main_point2.py 每个候选写一条 AttackOutcome JSONL)。

  raw_sum        tamper_global                     篡改 W_sum^raw → LHH 等式
  manifest_replay tamper_manifest + replay         注入不存在组包 / 重发已消费组包
  weight_source  weight_forge/weight_relabel/
                 source_downgrade                  伪造/改标/降级权重来源 (V-PDA)
  comp_param     alpha_tamper                      篡改 VCM 补偿衰减系数 α (σ>0)

验收口径 (计划): 每类攻击拒绝率 100% (detected/attempted), clean control
误拒率 0% (detected/attempts), 并报告 checkpoint 验证时间与通信量。

说明: 攻击实验必须 --verify (fail-fast), 攻击单元数由 --units 控制
(默认 100); 本机为 CPU 时建议先用 --units 5 冒烟。

用法:
  python run_attack_evidence.py --dry-run
  python run_attack_evidence.py --units 100 --seed 42
  python run_attack_evidence.py --units 5 --model Tiny     # 快速冒烟
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

ROOT_DEFAULT = "results_thesis_core_attack"

# 类 -> (成员攻击列表, 额外协议参数, 用途说明)
CLASSES = {
    "raw_sum": (["tamper_global"],
                ["--vcm", "--gamma-vcm", "1.0"],
                "篡改 W_sum^raw (结果绑定, LHH 等式)"),
    "manifest_replay": (["tamper_manifest", "replay"],
                        ["--vcm", "--gamma-vcm", "1.0"],
                        "篡改 manifest / 重放已消费组包 (exactly-once)"),
    "weight_source": (["weight_forge", "weight_relabel", "source_downgrade"],
                      ["--vcm", "--gamma-vcm", "1.0",
                       "--participation-debias", "ipw"],
                      "伪造/改标/降级权重来源 (V-PDA 独立复算)"),
    "comp_param": (["alpha_tamper"],
                   ["--vcm", "--gamma-vcm", "1.0", "--staleness-sigma", "0.5"],
                   "篡改 VCM 补偿衰减系数 α (公开规则复算)"),
}

BASE = ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
        "--max-checkpoints", "100"]


def _log_paths(root, tag):
    run_dir = os.path.join(root, tag)
    return run_dir, os.path.join(run_dir, f"attack_outcomes_{tag}.jsonl")


def _build_cmd(tag, root, attack, extra, args, max_ticks):
    run_dir = os.path.join(root, tag)
    cmd = [sys.executable, os.path.abspath("main_point2.py"),
           "--dataset", args.dataset, "--model", args.model,
           "--verify", "--deterministic",
           "--seed", str(args.seed),
           "--num-edge-servers", str(args.edge_servers),
           "--users-per-edge", str(args.users_per_edge),
           "--output-dir", os.path.abspath(run_dir), "--run-id", tag,
           "--force", "--max-ticks", str(max_ticks)] + BASE + list(extra)
    if attack != 'none':
        cmd += ["--attack", attack]
    else:
        # clean control: 显式指定 attack-log 才会在无攻击时也逐候选记录
        cmd += ["--attack-log", os.path.abspath(_log_paths(root, tag)[1])]
    return cmd


def _run(cmd, log_path):
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8', errors='replace') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc, time.time() - t0


def _read_outcomes(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def _read_costs(run_dir, tag):
    """验证耗时/通信量 (取自 main_point2 的 meta)."""
    hits = glob.glob(os.path.join(run_dir, f"point2_meta_{tag}.json"))
    if not hits:
        return {}
    try:
        with open(hits[0], encoding='utf-8') as f:
            m = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    n_ckp = max(1, int(m.get('checkpoint_attempts') or 1))
    return {
        'verify_seconds_total': m.get('verify_seconds_total'),
        'verify_seconds_per_attempt': ((m.get('verify_seconds_total') or 0.0)
                                       / n_ckp),
        'comm_total_bytes': m.get('comm_total_bytes'),
        'comm_bytes_per_attempt': ((m.get('comm_total_bytes') or 0) / n_ckp),
        'attempts': n_ckp,
    }


def run_class(cls, members, extra, args, root):
    """跑一类攻击: 成员攻击平分 units, 每成员一个 run."""
    units_total = args.units
    per_member = max(1, -(-units_total // len(members)))   # ceil
    recs, runs = [], []
    for attack in members:
        tag = f"{cls}_{attack}_s{args.seed}"
        run_dir, log_jsonl = _log_paths(root, tag)
        os.makedirs(run_dir, exist_ok=True)
        # 被拒候选不推进 checkpoint 计数 → 每个 tick 一个攻击单元;
        # 预留 warmup (启动组包/调度) ticks
        # replay 只有正常 checkpoint 成功提交后才产生一个攻击单元，触发频率
        # 低于“每 tick 一个”的拒绝型攻击，单独放宽 tick 预算。
        max_ticks = ((per_member * 6 if attack == 'replay' else per_member)
                     + args.warmup_ticks)
        cmd = _build_cmd(tag, root, attack, extra, args, max_ticks)
        if args.dry_run:
            print(f"[{tag}] " + ' '.join(cmd))
            continue
        rc, wall = _run(cmd, os.path.join(run_dir, f"run_{tag}.log"))
        got = _read_outcomes(log_jsonl)
        costs = _read_costs(run_dir, tag)
        runs.append({'tag': tag, 'attack': attack, 'cls': cls, 'rc': rc,
                     'wall_s': round(wall, 1), 'units': len(got),
                     'costs': costs})
        recs.extend(got)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}: rc={rc} "
              f"units={len(got)} wall={wall:.0f}s", flush=True)
    return recs, runs


def summarize_class(cls, recs, runs, units_wanted):
    """按类别汇总: 攻击单元/检出/被采纳/不适用 + 阶段分布."""
    attempted = [r for r in recs if r.get('attempted')]
    inapplicable = [r for r in recs if not r.get('applicable')]
    detected = [r for r in attempted if r.get('detected')]
    accepted = [r for r in attempted if r.get('accepted') and not r.get('detected')]
    stages = Counter(r.get('detection_stage') for r in detected)
    attempted_by_attack = Counter(r.get('attack') for r in attempted)
    rate = (len(detected) / len(attempted)) if attempted else None
    return {
        'class': cls,
        'units_recorded': len(recs),
        'units_wanted': units_wanted,
        'attempted': len(attempted),
        'applicable': len(recs) - len(inapplicable),
        'inapplicable': len(inapplicable),
        'detected': len(detected),
        'accepted_attack': len(accepted),
        'rejection_rate': rate,
        'detection_stages': dict(stages),
        'attempted_by_attack': dict(attempted_by_attack),
        'runs': runs,
    }


def summarize_clean(recs, runs):
    attempts = [r for r in recs if r.get('requested') is False]
    false_rejects = [r for r in attempts if r.get('detected')]
    rate = (len(false_rejects) / len(attempts)) if attempts else None
    return {
        'class': 'clean_control', 'units_recorded': len(recs),
        'attempted': len(attempts), 'detected': len(false_rejects),
        'false_reject_rate': rate, 'runs': runs,
    }


def render_md(summary, args):
    lines = ["# 策略绑定/结果绑定安全证据 (thesis-core attack evidence)", ""]
    lines.append(f"- 数据集/模型: {args.dataset} / {args.model}; "
                 f"seed={args.seed}; 拓扑 {args.edge_servers} ES × "
                 f"{args.users_per_edge} 用户; 每类目标 {args.units} 个攻击单元")
    lines.append(f"- 攻击实验强制 --verify (fail-fast), 单元 = 一次 checkpoint 尝试")
    lines.append("")
    lines.append("| 类别 | 攻击 | 单元(记录/目标) | 已尝试 | 检出 | 攻击被采纳 | "
                 "拒绝率 | 不适用 | 检出阶段 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summary['classes']:
        attacks = '+'.join(r['attack'] for r in s['runs']) or '—'
        rate = ('—' if s['rejection_rate'] is None
                else f"{s['rejection_rate']*100:.1f}%")
        stages = ', '.join(f"{k}:{v}" for k, v in
                           sorted(s['detection_stages'].items())) or '—'
        lines.append(f"| {s['class']} | {attacks} | "
                     f"{s['units_recorded']}/{s['units_wanted']} | "
                     f"{s['attempted']} | {s['detected']} | "
                     f"{s['accepted_attack']} | {rate} | {s['inapplicable']} | "
                     f"{stages} |")
    c = summary.get('clean')
    if c is not None:
        frate = ('—' if c['false_reject_rate'] is None
                 else f"{c['false_reject_rate']*100:.1f}%")
        lines.append(f"| clean_control | (无攻击) | {c['units_recorded']}/{args.units} | "
                     f"{c['attempted']} | {c['detected']} | 0 | 误拒率 {frate} | — | — |")
    lines.append("")
    lines.append("## 开销 (checkpoint 验证时间与协议通信量)")
    lines.append("")
    lines.append("| 运行 | 攻击 | 单元 | 验证耗时/单元 (s) | 通信量/单元 (KiB) | 本机耗时 (s) |")
    lines.append("|---|---|---|---|---|---|")
    rows_for_cost = list(summary['classes'])
    if c is not None:
        rows_for_cost.append(c)
    for row in rows_for_cost:
        for r in row['runs']:
            cst = r.get('costs') or {}
            v = cst.get('verify_seconds_per_attempt')
            cb = cst.get('comm_bytes_per_attempt')
            lines.append(f"| {r['tag']} | {r.get('attack', 'none')} | {r['units']} | "
                         f"{'—' if v is None else f'{v:.4f}'} | "
                         f"{'—' if cb is None else f'{cb/1024:.1f}'} | {r['wall_s']} |")
    lines.append("")
    # 验收判定 (计划口径)
    bad = []
    for s in summary['classes']:
        member_attacks = {r['attack'] for r in s['runs']}
        covered = {a for a, n in s['attempted_by_attack'].items() if n > 0}
        run_failed = any(r.get('rc') != 0 for r in s['runs'])
        if (s['attempted'] < s['units_wanted']
                or s['rejection_rate'] is None
                or s['rejection_rate'] < 1.0
                or s['accepted_attack'] != 0
                or run_failed
                or not member_attacks.issubset(covered)):
            bad.append(s['class'])
    if c is not None:
        clean_failed = any(r.get('rc') != 0 for r in c['runs'])
        if (c['attempted'] < args.units
                or c['false_reject_rate'] != 0.0
                or clean_failed):
            bad.append('clean_control')
    lines.append("## 验收判定")
    lines.append("")
    if bad:
        lines.append(f"- ⚠ 未达标: {bad} (要求攻击拒绝率 100%、clean 误拒率 0%)")
    else:
        lines.append("- ✅ 达标: 每类适用攻击单元达到目标、各成员攻击均有覆盖、"
                     "攻击拒绝率 100%, clean control 数量达标且误拒率 0%")
    lines.append("")
    lines.append("> 注: 单元数以实际注入的 attempted 计，必须达到目标值；"
                 "被拒候选不推进版本/消费 token, 由 AttackOutcome 的 "
                 "state_before/state_after 佐证。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="安全证据批次")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--dataset", default="MNIST",
                    choices=["MNIST", "FashionMNIST", "CIFAR10"])
    ap.add_argument("--model", default="Tiny",
                    help="证据批次用小模型 (默认 Tiny; 主表精度实验不用)")
    ap.add_argument("--edge-servers", type=int, default=5)
    ap.add_argument("--users-per-edge", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--units", type=int, default=100,
                    help="每类攻击单元数 (clean control 同规模)")
    ap.add_argument("--warmup-ticks", type=int, default=30,
                    help="每个 run 预留的启动/调度 ticks")
    ap.add_argument("--classes", nargs="+", default=list(CLASSES) + ['clean'],
                    choices=list(CLASSES) + ['clean'])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.root
    if not args.dry_run:
        os.makedirs(root, exist_ok=True)
    classes = [c for c in args.classes if c != 'clean']
    want_clean = ('clean' in args.classes) or not args.classes

    summary = {'classes': [], 'clean': None,
               'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               'args': vars(args)}

    for cls in classes:
        members, extra, _desc = CLASSES[cls]
        recs, runs = run_class(cls, members, extra, args, root)
        if args.dry_run:
            continue
        summary['classes'].append(
            summarize_class(cls, recs, runs, args.units))
        # 攻击单元原始记录留档 (便于复核检出阶段与状态快照)
        with open(os.path.join(root, f"outcomes_{cls}.jsonl"), 'w',
                  encoding='utf-8') as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if want_clean and not args.dry_run:
        tag = f"clean_s{args.seed}"
        run_dir, log_jsonl = _log_paths(root, tag)
        os.makedirs(run_dir, exist_ok=True)
        # clean control: 100 次成功提交 (无攻击, 逐候选记录)
        max_ticks = args.units * 6 + args.warmup_ticks
        cmd = _build_cmd(tag, root, 'none', CLASSES['raw_sum'][1], args,
                         max_ticks)
        rc, wall = _run(cmd, os.path.join(run_dir, f"run_{tag}.log"))
        recs = _read_outcomes(log_jsonl)
        clean_run = {'tag': tag, 'attack': 'none', 'cls': 'clean', 'rc': rc,
                     'wall_s': round(wall, 1), 'units': len(recs),
                     'costs': _read_costs(run_dir, tag)}
        summary['clean'] = summarize_clean(recs, [clean_run])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}: rc={rc} "
              f"units={len(recs)} wall={wall:.0f}s", flush=True)

    if args.dry_run:
        return

    json_path = os.path.join(root, "attack_evidence.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    md_path = os.path.join(root, "attack_evidence.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(render_md(summary, args))
    print(f"证据汇总: {json_path}")
    print(f"证据报告: {md_path}")


if __name__ == "__main__":
    main()
