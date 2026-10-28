"""新增攻击类批次 (盲审补全执行计划 阶段 4.4).

两类新增攻击 × 100 适用单元 × seeds {42..46}:
  drop_replace     漏聚合组包 (LHH 等式检出)
  history_missing  省略共识历史 W(v) (Step 0b fail-closed; v≥1 守卫修复后可测)

clean control 不重跑: 既有 results_thesis_core_attack/clean_s{42..46} 与
新类使用完全相同的协议参数 (--vcm --gamma-vcm 1.0) 与单元语义
(逐 checkpoint 尝试记录), 直接复用 (等量对照, 见 CLAIMS_MAP)。

产物隔离: 落在 results_thesis_core_attack_v2/, 每类每 seed 留
outcomes_{cls}_s{seed}.jsonl 与 _part_s{seed}_new.json, 末尾合并出
attack_evidence_v2.{json,md} (拒绝率 100% / 单元数达标验收口径)。

用法:
  python run_attack_v2.py --dry-run
  python run_attack_v2.py                # 全量 2 类 × 5 seeds = 10 runs
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
V2_ROOT = os.path.join(ROOT, "results_thesis_core_attack_v2")
CLASSES = ["drop_replace", "history_missing"]
SEEDS = [42, 43, 44, 45, 46]


def run_seed_class(cls, seed, units, dry_run=False):
    """单类单 seed 一次 run_attack_evidence 调用 (独立 root 防覆盖)."""
    tmp_root = os.path.join(ROOT, "results_attack_v2_tmp")
    cmd = [sys.executable, os.path.join(ROOT, "run_attack_evidence.py"),
           "--root", tmp_root, "--units", str(units), "--seed", str(seed),
           "--classes", cls]
    if dry_run:
        print(f"[{cls}_s{seed}] " + ' '.join(cmd))
        return None
    log = os.path.join(ROOT, f"attack_v2_{cls}_s{seed}.log")
    with open(log, 'w', encoding='utf-8', errors='replace') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    src = os.path.join(tmp_root, "attack_evidence.json")
    if not os.path.exists(src):
        print(f"[{cls}_s{seed}] rc={rc} 但无 JSON, 跳过")
        return None
    with open(src, encoding='utf-8') as f:
        part = json.load(f)
    os.makedirs(V2_ROOT, exist_ok=True)
    shutil.copy(src, os.path.join(V2_ROOT, f"_part_s{seed}_{cls}.json"))
    # 原始 outcome 留档 (per-seed 命名, 防下次覆盖)
    for cls_out in (cls,):
        src_out = os.path.join(tmp_root, f"outcomes_{cls_out}.jsonl")
        if os.path.exists(src_out):
            shutil.copy(src_out, os.path.join(
                V2_ROOT, f"outcomes_{cls_out}_s{seed}.jsonl"))
    # run 目录也迁入 v2 (审计完整产物)
    for attack in {'drop_replace': ['drop_replace_pkg'],
                   'history_missing': ['history_missing']}[cls]:
        run_dir = os.path.join(tmp_root, f"{cls}_{attack}_s{seed}")
        dst = os.path.join(V2_ROOT, f"{cls}_{attack}_s{seed}")
        if os.path.isdir(run_dir) and not os.path.exists(dst):
            shutil.copytree(run_dir, dst)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {cls}_s{seed}: rc={rc} "
          f"attempted={sum(c.get('attempted', 0) for c in part['classes'])}",
          flush=True)
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--classes", nargs="+", default=CLASSES,
                    choices=CLASSES)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    parts = []
    for seed in args.seeds:
        for cls in args.classes:
            part = run_seed_class(cls, seed, args.units, args.dry_run)
            if part is not None:
                parts.append(part)

    if args.dry_run or not parts:
        return 0

    merged = {
        'classes': [],
        'clean': None,   # clean control 复用 results_thesis_core_attack 的
                         # clean_s{42..46} (同协议参数), 合并见 phase 6 对账
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'args': {**parts[0].get('args', {}), 'root': V2_ROOT},
        'merged_from': [f"{p['args']['seed']}:{c['class']}"
                        for p in parts for c in p.get('classes', [])],
    }
    for part in parts:
        merged['classes'].extend(part.get('classes', []))

    json_path = os.path.join(V2_ROOT, "attack_evidence_v2.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2, allow_nan=False)

    import run_attack_evidence as rae
    ns = argparse.Namespace(**merged['args'])
    md = rae.render_md(merged, ns)
    md_path = os.path.join(V2_ROOT, "attack_evidence_v2.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    verdict = [l for l in md.splitlines() if '达标' in l]
    print(f"合并: {json_path}\n{md_path}")
    print("判定:", verdict[0].strip() if verdict else "?")
    return 0


if __name__ == '__main__':
    sys.exit(main())
