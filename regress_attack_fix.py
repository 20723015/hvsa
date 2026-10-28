"""history_missing v≥1 守卫修复的回归证明 (盲审补全执行计划 阶段 4.5).

修复唯一触及 main_point2.py 的 history_missing 注入分支 —— 该分支仅在
--attack history_missing 时可达, 性能路径 (--attack none) 与其他攻击类
零影响。本脚本以可复核的方式证明"既有单元行为零变化":

  1. 在 CPU (CUDA_VISIBLE_DEVICES="") 上原配置重跑 raw_sum 类 seed 42
     (run_attack_evidence.py --classes raw_sum --seed 42 --units 100);
  2. 与存档 results_thesis_core_attack/raw_sum_tamper_global_s42/ 的
     AttackOutcome JSONL 逐条逐字段对拍 (schedule/outcome 字段全部由
     tick/rng 整数序列决定, 与浮点设备无关; acc 不在 AttackOutcome 内);
  3. 判定写入 results_regress_attack/REGRESSION_VERDICT.txt。

退出码: 0=逐条一致, 1=不一致。
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(ROOT, "results_thesis_core_attack",
                       "raw_sum_tamper_global_s42",
                       "attack_outcomes_raw_sum_tamper_global_s42.jsonl")
REGROOT = os.path.join(ROOT, "results_regress_attack")
RERUN = os.path.join(REGROOT, "raw_sum_tamper_global_s42",
                     "attack_outcomes_raw_sum_tamper_global_s42.jsonl")


def read_jsonl(path):
    recs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def diff_record(a, b, path=""):
    """递归比较两条 outcome 记录, 返回差异列表."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: 仅单侧存在")
            else:
                diffs.extend(diff_record(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: 列表长度 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(diff_record(x, y, f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def main():
    os.makedirs(REGROOT, exist_ok=True)
    if not os.path.exists(ARCHIVE):
        print(f"[abort] 存档不存在: {ARCHIVE}")
        return 1

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, os.path.join(ROOT, "run_attack_evidence.py"),
           "--root", REGROOT, "--units", "100", "--seed", "42",
           "--classes", "raw_sum"]
    print("[regress] CPU 重跑:", ' '.join(cmd), flush=True)
    log = os.path.join(REGROOT, "regress_raw_sum_s42.log")
    with open(log, 'w', encoding='utf-8', errors='replace') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                             env=env, cwd=ROOT)
    print(f"[regress] 重跑 rc={rc}", flush=True)

    if not os.path.exists(RERUN):
        print(f"[regress] 重跑无产物: {RERUN} (见 {log})")
        return 1

    arch, rerun = read_jsonl(ARCHIVE), read_jsonl(RERUN)
    print(f"[regress] 存档 {len(arch)} 条, 重跑 {len(rerun)} 条", flush=True)
    all_diffs = []
    if len(arch) != len(rerun):
        all_diffs.append(f"记录数: 存档 {len(arch)} != 重跑 {len(rerun)}")
    for i, (a, b) in enumerate(zip(arch, rerun)):
        ds = diff_record(a, b, f"rec[{i}]")
        if ds:
            all_diffs.extend(ds[:5])

    verdict_path = os.path.join(REGROOT, "REGRESSION_VERDICT.txt")
    with open(verdict_path, 'w', encoding='utf-8') as f:
        f.write("=== history_missing v≥1 守卫修复回归判定 (raw_sum s42 整类) ===\n")
        f.write(f"存档: {ARCHIVE} ({len(arch)} 条)\n")
        f.write(f"重跑: {RERUN} ({len(rerun)} 条, CPU)\n")
        if all_diffs:
            f.write(f"判定: FAIL — {len(all_diffs)} 处差异\n")
            for d in all_diffs[:20]:
                f.write(f"  - {d}\n")
        else:
            f.write("判定: PASS — 逐条逐字段零差异, 既有单元行为零变化\n")
    print(open(verdict_path, encoding='utf-8').read())
    return 0 if not all_diffs else 1


if __name__ == "__main__":
    sys.exit(main())
