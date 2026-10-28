"""
研究点2 批量消融实验运行器

覆盖文档 THESIS_POINT2_STALENESS_AWARE_REUSE.md Sec 8.1 的全部消融项:
  A. 5组主实验
  B. 10项消融
  C. 3类攻击实验 (独立于性能基线)

用法:
  python run_all_ablations.py                        # 全部运行 (需要 torch)
  python run_all_ablations.py --dry-run              # 仅打印命令，不执行
  python run_all_ablations.py --dataset CIFAR10      # 指定数据集
  python run_all_ablations.py --quick                # 快速模式 (少量 rounds)
  python run_all_ablations.py --skip-ablation 4-10   # 跳过指定编号

输出:
  results/                           # 所有 NPZ 数据和日志
  results/summary.json               # 汇总表
  results/summary.txt                # 可读汇总表
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

# ============================================================
# 实验定义
# ============================================================

# 五组主实验 (文档 L830-834)
MAIN_GROUPS = [
    {"id": "G1", "label": "Sync-HVSA",
     "desc": "研究点1同步基线，非本模拟",
     "args": "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule none --gamma-vcm 0"},
    {"id": "G2", "label": "Grouped+Inter-Group Sync",
     "desc": "组内训练，组间等所有组到齐 (K=M=5)",
     "args": "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule none --gamma-vcm 0"},
    {"id": "G3", "label": "Grouped+Buffer/Checkpoint Semi-Async",
     "desc": "K<M, buffer+checkpoint, 无 stale reuse, 无 VCM",
     "args": "-K 3 --s-max 3 --reuse-rule none --gamma-vcm 0"},
    {"id": "G4", "label": "G3+Intra-Group Stale Reuse",
     "desc": "K<M, buffer+checkpoint, reuse=ours, 无 VCM",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --gamma-vcm 0"},
    {"id": "G5", "label": "G4+Direct-VCM (完整方案)",
     "desc": "K<M, buffer+checkpoint, reuse=ours, VCM=on, γ=1.0",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0"},
]

# 十项消融 (文档 L900-911)
ABLATIONS = [
    # 1. reuse rule 三规则
    {"id": "A1", "label": "ablation-reuse-rule",
     "desc": "No Reuse vs MAFSA Original vs Ours Decayed-Stale",
     "variants": [
        ("none",  "--reuse-rule none"),
        ("mafsa", "--reuse-rule mafsa --staleness-beta 0.3"),
        ("ours",  "--reuse-rule ours --rho-max 0.5 --staleness-beta 0.3"),
    ],
     "base_args": "-K 3 --s-max 3 --upload-dropout-rate 0.2 --gamma-vcm 0"},

    # 2. No Compensation
    {"id": "A2", "label": "ablation-no-comp",
     "desc": "No Compensation (γ_vcm=0)",
     "variants": [("gamma0", "--gamma-vcm 0")],
     "base_args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2"},

    # 3. Direct-VCM
    {"id": "A3", "label": "ablation-vcm",
     "desc": "Direct-VCM (γ_vcm=1.0)",
     "variants": [("gamma1.0", "--vcm --gamma-vcm 1.0")],
     "base_args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2"},

    # 4. reuse × compensation 2×2
    {"id": "A4", "label": "ablation-reuse-x-comp",
     "desc": "reuse × compensation 2×2 消融",
     "variants": [
        ("noReuse_noComp",  "--reuse-rule none --gamma-vcm 0"),
        ("noReuse_Comp",    "--reuse-rule none --vcm --gamma-vcm 1.0"),
        ("Reuse_noComp",    "--reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --gamma-vcm 0"),
        ("Reuse_Comp",      "--reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --vcm --gamma-vcm 1.0"),
    ],
     "base_args": "-K 3 --s-max 3 --upload-dropout-rate 0.2"},

    # 5. ρ_max, β_c 敏感性
    {"id": "A5", "label": "ablation-rho-beta",
     "desc": "ρ_max ∈ {0.25, 0.5} × β_c ∈ {0.1, 0.3, 0.5}",
     "variants": [
        ("r0.25_b0.1", "--reuse-rule ours --rho-max 0.25 --staleness-beta 0.1"),
        ("r0.25_b0.3", "--reuse-rule ours --rho-max 0.25 --staleness-beta 0.3"),
        ("r0.25_b0.5", "--reuse-rule ours --rho-max 0.25 --staleness-beta 0.5"),
        ("r0.5_b0.1",  "--reuse-rule ours --rho-max 0.5  --staleness-beta 0.1"),
        ("r0.5_b0.3",  "--reuse-rule ours --rho-max 0.5  --staleness-beta 0.3"),
        ("r0.5_b0.5",  "--reuse-rule ours --rho-max 0.5  --staleness-beta 0.5"),
    ],
     "base_args": "-K 3 --s-max 3 --upload-dropout-rate 0.2 --gamma-vcm 0"},

    # 6. γ_vcm 敏感性
    {"id": "A6", "label": "ablation-gamma-vcm",
     "desc": "γ_vcm ∈ {0.5, 0.8, 1.0}",
     "variants": [
        ("g0.5", "--vcm --gamma-vcm 0.5"),
        ("g0.8", "--vcm --gamma-vcm 0.8"),
        ("g1.0", "--vcm --gamma-vcm 1.0"),
    ],
     "base_args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2"},

    # 7. K 敏感性
    {"id": "A7", "label": "ablation-K",
     "desc": "K ∈ {2, 3, 5}",
     "variants": [
        ("K2", "-K 2"),
        ("K3", "-K 3"),
        ("K5", "-K 5"),
    ],
     "base_args": "--s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0"},

    # 8. s_max 敏感性
    {"id": "A8", "label": "ablation-smax",
     "desc": "s_max ∈ {2, 3, 5}",
     "variants": [
        ("s2", "--s-max 2"),
        ("s3", "--s-max 3"),
        ("s5", "--s-max 5"),
    ],
     "base_args": "-K 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0"},

    # 9. policy 对比
    {"id": "A9", "label": "ablation-policy",
     "desc": "FIFO vs Fresh-first vs Fairness-aware Fresh-first",
     "variants": [
        ("fifo",      "--policy fifo"),
        ("fresh",     "--policy fresh_first"),
        ("fair",      "--policy fair_fresh_first --D-fair 2"),
    ],
     "base_args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0"},

    # 10. adaptive s_max
    {"id": "A10", "label": "ablation-adaptive-smax",
     "desc": "adaptive s_max off vs on",
     "variants": [
        ("off", ""),  # default
        ("on",  ""),  # placeholder — 需要代码支持 adaptive s_max
    ],
     "base_args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0",
     "note": "adaptive s_max 代码待实现，当前仅占位"},
]

# 机制交互检验 (THESIS_FULL_DESIGN 4.5.4 I1/I2): 各机制与架构开关的耦合
# I1: VCM 增益在 checkpoint on/off 下应显著不同 (补偿仅对 mixed-version 生效)
# I2: reuse 增益在 buffer on/off 下应显著不同 (复用机会随半异步时间线增多)
INTERACTION_CHECKS = [
    {"id": "I1", "label": "interaction-vcm-x-checkpoint",
     "desc": "VCM 增益 × checkpoint 开关 (补偿仅对 mixed-version 生效)",
     "variants": [
        ("ckptOff_vcmOff", "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule none --gamma-vcm 0"),
        ("ckptOff_vcmOn",  "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule none --vcm --gamma-vcm 1.0"),
        ("ckptOn_vcmOff",  "-K 3 --s-max 3 --reuse-rule none --gamma-vcm 0"),
        ("ckptOn_vcmOn",   "-K 3 --s-max 3 --reuse-rule none --vcm --gamma-vcm 1.0"),
    ],
     "base_args": ""},
    {"id": "I2", "label": "interaction-reuse-x-buffer",
     "desc": "stale reuse 增益 × buffer 开关 (复用机会随半异步时间线增多)",
     "variants": [
        ("bufOff_reuseOff", "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule none --gamma-vcm 0"),
        ("bufOff_reuseOn",  "-K 5 --s-max 10 --delta-timeout 0 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --gamma-vcm 0"),
        ("bufOn_reuseOff",  "-K 3 --s-max 3 --reuse-rule none --gamma-vcm 0"),
        ("bufOn_reuseOn",   "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --gamma-vcm 0"),
    ],
     "base_args": ""},
]

# 三类攻击实验 (以 G5 完整方案为基座)
ATTACKS = [
    {"id": "ATK1", "label": "attack-tamper-global",
     "desc": "篡改全局模型",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0 --verify --attack tamper_global"},
    {"id": "ATK2", "label": "attack-tamper-manifest",
     "desc": "篡改 checkpoint manifest",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0 --verify --attack tamper_manifest"},
    {"id": "ATK3", "label": "attack-drop-replace-pkg",
     "desc": "漏聚合/替换组包",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0 --verify --attack drop_replace_pkg"},
    {"id": "ATK4", "label": "attack-replay",
     "desc": "重放已消费组包 (Aggregated_IDs exactly-once)",
     "args": "-K 3 --s-max 3 --reuse-rule ours --rho-max 0.5 --staleness-beta 0.3 --upload-dropout-rate 0.2 --vcm --gamma-vcm 1.0 --verify --attack replay"},
]


# ============================================================
# 运行器
# ============================================================

def run_one(label: str, extra_args: str, dataset: str, out_dir: str,
            quick: bool, dry_run: bool, noniid: bool = False,
            alpha: float = 0.5, seed: int = 42) -> Optional[dict]:
    """运行一次 main_point2.py, 返回结果摘要."""
    max_ckpt = "5" if quick else "20"
    max_ticks = "80" if quick else "300"
    partition_args = "--noniid --alpha " + str(alpha) if noniid else ""
    cmd = (
        f"python main_point2.py "
        f"--dataset {dataset} "
        f"--seed {seed} "
        f"--max-checkpoints {max_ckpt} "
        f"--max-ticks {max_ticks} "
        f"{partition_args} "
        f"{extra_args}"
    )
    print(f"\n{'='*70}")
    print(f"[{label}]")
    print(f"  {cmd}")
    if dry_run:
        print("  (dry-run, skipped)")
        return None

    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"{label}.log")
    start = time.time()

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=f, stderr=subprocess.STDOUT, timeout=3600,
            )
        elapsed = time.time() - start
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = 3600
        ok = False

    return {
        "label": label, "ok": ok, "elapsed_s": round(elapsed, 1),
        "cmd": cmd, "log": log_file,
    }


def collect_npz_results(out_dir: str) -> List[dict]:
    """扫描 out_dir 下的 NPZ 文件, 提取关键指标."""
    import glob
    rows = []
    for npz_path in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        try:
            d = np.load(npz_path, allow_pickle=True)
            acc = d.get("acc_history", None)
            final_acc = float(d.get("final_acc", 0))
            wall = float(d.get("wall_clock", 0))
            avg_wait = float(d.get("avg_wait", 0))
            comp_gain = float(d.get("comp_gain", 0))
            total_stale = int(d.get("total_stale_reuses", 0))
            expired = int(d.get("expired_buffer_count", 0))
            fairness = float(d.get("fairness_var", 0))
            verify_pass = int(d.get("verify_pass_count", 0))
            verify_total = int(d.get("verify_total_count", 0))
            attack_detected = float(d.get("attack_detected", 0))
            config = str(d.get("config", ""))
            rows.append({
                "file": os.path.basename(npz_path),
                "final_acc": round(final_acc, 2),
                "wall_s": round(wall, 1),
                "avg_wait": round(avg_wait, 1),
                "comp_gain": round(comp_gain, 4),
                "total_stale": total_stale,
                "expired": expired,
                "fairness_var": round(fairness, 4),
                "verify_pass": verify_pass,
                "verify_total": verify_total,
                "attack_detected": attack_detected,
                "is_attack": ("attack=" in config and "attack='none'" not in config),
                "config": config,
                "acc_history": (
                    acc.tolist() if acc is not None and len(acc.shape) > 0 else []
                ),
            })
        except Exception as e:
            print(f"  WARNING: 无法读取 {npz_path}: {e}")
    return rows


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="研究点2 批量消融实验")
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed shared by all runs (IID/Non-IID comparable)")
    parser.add_argument("--noniid", action="store_true",
                        help="run the ablation matrix with Dirichlet Non-IID data")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration used with --noniid")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令")
    parser.add_argument("--quick", action="store_true", help="快速模式 (少量 rounds)")
    parser.add_argument("--out-dir", default=None, help="输出目录 (默认 results_YYYYMMDD-HHMMSS)")
    parser.add_argument("--skip-ablation", default="", help="跳过指定消融编号，如 '4-10'")
    parser.add_argument("--only", default="", help="仅运行指定块: main/ablation/attack")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or f"results_{args.dataset}_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    skip_range = set()
    if args.skip_ablation:
        for part in args.skip_ablation.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                skip_range.update(range(int(lo), int(hi) + 1))
            else:
                skip_range.add(int(part))

    all_results = []

    def run_block(title, items):
        nonlocal all_results
        print(f"\n{'#'*70}")
        print(f"# {title}")
        print(f"{'#'*70}")
        for item in items:
            if item.get("skip"):
                continue
            if "variants" in item:
                # 消融项: 每个 variant 跑一次
                for vname, vargs in item["variants"]:
                    label = f"{item['id']}_{vname}"
                    extra = f"{item['base_args']} {vargs}".strip()
                    r = run_one(label, extra, args.dataset, out_dir,
                                args.quick, args.dry_run, args.noniid, args.alpha,
                                args.seed)
                    if r:
                        r["group"] = item["id"]
                        r["desc"] = item["desc"]
                        r["variant"] = vname
                        all_results.append(r)
                    if args.dry_run:
                        print(f"  variant: {vname} | {vargs}")
            else:
                r = run_one(item["id"], item["args"], args.dataset, out_dir,
                            args.quick, args.dry_run, args.noniid, args.alpha,
                            args.seed)
                if r:
                    r["group"] = item["id"]
                    r["desc"] = item.get("desc", "")
                    r["variant"] = ""
                    all_results.append(r)

    # --- 执行 ---
    if not args.only or args.only == "main":
        run_block("五组主实验 (文档 L830-834)", MAIN_GROUPS)

    if not args.only or args.only == "ablation":
        run_block("机制交互检验 (协同证据 I1/I2)", INTERACTION_CHECKS)
        for i, abl in enumerate(ABLATIONS, 1):
            if i in skip_range:
                print(f"\n--- 跳过消融 {i}: {abl['label']} ---")
                continue
            if abl.get("note"):
                print(f"\n--- NOTE: {abl['note']} ---")
            run_block(f"消融 {i}/10: {abl['desc']} ({abl['label']})", [abl])

    if not args.only or args.only == "attack":
        run_block("攻击实验 (独立于性能基线)", ATTACKS)

    # --- 汇总 ---
    print(f"\n{'='*70}")
    print(f"全部完成: {len(all_results)} 次运行")
    if args.dry_run:
        print("(dry-run, 无实际输出)")
        return

    ok_count = sum(1 for r in all_results if r["ok"])
    fail_count = len(all_results) - ok_count
    print(f"  成功: {ok_count}  失败: {fail_count}")

    # 扫描 NPZ 汇总
    npz_rows = collect_npz_results(out_dir)
    summary = {
        "dataset": args.dataset,
        "timestamp": ts,
        "out_dir": out_dir,
        "total_runs": len(all_results),
        "ok": ok_count,
        "fail": fail_count,
        "results": [
            {"label": r["label"], "ok": r["ok"], "elapsed_s": r["elapsed_s"]}
            for r in all_results
        ],
        "npz_metrics": npz_rows,
    }

    # 写汇总 JSON
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")

    # 写可读文本汇总
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"研究点2 批量消融实验  数据集={args.dataset}  {ts}\n")
        f.write(f"{'='*60}\n")
        if npz_rows:
            f.write(f"{'实验':<40} {'Acc%':>8} {'墙钟s':>8} {'等待':>6} {'CompGain':>10} {'Stale':>6} {'过期':>5} {'验证':>7} {'检率%':>7}\n")
            f.write(f"{'-'*40} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*6} {'-'*5} {'-'*7} {'-'*7}\n")
            for row in npz_rows:
                vt = row['verify_total']
                det = f"{row['attack_detected']*100:.0f}" if (vt > 0 and row['is_attack']) else "-"
                f.write(
                    f"{row['file']:<40} {row['final_acc']:>8.2f} {row['wall_s']:>8.1f} "
                    f"{row['avg_wait']:>6.1f} {row['comp_gain']:>10.4f} "
                    f"{row['total_stale']:>6} {row['expired']:>5} "
                    f"{row['verify_pass']:>3}/{vt:>3} {det:>7}\n"
                )
        f.write(f"\n{'='*60}\n")
        f.write(f"运行汇总: {ok_count}/{len(all_results)} 成功\n")
    print(f"文本汇总已保存: {txt_path}")


if __name__ == "__main__":
    main()
