"""Summarize complete plaintext pairs without pooling unfinished experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def describe(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f}±{values.std(ddof=1):.2f}" if len(values) > 1 else f"{values[0]:.2f}"


def analyze(root):
    plan = json.loads((root / "screen_plan.json").read_text(encoding="utf-8"))
    cells = {}
    missing = []
    for dataset in plan["datasets"]:
        cells[dataset] = []
        for seed in plan["seeds"]:
            directory = root / f"{dataset}_NI_s{seed}"
            meta_path = directory / "pair_meta.json"
            if not meta_path.exists():
                missing.append(directory.name)
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["status"] != "complete":
                missing.append(directory.name)
                continue
            if not meta["shared_trace"] or not meta["paired_minibatch_rng"]:
                raise RuntimeError(f"unpaired experiment: {directory.name}")
            arrays = {}
            for arm in ("increment", "calibration"):
                with np.load(directory / f"{arm}.npz") as data:
                    arrays[arm] = {key: data[key] for key in data.files}
            baseline, calibrated = arrays["increment"], arrays["calibration"]
            for key in ("tick", "executed_updates", "executed_sgd_steps",
                        "executed_samples", "adopted_updates", "adopted_sgd_steps",
                        "adopted_samples"):
                if not np.array_equal(baseline[key], calibrated[key]):
                    raise RuntimeError(f"paired budget mismatch: {directory.name}/{key}")
            if baseline["accuracy"][0] != calibrated["accuracy"][0]:
                raise RuntimeError("cold-start trajectories differ")
            if len(calibrated["accuracy"]) != plan["max_checkpoints"]:
                raise RuntimeError(f"wrong checkpoint count: {directory.name}")
            gain = calibrated["accuracy"] - baseline["accuracy"]
            cells[dataset].append(dict(seed=seed, metadata=meta, arrays=arrays,
                                       final_gain=float(gain[-1]),
                                       last5_gain=float(gain[-5:].mean())))
    hashes = {row["metadata"]["scientific_config"]["code_hash"]
              for rows in cells.values() for row in rows}
    if len(hashes) > 1:
        raise RuntimeError("cannot pool multiple source versions")
    lines = ["# 历史增量差分校准：明文开发筛选", "",
             "日期：2026-10-08。以下只统计完整实验；非独立留出验证，非安全协议性能。", "",
             "统一设置：5 组×4 用户，FIFO，K=3，s_max=3，delay=1–5 ticks，",
             "训练前掉线率 0.1，上传丢包率 0，Dirichlet α=0.5；CNN、本地 SGD momentum=0.9；",
             f"{plan['max_checkpoints']} checkpoints。两个方法共享逐事件调度及 minibatch RNG。", "",
             "权重使用真实样本数；模型除以入选样本量、缓存除以总体样本量。缓存在采纳提交后生效。", "",
             "± 为种子样本标准差，增益为同种子配对差（pp）。95% 区间是探索性配对 t 区间，",
             "3 个开发种子不足以完成稳定性确认。测试集分数用于固定候选的探索观察，没有调参；",
             "正式选择参数和最终评价仍需独立验证规则。", "",
             "| 数据集 | 完成配对 | 增量聚合 | 缓存校准 | 最终配对增益 | 95% 配对区间 | 正种子 | 最差种子 | last-5 配对增益 |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    summary = {"phase": "development_screen", "missing": missing, "cells": {},
               "source_hash": next(iter(hashes), None),
               "screening_gate_type": "basic final-point gate, not stability confirmation"}
    for dataset, rows in cells.items():
        if not rows:
            continue
        base = [r["metadata"]["summary"]["increment"]["final_accuracy"] for r in rows]
        calibrated = [r["metadata"]["summary"]["calibration"]["final_accuracy"] for r in rows]
        gains = np.array([r["final_gain"] for r in rows])
        last5 = [r["last5_gain"] for r in rows]
        half = (float(t.ppf(.975, len(gains)-1) * gains.std(ddof=1) / np.sqrt(len(gains)))
                if len(gains) > 1 else None)
        interval = (f"[{gains.mean()-half:.2f}, {gains.mean()+half:.2f}]" if half is not None else "—")
        wins = int((gains > 0).sum())
        screen_pass = len(rows) == len(plan["seeds"]) and gains.mean() >= .5 and wins >= 2
        lines.append(f"| {dataset} | {len(rows)} | {describe(base)} | {describe(calibrated)} | "
                     f"{describe(gains)} | {interval} | {wins}/{len(rows)} | {gains.min():+.2f} | {describe(last5)} |")
        summary["cells"][dataset] = dict(n=len(rows), mean_gain=float(gains.mean()), wins=wins,
                                        mean_last5_gain=float(np.mean(last5)),
                                        worst_gain=float(gains.min()), screening_gate_met=bool(screen_pass),
                                        seeds=[r["seed"] for r in rows],
                                        gains=gains.tolist(), last5_gains=last5)
    lines += ["", "## 配对曲线与缓存诊断", "",
              f"![配对增益、缓存年龄、残差与校准项大小]({(root / 'screen_diagnostics.png').as_posix()})", "",
              "曲线只包含已完成的配对。缓存年龄、残差和校准项大小是诊断代理，",
              "不能单独识别倒退原因，也不是缺席客户端新鲜更新误差的直接测量。", "",
              "## 每种子与诊断", "",
              "缓存年龄按当前 checkpoint 版本减缓存产生时的训练基版本；从未参与的缓存单独统计，",
              "年龄均值不把它们记为零。年龄是在本轮提交前测量；残差范数是入选用户新旧增量差的样本加权 RMS。", "",
              "| 数据集/seed | tick | 执行 SGD 步 | 采纳 SGD 步 | 最终增益 pp | last-5 pp | 最终总体缓存年龄 | 最终总体未见样本占比 | 最终残差 RMS | 最大不变量误差 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    max_equivalence = max_invariant = 0.0
    for dataset, rows in cells.items():
        for row in rows:
            cal = row["arrays"]["calibration"]
            meta = row["metadata"]
            invariant = float(cal["invariant_max_abs_error"].max())
            max_invariant = max(max_invariant, invariant)
            max_equivalence = max(max_equivalence, meta["max_vcm_equivalence_error"])
            lines.append(f"| {dataset}/{row['seed']} | {cal['tick'][-1]} | "
                         f"{cal['executed_sgd_steps'][-1]} | {cal['adopted_sgd_steps'][-1]} | "
                         f"{row['final_gain']:+.2f} | {row['last5_gain']:+.2f} | "
                         f"{cal['global_cache_age_mean'][-1]:.2f} | "
                         f"{cal['global_unseen_mass_fraction'][-1]:.1%} | "
                         f"{cal['residual_rms_norm'][-1]:.3f} | {invariant:.2e} |")
    lines += ["", "## 可主张范围", "",
              f"显式增量与同集合默认 VCM 的最大逐坐标差：{max_equivalence:.3e}；",
              f"提交后缓存不变量最大逐坐标差：{max_invariant:.3e}。", "",
              "该检查验证明文公式和状态记账，没有验证 TA 回执、通知丢失、持久化恢复或多轮隐私。",
              "模拟时间与实际执行/采纳训练量在每个配对中相同。两个方法在同一进程交替训练，",
              "总墙钟和单臂训练计时不作为端到端加速证据。", "",
              "冷启动缓存为零，两臂首个 checkpoint 必须完全相同。当前未增加任何衰减、EMA、",
              "归一化或自适应阈值，避免混淆单模块效果。原计划的基本终点门槛为平均最终增益",
              "≥0.5pp 且至少 2/3 个种子为正，同时检查 last-5 和最差种子。",
              "`screening_gate_met` 只编码上述终点数值条件，没有编码稳定性结论。",
              "若基本门槛未通过，先按负结果处理；若通过但末段均值为负，也不能据此直接晋级。", "",
              "## 复现", "",
              "入口：`D:\\Anaconda\\python.exe run_cache_screen.py`；已有且配置匹配的完整配对会跳过。",
              "统计：`D:\\Anaconda\\python.exe analyze_cache_screen.py`。冻结命令见同目录",
              "`screen_plan.json`，每配对的训练配置、初始模型/划分/事件摘要和环境见 `pair_meta.json`。",
              "本次执行源码快照为 `source_snapshot.zip`，附环境与逐文件摘要清单 `run_environment.json`。",
              f"本次训练源码 SHA-256：`{summary['source_hash']}`。"]
    if missing:
        lines += ["", "未完成配对：" + ", ".join(missing)]
    (root / "screen_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "screen_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    available = [(name, rows) for name, rows in cells.items() if rows]
    if available:
        fig, axes = plt.subplots(4, len(available), figsize=(6*len(available), 12), squeeze=False)
        for column, (dataset, rows) in enumerate(available):
            for row in rows:
                baseline, cal = row["arrays"]["increment"], row["arrays"]["calibration"]
                cp = cal["checkpoint"]
                axes[0, column].plot(cp, cal["accuracy"]-baseline["accuracy"], label=f"seed {row['seed']}")
                ages = np.where(cal["global_cache_age_mean"] >= 0, cal["global_cache_age_mean"], np.nan)
                axes[1, column].plot(cp, ages, label=f"seed {row['seed']}")
                axes[2, column].plot(cp, cal["residual_rms_norm"], label=f"seed {row['seed']}")
                axes[3, column].plot(cp, cal["correction_to_increment_ratio"], label=f"seed {row['seed']}")
            axes[0, column].axhline(0, color="black", linewidth=.7)
            axes[0, column].set(title=dataset, ylabel="Paired accuracy gain (pp)")
            axes[1, column].set(ylabel="Weighted global cache age (before commit)")
            axes[2, column].set(ylabel="Selected residual RMS norm")
            axes[3, column].set(xlabel="Checkpoint (matched work within each seed)",
                                ylabel="Correction norm / increment norm")
            for axis in axes[:, column]:
                axis.grid(alpha=.2)
                axis.legend()
        fig.suptitle("Plaintext development screen: cached update calibration")
        fig.tight_layout()
        fig.savefig(root / "screen_diagnostics.png", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results_cache_calibration/screen_20261008")
    analyze(Path(parser.parse_args().root).resolve())
