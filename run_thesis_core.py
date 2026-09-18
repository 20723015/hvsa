"""论文主表批跑入口 (thesis-core): 单一代码/配置/种子口径的串行确定性矩阵.

计划 ("稳过盲审"四周补强计划第 2 周):
  主矩阵  G2/G3/G3v/G5 × {MNIST, FashionMNIST, CIFAR10} × {IID, NI(α=0.5)}
          × seeds {42..46}                                        = 120 runs
  参考    G1 (FedAvg 同步) × 3 数据集 × 2 分布 × 5 seeds           =  30 runs
  消融    GFB (组包级 K=1, 同模拟器) × {FashionMNIST,
          CIFAR10} × 2 分布 × 5 seeds                             =  20 runs
  复现    G3v CIFAR10 × 2 分布 × seed 42 各再跑一次 (逐值可复现校验)
                                                                  =   2 runs
  可选    --with-sigma-ablation: G3s (仅 σ 幂衰减) × 3 × 2 × 5      =  30 runs

强约束 (主表证据口径, fail-fast):
  - 每个任务强制注入 --deterministic (严格确定性模式);
  - 默认串行执行 (workers=1): 并行会污染本机计时与 GPU 轨迹, 需显式
    --allow-parallel 才允许并行;
  - 每次运行独立输出目录 <root>/<tag>/, npz/meta/log 同目录存放;
  - manifest (<root>/thesis_core_manifest.jsonl) 记录配置摘要、acc 曲线摘要、
    最终 tick、执行/采纳更新数、验证耗时、通信量与本机运行时间;
  - 断点续跑: 已有同 config_hash 的成功记录自动跳过 (--force 重跑)。

产物目录与旧批次 (results_*、ASIG_* 等) 完全分离, 旧批次只作探索记录,
不进入论文主表统计。

用法:
  python run_thesis_core.py plan                      # 打印完整计划
  python run_thesis_core.py run                       # 全量串行执行 (可中断续跑)
  python run_thesis_core.py run --arms G3v --datasets CIFAR10 --seeds 42
  python run_thesis_core.py verify-repeat             # 复现轨迹逐值校验
"""

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT_DEFAULT = "results_thesis_core"
SUMMARY = "thesis_core_manifest.jsonl"

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
PARTS = [("IID", []), ("NI", ["--noniid", "--alpha", "0.5"])]
BASELINE_DATASETS = ["FashionMNIST", "CIFAR10"]
REPEAT_SEED = 42

# 主表臂 (G3v = 主方法), G1 = 同步参考, GFB = 组包级 K=1 异步消融
MAIN_ARMS = ["G2", "G3", "G3v", "G5"]
REF_ARMS = ["G1"]
BASELINE_ARMS = ["GFB"]
REPEAT_ARMS = ["G3v"]

# arm -> (脚本, 额外参数, 是否附加 COMMON)
ARMS = {
    "G1":  ("fedavg_point2.py", ["--rounds", "20"], False),
    "G2":  ("main_point2.py", ["-K", "5", "--s-max", "3", "--reuse-rule", "none"], True),
    "G3":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none"], True),
    "G3s": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--staleness-sigma", "0.5"], True),
    "G3v": ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--vcm", "--gamma-vcm", "1.0"], True),
    "G5":  ("main_point2.py", ["-K", "3", "--s-max", "3", "--reuse-rule", "none",
                               "--vcm", "--gamma-vcm", "1.0",
                               "--staleness-sigma", "0.5"], True),
    # 组包级 K=1 异步消融: 任一组包到达即聚合，陈旧度仍受 s_max 约束。
    # 它只用于观察 checkpoint 阈值，不作为逐客户端 FedBuff 复现或竞争基线。
    "GFB": ("main_point2.py", ["-K", "1", "--s-max", "3", "--reuse-rule", "none"], True),
}

COMMON = ["--max-checkpoints", "20", "--max-ticks", "300"]

# 会影响实验语义的本地源码快照。config_hash 只覆盖命令行科学配置，不能
# 识别长批跑期间的未提交源码变化；因此 runner 另行把这些文件的内容摘要
# 纳入 job_fingerprint。列表保守地覆盖主入口及其协议/数据依赖。
CODE_FILES = [
    "run_thesis_core.py", "main_point2.py", "fedavg_point2.py",
    "cloud.py", "edge.py", "ta.py", "user.py", "utils.py",
    "models.py", "datasets.py", "point2_types.py", "commit_tx.py",
]

ARM_NAMES = {
    "G1": "FedAvg 同步参考",
    "G2": "组间同步 (K=5, 无机制)",
    "G3": "①检查点式组级半异步 (K=3)",
    "G3s": "①+σ 幂衰减 (仅 σ, 消融)",
    "G3v": "①+Direct-VCM 后置补偿 (主方法)",
    "G5": "①+VCM+σ",
    "GFB": "组包级 K=1 异步消融 (非 FedBuff 复现)",
}


def run_tag(arm, ds, part, seed, rep=False):
    return f"{arm}_{ds}_{part}_s{seed}" + ("_rep" if rep else "")


def build_plan(arms, datasets, parts=None, seeds=None,
               with_sigma_ablation=False):
    """返回 [(tag, arm, dataset, part, seed, rep, script, extra_args)]."""
    arms = list(arms)
    datasets = list(datasets)
    parts = [p for p in PARTS if not parts or p[0] in parts]
    seeds = list(seeds)
    if with_sigma_ablation and 'G3s' not in arms:
        arms = arms + ['G3s']

    jobs = []

    def add(a, ds, part, seed, part_args, rep=False):
        script, extra, use_common = ARMS[a]
        args = list(extra) + (COMMON if use_common else []) + list(part_args)
        jobs.append((run_tag(a, ds, part, seed, rep), a, ds, part, seed, rep,
                     script, args))

    for a in arms:
        ds_list = ([d for d in datasets if d in BASELINE_DATASETS]
                   if a == 'GFB' else datasets)
        for ds in ds_list:
            for part, part_args in parts:
                for s in seeds:
                    add(a, ds, part, s, part_args)
                    # 复现校验对: 主表 run 之外再跑一次同配置 (CIFAR10, seed 42)
                    if a in REPEAT_ARMS and ds == 'CIFAR10' and s == REPEAT_SEED:
                        add(a, ds, part, s, part_args, rep=True)
    return jobs


def default_plan(with_sigma_ablation=False):
    return build_plan(MAIN_ARMS + REF_ARMS + BASELINE_ARMS, DATASETS, None,
                      SEEDS, with_sigma_ablation)


def _load_summary(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _code_digest():
    h = hashlib.sha256()
    for path in sorted(CODE_FILES):
        h.update(path.encode('utf-8'))
        try:
            with open(path, 'rb') as f:
                h.update(f.read())
        except OSError:
            h.update(b'<missing>')
    return h.hexdigest()[:16]


def _build_cmd(tag, ds, seed, script, extra, root, force=False,
               extra_args=None):
    run_dir = os.path.join(root, tag)
    cmd = [sys.executable, os.path.abspath(script),
           '--dataset', ds, '--seed', str(seed),
           '--output-dir', os.path.abspath(run_dir),
           '--run-id', tag, '--deterministic'] + list(extra)
    if extra_args:
        cmd += list(extra_args)
    if force:
        cmd.append('--force')
    return cmd


def _job_fingerprint(job, extra_args=None, code_digest=None):
    """tag 之外的科学任务 + 源码快照摘要；用于可靠断点续跑。"""
    _tag, arm, ds, part, seed, _rep, script, extra = job
    payload = {
        'arm': arm, 'dataset': ds, 'part': part, 'seed': int(seed),
        'script': os.path.basename(script), 'args': list(extra),
        'extra_args': list(extra_args or []),
        'code_digest': code_digest or _code_digest(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]


def _collect_products(run_dir, run_id):
    """提取主表所需字段 (配置摘要/曲线摘要/ticks/更新数/耗时/开销)."""
    import numpy as np
    rec = {}
    meta_path = os.path.join(run_dir, f"point2_meta_{run_id}.json")
    npz_hits = glob.glob(os.path.join(run_dir, f"point2_data_{run_id}.npz"))
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:  # noqa: BLE001
            meta = {}
        for k in ('config_hash', 'arr_digest', 'final_acc', 'wall_clock',
                  'deterministic', 'git_commit', 'git_dirty',
                  'adopted_packages', 'adopted_user_updates',
                  'group_packages_sealed', 'verify_seconds_total',
                  'comm_total_bytes', 'torch_num_threads'):
            rec[k] = meta.get(k)
    if npz_hits:
        rec['npz'] = os.path.basename(npz_hits[0])
        try:
            d = np.load(npz_hits[0], allow_pickle=True)
            acc = np.asarray(d.get('acc_history'), dtype=np.float64)
            # acc_history 为 (checkpoint_index, accuracy) 两列
            if acc.ndim == 2 and acc.shape[1] >= 2:
                acc = acc[:, 1]
            ticks = d.get('checkpoint_ticks')
            rec['checkpoints'] = int(len(acc))
            rec['final_tick'] = (int(ticks[-1])
                                 if ticks is not None and len(ticks) else None)
            if len(acc) and rec.get('final_acc') is None:
                rec['final_acc'] = float(acc[-1])
        except Exception:  # noqa: BLE001
            pass
    return rec


def _run_one(tag, arm, ds, part, seed, rep, script, extra, root, force=False,
             dry_run=False, extra_args=None):
    run_dir = os.path.join(root, tag)
    code_digest = _code_digest()
    job = (tag, arm, ds, part, seed, rep, script, extra)
    fingerprint = _job_fingerprint(job, extra_args, code_digest)
    cmd = _build_cmd(tag, ds, seed, script, extra, root, force, extra_args)
    if dry_run:
        print(f"[{tag}] " + ' '.join(cmd))
        return None
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"run_{tag}.log")
    t0 = time.time()
    with open(log_path, 'w', encoding='utf-8', errors='replace') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    rec = {'tag': tag, 'arm': arm, 'dataset': ds, 'part': part, 'seed': seed,
           'rep': bool(rep), 'rc': rc, 'runner_wall_s': round(elapsed, 1),
           'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'cmd': ' '.join(cmd), 'log': os.path.basename(log_path),
           'run_dir': os.path.abspath(run_dir),
           'extra_args': list(extra_args) if extra_args else None,
           'code_digest': code_digest, 'job_fingerprint': fingerprint}
    rec.update(_collect_products(run_dir, tag))
    with open(os.path.join(root, SUMMARY), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if rc == 0 and rec.get('final_acc') is not None:
        status = (f"acc={rec['final_acc']:.2f}% tick={rec.get('final_tick')} "
                  f"wall={elapsed:.0f}s")
    else:
        status = f"RC={rc} (见 {rec['log']})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}: {status}", flush=True)
    return rec


def _resolve(args, plan_cmd=False):
    arms = args.arms if args.arms else (MAIN_ARMS + REF_ARMS + BASELINE_ARMS +
                                        (['G3s'] if args.with_sigma_ablation
                                         else []))
    datasets = args.datasets if args.datasets else DATASETS
    seeds = args.seeds if args.seeds else SEEDS
    return build_plan(arms, datasets, args.parts, seeds,
                      args.with_sigma_ablation)


def cmd_plan(args):
    jobs = _resolve(args)
    print(f"共 {len(jobs)} 个任务 (串行执行, 强制 --deterministic)")
    by_arm = {}
    for _tag, arm, *_rest in jobs:
        by_arm[arm] = by_arm.get(arm, 0) + 1
    for arm, n in sorted(by_arm.items()):
        print(f"  {arm:<5} {n:>3} runs  {ARM_NAMES.get(arm, '')}")
    if args.verbose:
        for tag, *_ in jobs:
            print(f"  {tag}")


def cmd_run(args):
    jobs = _resolve(args)
    root = args.root
    os.makedirs(root, exist_ok=True)
    extra_args = (shlex.split(args.extra_args) if args.extra_args else None)
    if extra_args:
        if os.path.abspath(root) == os.path.abspath(ROOT_DEFAULT):
            raise SystemExit(
                "--extra-args 仅供冒烟，禁止写入正式 root；请显式使用 "
                "--root results_thesis_smoke")
        print(f"[warn] --extra-args 会改变配置 (config_hash 随之变化); 已记入 "
              f"manifest.extra_args 以便审计, 主表请勿使用: {extra_args}",
              flush=True)
    code_digest = _code_digest()
    done = {r.get('tag'): r.get('job_fingerprint')
            for r in _load_summary(os.path.join(root, SUMMARY))
            if r.get('rc') == 0 and r.get('job_fingerprint')}
    expected = {j[0]: _job_fingerprint(j, extra_args, code_digest) for j in jobs}
    todo = [j for j in jobs
            if args.force or done.get(j[0]) != expected[j[0]]]
    print(f"计划 {len(jobs)} 个任务, 待运行 {len(todo)} "
          f"(仅 tag+任务配置+源码摘要一致才跳过)", flush=True)
    if args.dry_run:
        for job in todo:
            _run_one(*job, root=root, dry_run=True, extra_args=extra_args)
        return
    if args.workers > 1 and not args.allow_parallel:
        print("[warn] thesis-core 默认串行; 并行需显式 --allow-parallel "
              "(并行会污染本机计时与确定性轨迹) → 强制 workers=1", flush=True)
        args.workers = 1
    results = []
    if args.workers == 1:
        for job in todo:
            # 进入 todo 说明旧产物不存在或指纹已变化；统一覆盖同 tag 旧产物，
            # 避免底层入口仅按自身 config_hash 跳过源码变化后的重跑。
            results.append(_run_one(*job, root=root, force=True,
                                    extra_args=extra_args))
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, *job, root=root, force=True,
                              extra_args=extra_args) for job in todo]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
    ok = sum(1 for r in results if r and r.get('rc') == 0)
    print(f"完成: {ok}/{len(results)} 成功; manifest → "
          f"{os.path.join(root, SUMMARY)}")


def cmd_verify_repeat(args):
    """复现校验: 主表 run 与 _rep run 的轨迹必须逐值一致 (严格确定性)."""
    import numpy as np
    root = args.root
    failures, pairs = [], 0
    rows = _load_summary(os.path.join(root, SUMMARY))
    latest = {r.get('tag'): r for r in rows if r.get('rc') == 0}
    for arm in REPEAT_ARMS:
        for part in ("IID", "NI"):
            tag = run_tag(arm, "CIFAR10", part, REPEAT_SEED)
            rep = tag + "_rep"
            a = glob.glob(os.path.join(root, tag, f"point2_data_{tag}.npz"))
            b = glob.glob(os.path.join(root, rep, f"point2_data_{rep}.npz"))
            if not a or not b:
                print(f"[skip] {tag}: 缺产物 (主表或 repeat 未跑)")
                continue
            pairs += 1
            da = np.load(a[0], allow_pickle=True)
            db = np.load(b[0], allow_pickle=True)
            acc_a, acc_b = da['acc_history'], db['acc_history']
            same = bool(np.array_equal(acc_a, acc_b))
            max_diff = (float(np.max(np.abs(acc_a - acc_b)))
                        if len(acc_a) else 0.0)
            ticks_same = bool(np.array_equal(da['checkpoint_ticks'],
                                             db['checkpoint_ticks']))
            fp_same = (latest.get(tag, {}).get('job_fingerprint') is not None
                       and latest.get(tag, {}).get('job_fingerprint')
                       == latest.get(rep, {}).get('job_fingerprint'))
            cfg_same = (latest.get(tag, {}).get('config_hash') is not None
                        and latest.get(tag, {}).get('config_hash')
                        == latest.get(rep, {}).get('config_hash'))
            ok = same and ticks_same and fp_same and cfg_same
            print(f"[{'PASS' if ok else 'FAIL'}] {tag}: acc 逐值一致={same} "
                  f"(max|Δ|={max_diff:.3e}), ticks 一致={ticks_same}, "
                  f"任务指纹一致={fp_same}, config_hash 一致={cfg_same}")
            if not ok:
                failures.append(tag)
    if pairs == 0:
        print("没有可校验的复现对; 先运行: python run_thesis_core.py run "
              "--arms G3v --datasets CIFAR10")
        return
    if failures:
        print(f"复现校验失败: {failures}")
        sys.exit(1)
    print(f"复现校验通过: {pairs} 对 (同 seed 同配置逐值一致)")


def main():
    ap = argparse.ArgumentParser(description="thesis-core 批跑入口")
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_common(p):
        p.add_argument("--root", default=ROOT_DEFAULT)
        p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
        p.add_argument("--datasets", nargs="+", default=None,
                       choices=DATASETS)
        p.add_argument("--parts", nargs="+", default=None,
                       choices=["IID", "NI"])
        p.add_argument("--seeds", nargs="+", type=int, default=None)
        p.add_argument("--with-sigma-ablation", action="store_true",
                       help="追加 G3s (仅 σ) 消融批次 (+30 runs)")

    p_plan = sub.add_parser('plan')
    add_common(p_plan)
    p_plan.add_argument("--verbose", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser('run')
    add_common(p_run)
    p_run.add_argument("--force", action="store_true",
                       help="忽略已完成记录重跑")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--allow-parallel", action="store_true")
    p_run.add_argument("--extra-args", default="",
                       help="附加到每个任务的额外参数 (仅用于流水线冒烟; "
                            "会改变 config_hash 并记入 manifest)")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser('verify-repeat')
    p_rep.add_argument("--root", default=ROOT_DEFAULT)
    p_rep.set_defaults(func=cmd_verify_repeat)

    args = ap.parse_args()
    if args.cmd in ('plan', 'run') and 'with_sigma_ablation' not in args:
        args.with_sigma_ablation = False
    args.func(args)


if __name__ == "__main__":
    main()
