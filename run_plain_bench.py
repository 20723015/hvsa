"""明文基线竞争批跑入口 (盲审补全执行计划 阶段 1.3 / 阶段 2).

机制克隆自 run_thesis_core.py (plan/run/manifest/断点续跑/code_digest),
产物落在独立 root (results_plain_bench/), 与主矩阵互不污染。

计划 (竞争基线正式实验, 阶段 2):
  明文基线  fedbuff / mafsa / dvafl (plain_bench.py)
            × {MNIST, FashionMNIST, CIFAR10} × {IID, NI(α=0.5)}
            × seeds {42..46}                             = 90 runs (全新)
  复用对照  g1_ref  = G1 (FedAvg 同步) 30 runs —— 复用 results_thesis_core
            ours_plain = G3v (冻结仿真器 verify-off) 30 runs —— 复用
            (阶段 0.2 对拍确认掩码对学习中性后; 见 docs/CLAIMS_MAP.md §4.2)

强约束 (与 run_thesis_core 一致, fail-fast):
  - 每个任务强制 --deterministic;
  - 默认串行 (workers=1);
  - 每 run 独立输出目录 <root>/<tag>/;
  - manifest (<root>/thesis_core_manifest.jsonl) 字段与主矩阵对齐,
    另含 arm/buffer_k/peak_rss_mb 明文特有字段;
  - 断点续跑: 同 config_hash + code_digest 的成功记录自动跳过 (--force 重跑)。

PARTS 自带四种分布: IID / NI(α=0.5, 主线) / NI01(α=0.1) / NI10(α=1.0),
后两者供阶段 3 边界扩展按需启用 (--parts NI01)。

用法:
  python run_plain_bench.py plan
  python run_plain_bench.py run --arms fedbuff --datasets MNIST --seeds 42
  python run_plain_bench.py run            # 全量 90 runs 串行
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

ROOT_DEFAULT = "results_plain_bench"
SUMMARY = "thesis_core_manifest.jsonl"

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["MNIST", "FashionMNIST", "CIFAR10"]
PARTS = [("IID", []),
         ("NI", ["--noniid", "--alpha", "0.5"]),
         ("NI01", ["--noniid", "--alpha", "0.1"]),
         ("NI10", ["--noniid", "--alpha", "1.0"])]
MAIN_PARTS = ("IID", "NI")     # 阶段 2 正式矩阵口径 (阶段 3 按需扩 NI01/NI10)

# 明文基线臂: plain_bench.py 的三臂, 聚合质量 buffer_k=12 对齐主方法
# G3v 每 checkpoint N_Q 均值 (11.2, 见 results_thesis_core G3v 实测)。
COMMON = ["--max-checkpoints", "20", "--max-ticks", "300",
          "--buffer-k", "12"]

ARMS = {
    "fedbuff": ("plain_bench.py", ["--arm", "fedbuff"], True),
    "mafsa":   ("plain_bench.py", ["--arm", "mafsa"], True),
    "dvafl":   ("plain_bench.py", ["--arm", "dvafl"], True),
}

ARM_NAMES = {
    "fedbuff": "FedBuff 逐客户端缓冲异步 (wait 口径公平加权)",
    "mafsa": "MAFSA Algorithm-1 掉线缓存 + 陈旧结果复用",
    "dvafl": "DVAFL 版本控制补偿 (矩校准 VCM)",
}

# 会影响实验语义的源码快照 (config_hash 之外的代码漂移检测)
CODE_FILES = [
    "run_plain_bench.py", "plain_bench.py",
    "models.py", "datasets.py", "utils.py", "main_point2.py",
]


def run_tag(arm, ds, part, seed):
    return f"{arm}_{ds}_{part}_s{seed}"


def build_plan(arms, datasets, parts=None, seeds=None):
    parts = [p for p in PARTS if not parts or p[0] in parts]
    jobs = []
    for a in arms:
        script, extra, use_common = ARMS[a]
        for ds in datasets:
            for part, part_args in parts:
                for s in seeds:
                    args = list(extra) + (COMMON if use_common else []) + list(part_args)
                    jobs.append((run_tag(a, ds, part, s), a, ds, part, s,
                                 script, args))
    return jobs


def default_plan():
    return build_plan(list(ARMS), DATASETS, list(MAIN_PARTS), SEEDS)


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


def _build_cmd(tag, ds, seed, script, extra, root, force=False):
    run_dir = os.path.join(root, tag)
    cmd = [sys.executable, os.path.abspath(script),
           '--dataset', ds, '--seed', str(seed),
           '--output-dir', os.path.abspath(run_dir),
           '--run-id', tag, '--deterministic'] + list(extra)
    if force:
        cmd.append('--force')
    return cmd


def _job_fingerprint(job, code_digest=None):
    _tag, arm, ds, part, seed, script, extra = job
    payload = {
        'arm': arm, 'dataset': ds, 'part': part, 'seed': int(seed),
        'script': os.path.basename(script), 'args': list(extra),
        'code_digest': code_digest or _code_digest(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]


def _collect_products(run_dir, run_id):
    """manifest 字段与 run_thesis_core._collect_products 对齐 (含明文特有字段)."""
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
                  'comm_total_bytes', 'torch_num_threads',
                  'arm', 'buffer_k', 'peak_rss_mb'):
            rec[k] = meta.get(k)
    if npz_hits:
        rec['npz'] = os.path.basename(npz_hits[0])
        try:
            d = np.load(npz_hits[0], allow_pickle=True)
            acc = np.asarray(d.get('acc_history'), dtype=np.float64)
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


def _run_one(tag, arm, ds, part, seed, script, extra, root, force=False,
             dry_run=False):
    run_dir = os.path.join(root, tag)
    code_digest = _code_digest()
    job = (tag, arm, ds, part, seed, script, extra)
    fingerprint = _job_fingerprint(job, code_digest)
    cmd = _build_cmd(tag, ds, seed, script, extra, root, force)
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
           'rc': rc, 'runner_wall_s': round(elapsed, 1),
           'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'cmd': ' '.join(cmd), 'log': os.path.basename(log_path),
           'run_dir': os.path.abspath(run_dir),
           'code_digest': code_digest, 'job_fingerprint': fingerprint}
    rec.update(_collect_products(run_dir, tag))
    with open(os.path.join(root, SUMMARY), 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if rc == 0 and rec.get('final_acc') is not None:
        status = (f"acc={rec['final_acc']:.2f}% tick={rec.get('final_tick')} "
                  f"wall={elapsed:.0f}s rss={rec.get('peak_rss_mb') or 0:.0f}MB")
    else:
        status = f"RC={rc} (见 {rec['log']})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag}: {status}", flush=True)
    return rec


def _resolve(args):
    arms = args.arms if args.arms else list(ARMS)
    datasets = args.datasets if args.datasets else DATASETS
    seeds = args.seeds if args.seeds else SEEDS
    parts = (args.parts if args.parts
             else [p[0] for p in PARTS if p[0] in MAIN_PARTS])
    return build_plan(arms, datasets, parts, seeds)


def cmd_plan(args):
    jobs = _resolve(args)
    print(f"共 {len(jobs)} 个任务 (串行执行, 强制 --deterministic)")
    by_arm = {}
    for _tag, arm, *_rest in jobs:
        by_arm[arm] = by_arm.get(arm, 0) + 1
    for arm, n in sorted(by_arm.items()):
        print(f"  {arm:<8} {n:>3} runs  {ARM_NAMES.get(arm, '')}")
    if args.verbose:
        for tag, *_ in jobs:
            print(f"  {tag}")


def cmd_run(args):
    jobs = _resolve(args)
    root = args.root
    os.makedirs(root, exist_ok=True)
    code_digest = _code_digest()
    done = {r.get('tag'): r.get('job_fingerprint')
            for r in _load_summary(os.path.join(root, SUMMARY))
            if r.get('rc') == 0 and r.get('job_fingerprint')}
    expected = {j[0]: _job_fingerprint(j, code_digest) for j in jobs}
    todo = [j for j in jobs
            if args.force or done.get(j[0]) != expected[j[0]]]
    print(f"计划 {len(jobs)} 个任务, 待运行 {len(todo)} "
          f"(仅 tag+任务配置+源码摘要一致才跳过)", flush=True)
    if args.dry_run:
        for job in todo:
            _run_one(*job, root=root, dry_run=True)
        return
    if args.workers > 1 and not args.allow_parallel:
        print("[warn] 默认串行; 并行需显式 --allow-parallel (并行会污染本机"
              "计时与确定性轨迹) → 强制 workers=1", flush=True)
        args.workers = 1
    results = []
    if args.workers == 1:
        for job in todo:
            # 进入 todo 说明旧产物不存在或指纹已变化; 统一覆盖同 tag 旧产物
            results.append(_run_one(*job, root=root, force=True))
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, *job, root=root, force=True)
                    for job in todo]
            for fut in cf.as_completed(futs):
                results.append(fut.result())
    ok = sum(1 for r in results if r and r.get('rc') == 0)
    print(f"完成: {ok}/{len(results)} 成功; manifest → "
          f"{os.path.join(root, SUMMARY)}")


def main():
    ap = argparse.ArgumentParser(description="明文基线批跑入口")
    sub = ap.add_subparsers(dest='cmd', required=True)

    def add_common(p):
        p.add_argument("--root", default=ROOT_DEFAULT)
        p.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
        p.add_argument("--datasets", nargs="+", default=None,
                       choices=DATASETS)
        p.add_argument("--parts", nargs="+", default=None,
                       choices=[p[0] for p in PARTS])
        p.add_argument("--seeds", nargs="+", type=int, default=None)

    p_plan = sub.add_parser('plan')
    add_common(p_plan)
    p_plan.add_argument("--verbose", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser('run')
    add_common(p_run)
    p_run.add_argument("--force", action='store_true',
                       help="忽略已完成记录重跑")
    p_run.add_argument("--dry-run", action='store_true')
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--allow-parallel", action='store_true')
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
