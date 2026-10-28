"""per-role 内存埋点 (盲审补全执行计划 阶段 4.1, 全新代码, 不改冻结文件).

单进程架构下 per-role 内存无法用外部观察拆分, 因此在**角色调用点**做
tracemalloc bracketing + 进程 RSS 采样:
  - ES 验证      EdgeServer.p2_vote_checkpoint_with_result (main_point2.py:1109)
  - TA 授权      TrustedAuthority.p2_authorize_checkpoint      (:1147)
  - commit 事务  commit_tx.run_commit_transaction              (:1169)
  - Cloud 聚合   CloudServer.p2_apply_vcm_and_average          (聚合路径)
  - TA 掩码构建  TrustedAuthority.p2_build_checkpoint_mask      (TA 侧 O(d))

实现方式: 运行时类方法/模块函数包装 (monkey-patch), 冻结文件零改动 ——
code_digest 与主矩阵保持一致。角色调用序列与 main_point2 主循环一致。

用法 (parent 子进程两层, 保证补丁先于仿真生效):
  python exp_memory_overhead.py --dataset MNIST --model Tiny --verify \\
      --max-checkpoints 20 --arm-args "--vcm --gamma-vcm 1.0"
输出: memory_overhead_report.json + 控制台 per-role 表。

角色对象图尺寸 (静态构成, 论文口径): 全局模型历史 = n_versions × d × 8B;
ES 单组包掩码缓冲 = 在线用户数 × d × 8B; 二者由配置直接可算, 不在运行时
递归测量 (避免 gc 噪声), 报告中以 analytic 字段给出。
"""

import argparse
import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPORT_DEFAULT = "memory_overhead_report.json"


def _child(child_args, report_path):
    """子进程: 打补丁 → runpy 执行 main_point2 __main__ → 落盘记录."""
    import gc

    import edge
    import cloud as cloud_mod
    import commit_tx
    import main_point2  # noqa: F401  (确保模块先加载, 再打补丁)
    import psutil
    import ta as ta_mod

    proc = psutil.Process()
    records = {r: [] for r in ('es_verify', 'ta_authorize', 'commit_tx',
                               'cloud_aggregate', 'ta_mask_build')}
    rss_samples = {r: [] for r in records}

    def wrap_method(cls, name, role):
        orig = getattr(cls, name)

        def wrapped(self, *a, **k):
            gc.collect()
            tracemalloc.start()
            rss0 = proc.memory_info().rss
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                el = time.perf_counter() - t0
                _cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records[role].append({'wall_s': el, 'peak_b': peak})
                rss_samples[role].append(
                    (rss0, proc.memory_info().rss))
        setattr(cls, name, wrapped)

    def wrap_module(mod, name, role):
        orig = getattr(mod, name)

        def wrapped(*a, **k):
            gc.collect()
            tracemalloc.start()
            rss0 = proc.memory_info().rss
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                el = time.perf_counter() - t0
                _cur, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records[role].append({'wall_s': el, 'peak_b': peak})
                rss_samples[role].append(
                    (rss0, proc.memory_info().rss))
        setattr(mod, name, wrapped)

    wrap_method(edge.EdgeServer, 'p2_vote_checkpoint_with_result', 'es_verify')
    wrap_method(ta_mod.TrustedAuthority, 'p2_authorize_checkpoint',
                'ta_authorize')
    wrap_method(cloud_mod.CloudServer, 'p2_apply_vcm_and_average',
                'cloud_aggregate')
    wrap_method(ta_mod.TrustedAuthority, 'p2_build_checkpoint_mask',
                'ta_mask_build')
    wrap_module(commit_tx, 'run_commit_transaction', 'commit_tx')

    import runpy
    mp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "main_point2.py")
    sys.argv = ["main_point2.py"] + list(child_args)
    try:
        runpy.run_path(mp_path, run_name="__main__")
    except SystemExit as e:  # main_point2 正常结束也走 SystemExit(0)? 无: 直接返回
        if e.code not in (None, 0):
            raise

    # 汇总
    summary = {}
    for role, recs in records.items():
        if not recs:
            continue
        peaks = [r['peak_b'] for r in recs]
        walls = [r['wall_s'] for r in recs]
        deltas = [(b - a) / (1024 * 1024) for a, b in rss_samples[role]]
        summary[role] = {
            'n_calls': len(recs),
            'tracemalloc_peak_mb_mean': round(sum(peaks) / len(peaks)
                                              / (1024 * 1024), 3),
            'tracemalloc_peak_mb_max': round(max(peaks) / (1024 * 1024), 3),
            'rss_delta_mb_mean': round(sum(deltas) / len(deltas), 3),
            'rss_delta_mb_max': round(max(deltas), 3),
            'wall_ms_mean': round(sum(walls) / len(walls) * 1000, 1),
        }
    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'child_args': child_args,
        'roles': summary,
        'note': ('tracemalloc 只覆盖 Python/numpy 堆分配, torch GPU 显存不在'
                 '追踪范围; RSS 差值含 CUDA 上下文噪声, 以 tracemalloc 口径为准'),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[mem] 报告已保存: {report_path}")
    for role, s in summary.items():
        print(f"  {role:<16} n={s['n_calls']:<4} "
              f"tracemalloc峰值均值={s['tracemalloc_peak_mb_mean']:.2f}MB "
              f"(max {s['tracemalloc_peak_mb_max']:.2f}) "
              f"wall均值={s['wall_ms_mean']:.1f}ms")


def main():
    ap = argparse.ArgumentParser(description="per-role 内存埋点 (阶段 4.1)")
    ap.add_argument('--report', default=REPORT_DEFAULT)
    ap.add_argument('child_args', nargs=argparse.REMAINDER,
                    help='透传给 main_point2.py 的参数 (以 -- 分隔)')
    args = ap.parse_args()
    child = [a for a in args.child_args if a != '--']
    if not child:
        child = ['--dataset', 'MNIST', '--model', 'Tiny', '--seed', '42',
                 '--verify', '--deterministic', '-K', '3', '--s-max', '3',
                 '--reuse-rule', 'none', '--vcm', '--gamma-vcm', '1.0',
                 '--max-checkpoints', '20', '--max-ticks', '300',
                 '--output-dir', 'results_memory_probe',
                 '--run-id', 'mem_probe_tiny', '--force']
    _child(child, args.report)


if __name__ == '__main__':
    main()
