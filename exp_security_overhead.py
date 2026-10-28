"""安全机制开销微基准 (论文开销表数据源).

实测 (纯 Python ecdsa, NIST P-256):
  - LHH.Hash 逐维标量乘成本, d ∈ {1000,2000,4000,8000} 线性外推到全维 582,026
  - Ed25519 签名/验签 (cryptography, C 实现)
  - PRG 掩码生成 (AES-CTR) 每维成本
口径: median(3 次), 单线程 (OMP=1 不影响纯 python)。与 docs/OVERHEAD_EXPERIMENT.md
的分析法口径一致 —— 全维真 LHH 端到端不可行 (单 Hash ~96s), 功能性实验在
Tiny (d=7850) 上全真进行, 开销按实测线性外推。

维度扫描扩展 (盲审补全执行计划 阶段 4.2):
  python exp_security_overhead.py --scan
  python exp_security_overhead.py --scan --dims 10000 100000 --warmup 3 --reps 10
  --scan 在 point2 协议对象 (TA/Cloud/ES verify-on) 上构造合成 checkpoint,
  3 预热 + 10 实测 (沿用 exp_n4_phase_breakdown.py 口径), 测量
  d × {G2(等权), G3v(VCM)} 的 per-checkpoint 验证/掩码/聚合耗时与 tracemalloc
  峰值内存, 输出 security_overhead_dimension_scan.json (5e5/1e6 真实 LHH
  单次约 80-165s, 预留整夜机时; 外推值保留作交叉校验)。

用法: python exp_security_overhead.py
"""

import argparse
import json
import statistics
import time
import tracemalloc

import numpy as np

from utils import LHH, PRG


def bench(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    print("== LHH.Hash (逐维 EC 标量乘, 纯 Python ecdsa NIST256p) ==")
    xs, ys = [], []
    for d in [1000, 2000, 4000, 8000]:
        pp = LHH.Gen(d)
        v = np.random.randint(-10**6, 10**6, d).astype(np.int64)
        t = bench(lambda: LHH.Hash(pp, v))
        xs.append(d)
        ys.append(t)
        print(f"  d={d:6d}: {t:8.3f}s  ({1e6*t/d:6.1f} us/dim)")
    # 线性拟合 (过原点): t = c · d
    c = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    d_full = 582026
    print(f"  拟合: t = {1e6*c:.1f} us/dim (R^2 见残差: "
          f"{max(abs(y - c*x) / y for x, y in zip(xs, ys)):.1%})")
    print(f"  外推 d={d_full}: 单次 Hash ≈ {c*d_full:.1f}s "
          f"({c*d_full/60:.1f} min)")
    # 端到端口径: 每 checkpoint = 3 ES × verify + ~3 package seal
    for n_hash, label in [(6, "每 checkpoint (3 verify + 3 seal)"),
                          (96, "每 run ~96 seals + 60 verify")]:
        print(f"  {label}: ≈ {n_hash} Hash → {n_hash*c*d_full/60:.1f} min")

    print("\n== Ed25519 (cryptography, C 实现) ==")
    from cryptography.hazmat.primitives.asymmetric import ed25519
    k = ed25519.Ed25519PrivateKey.generate()
    pub = k.public_key()
    msg = b"x" * 64
    t_sign = bench(lambda: k.sign(msg), 20)
    sig = k.sign(msg)
    t_ver = bench(lambda: pub.verify(sig, msg), 20)
    print(f"  sign: {1e6*t_sign:.1f} us   verify: {1e6*t_ver:.1f} us"
          f"  (每 run ~96 seals: {96*1e3*t_sign:.1f} ms)")

    print("\n== PRG 掩码 (HKDF + AES-CTR, 每用户每包 O(d)) ==")
    from utils import PRG
    seed = __import__('os').urandom(32)
    for d in [7850, 582026]:
        t = bench(lambda: PRG.mask_int(seed, "U-0", "edge_0:v0:p0", d), 3)
        print(f"  d={d}: {t*1e3:.1f} ms ({1e6*t/d:.2f} us/dim)")


# ============================================================
#  维度扫描 (阶段 4.2): point2 协议 per-checkpoint 开销 vs d
# ============================================================

def _build_harness(dim, arm, n_es=5, users_per_edge=4, seed=42):
    """构造 point2 协议合成 harness (verify-on, 真实掩码/LHH 链路).

    arm='G3v': 云端走 p2_apply_vcm_and_average (Direct-VCM);
    arm='G2':  等权聚合 (无补偿, p2_recover 后直接平均)。
    """
    import pickle
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from user import User
    from edge import EdgeServer
    from cloud import CloudServer
    from ta import TrustedAuthority

    ta = TrustedAuthority(gradient_dim=dim, ta_id='TA-01',
                          enable_verification=True)
    ta.point2_reset()
    cloud = CloudServer()
    cloud.point2_reset(np.zeros(dim, dtype=np.float64), initial_version=0)
    cloud._gradient_dim = dim   # main_point2 主循环每 tick 设置; harness 静态设置
    edges = []
    for i in range(n_es):
        es = EdgeServer(
            f"edge_{i}", cloud, ta, enable_verification=True,
            gamma_vcm=(1.0 if arm == 'G3v' else 0.0), vcm_mode='full')
        edges.append(es)
    users = []
    user_edge = {}
    ta_pub = ta.get_public_key_bytes()
    n_users = n_es * users_per_edge
    for i in range(n_users):
        u = User(f"U-{i}", None)
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        u.point2_init_cache()
        u.p2_set_sample_size(1000)
        users.append(u)
        user_edge[u.id] = edges[i // users_per_edge]
    rng = np.random.default_rng(seed)
    model_int = rng.integers(-1000, 1000, size=dim, dtype=np.int64)
    return ta, cloud, edges, users, user_edge, model_int


def _make_packages(harness, n_pkgs, seq):
    """每个 edge 封一个真实组包 (真掩码 + 真 LHH H_k + 真签名).

    seq: 每次合成 checkpoint 递增, 保证 package_token 全局唯一
    (cloud 的 exactly-once 去重会拒绝重复 token)。
    """
    import pickle
    ta, cloud, edges, users, user_edge, model_int = harness
    pkgs = []
    for pkg_idx, es in enumerate(edges[:n_pkgs]):
        base_v = 0
        token = f"edge_{pkg_idx}:v0:p{seq}"
        es.p2_start_package(base_version=base_v,
                            base_hash=cloud.p2_hash_current_model(),
                            package_token=token)
        for u in users:
            if user_edge[u.id] is not es:
                continue
            mask = PRG.mask_int(u._ta_seed, u.id, token, len(model_int))
            masked = model_int + mask
            es.p2_receive_masked_gradient(
                pickle.dumps({"user_id": u.id, "masked_gradient": masked}),
                token, base_v, sample_size=1000)
        pkg = es.p2_seal_group_package(tick=1)
        if pkg is not None:
            cloud.p2_receive_group_package(pkg, arrival_tick=1)
            pkgs.append(pkg)
    return pkgs


def _run_checkpoint_once(harness, arm, dim, seq):
    """跑一次完整 checkpoint 验证链路, 返回 {role: (耗时s, tracemalloc峰值B)}."""
    ta, cloud, edges, users, user_edge, model_int = harness
    out = {}

    def timed(role, fn):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            el = time.perf_counter() - t0
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            prev = out.get(role, (0.0, 0))
            out[role] = (prev[0] + el, max(prev[1], peak))

    pkgs = _make_packages(harness, len(edges), seq)
    selected = pkgs

    M_Q, H_mask_raw = timed('ta_mask_build',
                            lambda: ta.p2_build_checkpoint_mask(selected, None))
    raw_sum_int = timed('cloud_recover',
                        lambda: cloud.p2_recover_raw_sum(selected, M_Q))
    if arm == 'G3v':
        _c, final_model, _m = timed(
            'cloud_aggregate',
            lambda: cloud.p2_apply_vcm_and_average(
                selected, raw_sum_int, gamma_vcm=1.0, vcm_mode='full'))
    else:
        s_q = sum(p.S_k for p in selected)
        raw_float = raw_sum_int.astype(np.float64) / 1e6
        final_model = timed('cloud_aggregate',
                            lambda: raw_float / max(1, s_q))

    bundle = {
        'checkpoint_version': cloud.p2_current_version,
        'policy_config_digest': edges[0]._policy_digest,
        'selected_packages': selected,
        'raw_sum_int': raw_sum_int,
        'M_Q': M_Q,
        'H_mask_raw': H_mask_raw,
        'final_model': final_model,
        'gamma_vcm': (1.0 if arm == 'G3v' else 0.0),
        'vcm_mode': 'full',
        'alphas': None,
        'weights': None,
        'weight_source': 'none',
        'momentum': 0.0,
        'global_model_history': {0: cloud.p2_global_model_history[0]},
    }

    def do_verify():
        for e in edges:
            ok = e.p2_verify_checkpoint(bundle)
            assert ok, "合成 checkpoint 必须验证通过 (harness 自检)"
        return True

    timed('es_verify', do_verify)
    return out


def dimension_scan(dims, arms, warmup=3, reps=10, out_path=None,
                   reps_large=None):
    """d × arm 网格: 3 预热 + reps 实测, 每格记录 per-role 耗时/峰值内存.

    reps_large: d > 1e5 时改用的实测次数 (单次真实 LHH 在大维度达 80-280s,
    3+10 全量两臂约 30h+; 大维度降为 3+5 并在每格 reps 字段如实记录)。
    """
    results = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'warmup': warmup, 'reps': reps, 'reps_large': reps_large,
               'cells': []}
    for arm in arms:
        for d in dims:
            reps_here = (reps_large if (reps_large and d > 100_000) else reps)
            harness = _build_harness(d, arm)
            seq = 0
            for _ in range(warmup):
                _run_checkpoint_once(harness, arm, d, seq)
                seq += 1
            agg = {}
            for _ in range(reps_here):
                cell = _run_checkpoint_once(harness, arm, d, seq)
                seq += 1
                for role, (el, peak) in cell.items():
                    a = agg.setdefault(role, {'times': [], 'peaks': []})
                    a['times'].append(el)
                    a['peaks'].append(peak)
            row = {'arm': arm, 'dim': d, 'reps': reps_here, 'roles': {}}
            total_t, total_peak = 0.0, 0
            for role, a in agg.items():
                med_t = statistics.median(a['times'])
                max_peak = max(a['peaks'])
                row['roles'][role] = {
                    'median_s': round(med_t, 4),
                    'peak_traced_mb': round(max_peak / (1024 * 1024), 3)}
                total_t += med_t
                total_peak = max(total_peak, max_peak)
            row['checkpoint_total_median_s'] = round(total_t, 4)
            row['checkpoint_peak_traced_mb'] = round(total_peak / (1024 * 1024), 3)
            results['cells'].append(row)
            # 逐格落盘: 中断后已完成格保留 (整体 JSON 只在结尾有 out_path 时
            # 再写一次, 与逐格内容一致)
            if out_path:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [scan] arm={arm} d={d}: checkpoint 总耗时中位数 "
                  f"{total_t:.2f}s, 峰值内存 {total_peak / (1024 * 1024):.1f} MB",
                  flush=True)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"维度扫描 JSON: {out_path}")
    return results


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--scan", action="store_true",
                     help="运行维度扫描 (阶段 4.2), 否则执行原微基准")
    _ap.add_argument("--dims", type=int, nargs="+",
                     default=[10_000, 100_000, 500_000, 1_000_000])
    _ap.add_argument("--arms", nargs="+", default=["G2", "G3v"],
                     choices=["G2", "G3v"])
    _ap.add_argument("--warmup", type=int, default=3)
    _ap.add_argument("--reps", type=int, default=10)
    _ap.add_argument("--reps-large", type=int, default=None,
                     help="d>1e5 时的实测次数 (默认同 --reps)")
    _ap.add_argument("--out", default="security_overhead_dimension_scan.json")
    _args = _ap.parse_args()
    if _args.scan:
        dimension_scan(_args.dims, _args.arms, _args.warmup, _args.reps,
                       _args.out, reps_large=_args.reps_large)
    else:
        main()
