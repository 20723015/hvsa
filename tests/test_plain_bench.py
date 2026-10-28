"""明文基线仿真器单测 (盲审补全执行计划 阶段 1.4).

覆盖:
- 纯聚合函数手算期望值 (fedbuff 加权 / mafsa 混合 / dvafl Eq.(15)-(19));
- run_plain_bench 计划规模与 --deterministic 注入口径;
- Tiny 模型端到端冒烟 (三臂各一次, 断言产物契约与机制计数);
- 同 seed 确定性 (arr_digest 逐值一致)。

手算期望值来源: MAFSA Eq.(10)(11) 与 user.py:306-310 同式;
DVAFL Eq.(13)-(19) 原文 (reference/016-2026-DVAFL.pdf p.6)。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plain_bench as pb
import run_plain_bench as rpb


# ---------------------------------------------------------------- 纯函数

def _upd(flat, S=100, base=0, arrived=5, uid='u', idx=0):
    return pb.PlainUpdate(uid, idx, base, np.asarray(flat, dtype=np.float64),
                          0.0, S, arrived)


def test_fedbuff_weighted_average_hand():
    # S=(100,200), wait=(0,2), σ=0.5 → α=(1, 1/√3)
    u1 = _upd([1.0, 1.0], S=100, arrived=5)
    u2 = _upd([3.0, 3.0], S=200, arrived=3)
    a1 = 1.0 / (0 + 1) ** 0.5
    a2 = 1.0 / (2 + 1) ** 0.5
    w = pb.fedbuff_aggregate([u1, u2], [a1, a2])
    exp = (a1 * 100 * 1.0 + a2 * 200 * 3.0) / (a1 * 100 + a2 * 200)
    assert np.allclose(w, exp)
    # σ=0 退化为标准样本量加权平均
    w0 = pb.fedbuff_aggregate([u1, u2], [1.0, 1.0])
    assert np.allclose(w0, (100 * 1.0 + 200 * 3.0) / 300)


def test_mafsa_blend_hand():
    # α = exp(−β·E); E=2, β=0.3 → α=exp(−0.6); w = α·new + (1−α)·stale
    new = np.array([2.0, 0.0])
    stale = np.array([0.0, 4.0])
    b = pb.mafsa_blend(new, stale, E_i=2, beta=0.3)
    a = np.exp(-0.6)
    assert np.allclose(b, a * new + (1 - a) * stale)
    # E=0 → α=1 → 纯 fresh (混合只发生在 E>0, 语义与 user.py:287 一致)
    assert np.allclose(pb.mafsa_blend(new, stale, 0, 0.3), new)


def test_dvafl_moment_and_bias_correction_hand():
    # Eq.(15): m = λ·m + (1−λ)·‖g‖; 两次观测 ‖g‖=2, λ=0.9
    m1, c1 = pb.dvafl_moment_update(None, 0, 2.0, 0.9)
    assert np.isclose(m1, 0.2) and c1 == 1
    m2, c2 = pb.dvafl_moment_update(m1, c1, 2.0, 0.9)
    assert np.isclose(m2, 0.38) and c2 == 2
    # Eq.(19): m̂ = m/(1−λ^t); t=2 → 0.38/(1−0.81)=2.0
    assert np.isclose(pb.dvafl_bias_correct(m2, c2, 0.9), 2.0)
    # 冷启动: m=None 或 count=0 不校正
    assert pb.dvafl_bias_correct(None, 0, 0.9) is None


def test_dvafl_compensation_hand():
    hist = {0: np.array([0.0]), 1: np.array([1.0]), 2: np.array([2.0])}
    # ratio = m̂/M̂ = 1/2 (限幅内), G = W(2)−W(0) = 2, γ=1 → C=1
    wt, cn = pb.dvafl_compensation(np.array([5.0]), 0, 2, hist,
                                   m_hat=1.0, M_hat=2.0, gamma=1.0,
                                   moment_mode='moment1')
    assert np.allclose(wt, [6.0]) and np.isclose(cn, 1.0)
    # moment2: ratio² = 0.25 → C = 0.5
    wt2, _ = pb.dvafl_compensation(np.array([5.0]), 0, 2, hist,
                                   m_hat=1.0, M_hat=2.0, gamma=1.0,
                                   moment_mode='moment2')
    assert np.allclose(wt2, [5.5])
    # direct: 比值恒 1 (Eq.(14))
    wt3, _ = pb.dvafl_compensation(np.array([5.0]), 0, 2, hist,
                                   m_hat=1.0, M_hat=2.0, gamma=1.0,
                                   moment_mode='direct')
    assert np.allclose(wt3, [7.0])
    # fresh 更新 (r_i = v) 不补偿
    wt4, cn4 = pb.dvafl_compensation(np.array([5.0]), 2, 2, hist,
                                     m_hat=1.0, M_hat=2.0, gamma=1.0,
                                     moment_mode='moment1')
    assert np.allclose(wt4, [5.0]) and cn4 == 0.0
    # M̂ 冷启动 → 退化 Direct
    wt5, _ = pb.dvafl_compensation(np.array([5.0]), 0, 2, hist,
                                   m_hat=1.0, M_hat=None, gamma=1.0,
                                   moment_mode='moment1')
    assert np.allclose(wt5, [7.0])
    # ratio 限幅 [0,2]: m̂/M̂=4 → 2
    wt6, _ = pb.dvafl_compensation(np.array([5.0]), 0, 2, hist,
                                   m_hat=8.0, M_hat=2.0, gamma=1.0,
                                   moment_mode='moment1')
    assert np.allclose(wt6, [9.0])


def test_dvafl_aggregate_eq3_no_downweight():
    # Eq.(3): 纯样本量加权, 陈旧更新不降权 (DVAFL 与 FedAsync 的分界)
    w1, w2 = np.zeros(3) + 1.0, np.zeros(3) + 3.0
    out = pb.dvafl_aggregate([w1, w2], [100, 300])
    assert np.allclose(out, 1.0 * 0.25 + 3.0 * 0.75)


def test_dvafl_global_moment_hand():
    # Eq.(16): M = λ·M + (1−λ)·‖G‖
    M, c = pb.dvafl_global_moment_update(None, 0, 3.0, 0.9)
    assert np.isclose(M, 0.3) and c == 1
    M2, c2 = pb.dvafl_global_moment_update(M, c, 1.0, 0.9)
    assert np.isclose(M2, 0.9 * 0.3 + 0.1 * 1.0) and c2 == 2


# ---------------------------------------------------------------- 批跑计划

def test_plain_bench_plan_size_and_naming():
    jobs = rpb.default_plan()
    # 3 臂 × 3 数据集 × 2 分布 (IID/NI) × 5 seeds = 90 (阶段 2 全新 run 数)
    assert len(jobs) == 90
    by_arm = {}
    tags = set()
    for job in jobs:
        tag, arm, ds, part, seed, script, extra = job
        by_arm[arm] = by_arm.get(arm, 0) + 1
        tags.add(tag)
        assert script == 'plain_bench.py'
        # --deterministic 由 runner 统一注入, 计划参数内不得自带
        assert '--deterministic' not in extra
        assert '--buffer-k' in extra
    assert by_arm == {'fedbuff': 30, 'mafsa': 30, 'dvafl': 30}
    assert len(tags) == 90
    # PARTS 扩展: NI01/NI10 可按需启用 (阶段 3 边界)
    jobs01 = rpb.build_plan(['fedbuff'], ['MNIST'], ['NI01'], [42])
    assert len(jobs01) == 1 and jobs01[0][0] == 'fedbuff_MNIST_NI01_s42'
    assert '--alpha' in jobs01[0][6] and '0.1' in jobs01[0][6]


def test_plain_bench_arm_defaults():
    # 三臂默认超参与 PLAIN_BENCH_ADAPTATIONS.md 记录一致 (漂移须改文档)
    assert pb.ARM_DEFAULTS['fedbuff']['wait_sigma'] == 0.5
    assert pb.ARM_DEFAULTS['mafsa']['staleness_beta'] == 0.3
    assert pb.ARM_DEFAULTS['dvafl'] == {'gamma_vcm': 1.0,
                                        'moment_mode': 'moment1',
                                        'moment_lambda': 0.9}


# ------------------------------------------------- 端到端冒烟 (Tiny 模型)

def _run_arm(arm, tmp_path, extra=None, tag=None):
    tag = tag or f"{arm}_smoke"
    out = tmp_path / tag
    argv = [
        '--arm', arm, '--dataset', 'MNIST', '--model', 'Tiny',
        '--seed', '42', '--deterministic', '--max-checkpoints', '6',
        '--max-ticks', '60', '--buffer-k', '12',
        '--output-dir', str(out), '--run-id', tag,
    ] + (extra or [])
    pb.run_plain_simulation(_build_ns(argv))
    return out, tag


def _build_ns(argv):
    """复刻 __main__ 的 parser (仅测试用; 字段漂移由 npz 契约断言兜底)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    p.add_argument('--dataset', default='MNIST')
    p.add_argument('--model', default='Tiny')
    p.add_argument('--tag', default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noniid', action='store_true')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--max-checkpoints', type=int, default=6)
    p.add_argument('--max-ticks', type=int, default=60)
    p.add_argument('--min-delay', type=int, default=1)
    p.add_argument('--max-delay', type=int, default=5)
    p.add_argument('--group-by-speed', action='store_true')
    p.add_argument('--buffer-k', type=int, default=12)
    p.add_argument('--s-max', type=int, default=3)
    p.add_argument('--dropout-rate', type=float, default=0.1)
    p.add_argument('--dropout-burst', type=int, default=0)
    p.add_argument('--upload-dropout-rate', type=float, default=0.0)
    p.add_argument('--staleness-beta', type=float, default=0.3)
    p.add_argument('--wait-sigma', type=float, default=0.5)
    p.add_argument('--gamma-vcm', type=float, default=1.0)
    p.add_argument('--moment-mode', default='moment1')
    p.add_argument('--moment-lambda', type=float, default=0.9)
    p.add_argument('--output-dir', default='.')
    p.add_argument('--run-id', default='')
    p.add_argument('--force', action='store_true')
    return p.parse_args(argv)


def _load_products(out_dir, tag):
    npz = np.load(os.path.join(out_dir, f'point2_data_{tag}.npz'),
                  allow_pickle=True)
    with open(os.path.join(out_dir, f'point2_meta_{tag}.json'),
              encoding='utf-8') as f:
        meta = json.load(f)
    return npz, meta


def test_fedbuff_end_to_end_tiny(tmp_path):
    out, tag = _run_arm('fedbuff', tmp_path)
    npz, meta = _load_products(out, tag)
    acc = npz['acc_history']
    assert acc.ndim == 2 and acc.shape[0] == 6      # 6 次聚合
    assert acc[0, 1] < acc[-1, 1]                   # MNIST 上必须在学习
    assert npz['checkpoint_Ns'].tolist() == [12] * 6
    assert int(npz['user_updates_executed']) > 0
    assert float(npz['comp_gain']) == 0.0           # fedbuff 无补偿
    assert npz['comm_total_bytes'] == int(npz['comm_total_bytes'])
    assert meta['arm'] == 'fedbuff' and meta['verify_total'] == 0
    assert len(meta['arr_digest']) == 16


def test_mafsa_reuse_and_dvafl_comp_active(tmp_path):
    # 高掉线率让 mafsa 的 offline-train-and-cache 复用必然发生
    out, tag = _run_arm('mafsa', tmp_path,
                        extra=['--dropout-rate', '0.3'])
    npz, meta = _load_products(out, tag)
    assert int(np.sum(npz['stale_reuse_counts'])) > 0   # 复用人次 > 0
    assert float(npz['comp_gain']) == 0.0               # mafsa 无补偿
    # dvafl: 补偿激活
    out2, tag2 = _run_arm('dvafl', tmp_path)
    npz2, _ = _load_products(out2, tag2)
    assert float(npz2['comp_gain']) > 0.0


def test_same_seed_determinism_tiny(tmp_path):
    _, tag = _run_arm('fedbuff', tmp_path, tag='det_a')
    npz_a, meta_a = _load_products(tmp_path / 'det_a', tag)
    _run_arm('fedbuff', tmp_path, tag='det_b')
    npz_b, meta_b = _load_products(tmp_path / 'det_b', 'det_b')
    assert meta_a['arr_digest'] == meta_b['arr_digest']
    assert np.array_equal(npz_a['acc_history'], npz_b['acc_history'])


def test_noniid_partition_smoke_tiny(tmp_path):
    out, tag = _run_arm('fedbuff', tmp_path,
                        extra=['--noniid', '--alpha', '0.1'])
    npz, meta = _load_products(out, tag)
    assert npz['acc_history'].shape[0] >= 1
    assert meta['scientific_config']['noniid'] is True
