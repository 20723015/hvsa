"""论文主表基建的单元测试 (thesis-core 批跑 / 统计 / 安全证据).

覆盖不依赖训练的部分, 使批跑口径与统计口径可回归:
- run_thesis_core.build_plan: 计划规模、强制 --deterministic、repeat 对;
- thesis_stats: 配对统计 (mean/sd/CI/dz/正种子)、等时间取点与截断标记、
  acc_history 两列解析、NaN → null 的 JSON 合法化、verdict 判定门限;
- run_attack_evidence: 攻击单元汇总 (拒绝率/被采纳/不适用) 与 clean 误拒率;
- main_point2 的新指标字段: 由 _CommAccount 口径计算的通信量、执行/采纳计数。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_attack_evidence as rae
import run_thesis_core as rtc
import thesis_stats as ts
from main_point2 import _CommAccount


# ---------------------------------------------------------------- 批跑计划

def test_thesis_core_plan_size_and_determinism():
    jobs = rtc.default_plan()
    # 主矩阵 120 + 参考 30 + 基线 20 + repeat 2 = 172
    assert len(jobs) == 172
    by_arm = {}
    for _tag, arm, *_rest in jobs:
        by_arm[arm] = by_arm.get(arm, 0) + 1
    assert by_arm == {'G2': 30, 'G3': 30, 'G3v': 32, 'G5': 30,
                      'G1': 30, 'GFB': 20}
    # 每个任务都不得自带 --deterministic (由 runner 统一注入, 避免漏配)
    for _tag, _arm, _ds, _part, _seed, _rep, _script, extra in jobs:
        assert '--deterministic' not in extra


def test_thesis_core_repeat_pairs_only_seed42_cifar():
    jobs = rtc.default_plan()
    reps = [j for j in jobs if j[5]]
    assert len(reps) == 2
    for tag, arm, ds, _part, seed, _rep, _sc, _ex in reps:
        assert arm == 'G3v' and ds == 'CIFAR10' and seed == rtc.REPEAT_SEED
        assert tag.endswith('_rep')


def test_thesis_core_gfb_restricted_to_baseline_datasets():
    jobs = rtc.build_plan(['GFB'], rtc.DATASETS, None, [42])
    assert {j[2] for j in jobs} == set(rtc.BASELINE_DATASETS)


def test_thesis_core_run_injects_deterministic(tmp_path):
    calls = {}

    def fake_call(cmd, **kwargs):
        calls['cmd'] = cmd
        return 0

    import subprocess
    orig = subprocess.call
    subprocess.call = fake_call
    try:
        job = rtc.build_plan(['G3'], ['MNIST'], ['IID'], [42])[0]
        rtc._run_one(*job, root=str(tmp_path), force=False)
    finally:
        subprocess.call = orig
    assert '--deterministic' in calls['cmd']
    assert '--output-dir' in calls['cmd'] and '--run-id' in calls['cmd']
    # manifest 落盘且记录 arm/dataset/seed
    rows = [json.loads(l) for l in
            open(os.path.join(str(tmp_path), rtc.SUMMARY), encoding='utf-8')]
    assert rows and rows[0]['arm'] == 'G3' and rows[0]['seed'] == 42
    assert rows[0]['rc'] == 0
    assert rows[0]['job_fingerprint'] and rows[0]['code_digest']


def test_thesis_core_job_fingerprint_tracks_config_and_source(monkeypatch):
    job = rtc.build_plan(['G3'], ['MNIST'], ['IID'], [42])[0]
    a = rtc._job_fingerprint(job, None, code_digest='code-a')
    b = rtc._job_fingerprint(job, ['--model', 'Tiny'], code_digest='code-a')
    c = rtc._job_fingerprint(job, None, code_digest='code-b')
    assert len({a, b, c}) == 3


def test_thesis_core_smoke_extra_args_forbidden_in_formal_root():
    class Args:
        arms = ['G3']; datasets = ['MNIST']; parts = ['IID']; seeds = [42]
        with_sigma_ablation = False; root = rtc.ROOT_DEFAULT
        extra_args = '--model Tiny'; force = False; dry_run = True
        workers = 1; allow_parallel = False
    with pytest.raises(SystemExit, match='禁止写入正式 root'):
        rtc.cmd_run(Args())


# ---------------------------------------------------------------- 统计口径

def test_paired_stats_n5_and_verdict():
    a = {42: 80.0, 43: 81.0, 44: 82.0, 45: 83.0, 46: 84.0}
    b = {42: 78.0, 43: 79.0, 44: 80.0, 45: 81.0, 46: 82.0}
    st = ts.paired_stats(a, b)
    assert st['n'] == 5 and st['mean_diff'] == pytest.approx(2.0)
    assert st['sd_diff'] == pytest.approx(0.0)
    assert st['positives'] == 5
    assert np.isnan(st['dz']), 'sd=0 时 dz 未定义 (JSON 中规整为 null)'
    assert ts._verdict(st) == '稳定提升'

    # 混合情形: 配对差 [1, -1, 2, 0.5, 1.5] → mean 0.8, 4/5 为正
    a2 = {42: 10.0, 43: 10.0, 44: 10.0, 45: 10.0, 46: 10.0}
    b2 = {42: 9.0, 43: 11.0, 44: 8.0, 45: 9.5, 46: 8.5}
    st2 = ts.paired_stats(a2, b2)
    assert st2['mean_diff'] == pytest.approx(0.8)
    assert st2['positives'] == 4
    lo, hi = st2['ci95']
    assert lo < 0.8 < hi
    assert ts._verdict(st2) == '趋势(正)'

    # 即便四个 seed 全正也不能越过正式五种子门槛。
    st4 = ts.paired_stats({i: 2.0 for i in range(4)},
                          {i: 1.0 for i in range(4)})
    assert ts._verdict(st4) == '样本不足(n=4, 仅趋势)'


def test_paired_stats_t_critical_is_2_776_for_n5():
    # 通过 CI 宽度反推 t 临界: half = tc * sd / sqrt(n)
    diffs = [1.0, -1.0, 2.0, 0.5, 1.5]
    a = {i: v for i, v in enumerate(diffs)}
    b = {i: 0.0 for i in range(5)}
    st = ts.paired_stats(a, b)
    sd = np.std(diffs, ddof=1)
    half = st['ci95'][1] - st['mean_diff']
    assert half == pytest.approx(2.7764 * sd / np.sqrt(5), rel=1e-3)


def test_acc_at_budget_and_censoring():
    run = {'ticks': np.array([5, 10, 15]), 'acc': np.array([60.0, 70.0, 80.0])}
    v, censored = ts.acc_at_budget(run, 12)
    assert v == pytest.approx(70.0) and censored is True
    v, censored = ts.acc_at_budget(run, 15)
    assert v == pytest.approx(80.0) and censored is False
    v, censored = ts.acc_at_budget(run, 3)      # 早于第一个 checkpoint
    assert v == pytest.approx(60.0) and censored is True


def test_equal_time_uses_g3_budget_but_g2_as_comparison_base():
    def run(ticks, acc):
        return {'ticks': np.array(ticks), 'acc': np.array(acc)}
    by = {
        ('G2', 'MNIST', 'IID'): {42: run([5, 10, 15, 20], [50, 60, 70, 80])},
        ('G3', 'MNIST', 'IID'): {42: run([4, 8, 12], [55, 65, 75])},
        ('G3v', 'MNIST', 'IID'): {42: run([4, 8, 12], [57, 68, 78])},
    }
    out, budgets = ts.equal_time_stats(by, 'MNIST', 'IID')
    assert budgets == {42: 12}
    # @T=12: G2=60, G3=75, G3v=78
    assert out['G3']['mean_diff'] == pytest.approx(15.0)
    assert out['G3v']['mean_diff'] == pytest.approx(18.0)
    assert out['G3v']['base_mean_at_budget'] == pytest.approx(60.0)


def test_completeness_requires_exact_five_seed_main_matrix():
    by = {}
    for arm in ts.MAIN_ARMS:
        for ds in ts.DATASETS:
            for part in ts.PARTS:
                by[(arm, ds, part)] = {s: {} for s in (42, 43, 44, 45, 46)}
    assert ts.completeness_issues(by) == []
    by[('G5', 'CIFAR10', 'NI')].pop(46)
    issues = ts.completeness_issues(by)
    assert len(issues) == 1 and 'G5/CIFAR10/NI' in issues[0]


def test_acc_column_parses_main_point2_two_column_history():
    raw = np.array([[1, 65.38], [2, 79.33], [3, 82.51]])
    assert np.allclose(ts._acc_column(raw), [65.38, 79.33, 82.51])
    assert np.allclose(ts._acc_column(np.array([1.0, 2.0])), [1.0, 2.0])


def test_load_runs_reads_dir_and_skips_repeat(tmp_path):
    def write(tag, arm, ds, part, seed, accs, rep=False):
        d = tmp_path / tag
        d.mkdir()
        arr = np.array([[i + 1, a] for i, a in enumerate(accs)])
        np.savez(d / f"point2_data_{tag}.npz", acc_history=arr,
                 checkpoint_ticks=np.array([5 * (i + 1) for i in range(len(accs))]))
        meta = {'config_hash': f'h_{tag}', 'final_acc': float(accs[-1]),
                'wall_clock': 12.5, 'deterministic': True,
                'adopted_packages': 2, 'group_packages_sealed': 3,
                'verify_seconds_total': 1.5, 'comm_total_bytes': 1024}
        (d / f"point2_meta_{tag}.json").write_text(
            json.dumps(meta), encoding='utf-8')

    write('G3_MNIST_IID_s42', 'G3', 'MNIST', 'IID', 42, [60.0, 70.0])
    write('G3_MNIST_IID_s42_rep', 'G3', 'MNIST', 'IID', 42, [60.0, 70.0], True)
    write('G2_MNIST_IID_s42', 'G2', 'MNIST', 'IID', 42, [55.0, 65.0])
    runs = ts.load_runs(str(tmp_path))
    assert len(runs) == 2, 'repeat 运行不得进入统计'
    by = ts.index_runs(runs)
    assert set(by) == {('G3', 'MNIST', 'IID'), ('G2', 'MNIST', 'IID')}
    st = ts.arm_cell_stats(by, 'G3', 'MNIST', 'IID')
    assert st['mean'] == pytest.approx(70.0) and st['n'] == 1


def test_jsonable_replaces_nan_and_inf():
    out = ts._jsonable({'dz': float('nan'), 'ci': [float('inf'), 1.0],
                        'n': np.int64(3)})
    assert out == {'dz': None, 'ci': [None, 1.0], 'n': 3.0}
    json.dumps(out, allow_nan=False)


# ---------------------------------------------------------------- 安全证据

def _outcome(**kw):
    rec = {'attack': 'tamper_global', 'requested': True, 'applicable': True,
           'attempted': True, 'detected': True, 'accepted': False,
           'detection_stage': 'lhh', 'reason_code': 'verify_failed',
           'state_before': {}, 'state_after': {}}
    rec.update(kw)
    return rec


def test_summarize_class_rejection_rate_and_inapplicable():
    recs = [_outcome() for _ in range(3)]
    recs.append(_outcome(applicable=False, attempted=False, detected=False))
    recs.append(_outcome(detected=False, accepted=True, detection_stage='commit'))
    s = rae.summarize_class('raw_sum', recs, [{'tag': 't', 'attack': 'tamper_global',
                                               'units': 5, 'wall_s': 1, 'rc': 0}],
                            units_wanted=5)
    assert s['attempted'] == 4 and s['detected'] == 3
    assert s['accepted_attack'] == 1
    assert s['rejection_rate'] == pytest.approx(0.75)
    assert s['inapplicable'] == 1
    assert s['detection_stages'] == {'lhh': 3}
    assert s['attempted_by_attack'] == {'tamper_global': 4}


def test_attack_report_rejects_short_or_missing_member_coverage():
    short = rae.summarize_class(
        'manifest_replay', [_outcome(attack='tamper_manifest') for _ in range(2)],
        [{'tag': 'm', 'attack': 'tamper_manifest', 'units': 2, 'wall_s': 1, 'rc': 0},
         {'tag': 'r', 'attack': 'replay', 'units': 0, 'wall_s': 1, 'rc': 0}],
        units_wanted=5)
    clean = rae.summarize_clean(
        [_outcome(attack='none', requested=False, attempted=False,
                  detected=False, accepted=True) for _ in range(5)],
        [{'tag': 'c', 'attack': 'none', 'units': 5, 'wall_s': 1, 'rc': 0}])
    class Args:
        dataset = 'MNIST'; model = 'Tiny'; seed = 42
        edge_servers = 5; users_per_edge = 4; units = 5
    md = rae.render_md({'classes': [short], 'clean': clean}, Args())
    assert '未达标' in md and 'manifest_replay' in md


def test_summarize_clean_false_reject_rate():
    clean = [_outcome(attack='none', requested=False, attempted=False,
                      detected=False, accepted=True, detection_stage='commit')
             for _ in range(4)]
    clean.append(_outcome(attack='none', requested=False, detected=True,
                          accepted=False, detection_stage='lhh'))
    s = rae.summarize_clean(clean, [])
    assert s['attempted'] == 5 and s['detected'] == 1
    assert s['false_reject_rate'] == pytest.approx(0.2)


def test_attack_classes_cover_plan_four_families():
    # 计划要求四类: raw sum / manifest+replay / 权重来源 / 补偿参数
    assert set(rae.CLASSES) == {'raw_sum', 'manifest_replay',
                                'weight_source', 'comp_param'}
    assert 'tamper_global' in rae.CLASSES['raw_sum'][0]
    assert 'replay' in rae.CLASSES['manifest_replay'][0]
    assert 'weight_forge' in rae.CLASSES['weight_source'][0]
    assert 'alpha_tamper' in rae.CLASSES['comp_param'][0]
    # 补偿参数类必须开 σ (>0) 才可能产生 alphas
    assert '--staleness-sigma' in rae.CLASSES['comp_param'][1]
    # 权重来源类必须使权重存在 (V-PDA 或 σ-agg)
    assert '--participation-debias' in rae.CLASSES['weight_source'][1]


# ---------------------------------------------------------------- 通信量口径

def test_comm_account_formulas():
    c = _CommAccount()
    c.upload(100)
    assert c.bytes['upload'] == 800 and c.counts['upload'] == 1
    c.package(100, 4)          # 8·dim + 33 + 64 + 48·n_users
    assert c.bytes['package'] == 800 + 33 + 64 + 192
    c.manifest(100, 3)         # 16·dim + (33+64+48)·n_q
    assert c.bytes['manifest'] == 1600 + 145 * 3
    c.vote(5)
    c.receipt()
    c.notify(5)
    assert c.bytes['vote'] == 1280 and c.counts['vote'] == 5
    assert c.bytes['receipt'] == 320
    assert c.bytes['notify'] == 128 * 6 and c.counts['notify'] == 6
    assert c.total_bytes() == sum(c.bytes.values())
    assert set(c.to_dict()['bytes']) == set(_CommAccount.KEYS)


# ------------------------------------------------- FedAvg 加权 / verify-on

def test_fedavg_hierarchical_weighting_equals_flat_sample_average():
    """Non-IID 分层样本加权 == 直接全局样本量加权 (标准 FedAvg 语义)."""
    import torch
    import fedavg_point2 as fa
    states = [{'w': torch.tensor(float(v)), 'b': torch.tensor(float(-v))}
              for v in (10.0, 20.0, 30.0, 40.0)]
    sizes = [1, 3, 2, 6]                       # Dirichlet 不均衡样本数
    out = fa.fedavg_weighted_average(['w', 'b'], states, sizes, 2, 2)
    total = sum(sizes)
    for key, sign in (('w', 1.0), ('b', -1.0)):
        flat = sign * sum((10.0, 20.0, 30.0, 40.0)[i] * sizes[i]
                          for i in range(4)) / total
        assert float(out[key]) == pytest.approx(flat)   # 370/12 ≈ 30.83
    # 修复前的等权逐用户平均 (25.0) 在样本不均时是错误口径
    assert float(out['w']) != pytest.approx(25.0)
    # 样本量全同时退化为等权平均
    out_eq = fa.fedavg_weighted_average(
        ['w'], [{'w': s['w']} for s in states], [5, 5, 5, 5], 2, 2)
    assert float(out_eq['w']) == pytest.approx(25.0)


def test_p2_vote_with_result_verifies_exactly_once(monkeypatch):
    """verify-on 正常路径: 生成 vote 复用同一次完整验证, 不得重复验算."""
    import edge as edge_mod
    from utils import SIG
    es = edge_mod.EdgeServer.__new__(edge_mod.EdgeServer)
    es.id = 'es0'
    es._policy_digest = 'pd-42'
    es._p2_last_verified = None
    pub, es.my_sign_priv_key = SIG.gen()
    calls = []

    def fake_verify(bundle):
        calls.append(bundle.get('checkpoint_version'))
        return edge_mod.VerificationResult(passed=True, stage='all', reason='')

    monkeypatch.setattr(es, 'p2_verify_checkpoint_detailed', fake_verify)
    vote, res = es.p2_vote_checkpoint_with_result(
        {'checkpoint_version': 3, 'selected_packages': []})
    assert calls == [3], 'vote 生成必须恰好消费一次完整验证'
    assert res.passed and vote.passed
    assert es._p2_last_verified is not None
    assert SIG.verify(bytes.fromhex(vote.payload_digest()), vote.signature, pub)
