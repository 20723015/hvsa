"""策略绑定验证测试 (研究点 2, 阶段 0).

覆盖: ES 对 checkpoint 权重/补偿策略的独立复算与拒绝 —— 恶意云端不能
自行选择权重, 也不能把伪造权重改标为其他来源绕过复算 (edge.py Sec.4.6
策略绑定扩展)。

1. σ-agg 整数权重: ES 按公开陈旧度规则独立复算, 伪造权重被拒。
2. 改标攻击: V-PDA 权重改标为 staleness → 拒; 反向改标 → 拒。
3. 来源绑定: 权重来源由 ES 本地协议配置唯一确定 (expected_weight_source),
   降级为 none / 在合法规则间择优声明 → 来源核对即拒。
4. 等权 VCM 路径: 云端 alphas 与公开规则复算值比对, α/σ 篡改被拒。
5. Aσ (alignment): σ* 档位由公开量复算, 两阶段状态 (commit 才落地),
   锁定档位篡改被拒。
6. 声明策略参数 (γ/momentum) 与 ES 本地协议配置不一致 → 拒。
7. checkpoint 版本绑定: 谎报版本 → 拒。
8. wait 度量含云端调度状态, 不可策略绑定 → 拒 (fail-closed)。
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_vpda import DIM, SCALE, build_bundle, make_env, make_pkg

from utils import (AlignmentSigmaPolicy, QUANTIZE_SCALE, staleness_alpha,
                   staleness_int_weights, vpda_weights)


def _advance(cloud, v, delta=0.2):
    """构造历史版本 W(v) = W(v-1)+delta, 设 current_version=v."""
    cloud.p2_global_model_history[v] = (
        cloud.p2_global_model_history[v - 1] + delta)
    cloud.p2_current_version = v


def _bind_es(edges, cloud, committed_v):
    """ES 本地共识状态同步 (模拟此前 checkpoint 均已诚实提交)."""
    for e in edges:
        e._p2_committed_version = committed_v
        for k in range(1, committed_v + 1):
            e._p2_committed_history[k] = cloud.p2_global_model_history[k]


def _honest_agg_bundle(ta, cloud, selected, weights, alphas):
    """σ-agg 诚实路径: 加权掩码 → 加权 raw → γ=0 加权平均."""
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=weights)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=weights)
    _, final, _ = cloud.p2_weighted_average(selected, raw, weights)
    return build_bundle(cloud, ta, selected, raw, M_Q, H, final, weights,
                        gamma=0.0, alphas=alphas, weight_source='staleness')


def test_staleness_weights_independently_recomputed():
    """σ-agg: ES 按 version 度量独立复算整数权重; 伪造权重被拒."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=0.0,
                                       sigma_target='agg')
    e0, e1 = edges
    _advance(cloud, 1)                     # v=1, 包 base 0 → s_k=1
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]

    alphas = [staleness_alpha('version', 1 - p.base_version, 0.5)
              for p in selected]
    ws = staleness_int_weights(alphas)
    assert ws == [int(round(0.5 ** 0.5 * SCALE))] * 2

    bundle = _honest_agg_bundle(ta, cloud, selected, ws, alphas)
    assert e0.p2_verify_checkpoint(bundle) is True
    assert e1.p2_verify_checkpoint(bundle) is True

    # 伪造权重 (σ 取别的值 / 任选) → 复算不匹配 → 拒
    assert e0.p2_verify_checkpoint(
        dict(bundle, weights=[ws[0] + 1, ws[1]])) is False
    # 云端按 σ=0 生成等权 65536 冒充 → 拒
    assert e0.p2_verify_checkpoint(
        dict(bundle, weights=[SCALE, SCALE])) is False


def test_relabel_vpda_weights_as_staleness_rejected():
    """改标攻击: 伪造的 V-PDA 权重改标为 staleness 必须被独立复算拒绝."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=0.0,
                                       sigma_target='agg')
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]

    # 云端伪造 V-PDA 权重 (不均衡参与历史下 q≠1 → w≠65536)
    vpda_ws = [56757, 80257]
    bundle = _honest_agg_bundle(ta, cloud, selected, vpda_ws,
                                [1.0 / 2 ** 0.5] * 2)
    # 改标为 'staleness' (旧代码只查正性 → 放行) → 现在独立复算拒绝
    assert e0.p2_verify_checkpoint(
        dict(bundle, weight_source='staleness')) is False


def test_relabel_staleness_weights_as_vpda_rejected():
    """反向改标: 真实 σ 权重声明为 vpda 来源 → 来源绑定即拒 (expected=staleness)."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=0.0,
                                       sigma_target='agg')
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]

    alphas = [staleness_alpha('version', 1, 0.5)] * 2
    ws = staleness_int_weights(alphas)
    bundle = _honest_agg_bundle(ta, cloud, selected, ws, alphas)
    assert e0.p2_verify_checkpoint(
        dict(bundle, weight_source='vpda')) is False


def test_vcm_alphas_independently_recomputed():
    """等权 VCM 路径: σ>0 时 ES 复算 α 并与云端声明比对; α 篡改被拒."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=1.0)
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]

    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    alphas = [staleness_alpha('version', 1, 0.5)] * 2
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=1.0, vcm_mode='full', alphas=alphas)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=1.0, alphas=alphas, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is True

    # α 篡改 (全 1, 即关闭衰减) → 复算不匹配 → 拒
    assert e0.p2_verify_checkpoint(
        dict(bundle, alphas=[1.0, 1.0])) is False
    # 声明 σ 篡改 (α 按别的 σ 生成) → 拒
    alphas_other = [staleness_alpha('version', 1, 0.1)] * 2
    assert e0.p2_verify_checkpoint(
        dict(bundle, alphas=alphas_other)) is False
    # σ>0 却不提供 alphas → 拒
    assert e0.p2_verify_checkpoint(dict(bundle, alphas=None)) is False


def _asig_round(ta, cloud, edges, users, cloud_policy, round_idx, plain_step,
                cloud_sigma_override=None):
    """Aσ 单轮: 按云端流程生成 alphas 并构建 bundle.

    包基于 v-1 (s_k=1, 使不同 σ 档产生可区分的 α)。cos 由公开量决定
    (plain 同/反向 → cos ≈ ±1)。云端用独立 AlignmentSigmaPolicy 实例
    (与 main 相同的两阶段流程); cloud_sigma_override 模拟云端谎报锁定
    档位 (用别的 σ 生成 alphas)。
    """
    v = cloud.p2_current_version
    _bind_es(edges, cloud, committed_v=v)
    pkg_base = max(0, v - 1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[pkg_base].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], f"e0:p{round_idx}", pkg_base, base_hash, 200,
                  np.full(DIM, plain_step))
    p1 = make_pkg(e1, users[2:], f"e1:p{round_idx}", pkg_base, base_hash, 300,
                  np.full(DIM, plain_step))
    selected = [p0, p1]

    # 云端流程: 先恢复 raw sum (TA 掩码), 再由聚合量复算 cos
    # (与 edge._p2_alignment_cos 同公式镜像)
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    S_policy = max(1, sum(p.S_k for p in selected))
    raw_float = raw.astype(np.float64) / QUANTIZE_SCALE
    history = cloud.p2_global_model_history
    R = raw_float / S_policy
    B = (sum(p.S_k * np.asarray(history[p.base_version], dtype=np.float64)
             for p in selected) / S_policy)
    Wv = np.asarray(history[v], dtype=np.float64)
    C0 = (sum(p.S_k * 1.0 * (Wv - np.asarray(
        history[p.base_version], dtype=np.float64))
        for p in selected if p.base_version < v) / S_policy)
    D = R - B
    d_n, c_n = float(np.linalg.norm(D)), float(np.linalg.norm(C0))
    cos = float(np.dot(C0, D) / (c_n * d_n + 1e-12))
    cos_valid = c_n > 0.0 and d_n > 0.0

    sigma_eff = cloud_policy.tentative_eff(cos if cos_valid else None)
    sigma_for_alpha = (cloud_sigma_override
                       if cloud_sigma_override is not None else sigma_eff)
    alphas = [staleness_alpha('version', v - p.base_version, sigma_for_alpha)
              for p in selected]

    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=1.0, vcm_mode='full', alphas=alphas)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=1.0, alphas=alphas, weight_source='none')
    return bundle, (cos if cos_valid else None), sigma_eff


def test_asig_alignment_sigma_star_recomputation():
    """Aσ: σ* 档位由公开量复算 + 两阶段状态; 锁定档位篡改被拒."""
    ta, cloud, users, edges = make_env(sigma_policy='alignment',
                                       staleness_sigma=0.5, gamma_vcm=1.0)
    cloud_policy = AlignmentSigmaPolicy(
        sigma_fallback=0.5, sigma_high=-0.25, ema_beta=0.8, warmup=2,
        boost_threshold=0.9, mid_threshold=0.72)
    e0, e1 = edges

    # 轮 1: cos=+1 (探测期) → σ_eff=0.5; ES 通过, 双方状态落地
    _advance(cloud, 1)
    bundle1, cos1, eff1 = _asig_round(ta, cloud, edges, users, cloud_policy,
                                      0, 0.05)
    assert eff1 == 0.5
    assert e0.p2_verify_checkpoint(bundle1) is True
    assert e1.p2_verify_checkpoint(bundle1) is True
    committed = cloud.p2_current_version + 1
    final1 = bundle1['final_model']
    cloud.p2_global_model_history[committed] = final1.copy()
    cloud.p2_current_version = committed
    cloud_policy.commit_observation(cos1)
    for e in edges:
        e.p2_note_committed_checkpoint(bundle1['selected_packages'],
                                       committed_version=committed,
                                       final_model=final1)

    # 轮 2: W(2) = 轮 1 提交的 final1 (真实协议语义); probe_mean=1.0 ≥ 0.9
    # → ES 试探性锁定 σ*=-0.25; 云端诚实用 -0.25 → 通过
    bundle2, cos2, eff2 = _asig_round(ta, cloud, edges, users, cloud_policy,
                                      1, 0.05)
    assert eff2 == -0.25
    assert e0.p2_verify_checkpoint(bundle2) is True

    # σ* 篡改: 云端谎报锁定档位 (用 σ=0.5 生成 alphas) → ES 复算 -0.25 → 拒
    cloud_forged = AlignmentSigmaPolicy(
        sigma_fallback=0.5, sigma_high=-0.25, ema_beta=0.8, warmup=2,
        boost_threshold=0.9, mid_threshold=0.72)
    bundle2f, _, _ = _asig_round(ta, cloud, edges, users, cloud_forged,
                                 1, 0.05, cloud_sigma_override=0.5)
    assert e0.p2_verify_checkpoint(bundle2f) is False

    # 两阶段语义: 轮 2 从未 commit → 双方锁定状态均未落地
    assert cloud_policy.locked is None
    assert e0._alignment_policy.locked is None


def test_asig_low_consistency_locks_fallback():
    """Aσ: cos 反向 (NI 形态) → 锁定 fallback σ; 云端谎报高一致档 → 拒."""
    ta, cloud, users, edges = make_env(sigma_policy='alignment',
                                       staleness_sigma=0.5, gamma_vcm=1.0)
    cloud_policy = AlignmentSigmaPolicy(
        sigma_fallback=0.5, sigma_high=-0.25, ema_beta=0.8, warmup=2,
        boost_threshold=0.9, mid_threshold=0.72)
    e0, e1 = edges

    _advance(cloud, 1)
    bundle1, cos1, eff1 = _asig_round(ta, cloud, edges, users, cloud_policy,
                                      0, -0.05)   # plain 与补偿反向
    assert eff1 == 0.5
    assert e0.p2_verify_checkpoint(bundle1) is True
    committed = cloud.p2_current_version + 1
    final1 = bundle1['final_model']
    cloud.p2_global_model_history[committed] = final1.copy()
    cloud.p2_current_version = committed
    cloud_policy.commit_observation(cos1)
    for e in edges:
        e.p2_note_committed_checkpoint(bundle1['selected_packages'],
                                       committed_version=committed,
                                       final_model=final1)

    # 轮 2: W(2) = 轮 1 提交的 final1; cos 反向 → 试探性锁 fallback σ=0.5
    bundle2, cos2, eff2 = _asig_round(ta, cloud, edges, users, cloud_policy,
                                      1, -0.05)
    assert eff2 == 0.5, "cos 反向 (< 0.72) 应保持 fallback σ"
    assert e0.p2_verify_checkpoint(bundle2) is True
    # 谎报高一致档 (σ=-0.25) → 拒
    cloud_forged = AlignmentSigmaPolicy(
        sigma_fallback=0.5, sigma_high=-0.25, ema_beta=0.8, warmup=2,
        boost_threshold=0.9, mid_threshold=0.72)
    bundle2f, _, _ = _asig_round(ta, cloud, edges, users, cloud_forged,
                                 1, -0.05, cloud_sigma_override=-0.25)
    assert e0.p2_verify_checkpoint(bundle2f) is False


def test_source_downgrade_and_pinning():
    """来源绑定: 降级为 none / 在合法规则间择优声明 均被拒绝.

    权重数值复算只约束"数值对不对"; 来源绑定约束"该用哪条规则" ——
    预定来源由 ES 本地协议配置 (participation_debias/sigma_target/σ)
    唯一确定, 云端把 vpda/staleness 降级为 none (静默退回等权) 或在
    两条合法规则之间挑更有利的一条, 都在来源核对即被拒。
    """
    # 预定来源 = vpda
    ta, cloud, users, edges = make_env(participation_debias='ipw',
                                       gamma_vcm=0.0)
    e0, e1 = edges
    assert e0._expected_weight_source == 'vpda'
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    _, ws = vpda_weights([0, 0], all_counts=[0, 0])
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=ws)
    _, final, _ = cloud.p2_weighted_vcm_average(selected, raw, ws,
                                                gamma_vcm=0.0)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, ws,
                          gamma=0.0, weight_source='vpda')
    assert e0.p2_verify_checkpoint(bundle) is True

    # 降级攻击 1: 撤掉权重、声明等权 none → 拒 (静默退回等权聚合被拦)
    assert e0.p2_verify_checkpoint(
        dict(bundle, weights=None, weight_source='none')) is False
    # 降级攻击 2: 声明等权 + 伪造等权数值 (数值自洽也无关紧要) → 拒
    assert e0.p2_verify_checkpoint(
        dict(bundle, weights=[SCALE, SCALE], weight_source='none')) is False
    # 择源攻击: 预定 vpda 却声明 staleness → 来源核对即拒 (复算规则被钉死)
    assert e0.p2_verify_checkpoint(
        dict(bundle, weight_source='staleness')) is False

    # 预定来源 = staleness (σ-agg)
    ta2, cloud2, users2, edges2 = make_env(staleness_sigma=0.5, gamma_vcm=0.0,
                                           sigma_target='agg')
    e2 = edges2[0]
    assert e2._expected_weight_source == 'staleness'
    _advance(cloud2, 1)
    _bind_es(edges2, cloud2, committed_v=1)
    base_hash2 = hashlib.sha256(
        cloud2.p2_global_model_history[0].tobytes()).digest()
    q0 = make_pkg(edges2[0], users2[:2], "q0:p0", 0, base_hash2, 200,
                  np.full(DIM, 0.05))
    q1 = make_pkg(edges2[1], users2[2:], "q1:p0", 0, base_hash2, 300,
                  np.full(DIM, 0.08))
    sel2 = [q0, q1]
    alphas = [staleness_alpha('version', 1, 0.5)] * 2
    ws2 = staleness_int_weights(alphas)
    M_Q2, H2 = ta2.p2_build_checkpoint_mask(sel2, None, weights=ws2)
    raw2 = cloud2.p2_recover_raw_sum(sel2, M_Q2, weights=ws2)
    _, final2, _ = cloud2.p2_weighted_average(sel2, raw2, ws2)
    bundle2 = build_bundle(cloud2, ta2, sel2, raw2, M_Q2, H2, final2, ws2,
                           gamma=0.0, alphas=alphas, weight_source='staleness')
    assert e2.p2_verify_checkpoint(bundle2) is True
    # 降级攻击: 预定 staleness 降级为 none → 拒
    assert e2.p2_verify_checkpoint(
        dict(bundle2, weights=None, weight_source='none')) is False
    # 择源攻击: 预定 staleness 却声明 vpda → 来源核对即拒
    assert e2.p2_verify_checkpoint(
        dict(bundle2, weight_source='vpda')) is False


def test_wait_omitted_alphas_rejected():
    """回归 (P1): wait 度量下云端省略 alphas 不能把衰减策略静默关闭.

    修复前 _p2_recompute_policy_alphas 用 None 同时表示"合法无 α"和
    "wait 不可复算": 云端省略 alphas 并按无衰减计算最终模型 → 验证
    通过 (WAIT_OMIT_ALPHAS_ACCEPTED=True)。修复后复算失败与合法无 α
    严格区分 ((ok, alphas) 返回约定), 本场景必须拒绝。
    """
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=1.0,
                                       staleness_metric='wait')
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    # 云端按无衰减 (α=1) 计算最终模型, 且 bundle 不提供 alphas
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=1.0, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=1.0, alphas=None, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is False


def test_rel2_unverifiable_returns_failure_not_none():
    """rel2 缺共识历史时复算必须返回 (False, None) 而非合法无 α."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=1.0,
                                       staleness_metric='rel2')
    e0 = edges[0]
    _advance(cloud, 1)
    _advance(cloud, 2)
    _bind_es(edges, cloud, committed_v=2)
    h1 = hashlib.sha256(
        cloud.p2_global_model_history[1].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 1, h1, 200, np.full(DIM, 0.05))
    # 缺 history[0] (v=2、r_k=1 均在, W0 缺失 → rel2 不可复算)
    history_no_w0 = {k: w for k, w in cloud.p2_global_model_history.items()
                     if k != 0}
    ok, alphas = e0._p2_recompute_policy_alphas(
        [p0], 2, history_no_w0, np.zeros(DIM, dtype=np.int64))
    assert ok is False and alphas is None


def test_missing_current_version_history_rejected():
    """bundle 省略 history[v] → fail-closed (不能静默关闭 VCM 补偿策略)."""
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=1.0)
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    alphas = [staleness_alpha('version', 1, 0.5)] * 2
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=1.0, vcm_mode='full', alphas=alphas)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=1.0, alphas=alphas, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is True
    # 省略 history[v=1] (补偿参考系) → 拒
    trimmed = dict(bundle)
    trimmed['global_model_history'] = {k: w for k, w
                                       in bundle['global_model_history'].items()
                                       if k != 1}
    assert e0.p2_verify_checkpoint(trimmed) is False


def test_policy_digest_binding():
    """配置摘要绑定: manifest 缺失/错配 policy_config_digest → 拒 (fail-closed)."""
    ta, cloud, users, edges = make_env(gamma_vcm=0.5)
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is True
    # 错配 → 拒 (云端另选配置)
    assert e0.p2_verify_checkpoint(
        dict(bundle, policy_config_digest='0' * 64)) is False
    # 缺失 → 拒 (省略摘要 = 省略策略绑定本身)
    b = dict(bundle)
    del b['policy_config_digest']
    assert e0.p2_verify_checkpoint(b) is False


def test_ta_independent_weight_recompute():
    """TA 独立复算: 云端声明权重只作审计, 掩码以 TA 复算值为准."""
    from utils import PRG, vpda_weights
    ta, cloud, users, edges = make_env(participation_debias='ipw',
                                       gamma_vcm=0.0)
    ta.p2_set_policy_config(edges[0]._policy_config)
    _advance(cloud, 1)
    _advance(cloud, 2)
    _bind_es(edges, cloud, committed_v=2)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[2].tobytes()).digest()
    p0 = make_pkg(edges[0], users[:2], "e0:p0", 2, base_hash, 200,
                  np.full(DIM, 0.05))
    p1 = make_pkg(edges[1], users[2:], "e1:p1", 2, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    # 双侧参与历史同步 (与 main 的 commit 通知同源): edge_0 已入选 2 次。
    # 直接注入历史计数 (note 现在带版本严格递增校验, 仅用于真实提交流程)
    for e in edges:
        e._p2_participation_counts = {'edge_0': 2}
    ta._p2_participation_counts = {'edge_0': 2}
    ta._p2_committed_version = 2

    # 云端声明等权 (伪造) → TA 以独立复算值为准 (返回的 M_Q/H 均按复算值)
    M_Q_declared, H_declared = ta.p2_build_checkpoint_mask(
        selected, None, weights=[SCALE, SCALE])
    # 参照: 直接按 TA 复算权重构造
    _, ws_true = vpda_weights(
        [2, 0], kappa=0.5, wmin=0.5, wmax=2.0, scale=SCALE,
        all_counts=[ta._p2_participation_counts.get(e.id, 0)
                    for e in edges])
    M_ref = np.zeros(DIM, dtype=np.int64)
    for w, pkg in zip(ws_true, selected):
        for u in (users[:2] if pkg is p0 else users[2:]):
            M_ref = M_ref + w * PRG.mask_int(u._ta_seed, u.id,
                                             pkg.package_token, DIM)
    assert np.array_equal(M_Q_declared, M_ref), \
        "TA 掩码必须按独立复算权重构建, 不采信云端声明值"
    # audit_only (影子/诊断路径) 才按声明值
    M_Q_audit, _ = ta.p2_build_checkpoint_mask(selected, None,
                                               weights=[SCALE, SCALE],
                                               audit_only=True)
    assert not np.array_equal(M_Q_audit, M_ref)

    # 端到端: 按 TA 复算权重恢复/聚合 → ES 验证通过 (来源与数值全绑定)
    raw = cloud.p2_recover_raw_sum(selected, M_Q_declared, weights=ws_true)
    _, final, _ = cloud.p2_weighted_vcm_average(selected, raw, ws_true,
                                                gamma_vcm=0.0)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q_declared, H_declared,
                          final, ws_true, gamma=0.0, alphas=None,
                          weight_source='vpda')
    assert edges[0].p2_verify_checkpoint(bundle) is True


def test_policy_param_declaration_mismatch_rejected():
    """声明的 γ/momentum 与 ES 本地协议配置不一致 → 拒 (fail-closed)."""
    ta, cloud, users, edges = make_env(gamma_vcm=0.5)
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is True

    # γ 谎报 → 拒 (另一选聚合策略不被声明值偷换)
    assert e0.p2_verify_checkpoint(dict(bundle, gamma_vcm=0.7)) is False
    # mode 谎报 (γ>0 时) → 拒
    assert e0.p2_verify_checkpoint(dict(bundle, vcm_mode='span')) is False
    # momentum 谎报 → 拒
    assert e0.p2_verify_checkpoint(dict(bundle, momentum=0.3)) is False


def test_version_binding_rejects_mismatch():
    """checkpoint 版本绑定: bundle 谎报版本 → 拒."""
    ta, cloud, users, edges = make_env(gamma_vcm=0.5)
    e0, e1 = edges
    _advance(cloud, 1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, weight_source='none')
    # ES 本地已提交 v=0, bundle 声明 v=1 → 版本错位拒绝
    assert e0.p2_verify_checkpoint(bundle) is False


def test_wait_metric_not_bindable():
    """wait 度量含云端调度状态: ES 无法独立复算 → 一律拒绝 (fail-closed)."""
    # α 路径 (等权掩码域): 预定来源 none, 云端提供 wait-σ 的 alphas → 复算即拒
    ta, cloud, users, edges = make_env(staleness_sigma=0.5, gamma_vcm=1.0,
                                       staleness_metric='wait')
    e0, e1 = edges
    _advance(cloud, 1)
    _bind_es(edges, cloud, committed_v=1)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    alphas = [0.7, 0.7]
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=1.0, vcm_mode='full', alphas=alphas)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=1.0, alphas=alphas, weight_source='none')
    assert e0.p2_verify_checkpoint(bundle) is False

    # σ-agg 权重 + wait → 同样拒绝 (α 复算即失败); 来源绑定下预定来源
    # 为 staleness, 单独再建一个 σ-agg 环境使来源核对通过、落到 wait 复算
    ta2, cloud2, users2, edges2 = make_env(staleness_sigma=0.5, gamma_vcm=0.0,
                                           staleness_metric='wait',
                                           sigma_target='agg')
    e2 = edges2[0]
    _advance(cloud2, 1)
    _bind_es(edges2, cloud2, committed_v=1)
    base_hash2 = hashlib.sha256(
        cloud2.p2_global_model_history[0].tobytes()).digest()
    q0 = make_pkg(edges2[0], users2[:2], "q0:p0", 0, base_hash2, 200,
                  np.full(DIM, 0.05))
    q1 = make_pkg(edges2[1], users2[2:], "q1:p0", 0, base_hash2, 300,
                  np.full(DIM, 0.08))
    sel2 = [q0, q1]
    ws2 = staleness_int_weights([0.7, 0.7])
    M_Q2, H2 = ta2.p2_build_checkpoint_mask(sel2, None, weights=ws2)
    raw2 = cloud2.p2_recover_raw_sum(sel2, M_Q2, weights=ws2)
    _, final2, _ = cloud2.p2_weighted_average(sel2, raw2, ws2)
    bundle_w = build_bundle(cloud2, ta2, sel2, raw2, M_Q2, H2, final2, ws2,
                            gamma=0.0, alphas=[0.7, 0.7],
                            weight_source='staleness')
    assert e2.p2_verify_checkpoint(bundle_w) is False
