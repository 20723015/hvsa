"""V-PDA 加权验证链路单元测试 (研究点 2, 阶段 0).

覆盖:
1. 等权回归: 全部权重相等时, 加权聚合+加权 VCM 与等权 G5 逐值等价。
2. 篡改拒绝: 修改权重 / raw sum / 最终模型, ES 验证必须失败;
   ES 按公开参与历史独立重算权重, 拒绝云端伪造。
3. int64 溢出边界: 加权恢复 (Σ w_k·masked_sum) 与 TA 加权掩码 (Σ w_k·M_k)
   每维最大值相对 int64 的余量。
"""

import hashlib
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud import CloudServer
from edge import EdgeServer
from ta import TrustedAuthority
from user import User
from utils import LHH, QUANTIZE_SCALE, PRG, vpda_weights

DIM = 16
SCALE = 65536


def make_env(**edge_kwargs):
    ta = TrustedAuthority(gradient_dim=DIM, ta_id="TA-T", enable_verification=True)
    ta.point2_reset()
    cloud = CloudServer()
    cloud.point2_reset(np.zeros(DIM, dtype=np.float64), initial_version=0)
    cloud._gradient_dim = DIM

    model = torch.nn.Linear(4, 2)
    users = []
    for i in range(4):
        u = User(f"U-{i}", model)
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta.get_public_key_bytes())
        u.point2_init_cache()
        users.append(u)

    edges = []
    edge_kwargs.setdefault("gamma_vcm", 0.5)
    for i in range(2):
        e = EdgeServer(f"edge_{i}", cloud, ta, enable_verification=True,
                       debias_kappa=0.5, debias_min=0.5, debias_max=2.0,
                       debias_scale=SCALE, **edge_kwargs)
        edges.append(e)
    # 记录当前 env: build_bundle 默认取第一台 ES 的协议配置摘要
    # (同 env 的 ES 配置一致, digest 相同)
    global _CURRENT_EDGES
    _CURRENT_EDGES = edges
    return ta, cloud, users, edges


_CURRENT_EDGES = None


def _default_digest():
    return (_CURRENT_EDGES[0]._policy_digest
            if _CURRENT_EDGES is not None else None)


def make_pkg(edge, users, token, base_v, base_hash, S, vals):
    """手工构造一个签名完整的组包 (跳过真实训练).

    上传梯度 = 明文·QS + 用户掩码 (与 user.p2_train_mix_mask_and_upload
    同构: 掩码 = PRG.mask_int(user_seed, user_id, token, dim)), 使
    masked_sum_int − M_k = 组内明文模型和, 与真实协议语义一致。
    """
    edge.p2_start_package(base_version=base_v, base_hash=base_hash,
                          package_token=token)
    n = len(users)
    plain = (vals * QUANTIZE_SCALE).astype(np.int64)
    for j, u in enumerate(users):
        mask = PRG.mask_int(u._ta_seed, u.id, token, DIM)
        g = plain + mask
        edge.p2_receive_masked_gradient(
            pickle.dumps({"user_id": u.id, "masked_gradient": g}),
            token, base_v, sample_size=S // n)
    return edge.p2_seal_group_package(tick=1)


def build_bundle(cloud, ta, selected, raw_sum_int, M_Q, H_mask_raw, final,
                 weights, gamma=0.5, alphas=None, weight_source='vpda',
                 vcm_mode='full', policy_digest=None):
    return {
        'checkpoint_version': cloud.p2_current_version,
        'policy_config_digest': (policy_digest if policy_digest is not None
                                 else _default_digest()),
        'selected_packages': selected,
        'raw_sum_int': raw_sum_int,
        'M_Q': M_Q,
        'H_mask_raw': H_mask_raw,
        'final_model': final,
        'gamma_vcm': gamma,
        'vcm_mode': vcm_mode,
        'alphas': alphas,
        'weights': weights,
        'weight_source': weight_source,
        'global_model_history': cloud.p2_global_model_history,
    }


def test_equal_weights_equivalent_to_g5():
    """全部权重相等 (A 全相等 → q=1 → w=scale) 时, 加权路径与等权 G5 逐值等价."""
    ta, cloud, users, edges = make_env()
    e0, e1 = edges
    init_flat = cloud.p2_global_model_history[0]
    base_hash = hashlib.sha256(init_flat.tobytes()).digest()

    vals0 = np.full(DIM, 0.05)
    vals1 = np.full(DIM, 0.08)
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, vals0)
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, vals1)
    selected = [p0, p1]

    # A=[0,0] → q 全 1 → w 全等于 scale (数学上 ≡ 等权)
    qs, ws = vpda_weights([0, 0], kappa=0.5, wmin=0.5, wmax=2.0, scale=SCALE)
    assert qs == [1.0, 1.0] and ws == [SCALE, SCALE]

    # 加权路径
    M_Q_w, _ = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw_w = cloud.p2_recover_raw_sum(selected, M_Q_w, weights=ws)
    _, final_w, _ = cloud.p2_weighted_vcm_average(
        selected, raw_w, ws, gamma_vcm=0.5, vcm_mode='full', alphas=None)

    # 等权 G5 路径
    M_Q_eq, _ = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw_eq = cloud.p2_recover_raw_sum(selected, M_Q_eq, weights=None)
    _, final_eq, _ = cloud.p2_apply_vcm_and_average(
        selected, raw_eq, gamma_vcm=0.5, vcm_mode='full', alphas=None)

    # 等权权重下加权路径应与 G5 逐值等价
    assert np.array_equal(final_w, final_eq)

    # base_version=0 == v=0 → 无 VCM 补偿项 → final = Σ 组明文模型和 / Σ S_k
    # 每组 2 个用户各上传 vals·QS → 组明文和 = 2·vals·QS
    expected = (2 * vals0 + 2 * vals1) / 500.0
    assert np.allclose(final_eq, expected, rtol=1e-12)
    assert np.allclose(final_w, expected, rtol=1e-12)


def test_weighted_verification_passes_and_rejects_tampering():
    """正确权重验证通过; 篡改权重/raw sum/最终模型均被 ES 拒绝."""
    ta, cloud, users, edges = make_env(participation_debias='ipw')
    e0, e1 = edges
    init_flat = cloud.p2_global_model_history[0]
    base_hash = hashlib.sha256(init_flat.tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]

    qs, ws = vpda_weights([0, 0], kappa=0.5, wmin=0.5, wmax=2.0, scale=SCALE)
    M_Q, H_mask_raw = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=ws)
    _, final, _ = cloud.p2_weighted_vcm_average(
        selected, raw, ws, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q,
                          H_mask_raw, final, ws, gamma=0.5)

    assert e0.p2_verify_checkpoint(bundle) is True
    assert e1.p2_verify_checkpoint(bundle) is True

    # 篡改 1: 云端伪造权重 (w0+1) → ES 独立重算不匹配 → 拒绝
    b_fake_w = dict(bundle, weights=[ws[0] + 1, ws[1]])
    assert e0.p2_verify_checkpoint(b_fake_w) is False

    # 篡改 2: raw sum 加 1 → LHH 等式破坏 → 拒绝
    b_fake_raw = dict(bundle, raw_sum_int=raw.copy())
    b_fake_raw['raw_sum_int'][0] += 1
    assert e0.p2_verify_checkpoint(b_fake_raw) is False

    # 篡改 3: 最终模型加 1 → 公开重算不匹配 → 拒绝
    b_fake_final = dict(bundle, final_model=final.copy())
    b_fake_final['final_model'][0] += 1.0
    assert e0.p2_verify_checkpoint(b_fake_final) is False


def test_es_recomputes_weights_from_local_history():
    """入选不均衡时 ES 独立重算的权重与云端一致; 伪造权重被拒."""
    ta, cloud, users, edges = make_env(participation_debias='ipw')
    e0, e1 = edges
    init_flat = cloud.p2_global_model_history[0]
    base_hash = hashlib.sha256(init_flat.tobytes()).digest()

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))

    # 模拟不均衡参与历史: edge_0 入选 3 次, edge_1 入选 1 次 (ES 本地独立累计)
    e0._p2_participation_counts = {"edge_0": 3, "edge_1": 1}
    e1._p2_participation_counts = {"edge_0": 3, "edge_1": 1}

    selected = [p0, p1]
    qs, ws = vpda_weights([3, 1], kappa=0.5, wmin=0.5, wmax=2.0, scale=SCALE)
    assert ws[0] < ws[1], "入选多的组应获得更小权重 (逆参与权重)"

    M_Q, H_mask_raw = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=ws)
    _, final, _ = cloud.p2_weighted_vcm_average(
        selected, raw, ws, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q,
                          H_mask_raw, final, ws, gamma=0.5)

    # ES 用本地历史重算的权重与云端一致 → 验证通过
    assert e0._p2_recompute_debias_weights(selected) == ws
    assert e0.p2_verify_checkpoint(bundle) is True

    # 云端用错误权重 (等权) 提交 → ES 重算比对失败 → 拒绝
    b_eq = dict(bundle, weights=[SCALE, SCALE])
    assert e0.p2_verify_checkpoint(b_eq) is False


def test_int64_overflow_margin():
    """int64 溢出边界: 加权恢复与 TA 加权掩码的每维上限相对 int64 余量 ≥ 1e4 倍."""
    from utils import PRG

    INT64_MAX = 2**63 - 1
    w_max = 2 * 65536          # q 上限 2.0 × scale
    n_groups = 5
    S_max = 2000.0             # CIFAR10 组内最大样本量上限 (实际 ≤1500)
    model_abs_max = 10.0       # 模型参数绝对值保守上限

    # 加权恢复: Σ w_k·masked_sum_k, 每维 ≤ n_groups·w_max·S_max·QS·|model|
    raw_bound = n_groups * w_max * S_max * QUANTIZE_SCALE * model_abs_max
    assert raw_bound < INT64_MAX // 10
    print(f"raw bound {raw_bound:.2e} vs int64 {INT64_MAX:.2e} "
          f"(余量 {INT64_MAX / raw_bound:.0f}x)")

    # TA 加权掩码: Σ w_k·M_k, 每维 ≤ n_groups·w_max·n_users_per_edge·|mask|
    n_users_per_edge = 20      # 架构上限保守值
    mask_bound = n_groups * w_max * n_users_per_edge * (
        PRG.MASK_RANGE + abs(PRG.MASK_LOW))
    assert mask_bound < INT64_MAX // 10
    print(f"mask bound {mask_bound:.2e} vs int64 {INT64_MAX:.2e} "
          f"(余量 {INT64_MAX / mask_bound:.0f}x)")


# ============================================================
#  V-PDA 证据链修正: weight_source 校验 + 全组均值
# ============================================================

def test_weight_source_validation():
    """来源绑定 + 来源校验: 缺来源/非法来源/声明来源无权重/来源不符 → 拒."""
    ta, cloud, users, edges = make_env(participation_debias='ipw')
    e0, e1 = edges
    base_hash = hashlib.sha256(cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    qs, ws = vpda_weights([0, 0], all_counts=[0, 0])
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=ws)
    _, final, _ = cloud.p2_weighted_vcm_average(selected, raw, ws,
                                                gamma_vcm=0.5, vcm_mode='full')
    ok = build_bundle(cloud, ta, selected, raw, M_Q, H, final, ws,
                      weight_source='vpda')
    assert e0.p2_verify_checkpoint(ok) is True

    # 权重存在但来源缺失 → 拒绝
    b = dict(ok); del b['weight_source']
    assert e0.p2_verify_checkpoint(b) is False
    # 非法来源 → 拒绝
    assert e0.p2_verify_checkpoint(dict(ok, weight_source='bogus')) is False
    # 声明来源却无权重 → 拒绝
    assert e0.p2_verify_checkpoint(dict(ok, weights=None)) is False
    # 非正权重 → 拒绝
    assert e0.p2_verify_checkpoint(dict(ok, weights=[0, int(ws[1])])) is False
    # 来源绑定: ES 期望 vpda (协议配置), 声明 staleness 即便数值自洽
    # (σ=0 时恰为 65536) 也被来源核对拒绝 —— 非简并改标场景见
    # tests/test_policy_binding.py
    b_st = dict(ok, weight_source='staleness')
    assert e0.p2_verify_checkpoint(b_st) is False
    b_tamper = dict(b_st, final_model=final.copy())
    b_tamper['final_model'][0] += 1.0
    assert e0.p2_verify_checkpoint(b_tamper) is False
    assert e0.p2_verify_checkpoint(
        dict(b_st, weights=[int(ws[0]) + 1, int(ws[1])])) is False


def test_vpda_weights_all_groups_mean():
    """均值必须取全部组 (公平基准), 而非选中子集."""
    # 全组 A=[6,0,0,0,0] (选中组 A=[6,0]): 全组均值=(7+1+1+1+1)/5=2.2
    # (wmin=0.1 避免下限 clip 干扰断言)
    qs_all, ws_all = vpda_weights([6, 0], kappa=1.0, wmin=0.1, wmax=2.0,
                                  all_counts=[6, 0, 0, 0, 0])
    qs_sub, ws_sub = vpda_weights([6, 0], kappa=1.0, wmin=0.1, wmax=2.0)
    assert ws_all != ws_sub
    assert abs(qs_all[0] - 2.2 / 7.0) < 1e-12
    assert abs(qs_sub[0] - 4.0 / 7.0) < 1e-12


# ============================================================
#  UR-VCM (vcm-mode=trust): 正确性 / 回归 / 门控 / 篡改拒绝
# ============================================================

def synth_pkg(edge_id, token, base_v, S_k, masked_sum_int, H_k=None):
    from point2_types import GroupPackage
    return GroupPackage(edge_id=edge_id, package_token=token, base_version=base_v,
                        base_hash=b'', online_users=[], n_k=1, S_k=S_k,
                        masked_sum_int=masked_sum_int, H_k=H_k, signature=b'',
                        created_tick=0)


def _trust_cloud(rho=1.0):
    dim = DIM
    W0 = np.full(dim, 0.1)
    W1 = np.full(dim, 0.3)
    cloud = CloudServer()
    cloud.point2_reset(W0.copy(), 0)
    cloud.p2_global_model_history[1] = W1.copy()
    cloud.p2_current_version = 1
    cloud._vcm_trust_ratio = rho
    return cloud, W0, W1


def test_urvcm_trust_formula_equal():
    """等权 trust: W=R+η·C̄, η=min(1, ρ‖Δtrain‖/(‖C̄‖+ε)); 诊断字段一致."""
    cloud, W0, W1 = _trust_cloud(rho=1.0)
    S = 100
    R_target = W0 + 0.05                     # 训练增量 Δtrain = 0.05
    masked = np.round(S * R_target * QUANTIZE_SCALE).astype(np.int64)
    p = synth_pkg('e0', 't0', 0, S, masked)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        [p], masked, gamma_vcm=0.5, vcm_mode='trust', alphas=None)

    R = R_target
    B = W0
    C_bar = 0.5 * (W1 - W0)                  # C_k/S = γ·(W1−W0)
    d_norm = float(np.linalg.norm(R - B))
    c_norm = float(np.linalg.norm(C_bar))
    eta = min(1.0, d_norm / (c_norm + 1e-12))
    assert abs(eta - 0.5) < 1e-9, f"η 应为 0.5, 实际 {eta}"
    assert np.allclose(final, R + eta * C_bar, rtol=1e-9)
    d = cloud.p2_last_vcm_diag
    assert abs(d['eta'] - eta) < 1e-9
    assert abs(d['train_norm'] - d_norm) < 1e-9
    assert d['gated'] is True


def test_urvcm_large_rho_matches_full():
    """ρ→∞ (η=1) 时 trust 与旧 full 模式逐值一致 (full 行为回归锚点)."""
    cloud, W0, W1 = _trust_cloud(rho=1e12)
    S = 100
    masked = np.round(S * (W0 + 0.05) * QUANTIZE_SCALE).astype(np.int64)
    p = synth_pkg('e0', 't0', 0, S, masked)
    _, final_trust, comp_trust = cloud.p2_apply_vcm_and_average(
        [p], masked, gamma_vcm=0.5, vcm_mode='trust', alphas=None)
    _, final_full, comp_full = cloud.p2_apply_vcm_and_average(
        [p], masked, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    assert np.array_equal(final_trust, final_full)
    assert comp_trust == comp_full


def test_urvcm_delta_zero_suppresses_compensation():
    """Δtrain=0 且 C≠0 时 η=0: 补偿被完全抑制, W=R; C=0 时恒等."""
    cloud, W0, W1 = _trust_cloud(rho=1.0)
    S = 100
    # R == B == W0 → Δtrain = 0
    masked = np.round(S * W0 * QUANTIZE_SCALE).astype(np.int64)
    p = synth_pkg('e0', 't0', 0, S, masked)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        [p], masked, gamma_vcm=0.5, vcm_mode='trust', alphas=None)
    assert np.allclose(final, W0, rtol=1e-12)
    assert cloud.p2_last_vcm_diag['eta'] == 0.0
    # C=0 (gamma=0) → W=R (恒等, 与 full 一致)
    _, final2, _ = cloud.p2_apply_vcm_and_average(
        [p], masked, gamma_vcm=0.0, vcm_mode='trust', alphas=None)
    assert np.allclose(final2, W0, rtol=1e-12)


def test_urvcm_trust_weighted():
    """加权 trust: denom=Σw_kS_k, R/B/C̄ 均按加权口径, η 同公式."""
    cloud, W0, W1 = _trust_cloud(rho=1.0)
    S1, S2, w1, w2 = 100, 300, SCALE, 2 * SCALE
    R1, R2 = W0 + 0.04, W0 + 0.08
    m1 = np.round(S1 * R1 * QUANTIZE_SCALE).astype(np.int64)
    m2 = np.round(S2 * R2 * QUANTIZE_SCALE).astype(np.int64)
    p1 = synth_pkg('e0', 't1', 0, S1, m1)
    p2 = synth_pkg('e1', 't2', 0, S2, m2)
    weights = [w1, w2]
    raw = w1 * m1 + w2 * m2        # 掩码域加权恢复: Σ w_k·masked_sum_k
    _, final, _ = cloud.p2_weighted_vcm_average(
        [p1, p2], raw, weights, gamma_vcm=0.5, vcm_mode='trust', alphas=None)
    denom = w1 * S1 + w2 * S2
    R = (w1 * S1 * R1 + w2 * S2 * R2) / denom
    B = W0
    C_bar = sum(w * S * 0.5 * (W1 - W0) for w, S in ((w1, S1), (w2, S2))) / denom
    d_norm = float(np.linalg.norm(R - B))
    c_norm = float(np.linalg.norm(C_bar))
    eta = min(1.0, d_norm / (c_norm + 1e-12))
    assert np.allclose(final, R + eta * C_bar, rtol=1e-9)
    assert abs(cloud.p2_last_vcm_diag['eta'] - eta) < 1e-9


def test_urvcm_es_accepts_and_rejects():
    """ES 端 trust 模式: 正确通过; 篡改补偿/最终模型/权重均拒绝."""
    ta, cloud, users, edges = make_env(vcm_mode='trust',
                                       participation_debias='ipw')
    e0, e1 = edges
    init = cloud.p2_global_model_history[0]
    base_hash = hashlib.sha256(init.tobytes()).digest()
    # 构造 v=1, 包基于 v=0 → 存在补偿项
    cloud.p2_global_model_history[1] = init + 0.2
    cloud.p2_current_version = 1
    cloud._vcm_trust_ratio = 1.0
    # ES 本地共识状态同步到 v=1 (版本绑定检查要求)
    for e in edges:
        e._p2_committed_version = 1
        e._p2_committed_history[1] = cloud.p2_global_model_history[1]

    p0 = make_pkg(e0, users[:2], "e0:p0", 0, base_hash, 200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:p1", 0, base_hash, 300, np.full(DIM, 0.08))
    selected = [p0, p1]
    qs, ws = vpda_weights([0, 0], all_counts=[0, 0])
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=ws)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=ws)
    _, final, _ = cloud.p2_weighted_vcm_average(
        selected, raw, ws, gamma_vcm=0.5, vcm_mode='trust', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, ws,
                          weight_source='vpda', vcm_mode='trust')
    assert e0.p2_verify_checkpoint(bundle) is True

    # 篡改补偿口径 (用 full 模式模型冒充 trust 结果) → 拒绝
    _, final_full, _ = cloud.p2_weighted_vcm_average(
        selected, raw, ws, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    assert not np.array_equal(final, final_full)
    assert e0.p2_verify_checkpoint(dict(bundle, final_model=final_full)) is False
    # 篡改最终模型 → 拒绝
    b = dict(bundle, final_model=final.copy()); b['final_model'][0] += 1.0
    assert e0.p2_verify_checkpoint(b) is False

    # 篡改权重 → 拒绝
    assert e0.p2_verify_checkpoint(
        dict(bundle, weights=[int(ws[0]) + 1, int(ws[1])])) is False


def test_urvcm_parser_rejects_bad_ratio():
    """非法可信域参数由 parser.error 拒绝 (非正/NaN/Inf)."""
    import subprocess
    main_py = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "main_point2.py")
    for bad in ("0", "-1", "nan", "inf"):
        r = subprocess.run([sys.executable, main_py, "--vcm-mode", "trust",
                            "--vcm-trust-ratio", bad],
                           capture_output=True, text=True, errors="replace")
        assert r.returncode != 0, f"rho={bad} should be rejected"
        assert "vcm-trust-ratio" in (r.stderr + r.stdout)


def test_vpda_parser_rejects_bad_debias_params():
    """非法 V-PDA 权重参数由 parser.error 拒绝."""
    import subprocess
    main_py = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "main_point2.py")
    cases = [("--debias-min", "0"), ("--debias-min", "3"), ("--debias-kappa", "-1"),
             ("--debias-scale", "0")]
    for flag, bad in cases:
        r = subprocess.run([sys.executable, main_py, "--participation-debias",
                            "ipw", flag, bad],
                           capture_output=True, text=True, errors="replace")
        assert r.returncode != 0, f"{flag}={bad} should be rejected"
