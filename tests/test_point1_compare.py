"""研究点一对比实验的协议正确性测试 (E1/E5 小维度快速回归).

覆盖 protocol_mapping.md 的映射契约:
- KA.gen() 接口顺序 (pub, priv) 与对称性 (计划 §3 / mapping §2.3);
- GPVO 成对掩码两端同流抵消、TA 掩码 user/TA 对称;
- RSA 型承诺同态 (Commit(∏m,Στ) = ∏Commit) 与按原文重算式 Decommit;
- GPVO 完整回合 (Alg.2-5) 去掩码和逐维等于明文参考, 含掉线补偿与空组;
- HVSA 真实单 TA 路径 (真实实体类) 去掩码和逐维等于明文参考;
- E5 篡改/重放: HVSA签名绑定可检出; GPVO兼容变体固定密钥下同集合重放
  可通过, 新初始化则另测; 不将固定密钥选择归因于原论文。
- GPVO兼容变体的结果和承诺可协调修改; 这是已知实现限制的回归记录,
  通过该测试不代表协议通过完整性验收。
"""

import copy
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from experiments.point1_compare.adapters.common import (  # noqa: E402
    gen_client_vectors,
    reference_sum,
)
from experiments.point1_compare.adapters.gpvo_full import (  # noqa: E402
    GPVOSession,
    com_aggregate,
    com_commit,
    com_decommit,
    com_param_gen,
    run_gpvo_round,
)
from experiments.point1_compare.adapters.hvsa_point1 import HVSAProtocol  # noqa: E402
from utils import KA, LHH, PRG  # noqa: E402

D = 32
AMP = 1_000_000_000
MIX = {"uniform": 0.8, "zero": 0.1, "boundary": 0.1}


@pytest.fixture(scope="module")
def com_params():
    return com_param_gen(1024)  # 测试用小模数; 同态代数与正式 3072-bit 一致


def _ref(vecs, online):
    return reference_sum([vecs[u] for u in sorted(online)])


def _rng(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ 接口与原语

def test_ka_pub_priv_order_and_symmetry():
    """计划 §3: KA.gen() 第一返回值为公钥; agree(priv_a, pub_b) == agree(priv_b, pub_a)."""
    pub_a, priv_a = KA.gen()
    pub_b, priv_b = KA.gen()
    assert len(pub_a) == len(priv_a) == 32
    assert KA.agree(priv_a, pub_b) == KA.agree(priv_b, pub_a)


def test_gpvo_pairwise_mask_cancellation(com_params):
    """mapping §2.2-4: 成对掩码两端用同一对称标签派生同一流; TA 掩码 user/TA 对称."""
    session = GPVOSession(D, [["u0", "u1", "u2"]], com_params=com_params)
    # 两端派生的成对掩码必须完全一致 (否则残留无法抵消)
    s_01_user = session.pair_key_user("u0", "u1")
    s_01_peer = session.pair_key_user("u1", "u0")
    assert s_01_user == s_01_peer
    from experiments.point1_compare.adapters.gpvo_full import _pair_mask
    assert np.array_equal(_pair_mask(s_01_user, 0, "u0", "u1", D),
                          _pair_mask(s_01_peer, 0, "u1", "u0", D))
    # TA 侧与用户侧的 s_{i,TA} 一致
    s_ta_user = KA.agree(session._user_priv["u0"], session._ta_pub)
    assert session._ta_shared["u0"] == s_ta_user
    from experiments.point1_compare.adapters.gpvo_full import _ta_mask
    assert np.array_equal(_ta_mask(s_ta_user, "u0", D),
                          PRG.mask_int(session._ta_shared["u0"], "gpvo-ta|u0", "static_per_paper", D))


def test_gpvo_commitment_homomorphism(com_params):
    """定理 1 代数: ∏Commit(m_k, τ_k) = Commit(∏m_k, Στ_k); Decommit 按原文重算."""
    p = com_params
    ms = [12345, 67890, 999]
    taus = [p.n2 // 7, p.n2 // 11, 424242]
    c_prod = com_aggregate(p, [com_commit(p, m, t) for m, t in zip(ms, taus)])
    m_prod = 1
    for m in ms:
        m_prod = (m_prod * m) % p.n2
    assert c_prod == com_commit(p, m_prod, sum(taus))
    assert com_decommit(p, c_prod, m_prod, sum(taus))
    assert not com_decommit(p, c_prod, (m_prod + 1) % p.n2, sum(taus))
    assert not com_decommit(p, c_prod, m_prod, sum(taus) + 1)


# ------------------------------------------------------------------ GPVO 完整回合

def _gpvo_case(groups, online, seed=1, com_params=None):
    session = GPVOSession(D, groups, com_params=com_params)
    vecs = gen_client_vectors(_rng(seed), sum(len(g) for g in groups), D, MIX, AMP)
    res = run_gpvo_round(session, vecs, set(online), round_t=7)
    ref = _ref(vecs, online)
    return res, ref


def test_gpvo_round_matches_reference(com_params):
    groups = [["u0", "u1", "u2", "u3"], ["u4", "u5", "u6"], ["u7", "u8"]]
    res, ref = _gpvo_case(groups, [f"u{i}" for i in range(9)], com_params=com_params)
    assert res.completed and res.verified
    assert np.array_equal(res.raw_sum, ref)


def test_gpvo_dropout_compensation(com_params):
    # 单用户掉线
    res, ref = _gpvo_case([["u0", "u1", "u2", "u3"], ["u4", "u5"]],
                          ["u0", "u2", "u3", "u4", "u5"], com_params=com_params)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)
    # 相邻两用户连续掉线 (环定向案例)
    res, ref = _gpvo_case([["u0", "u1", "u2", "u3"], ["u4", "u5"]],
                          ["u0", "u3", "u4", "u5"], com_params=com_params)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)
    # m_k=2 退化 (i-1 = i+1 为同一邻居, 两掩码项自相消)
    res, ref = _gpvo_case([["u0", "u1"], ["u2", "u3", "u4"]], ["u0", "u3", "u4"],
                          com_params=com_params)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)
    # 组内仅剩 1 人且两侧邻居均掉线
    res, ref = _gpvo_case([["u0", "u1", "u2", "u3", "u4"], ["u5", "u6"]],
                          ["u2", "u5", "u6"], com_params=com_params)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)


def test_gpvo_empty_group_skipped(com_params):
    """组内无人上传: 该边缘不参与本轮 (mapping §2.2); 其余组仍正确."""
    res, ref = _gpvo_case([["u0", "u1", "u2"], ["u3", "u4"]], ["u0", "u1", "u2"],
                          com_params=com_params)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)
    assert res.n_uploaded == 3


# ------------------------------------------------------------------ HVSA 真实路径

def _hvsa_round(groups, online, seed=1):
    proto = HVSAProtocol(D, groups)
    vecs = gen_client_vectors(_rng(seed), sum(len(g) for g in groups), D, MIX, AMP)
    res = proto.run_round(vecs, set(online), round_t=3)
    return proto, vecs, res, _ref(vecs, online)


def test_hvsa_round_matches_reference():
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    proto, vecs, res, ref = _hvsa_round(groups, [f"u{i}" for i in range(6)])
    assert res.completed and res.verified
    assert np.array_equal(res.raw_sum, ref)
    assert res.extra["per_edge_verified"] == {"edge_0": True, "edge_1": True}
    # 消息字节可计量: 每用户一条 user->edge, 每边缘 edge->cloud + edge->ta
    links = {}
    for m in res.messages:
        links[m.link] = links.get(m.link, 0) + 1
    assert links["user->edge"] == 6
    assert links["edge->cloud"] == 2
    assert links["edge->ta"] == 2
    assert links["ta->cloud"] == 1
    assert all(m.num_bytes > 0 for m in res.messages)
    assert links["cloud->edge"] == 2
    assert links["ta->edge"] == 4
    assert links["edge->user"] == 6
    assert len({m.receiver for m in res.messages if m.link == "edge->user"}) == 6


def test_hvsa_tamper_rejected():
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    proto, vecs, res, ref = _hvsa_round(groups, [f"u{i}" for i in range(6)])
    honest_hl = copy.deepcopy(proto.cs._collected_hash_data)
    es = proto.edges[0]

    # 修改明文聚合和 (单坐标 / 多坐标 / 整体缩放) → 全部拒绝
    tw = res.raw_sum.copy(); tw[0] += 7
    assert not es.verify_global_result(tw, honest_hl)
    tw = res.raw_sum.copy(); tw[:5] += np.arange(1, 6)
    assert not es.verify_global_result(tw, honest_hl)
    tw = (res.raw_sum.astype(np.float64) * 1.01).astype(np.int64)
    assert not es.verify_global_result(tw, honest_hl)

    # 修改证据 (哈希 / 签名 / 上下文字段) → 全部拒绝
    hl = copy.deepcopy(honest_hl)
    hl[0]["lhh_hash"] = LHH.Hash(LHH.Gen(D), np.array([1], dtype=np.int64))
    assert not es.verify_global_result(res.raw_sum, hl)
    hl = copy.deepcopy(honest_hl)
    hl[0]["signature"] = bytes((hl[0]["signature"][0] ^ 0xFF,)) + hl[0]["signature"][1:]
    assert not es.verify_global_result(res.raw_sum, hl)
    hl = copy.deepcopy(honest_hl)
    hl[0]["round"] = 999
    assert not es.verify_global_result(res.raw_sum, hl)
    hl = copy.deepcopy(honest_hl)
    hl[0]["base_hash"] = bytes(32)
    assert not es.verify_global_result(res.raw_sum, hl)
    hl = copy.deepcopy(honest_hl)
    hl[0]["online_users"] = list(hl[0]["online_users"]) + ["ghost"]
    assert not es.verify_global_result(res.raw_sum, hl)

    # 漏组 / 重复组 → 全部拒绝
    assert not es.verify_global_result(res.raw_sum, honest_hl[1:])
    assert not es.verify_global_result(res.raw_sum, honest_hl + [copy.deepcopy(honest_hl[0])])

    # 重放历史结果 (旧结果+旧证据在新轮呈现) → 签名轮次绑定拒绝
    vecs2 = gen_client_vectors(_rng(99), 6, D, MIX, AMP)
    proto.run_round(vecs2, set(f"u{i}" for i in range(6)), round_t=4)
    assert not es.verify_global_result(res.raw_sum, honest_hl)


def test_hvsa_coordinated_omission_and_duplication_rejected():
    """Use only cloud-received sums and evidence; valid entry signatures persist."""
    import pickle

    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    proto = HVSAProtocol(D, groups)
    received = []
    original = proto.cs.receive_pre_aggregation

    def capture(raw):
        received.append(pickle.loads(raw))
        return original(raw)

    proto.cs.receive_pre_aggregation = capture
    vectors = gen_client_vectors(_rng(23), 6, D, MIX, AMP)
    result = proto.run_round(vectors, set(vectors), 1)
    evidence = copy.deepcopy(proto.cs._collected_hash_data)
    group_sum = received[0]["pre_aggregated_gradient"]
    verifier = proto.edges[0]
    assert result.verified
    assert verifier.verify_global_result(result.raw_sum, list(reversed(evidence)))
    assert not verifier.verify_global_result(result.raw_sum - group_sum, evidence[1:])
    assert not verifier.verify_global_result(result.raw_sum + group_sum,
                                             evidence + [copy.deepcopy(evidence[0])])


def test_hvsa_controls_preserve_sum_and_label_missing_verification():
    """E6 controls change work without silently claiming equal security."""
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    vectors = gen_client_vectors(_rng(31), 6, D, MIX, AMP)
    online = set(vectors)
    reference = _ref(vectors, online)
    control = HVSAProtocol(D, groups, mask_layers=3)
    result = control.run_round(vectors, online, 1)
    assert result.verified and np.array_equal(result.raw_sum, reference)
    assert sum(m.link == "user->edge" for m in result.messages) == 6
    with pytest.raises(ValueError, match="all clients"):
        control.run_round(vectors, online - {"u0"}, 2)
    off = HVSAProtocol(D, groups, enable_verification=False)
    result = off.run_round(vectors, online, 1, model_recipients=online - {"u0"})
    assert result.completed and result.verified is None
    assert np.array_equal(result.raw_sum, reference)
    assert sum(m.link == "edge->user" for m in result.messages) == 5


def test_hvsa_rejects_noncanonical_output_domain():
    """Authentic evidence cannot authorize a different numeric representation."""
    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    proto, vectors, result, reference = _hvsa_round(groups, [f"u{i}" for i in range(6)])
    evidence = copy.deepcopy(proto.cs._collected_hash_data)
    verifier = proto.edges[0]
    aliased = result.raw_sum.astype(object)
    aliased[0] += LHH.CURVE.order
    floating = result.raw_sum.astype(np.float64)
    floating[0] += .5 if floating[0] >= 0 else -.5
    assert not verifier.verify_global_result(aliased, evidence)
    assert not verifier.verify_global_result(floating, evidence)
    assert not verifier.verify_global_result(result.raw_sum.reshape(1, -1), evidence)
    assert not verifier.verify_global_result(result.raw_sum[:-1], evidence)
    assert not verifier.verify_global_result(np.abs(result.raw_sum).astype(np.uint64), evidence)
    assert verifier.verify_global_result(result.raw_sum.astype(">i8"), evidence)


def test_gpvo_tamper_rejected(com_params):
    from experiments.point1_compare.run_compare import _gpvo_verify_only
    groups = [["u0", "u1", "u2"], ["u3", "u4"]]
    online = {f"u{i}" for i in range(5)}
    session = GPVOSession(D, groups, com_params=com_params)
    vecs = gen_client_vectors(_rng(1), 5, D, MIX, AMP)
    res = run_gpvo_round(session, vecs, online, round_t=7)
    ref = _ref(vecs, online)
    assert res.completed and res.verified and np.array_equal(res.raw_sum, ref)
    art = res.extra["artifacts"]
    h_ta = art["ta"]["h_ta"]
    r = art["ta"]["r"]

    # 修改 x_agg → Decommit 拒绝
    tx = res.raw_sum.copy(); tx[0] += 7
    assert not _gpvo_verify_only(session, tx, art["cloud"]["c_agg"], h_ta, r)
    tx = (res.raw_sum.astype(np.float64) * 1.01).astype(np.int64)
    assert not _gpvo_verify_only(session, tx, art["cloud"]["c_agg"], h_ta, r)
    # 修改证据 c_agg → 拒绝
    assert not _gpvo_verify_only(session, res.raw_sum, art["cloud"]["c_agg"] ^ 0xFF, h_ta, r)
    # 漏组 / 重复组 → 拒绝
    factors = [art["edge"][k]["c_k"] for k in sorted(art["edge"])]
    c_drop = com_aggregate(com_params, factors[1:])
    assert not _gpvo_verify_only(session, res.raw_sum, c_drop, h_ta, r)
    c_dup = (art["cloud"]["c_agg"] * factors[0]) % com_params.n2
    assert not _gpvo_verify_only(session, res.raw_sum, c_dup, h_ta, r)

    # 重放: 同一部署内 (相同 τ_k/密钥, 静态掩码), 旧结果+旧证据在相同在线集合
    # 的新轮呈现 → 当前固定密钥变体接受; 不代表每轮新初始化的原论文行为。
    vecs2 = gen_client_vectors(_rng(99), 5, D, MIX, AMP)
    res2 = run_gpvo_round(session, vecs2, online, round_t=8)
    art2 = res2.extra["artifacts"]
    assert _gpvo_verify_only(session, res.raw_sum, art["cloud"]["c_agg"],
                             art2["ta"]["h_ta"], art2["ta"]["r"])
    # 在线集合变化 → ξ 变化 → 检出
    res3 = run_gpvo_round(session, vecs2, {"u0", "u1", "u2", "u3"}, round_t=9)
    art3 = res3.extra["artifacts"]
    assert not _gpvo_verify_only(session, res.raw_sum, art["cloud"]["c_agg"],
                                 art3["ta"]["h_ta"], art3["ta"]["r"])


def test_gpvo_fresh_initialization_rejects_old_round(com_params):
    """Fresh DH/commitment randomness changes the replay result; key lifecycle matters."""
    from experiments.point1_compare.run_compare import _gpvo_verify_only

    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    online = {u for g in groups for u in g}
    vecs = gen_client_vectors(_rng(19), 6, D, MIX, AMP)
    old = run_gpvo_round(GPVOSession(D, groups, com_params=com_params),
                         vecs, online, round_t=1)
    fresh_session = GPVOSession(D, groups, com_params=com_params)
    current = run_gpvo_round(fresh_session, vecs, online, round_t=2)
    assert old.verified and current.verified
    old_art = old.extra["artifacts"]
    current_art = current.extra["artifacts"]
    assert not _gpvo_verify_only(fresh_session, old.raw_sum,
                                 old_art["cloud"]["c_agg"],
                                 current_art["ta"]["h_ta"], current_art["ta"]["r"])


def test_gpvo_compatibility_variant_accepts_coordinated_tamper(com_params):
    """Known limitation: public homomorphism changes x and c together without TA secrets."""
    from experiments.point1_compare.adapters.gpvo_full import lhf_hash
    from experiments.point1_compare.run_compare import _gpvo_verify_only

    groups = [["u0", "u1", "u2"], ["u3", "u4", "u5"]]
    online = {u for g in groups for u in g}
    session = GPVOSession(D, groups, com_params=com_params)
    vecs = gen_client_vectors(_rng(21), 6, D, MIX, AMP)
    honest = run_gpvo_round(session, vecs, online, round_t=1)
    assert honest.verified
    art = honest.extra["artifacts"]
    delta = np.zeros(D, dtype=np.int64)
    delta[0] = 7
    fake_sum = honest.raw_sum + delta
    # Construction uses public parameters, the cloud commitment and chosen delta.
    # Authentic TA aid is only used by the verifier, never by the attacker.
    fake_commitment = (art["cloud"]["c_agg"]
                       * pow(lhf_hash(com_params, delta), com_params.e, com_params.n2)) % com_params.n2
    assert not np.array_equal(fake_sum, honest.raw_sum)
    assert _gpvo_verify_only(session, fake_sum, fake_commitment,
                             art["ta"]["h_ta"], art["ta"]["r"])
