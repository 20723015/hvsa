"""TA 权威授权流测试: vote → CheckpointReceipt → 提交/推进 (验收计划第 1-3 天).

覆盖验收门槛中的负例:
- token 未注册 / 已消费 (重放) → TA 授权拒绝;
- 常规触发 |Q| ≠ K、超时触发 |Q| ≥ K → 拒绝;
- 组包陈旧度 > s_max → 拒绝;
- checkpoint 版本跳变 → 拒绝;
- vote 篡改 (passed/token digest/签名) → 拒绝;
- 重放 receipt: Cloud 重启后凭旧 receipt 提交无法让 ES 推进 (版本严格 +1);
- receipt 签名篡改 → ES 不推进任何状态 (版本/参与计数/Aσ 状态)。

P0 闭环 (finalize 可验证/持久化/幂等/可恢复):
- 伪造或未经 prepare 的 receipt 无法写 TA ledger;
- 相同 receipt 重复 finalize 幂等成功, 同 id 不同内容才拒绝;
- 同版本换候选 → prepare 记录替换, 旧 receipt 作废;
- confirm_receipt 核对完整 receipt (签名 + prepare 记录 + 已 finalize);
- stage → TA finalize → publish → notify 四阶段崩溃恢复均收敛,
  且不重复消费 token / 不重复计入参与历史。
"""

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_vpda import DIM, build_bundle, make_env, make_pkg

from commit_tx import (CrashInjected, recover_staged_commit,
                       run_commit_transaction)
from point2_types import (CheckpointReceipt, ProtocolConfig, result_digest,
                          tokens_digest)
from utils import SIG


def _full_env(stage_path=None, **kwargs):
    cfg = ProtocolConfig(K=2, s_max=3, gamma_vcm=0.5)
    ta, cloud, users, edges = make_env(policy_config=cfg,
                                       gamma_vcm=0.5, **kwargs)
    ta.p2_set_policy_config(cfg)
    cloud.p2_set_receipt_pub_key(ta.get_receipt_public_key())
    if stage_path is not None:
        cloud.p2_set_stage_path(stage_path)
    return ta, cloud, users, edges, cfg


def _prepare_round(ta, cloud, users, edges, cfg, v, base_v=None,
                   seq=0, trigger_mode='normal'):
    """准备一个候选: 封包 → bundle → votes → TA 授权 (prepare, 不提交)."""
    base_v = v if base_v is None else base_v
    cloud.p2_global_model_history.setdefault(v, cloud.p2_global_model_history[0])
    cloud.p2_current_version = v
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[base_v].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], f"e0:v{base_v}:p{seq}", base_v, base_hash,
                  200, np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], f"e1:v{base_v}:p{seq}", base_v, base_hash,
                  300, np.full(DIM, 0.08))
    selected = [p0, p1]
    cloud.p2_receive_group_package(p0)
    cloud.p2_receive_group_package(p1)
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]
    receipt = ta.p2_authorize_checkpoint(
        selected, votes, checkpoint_version=v,
        trigger_mode=trigger_mode, config=cfg)
    return selected, bundle, votes, receipt


def _checkpoint_round(ta, cloud, users, edges, cfg, v, base_v=None,
                      seq=0, trigger_mode='normal'):
    """完整一轮: prepare → stage → TA finalize → publish → notify (事务)."""
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v, base_v=base_v, seq=seq,
        trigger_mode=trigger_mode)
    final = bundle['final_model']
    run_commit_transaction(
        cloud, ta, edges, selected_packages=selected, final_model=final,
        raw_sum_int=bundle['raw_sum_int'], M_Q=bundle['M_Q'],
        H_mask_raw=bundle['H_mask_raw'], corrected_sum=final * 500,
        comp_map={}, receipt=receipt)
    return selected, receipt


def test_happy_path_vote_receipt_commit():
    """诚实路径: 全 ES vote 验签通过 → receipt 签发 → 提交 → ES 推进."""
    ta, cloud, users, edges, cfg = _full_env()
    assert cloud.p2_current_version == 0
    _checkpoint_round(ta, cloud, users, edges, cfg, v=0)
    assert cloud.p2_current_version == 1
    for e in edges:
        assert e._p2_committed_version == 1
        assert e._p2_participation_counts == {'edge_0': 1, 'edge_1': 1}
    assert ta.ledger.next_checkpoint_version() == 1
    # 第二轮继续正常推进
    _checkpoint_round(ta, cloud, users, edges, cfg, v=1, seq=1)
    assert cloud.p2_current_version == 2
    assert ta.ledger.next_checkpoint_version() == 2


def test_unregistered_token_rejected():
    """token 未向 TA 注册 (旁路封包) → 授权拒绝."""
    from point2_types import GroupPackage
    from utils import LHH
    ta, cloud, users, edges, cfg = _full_env()
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], "e0:v0:p0", 0, base_hash, 200,
                  np.full(DIM, 0.05))
    # p1 真实封包 (签名合法), 但模拟其 token 未注册 (直接从 ledger 撤销)
    p1 = make_pkg(e1, users[2:], "e1:v0:p0", 0, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]
    # 模拟 p1 的 token 未注册 (直接从 ledger 撤销注册)
    ta.ledger._conn.execute('DELETE FROM registered_tokens WHERE token = ?',
                            (p1.package_token,))
    ta.ledger._conn.commit()
    assert not ta.ledger.is_registered(p1.package_token)
    try:
        ta.p2_authorize_checkpoint(selected, votes, checkpoint_version=0,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('未注册 token 必须被拒绝')
    except ValueError as e:
        assert '未向 TA 注册' in str(e)


def test_replayed_token_rejected():
    """同一 token 第二次授权 (重放) → ledger 主键约束拒绝."""
    ta, cloud, users, edges, cfg = _full_env()
    _checkpoint_round(ta, cloud, users, edges, cfg, v=0)
    # 第二轮重用同一批 token (已消费)
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[1].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], "e0:v0:p0", 1, base_hash, 200,
                  np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:v0:p0", 1, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]
    try:
        ta.p2_authorize_checkpoint(selected, votes, checkpoint_version=1,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('已消费 token 必须被拒绝')
    except ValueError as e:
        assert '已消费' in str(e)


def test_schedule_constraints_rejected():
    """调度约束负例: normal |Q|≠K、timeout |Q|≥K、陈旧度>s_max、版本跳变."""
    from point2_types import GroupPackage
    from utils import LHH
    ta, cloud, users, edges, cfg = _full_env()
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], "e0:v0:p0", 0, base_hash, 200,
                  np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:v0:p0", 0, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]

    # normal 触发 |Q|=2=K ✓, 但 timeout 语义下 |Q|≥K → 拒
    try:
        ta.p2_authorize_checkpoint(selected, votes, checkpoint_version=0,
                                   trigger_mode='timeout', config=cfg)
        raise AssertionError('timeout 下 |Q|>=K 必须被拒绝')
    except ValueError as e:
        assert '超时触发' in str(e)

    # 版本跳变 (声明 v1, ledger 期望 v0) → 拒
    try:
        ta.p2_authorize_checkpoint(selected, votes, checkpoint_version=1,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('版本跳变必须被拒绝')
    except ValueError as e:
        assert '版本错位' in str(e)

    # 单包 (|Q|=1≠K) normal → 拒
    selected1 = [p0]
    M_Q1, H1 = ta.p2_build_checkpoint_mask(selected1, None, weights=None)
    raw1 = cloud.p2_recover_raw_sum(selected1, M_Q1, weights=None)
    _, final1, _ = cloud.p2_apply_vcm_and_average(
        selected1, raw1, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle1 = build_bundle(cloud, ta, selected1, raw1, M_Q1, H1, final1,
                           None, gamma=0.5, alphas=None,
                           weight_source='none')
    votes1 = [e.p2_vote_checkpoint(bundle1) for e in edges]
    try:
        ta.p2_authorize_checkpoint(selected1, votes1, checkpoint_version=0,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('normal 下 |Q|!=K 必须被拒绝')
    except ValueError as e:
        assert '常规触发' in str(e)


def test_vote_tampering_rejected():
    """vote 篡改 (改 passed/token digest/伪造签名) → 授权拒绝."""
    from point2_types import CheckpointVote
    ta, cloud, users, edges, cfg = _full_env()
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    e0, e1 = edges
    p0 = make_pkg(e0, users[:2], "e0:v0:p0", 0, base_hash, 200,
                  np.full(DIM, 0.05))
    p1 = make_pkg(e1, users[2:], "e1:v0:p0", 0, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]

    # 篡改 1: token digest 改动 (载荷变化 → 验签失败)
    bad = CheckpointVote(
        voter=votes[0].voter, checkpoint_version=votes[0].checkpoint_version,
        token_digest='0' * 64, config_digest=votes[0].config_digest,
        passed=True, result_digest=votes[0].result_digest,
        signature=votes[0].signature)
    try:
        ta.p2_authorize_checkpoint(selected, [bad, votes[1]],
                                   checkpoint_version=0,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('篡改 vote 必须被拒绝')
    except ValueError as e:
        assert '签名无效' in str(e) or 'token 摘要' in str(e)

    # 篡改 2: 无签名 vote
    unsigned = CheckpointVote(
        voter=votes[0].voter, checkpoint_version=votes[0].checkpoint_version,
        token_digest=votes[0].token_digest,
        config_digest=votes[0].config_digest, passed=True,
        result_digest=votes[0].result_digest)
    try:
        ta.p2_authorize_checkpoint(selected, [unsigned, votes[1]],
                                   checkpoint_version=0,
                                   trigger_mode='normal', config=cfg)
        raise AssertionError('未签名 vote 必须被拒绝')
    except ValueError as e:
        assert '签名无效' in str(e)


def test_cloud_restart_replay_cannot_advance_es():
    """Cloud 重启 + 重放旧 receipt: ES 版本严格 +1 校验拒绝推进."""
    ta, cloud, users, edges, cfg = _full_env()
    selected, receipt = _checkpoint_round(ta, cloud, users, edges, cfg, v=0)
    assert all(e._p2_committed_version == 1 for e in edges)

    # 模拟 Cloud 重启: 新 CloudServer 从 v0 重新开始 (本地状态丢失)
    from cloud import CloudServer
    cloud2 = CloudServer()
    cloud2.point2_reset(cloud.p2_global_model_history[0].copy(), 0)
    cloud2._gradient_dim = DIM
    cloud2.p2_set_receipt_pub_key(ta.get_receipt_public_key())
    # 重启后的 Cloud 重建本地缓存 (攻击者重放的包进入 buffer), 并用
    # 旧 receipt 重放提交 —— P0-1 结果摘要绑定使其在 commit 层即被拒绝
    # (重放者无法复现诚实候选的 raw sum / M_Q / final model 摘要)
    for p in selected:
        cloud2.p2_receive_group_package(p)
    import pytest as _pytest
    with _pytest.raises(ValueError) as ei:
        cloud2.p2_commit_checkpoint(
            selected, np.zeros(DIM), np.zeros(DIM, dtype=np.int64),
            np.zeros(DIM, dtype=np.int64), None, np.zeros(DIM), {},
            receipt=receipt)
    assert ('结果摘要' in str(ei.value)
            or 'receipt 版本' in str(ei.value)
            or 'token 集合' in str(ei.value))
    # 权威状态 (TA ledger) 未二次推进: token 已消费、版本停在 1;
    # 全局共识仍为 v1, ES 版本仍为 1
    assert ta.ledger.is_consumed(selected[0].package_token)
    assert ta.ledger.next_checkpoint_version() == 1
    assert cloud.p2_current_version == 1
    for e in edges:
        assert e._p2_committed_version == 1, '重放不得推进 ES 状态'


def test_es_rejects_tampered_receipt():
    """receipt 签名篡改 → ES 不推进版本/参与计数/策略状态."""
    from point2_types import CheckpointReceipt, tokens_digest
    ta, cloud, users, edges, cfg = _full_env()
    selected, receipt = _checkpoint_round(ta, cloud, users, edges, cfg, v=0)
    assert all(e._p2_committed_version == 1 for e in edges)
    forged = CheckpointReceipt(
        receipt_id=receipt.receipt_id, checkpoint_version=1,
        token_digest=tokens_digest([p.package_token for p in selected]),
        selected_tokens=receipt.selected_tokens, trigger_mode='normal',
        signature=b'\x00' * 64)
    e0 = edges[0]
    e0.p2_note_committed_checkpoint(
        selected, committed_version=2, final_model=np.zeros(DIM),
        receipt=forged)
    assert e0._p2_committed_version == 1, '坏 receipt 不得推进状态'
    assert e0._p2_participation_counts == {'edge_0': 1, 'edge_1': 1}


# ============================================================
#  P0-1 结果摘要绑定 / P0-2 prepare-finalize 回归
# ============================================================

def test_model_swap_after_receipt_rejected():
    """P0-1: 取得诚实 receipt 后替换 raw_sum/final_model → commit 拒绝."""
    import pytest as _pytest
    ta, cloud, users, edges, cfg = _full_env()
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    final = bundle['final_model']
    # 场景 A: 偷换 final_model
    swapped = final.copy()
    swapped[0] += 1.0
    with _pytest.raises(ValueError) as ei:
        cloud.p2_commit_checkpoint(
            selected, swapped, bundle['raw_sum_int'], bundle['M_Q'],
            bundle['H_mask_raw'], swapped * 500, {}, receipt=receipt)
    assert '结果摘要' in str(ei.value)
    # 场景 B: 偷换 raw sum
    tampered_raw = bundle['raw_sum_int'].copy()
    tampered_raw[0] += 1
    with _pytest.raises(ValueError) as ei:
        cloud.p2_commit_checkpoint(
            selected, final, tampered_raw, bundle['M_Q'],
            bundle['H_mask_raw'], final * 500, {}, receipt=receipt)
    assert '结果摘要' in str(ei.value)
    # 权威状态未被这次失败尝试污染
    assert not ta.ledger.is_consumed(selected[0].package_token)
    assert ta.ledger.next_checkpoint_version() == 0


def test_commit_failure_does_not_advance_ledger():
    """P0-2: prepare 授权不消费 token; 提交失败时 ledger 不推进."""
    import pytest as _pytest
    ta, cloud, users, edges, cfg = _full_env()
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    # prepare 之后: token 仍未消费、版本未推进 (与旧实现的关键差异)
    assert not ta.ledger.is_consumed(selected[0].package_token)
    assert ta.ledger.next_checkpoint_version() == 0
    assert not ta.p2_confirm_receipt(receipt.receipt_id)
    # 提交失败 (buffer 被清空 → commit 守卫拒绝)
    cloud.p2_package_buffer.clear()
    with _pytest.raises(ValueError):
        cloud.p2_commit_checkpoint(
            selected, bundle['final_model'], bundle['raw_sum_int'],
            bundle['M_Q'], bundle['H_mask_raw'], bundle['final_model'] * 500,
            {}, receipt=receipt)
    # ledger 权威状态仍不推进; ES 也不推进
    assert not ta.ledger.is_consumed(selected[0].package_token)
    assert ta.ledger.next_checkpoint_version() == 0
    assert all(e._p2_committed_version == 0 for e in edges)
    # 之后正常提交 + finalize 才推进
    for p in selected:
        cloud.p2_receive_group_package(p)
    cloud.p2_commit_checkpoint(
        selected, bundle['final_model'], bundle['raw_sum_int'], bundle['M_Q'],
        bundle['H_mask_raw'], bundle['final_model'] * 500, {}, receipt=receipt)
    ta.p2_finalize_checkpoint(receipt)
    assert ta.ledger.is_consumed(selected[0].package_token)
    assert ta.ledger.next_checkpoint_version() == 1
    assert ta.p2_confirm_receipt(receipt.receipt_id)


def test_double_finalize_idempotent_conflict_rejected():
    """相同 receipt 重复 finalize → 幂等成功; 同 id 不同内容 → 拒绝."""
    import pytest as _pytest
    from point2_types import canonical_digest
    ta, cloud, users, edges, cfg = _full_env()
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    assert ta.p2_finalize_checkpoint(receipt) == 'committed'
    # 幂等重放: 不再抛错, 不重复消费, 不推进版本
    assert ta.p2_finalize_checkpoint(receipt) == 'already'
    assert ta.ledger.next_checkpoint_version() == 1
    n_consumed = ta.ledger._conn.execute(
        'SELECT COUNT(*) FROM consumed_tokens').fetchone()[0]
    assert n_consumed == 2
    assert ta.p2_confirm_receipt(receipt)

    # 冲突: 同 receipt_id 但内容不同 (result_digest 被改) 且用 TA 私钥重签
    # —— 签名有效但摘要与 prepare 记录不符 → 拒绝 (不同于"同 receipt 幂等")
    payload = dict(receipt.payload())
    payload['result_digest'] = 'f' * 64
    conflicting = CheckpointReceipt(
        receipt_id=payload['receipt_id'],
        checkpoint_version=payload['checkpoint_version'],
        token_digest=payload['token_digest'],
        selected_tokens=payload['selected_tokens'],
        trigger_mode=payload['trigger_mode'],
        result_digest=payload['result_digest'],
        signature=SIG.sign(bytes.fromhex(canonical_digest(payload)),
                           ta._receipt_priv_key))
    with _pytest.raises(ValueError) as ei:
        ta.p2_finalize_checkpoint(conflicting)
    assert '不符' in str(ei.value)
    assert not ta.p2_confirm_receipt(conflicting)
    # 权威状态未被冲突 receipt 改动
    assert ta.ledger.next_checkpoint_version() == 1
    assert n_consumed == ta.ledger._conn.execute(
        'SELECT COUNT(*) FROM consumed_tokens').fetchone()[0]


def test_s_max_violation_rejected():
    """真实 s_max 负例: 组包陈旧度 v−t_k > s_max → 授权拒绝."""
    import pytest as _pytest
    from point2_types import ProtocolConfig
    cfg0 = ProtocolConfig(K=2, s_max=0, gamma_vcm=0.5)
    ta, cloud, users, edges = make_env(policy_config=cfg0, gamma_vcm=0.5)
    ta.p2_set_policy_config(cfg0)
    cloud.p2_set_receipt_pub_key(ta.get_receipt_public_key())
    # v=1, 包基于 v=0 → age=1 > s_max=0
    cloud.p2_global_model_history[1] = cloud.p2_global_model_history[0] + 0.2
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    for e in edges:
        e._p2_committed_version = 1
        e._p2_committed_history[1] = cloud.p2_global_model_history[1]
    cloud.p2_current_version = 1
    p0 = make_pkg(edges[0], users[:2], "e0:v0:p0", 0, base_hash, 200,
                  np.full(DIM, 0.05))
    p1 = make_pkg(edges[1], users[2:], "e1:v0:p0", 0, base_hash, 300,
                  np.full(DIM, 0.08))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]
    with _pytest.raises(ValueError) as ei:
        ta.p2_authorize_checkpoint(
            selected, votes, checkpoint_version=1, trigger_mode='normal',
            config=cfg0)
    assert '陈旧度' in str(ei.value)



def test_es_rejects_receipt_result_mismatch():
    """ES 推进校验: receipt 摘要/模型与本地已验证候选不符 → 不推进."""
    from point2_types import CheckpointReceipt, canonical_digest
    from utils import SIG
    ta, cloud, users, edges, cfg = _full_env()
    selected1, bundle1, votes1, receipt1 = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    final1 = bundle1['final_model']
    ta.p2_finalize_checkpoint(receipt1)

    # 分支 1: 摘要不符但签名有效 (模拟用另一候选的合法 receipt 冒充)
    payload = dict(receipt1.payload())
    payload['result_digest'] = 'f' * 64
    forged = CheckpointReceipt(
        receipt_id=payload['receipt_id'],
        checkpoint_version=payload['checkpoint_version'],
        token_digest=payload['token_digest'],
        selected_tokens=payload['selected_tokens'],
        trigger_mode=payload['trigger_mode'],
        result_digest=payload['result_digest'],
        signature=SIG.sign(bytes.fromhex(canonical_digest(payload)),
                           ta._receipt_priv_key))
    for e in edges:
        e.p2_note_committed_checkpoint(
            selected1, committed_version=1, final_model=final1,
            receipt=forged)
    assert all(e._p2_committed_version == 0 for e in edges),         '结果摘要不符不得推进'

    # 分支 2: receipt 合法但通知的 final_model 被替换 → 不推进
    swapped = final1.copy()
    swapped[0] += 1.0
    for e in edges:
        e.p2_note_committed_checkpoint(
            selected1, committed_version=1, final_model=swapped,
            receipt=receipt1)
    assert all(e._p2_committed_version == 0 for e in edges),         '模型被调包不得推进'

    # 分支 3: 合法 receipt + 一致模型 → 正常推进
    for e in edges:
        e.p2_note_committed_checkpoint(
            selected1, committed_version=1, final_model=final1,
            receipt=receipt1)
    assert all(e._p2_committed_version == 1 for e in edges)


# ============================================================
#  P0 finalize 闭环: 伪造拒绝 / prepare 替换 / 全量 confirm / 四阶段崩溃恢复
# ============================================================

def _unissued_receipt(ta, selected, raw, M_Q, final, v=0):
    """构造"签名合法但未经 prepare"的 receipt (模拟签名旁路/泄露)."""
    tdig = tokens_digest([p.package_token for p in selected])
    rdig = result_digest(v, tdig, raw, M_Q, final)
    r = CheckpointReceipt(
        receipt_id=f"{ta.id}:v{v}:{tdig[:16]}", checkpoint_version=v,
        token_digest=tdig,
        selected_tokens=sorted(p.package_token for p in selected),
        trigger_mode='normal', result_digest=rdig)
    return CheckpointReceipt(
        receipt_id=r.receipt_id, checkpoint_version=r.checkpoint_version,
        token_digest=r.token_digest, selected_tokens=r.selected_tokens,
        trigger_mode=r.trigger_mode, result_digest=r.result_digest,
        signature=SIG.sign(bytes.fromhex(r.payload_digest()),
                           ta._receipt_priv_key))


def _candidate(seq, ta, cloud, users, edges):
    """构造一个完整候选 (包 + 掩码 + 聚合 + bundle + votes), 不入 buffer."""
    base_hash = hashlib.sha256(
        cloud.p2_global_model_history[0].tobytes()).digest()
    p0 = make_pkg(edges[0], users[:2], f"e0:v0:p{seq}", 0, base_hash, 200,
                  np.full(DIM, 0.05 + 0.01 * seq))
    p1 = make_pkg(edges[1], users[2:], f"e1:v0:p{seq}", 0, base_hash, 300,
                  np.full(DIM, 0.08 + 0.01 * seq))
    selected = [p0, p1]
    M_Q, H = ta.p2_build_checkpoint_mask(selected, None, weights=None)
    raw = cloud.p2_recover_raw_sum(selected, M_Q, weights=None)
    _, final, _ = cloud.p2_apply_vcm_and_average(
        selected, raw, gamma_vcm=0.5, vcm_mode='full', alphas=None)
    bundle = build_bundle(cloud, ta, selected, raw, M_Q, H, final, None,
                          gamma=0.5, alphas=None, weight_source='none')
    votes = [e.p2_vote_checkpoint(bundle) for e in edges]
    return selected, bundle, votes


def test_forged_finalize_rejected():
    """P0-1: 伪造/未经 prepare 的 receipt 不能写 TA ledger."""
    ta, cloud, users, edges, cfg = _full_env()
    selected, bundle, votes = _candidate(0, ta, cloud, users, edges)

    # (a) 签名有效但未经 prepare (TA 私钥旁路签发): 无 prepare 记录 → 拒绝
    unissued = _unissued_receipt(ta, selected, bundle['raw_sum_int'],
                                 bundle['M_Q'], bundle['final_model'])
    with pytest.raises(ValueError) as ei:
        ta.p2_finalize_checkpoint(unissued)
    assert '无 prepare 记录' in str(ei.value)
    assert not ta.p2_confirm_receipt(unissued)
    assert ta.ledger.next_checkpoint_version() == 0
    assert not ta.ledger.is_consumed(selected[0].package_token)

    # (b) 签名无效 (伪造签名): 直接拒绝
    forged = CheckpointReceipt(
        receipt_id=unissued.receipt_id, checkpoint_version=0,
        token_digest=unissued.token_digest,
        selected_tokens=unissued.selected_tokens, trigger_mode='normal',
        result_digest=unissued.result_digest, signature=b'\x00' * 64)
    with pytest.raises(ValueError) as ei:
        ta.p2_finalize_checkpoint(forged)
    assert '签名无效' in str(ei.value)
    with pytest.raises(ValueError):
        ta.p2_finalize_checkpoint(object())
    assert ta.ledger.next_checkpoint_version() == 0
    assert not ta.ledger.is_consumed(selected[0].package_token)


def test_unissued_receipt_transaction_no_divergence(tmp_path):
    """P0-2 核心: finalize 被拒时 Cloud/TA/ES 均不推进 (旧分叉不复现).

    stage 只做只读校验: 该 receipt 的签名/版本/token/结果摘要都合法, 因此
    stage 通过; 但 TA ledger 无 prepare 记录 → finalize 拒绝 → 事务丢弃
    stage, Cloud 版本/buffer 与 TA/ES 都不变。
    """
    ta, cloud, users, edges, cfg = _full_env(
        stage_path=str(tmp_path / 'stage.journal'))
    selected, bundle, votes = _candidate(0, ta, cloud, users, edges)
    unissued = _unissued_receipt(ta, selected, bundle['raw_sum_int'],
                                 bundle['M_Q'], bundle['final_model'])
    for p in selected:
        cloud.p2_receive_group_package(p)
    with pytest.raises(ValueError) as ei:
        run_commit_transaction(
            cloud, ta, edges, selected_packages=selected,
            final_model=bundle['final_model'], raw_sum_int=bundle['raw_sum_int'],
            M_Q=bundle['M_Q'], H_mask_raw=bundle['H_mask_raw'],
            corrected_sum=bundle['final_model'] * 500, comp_map={},
            receipt=unissued)
    assert '无 prepare 记录' in str(ei.value)
    # 三方一致停在 v0: 无 "Cloud 已提交 / TA 未落地" 分叉
    assert cloud.p2_current_version == 0
    assert len(cloud.p2_package_buffer) == 2, 'buffer 未被消费'
    assert cloud.p2_staged_checkpoint() is None, 'stage 已丢弃'
    assert not os.path.exists(str(tmp_path / 'stage.journal'))
    assert ta.ledger.next_checkpoint_version() == 0
    assert not ta.ledger.is_consumed(selected[0].package_token)
    assert all(e._p2_committed_version == 0 for e in edges)


def test_prepare_replacement_supersedes_receipt():
    """同版本换候选: prepare 记录被新候选替换, 旧 receipt 作废."""
    ta, cloud, users, edges, cfg = _full_env()
    sel_a, bundle_a, votes_a = _candidate(0, ta, cloud, users, edges)
    receipt_a = ta.p2_authorize_checkpoint(
        sel_a, votes_a, checkpoint_version=0, trigger_mode='normal', config=cfg)
    sel_b, bundle_b, votes_b = _candidate(1, ta, cloud, users, edges)
    receipt_b = ta.p2_authorize_checkpoint(
        sel_b, votes_b, checkpoint_version=0, trigger_mode='normal', config=cfg)
    assert receipt_a.receipt_id != receipt_b.receipt_id

    # 旧 receipt 已作废: 无对应 prepare 记录 → 拒绝, 且不消费其 token
    with pytest.raises(ValueError) as ei:
        ta.p2_finalize_checkpoint(receipt_a)
    assert '作废' in str(ei.value) or '无 prepare 记录' in str(ei.value)
    assert not ta.ledger.is_consumed(sel_a[0].package_token)
    assert not ta.p2_confirm_receipt(receipt_a)

    # 当前候选正常落地
    assert ta.p2_finalize_checkpoint(receipt_b) == 'committed'
    assert ta.ledger.next_checkpoint_version() == 1
    assert ta.ledger.is_consumed(sel_b[0].package_token)
    assert not ta.ledger.is_consumed(sel_a[0].package_token)
    assert ta.p2_confirm_receipt(receipt_b)
    # 已 finalize 的版本不可再 prepare (不可回退)
    with pytest.raises(ValueError):
        ta.p2_authorize_checkpoint(sel_a, votes_a, checkpoint_version=0,
                                   trigger_mode='normal', config=cfg)


def test_confirm_receipt_requires_full_match():
    """confirm_receipt 核对完整 receipt (签名 + prepare 记录 + 已 finalize)."""
    ta, cloud, users, edges, cfg = _full_env()
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    # 未 finalize → 即使 prepare 记录存在也不得确认
    assert not ta.p2_confirm_receipt(receipt)
    ta.p2_finalize_checkpoint(receipt)
    assert ta.p2_confirm_receipt(receipt)
    assert ta.p2_confirm_receipt(receipt.receipt_id)   # 兼容: 仅 id 查询
    # 同 id 不同内容 (对篡改后的载荷重新签名 → 签名有效) → 摘要与 prepare
    # 记录不符, 不得确认 (只查 receipt_id 的实现会误判为真)
    from point2_types import canonical_digest
    payload = dict(receipt.payload())
    payload['result_digest'] = 'a' * 64
    tampered = CheckpointReceipt(
        receipt_id=payload['receipt_id'],
        checkpoint_version=payload['checkpoint_version'],
        token_digest=payload['token_digest'],
        selected_tokens=payload['selected_tokens'],
        trigger_mode=payload['trigger_mode'],
        result_digest=payload['result_digest'],
        signature=SIG.sign(bytes.fromhex(canonical_digest(payload)),
                           ta._receipt_priv_key))
    assert not ta.p2_confirm_receipt(tampered)
    with pytest.raises(ValueError) as ei:
        ta.p2_finalize_checkpoint(tampered)
    assert '不符' in str(ei.value)
    # 未知 id / 非 receipt 对象
    assert not ta.p2_confirm_receipt('TA-01:v9:' + '0' * 16)
    assert not ta.p2_confirm_receipt(None)


@pytest.mark.parametrize('phase', ['stage', 'finalize', 'publish', 'notify',
                                   'notify_first'])
def test_four_phase_crash_recovery(tmp_path, phase):
    """四阶段崩溃恢复: 任一阶段边界崩溃后重放收敛, 且不重复应用变更.

    stage / finalize / publish / notify 各阶段完成后崩溃 (notify_first =
    通知首个 ES 后崩溃, 复现 "Cloud v1 / TA v1 / ES v0" 分叉), 恢复必须:
    版本三方一致、token 恰好消费一次、参与历史不重复计入、日志单条、
    journal 清理、重复恢复为 no-op。
    """
    journal = str(tmp_path / f'{phase}.journal')
    ta, cloud, users, edges, cfg = _full_env(stage_path=journal)
    selected, bundle, votes, receipt = _prepare_round(
        ta, cloud, users, edges, cfg, v=0)
    final = bundle['final_model']
    kwargs = dict(selected_packages=selected, final_model=final,
                  raw_sum_int=bundle['raw_sum_int'], M_Q=bundle['M_Q'],
                  H_mask_raw=bundle['H_mask_raw'], corrected_sum=final * 500,
                  comp_map={}, receipt=receipt)
    with pytest.raises(CrashInjected):
        run_commit_transaction(cloud, ta, edges, crash_after=phase, **kwargs)

    # 崩溃点状态: 只允许"未推进"或"部分推进", 不得是静默分叉的终态
    if phase == 'stage':
        assert cloud.p2_current_version == 0
        assert ta.ledger.next_checkpoint_version() == 0
        assert not ta.ledger.is_consumed(selected[0].package_token)
    elif phase == 'finalize':
        assert cloud.p2_current_version == 0, 'finalize 后 publish 尚未执行'
        assert ta.ledger.next_checkpoint_version() == 1
        assert all(e._p2_committed_version == 0 for e in edges)
    elif phase == 'publish':
        assert cloud.p2_current_version == 1
        assert all(e._p2_committed_version == 0 for e in edges), 'ES 尚未通知'
    elif phase == 'notify_first':
        assert cloud.p2_current_version == 1
        assert edges[0]._p2_committed_version == 1
        assert edges[1]._p2_committed_version == 0, '复现 Cloud v1/ES v0 分叉'

    # 恢复: 幂等重放收敛
    result = recover_staged_commit(cloud, ta, edges)
    assert result is not None and result.version == 0
    assert cloud.p2_current_version == 1
    assert ta.ledger.next_checkpoint_version() == 1
    assert all(e._p2_committed_version == 1 for e in edges)
    assert all(e._p2_participation_counts == {'edge_0': 1, 'edge_1': 1}
               for e in edges), '恢复不得重复计入参与历史'
    assert len(cloud.p2_checkpoint_log) == 1
    assert ta.ledger._conn.execute(
        'SELECT COUNT(*) FROM consumed_tokens').fetchone()[0] == 2
    assert not os.path.exists(journal), '恢复完成后 journal 应清理'
    assert _no_staged_state(cloud)
    # 再恢复一次是 no-op
    assert recover_staged_commit(cloud, ta, edges) is None


def _no_staged_state(cloud) -> bool:
    return (cloud.p2_staged_checkpoint() is None
            and cloud.p2_load_staged_journal() is None)


# ============================================================
#  持久化: prepared_checkpoints 与 TA receipt 密钥跨进程续用
# ============================================================

def test_prepared_records_persist_across_reopen(tmp_path):
    """prepared_checkpoints / consumed_tokens 落在持久化 DB, 重开后仍可核对."""
    from ta import TALedger
    db = str(tmp_path / 'ta_ledger.sqlite')
    ledger = TALedger(db)
    ledger.register_token('t0', 'edge_0', 0)
    ledger.register_token('t1', 'edge_1', 0)
    blob = b'receipt-bytes'
    ledger.prepare_checkpoint(
        version=0, receipt_id='rid0', receipt_digest='d' * 64,
        receipt_pickle=blob, token_digest='t' * 64, trigger_mode='normal',
        result_digest='r' * 64, tokens=['t0', 't1'],
        token_meta=[['t0', 'edge_0', 0], ['t1', 'edge_1', 0]])

    # 重开 (模拟进程重启): prepare 记录在, 且未被标记 finalize
    ledger2 = TALedger(db)
    row = ledger2.get_prepared('rid0')
    assert row is not None and row['receipt_digest'] == 'd' * 64
    assert row['receipt_pickle'] == blob
    assert row['token_meta'] == [['t0', 'edge_0', 0], ['t1', 'edge_1', 0]]
    assert not row['finalized']

    assert ledger2.commit_prepared('rid0') == 'committed'
    ledger3 = TALedger(db)
    row3 = ledger3.get_prepared('rid0')
    assert row3['finalized']
    assert ledger3.next_checkpoint_version() == 1
    assert ledger3.is_consumed('t0') and ledger3.is_consumed('t1')
    # token 消费记录带真实 edge/base_version 元数据
    meta = ledger3._conn.execute(
        'SELECT edge_id, base_version FROM consumed_tokens WHERE token = ?',
        ('t1',)).fetchone()
    assert tuple(meta) == ('edge_1', 0)
    # 重复 finalize 幂等
    assert ledger3.commit_prepared('rid0') == 'already'


def test_ta_receipt_keys_persist_with_ledger(tmp_path):
    """--ta-state-path 下 TA 密钥 load-or-create: 重启后同一公钥、同一 ledger."""
    from ta import TrustedAuthority
    db = str(tmp_path / 'ta.sqlite')
    ta1 = TrustedAuthority(gradient_dim=8, ta_id='TA-P', ledger_path=db)
    ta1.point2_reset()
    pub1 = ta1.get_receipt_public_key()
    ta1.ledger.register_token('t0', 'edge_0', 0)
    ta1.ledger.prepare_checkpoint(
        version=0, receipt_id='rid0', receipt_digest='d' * 64,
        receipt_pickle=b'x', token_digest='t' * 64, trigger_mode='normal',
        result_digest='r' * 64, tokens=['t0'],
        token_meta=[['t0', 'edge_0', 0]])

    # 同一 TA 状态路径重启 → 同一签名密钥 + 可见的 prepare 记录
    ta2 = TrustedAuthority(gradient_dim=8, ta_id='TA-P', ledger_path=db)
    assert ta2.get_receipt_public_key() == pub1
    assert ta2.ledger.get_prepared('rid0') is not None
    # 旧 TA 签发的载荷仍可由新实例验签
    sig = SIG.sign(b'payload', ta1._receipt_priv_key)
    assert SIG.verify(b'payload', sig, ta2.get_receipt_public_key())
