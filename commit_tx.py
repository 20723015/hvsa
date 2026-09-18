"""Point2 提交事务 (P0-2): stage → TA finalize → publish → notify.

四阶段与故障语义:
1. stage    —— Cloud 只读校验候选 (receipt 签名/版本/token/结果摘要绑定,
               buffer 原包仍在) 并把"提交成功要应用的变更"暂存 (可写
               journal), 不改任何权威状态;
2. finalize —— TA 核对 prepare 记录后, 把版本记录与全部 token 消费原子
               写入权威 ledger (幂等: 相同 receipt 重复 finalize 返回
               already, 不重复消费);
3. publish  —— Cloud 应用暂存变更 (buffer 消费 / 版本推进 / 模型历史),
               幂等: 版本已推进时直接返回已记录结果;
4. notify   —— 通知 TA 与各 ES 推进共识状态 (幂等: 通知版本 ≤ 已落地
               版本即跳过, 不重复计入参与历史)。

关键性质:
- finalize 被协议性拒绝 (ValueError) → 丢弃 stage 并抛出; Cloud 未应用
  任何变更, Cloud/TA/ES 三者不分叉 (旧实现的 P0: Cloud 已提交而 TA 未
  落地);
- finalize 成功后的任一点崩溃 → journal + 幂等阶段支持重放完成
  (Cloud v+1 / TA v+1 / ES v0 这类分叉由 `recover_staged_commit` 收敛);
- 瞬时故障 (非协议性异常, 如数据库抖动) 在 `max_attempts` 内就地重试,
  每阶段幂等因此重放安全。
"""

import logging

logger = logging.getLogger(__name__)


class CrashInjected(Exception):
    """测试用故障注入: 在指定阶段边界模拟进程崩溃 (不触发就地重试)."""


def _maybe_crash(crash_after, phase: str) -> None:
    if crash_after == phase:
        raise CrashInjected(f"模拟崩溃: {phase} 阶段完成后")


def run_commit_transaction(cloud, ta, edges, *,
                           selected_packages=None,
                           final_model=None, raw_sum_int=None, M_Q=None,
                           H_mask_raw=None, corrected_sum=None, comp_map=None,
                           receipt=None, stage=None,
                           crash_after=None, max_attempts=3):
    """执行四阶段提交事务, 返回 CheckpointResult.

    Args:
        cloud, ta, edges: 协议参与者。
        selected_packages, final_model, raw_sum_int, M_Q, H_mask_raw,
            corrected_sum, comp_map, receipt: stage 阶段输入 (提供了 stage
            时可省略)。
        stage: 复用已暂存候选 (崩溃恢复路径; 跳过 stage 阶段)。
        crash_after: 测试注入点, ∈ {'stage', 'finalize', 'publish',
            'notify', 'notify_first'}; 'notify_first' = 通知首个 ES 后崩溃
            (复现 ES 版本落后于 Cloud/TA 的分叉)。
        max_attempts: 瞬时故障最大尝试次数。

    Raises:
        ValueError: 协议性拒绝 (stage 校验失败 / TA finalize 核对失败);
            Cloud 权威状态未推进。
    """
    if stage is None:
        if selected_packages is None:
            raise ValueError("run_commit_transaction: 缺少 stage 候选参数")
        stage = cloud.p2_stage_checkpoint(
            selected_packages, final_model, raw_sum_int, M_Q, H_mask_raw,
            corrected_sum, comp_map, receipt=receipt)
        _maybe_crash(crash_after, 'stage')

    phase = 'finalize'
    result = None
    attempt = 0
    while True:
        try:
            if phase == 'finalize':
                if stage.receipt is not None:
                    ta.p2_finalize_checkpoint(stage.receipt)
                _maybe_crash(crash_after, 'finalize')
                phase = 'publish'

            if phase == 'publish':
                result = cloud.p2_publish_staged(stage)
                _maybe_crash(crash_after, 'publish')
                phase = 'notify'

            if phase == 'notify':
                committed_v = result.version + 1
                ta.p2_note_committed_checkpoint(
                    stage.selected_packages, committed_version=committed_v)
                for i, e in enumerate(edges):
                    e.p2_note_committed_checkpoint(
                        stage.selected_packages, committed_version=committed_v,
                        final_model=stage.final_model, receipt=stage.receipt)
                    if crash_after == 'notify_first' and i == 0:
                        raise CrashInjected("模拟崩溃: 通知首个 ES 后")
                _maybe_crash(crash_after, 'notify')
                cloud.p2_complete_staged(stage)
                return result

        except CrashInjected:
            raise
        except ValueError:
            # 协议性拒绝只可能来自 stage 校验或 TA finalize 核对: 此时
            # Cloud 未应用变更, 权威状态未推进 → 丢弃 stage (fail-closed)
            if phase == 'finalize':
                cloud.p2_discard_staged(stage)
            raise
        except Exception as e:      # 瞬时故障: 幂等重放当前阶段
            attempt += 1
            if attempt >= max_attempts:
                logger.error(f"commit 事务在 {phase} 阶段连续失败 "
                             f"{attempt} 次, 放弃: {type(e).__name__}: {e}")
                raise
            logger.warning(f"commit 事务 {phase} 阶段瞬时故障 "
                           f"({type(e).__name__}: {e}); 第 {attempt} 次重试 "
                           f"(阶段幂等, 重放安全)")


def recover_staged_commit(cloud, ta, edges, *, crash_after=None,
                          max_attempts=3):
    """崩溃恢复: 完成 (或安全放弃) 上次未完成的提交事务.

    Returns:
        CheckpointResult  —— 已恢复完成;
        None              —— 无待恢复候选, 或 TA 核对失败已安全放弃
                             (Cloud 未推进, 无分叉)。
    """
    stage = cloud.p2_staged_checkpoint() or cloud.p2_load_staged_journal()
    if stage is None:
        return None
    logger.warning(f"检测到未完成的 staged checkpoint {stage.stage_id} "
                   f"(v{stage.base_version}), 尝试恢复完成")
    try:
        return run_commit_transaction(cloud, ta, edges, stage=stage,
                                      crash_after=crash_after,
                                      max_attempts=max_attempts)
    except ValueError as e:
        logger.error(f"恢复失败 ({e}); 丢弃 staged checkpoint "
                     f"(Cloud 未推进, 无分叉)")
        cloud.p2_discard_staged(stage, reason=str(e))
        return None
