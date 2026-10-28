# 研究点二：半异步checkpoint与版本补偿

本页把已实现协议、已有实验和最新设计分开。现有代码里存在多种VCM/moment候选，不能仅凭参数名称就认定已经实现10月12日的新方案。

## 协议主链

1. [`main_point2.py`](../../main_point2.py)：事件调度、本地训练、组包buffer、checkpoint触发与结果输出。
2. [`user.py`](../../user.py)：训练、量化、掩码上传；[`cloud.py`](../../cloud.py)：缓存组包、冻结选中集合、构造候选。
3. [`ta.py`](../../ta.py)：授权、总掩码、token消费和提交记录；[`edge.py`](../../edge.py)：独立核验输入、策略及最终模型，签署投票。
4. [`commit_tx.py`](../../commit_tx.py)、[`point2_types.py`](../../point2_types.py)：候选、投票、收据、提交与恢复。

## 按实验分类

| 分类 | 入口/实现 | 配套分析/测试 | 状态与作用 |
| --- | --- | --- | --- |
| 半异步主矩阵与同步参考 | [`run_thesis_core.py`](../../run_thesis_core.py)、`main_point2.py`、`fedavg_point2.py` | `thesis_stats.py`、`thesis_tables.py`、`tests/test_thesis_pipeline.py` | 现有主矩阵；臂与规模见[原流水线](../../docs/THESIS_CORE_PIPELINE.md) |
| 版本对齐/陈旧复用对照 | [`run_point_matrix.py`](../../run_point_matrix.py)、`run_reuse_stages.py`、`run_reuse_viability.py` | `analyze_point_matrix.py`、`analyze_reuse.py`、`analyze_reuse_viability.py` | 历史VCM/复用比较，不是新checkpoint矩校准的效果证据 |
| 明文竞争基线 | [`plain_bench.py`](../../plain_bench.py)、`run_plain_bench.py`、`plain_cache_bench.py` | `plain_bench_stats.py`、`tests/test_plain_bench.py`、[适配说明](../../docs/PLAIN_BENCH_ADAPTATIONS.md) | FedBuff/MAFSA/DVAFL的本地适配；阅读拓扑、权重和通信口径差异 |
| 策略与结果完整性攻击 | [`run_attack_evidence.py`](../../run_attack_evidence.py)、`run_seed_attack_chunked.py`、`run_attack_v2.py` | `regress_attack_fix.py`、`analyze_security.py`、`tests/test_policy_binding.py`、`test_receipt_flow.py` | 结果篡改、manifest重放、权重/补偿参数与历史缺失等；适用单元与诚实对照分开计量 |
| 提交与状态恢复 | [`commit_tx.py`](../../commit_tx.py)、`ta.py`、`edge.py` | `tests/test_receipt_flow.py`、`tests/test_ra_cma_chain.py` | 现有提交收据链路；新矩状态仍需进一步接入 |
| 机制消融/敏感性 | [`run_all_ablations.py`](../../run_all_ablations.py)、`run_tier1_batch.py`、`thesis_ablation_extra.py` | `analyze_tier1.py`、`analyze_tier2.py` | K、陈旧度上界、补偿强度、调度等旧机制实验 |
| V-PDA/URVCM/CMA/Aσ候选 | `run_vpda_stages.py`、`run_urvcm_stages.py`、`run_cma_stages.py`、`run_asig_verify.py` | `analyze_vpda.py`、`analyze_urvcm.py`、`analyze_cma.py`、`tests/test_vpda.py`、`test_ra_cma_chain.py` | 历史候选与诊断，各自按原设计解释 |
| 缓存差分校准开发筛查 | [`cache_calibration.py`](../../cache_calibration.py)、`run_cache_screen.py` | `analyze_cache_screen.py`、`tests/test_cache_calibration.py` | 开发配对筛查；没有确认稳定收益，不列为已验证主贡献 |
| 时间/内存/通信开销 | [`exp_security_overhead.py`](../../exp_security_overhead.py)、`exp_memory_overhead.py`、`exp_comm_per_role.py`、`examples/exp_n4_phase_breakdown.py` | [开销说明](../../docs/OVERHEAD_EXPERIMENT.md) | 区分微基准、实际维度扫描及旧外推结果 |
| 隐私攻击与分布展示 | [`exp_n2_dlg_inversion.py`](../../exp_n2_dlg_inversion.py)、[`exp_n7_masked_distribution.py`](../../exp_n7_masked_distribution.py) | `run_attack_v2.py`与对应原始产物 | DLG重建、掩码分布是辅助证据，不能代替密码安全证明 |

## 最新方案：checkpoint更新矩校准

- [正式设计稿](../../docs/POINT2_CHECKPOINT_MOMENT_DESIGN_20261012.md)
- [机器可读设计参数](../../docs/point2_checkpoint_moment_v0_20261012.json)
- [数学检查记录](../../docs/point2_checkpoint_moment_mathcheck_20261012.json)

核心形式为`Δ = R − A`、`W_next = R + ρ (W − A)`：R是已验证的checkpoint聚合均值，A是所选基模型加权均值；边缘使用整体增量与历史位移确定补偿量，并限制额外残差。客户端不新增统计任务，TA继续只释放checkpoint总掩码，模型/历史状态/计数联合提交。

**当前文件明确记为设计阶段**：`protocol_integration_complete=false`、`training_evaluated=false`。已有数学检查不等于完整安全链路或训练效果通过。旧`--vcm`、`--momentum`、明文DVAFL或`ema-decay`结果不能直接填入这个新方案的实验表。

已有输出通常在`results_thesis_core/`、`results_thesis_core_attack/`、`results_plain_bench/`等独立目录，未随代码审阅分支上传。
