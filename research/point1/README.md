# 研究点一：分层可验证安全聚合

研究对象是客户端、边缘、云与可信TA协作的完整安全聚合链路。先读[HVSA真实实体适配器](../../experiments/point1_compare/adapters/hvsa_point1.py)，再沿其调用进入四角色代码；跨方案参数与边界以[决策稿](../../docs/POINT1_BASELINE_DECISIONS_20261019.md)为准。

## 按实验分类

表中的脚本名均位于`experiments/point1_compare/`；完整路径与测试清单见[目录清单](../catalog.json)。

| 编号/目的 | 实验入口 | 配置或设计 | 核验、统计与主要测试 |
| --- | --- | --- | --- |
| E1 聚合正确性 | [`run_compare.py`](../../experiments/point1_compare/run_compare.py)（`e1`） | [`configs/mvp.json`](../../experiments/point1_compare/configs/mvp.json)；含零向量、负数、边界值与完整维度回放 | `validate_results.py`；`tests/test_point1_compare.py` |
| E2 HVSA计算/通信规模 | [`run_performance.py`](../../experiments/point1_compare/run_performance.py)（`e2`） | [`configs/full.json`](../../experiments/point1_compare/configs/full.json)，人数/维度/边缘数扫描 | `report_cost_matrix.py`、`validate_results.py`；`test_point1_performance.py` |
| E2 跨方案正式实例化成本 | [`run_decided_comparison.py`](../../experiments/point1_compare/run_decided_comparison.py)、`run_decided_batch.py` | [`baseline_decisions_20261019.json`](../../experiments/point1_compare/configs/baseline_decisions_20261019.json) | `report_decided_matrix.py`；`test_point1_decided_artifacts.py`、`test_point1_decided_batch.py` |
| E2 完整跨方案N/d/M扫描 | [`run_full_scale_point.py`](../../experiments/point1_compare/run_full_scale_point.py)、`run_full_scale_batch.py` | [`full_scale_contract.py`](../../experiments/point1_compare/full_scale_contract.py)、[冻结规格](../../docs/POINT1_FULL_SCALE_SCAN_STAGE_20261026.md)；10几何点/28格，每格3预热+10正式 | `validate_full_scale.py`、`report_full_scale.py`；`test_point1_full_scale.py`。快照时最后一点仍运行，不标作全量验收完成 |
| E3 多轮及初始化摊销 | [`run_performance.py`](../../experiments/point1_compare/run_performance.py)（`e3`）、[`run_large_multirun.py`](../../experiments/point1_compare/run_large_multirun.py) | `large_multirun_contract.py`；固定/变化在线集合 | `validate_large_multirun.py`、`report_large_multirun.py`；`test_point1_large_multirun.py` |
| E3 HVSA同人数对照 | [`run_hvsa_paired40.py`](../../experiments/point1_compare/run_hvsa_paired40.py) | 固定40人/变化40人，各50轮 | `report_hvsa_paired40.py`；`test_point1_paired40.py` |
| E3/E4 小规模与分段接续 | `run_decided_trajectories.py`、`continue_decided_trajectories.py` | `trajectory_segment_contract.py`；分段状态不能伪装为连续单密钥状态 | `validate_decided_trajectories.py`、`validate_trajectory_segments.py`；`report_trajectory_segments.py` |
| E4 随机/分阶段掉线 | [`run_performance.py`](../../experiments/point1_compare/run_performance.py)（`e4`）、[`run_large_dropout.py`](../../experiments/point1_compare/run_large_dropout.py) | `large_dropout_contract.py`；区分上传前与上传后离线 | `validate_large_dropout.py`、`report_large_dropout.py`；`test_point1_large_dropout.py` |
| E4 定向边界 | [`run_directed_dropout.py`](../../experiments/point1_compare/run_directed_dropout.py) | [`directed_dropout_contract.py`](../../experiments/point1_compare/directed_dropout_contract.py)；空组、阈值、相邻/分散缺席 | `validate_directed_dropout.py`、`report_directed_dropout.py`；`test_point1_directed_dropout.py` |
| E5 篡改与上下文绑定 | [`run_integrity.py`](../../experiments/point1_compare/run_integrity.py)、`run_compare.py`（`e5`）、[`run_decided_integrity.py`](../../experiments/point1_compare/run_decided_integrity.py) | 诚实对照、重放、证据/输入域及生产参数异常 | `validate_decided_integrity.py`、`report_decided_integrity.py`；`test_point1_boundary.py`、`test_point1_decided_integrity.py` |
| E6 掩码/验证消融 | [`run_performance.py`](../../experiments/point1_compare/run_performance.py)（`e6`） | 单掩码、三路掩码控制、关闭验证；这些是消融臂，不是GPVO复现 | `report_ablation.py`；`test_point1_derivation.py`、`test_point1_measurement.py` |
| E7 MNIST真实训练 | [`run_training.py`](../../experiments/point1_compare/run_training.py) | Linear(784,10)，原始整数和与同编码明文参考逐轮核对 | `validate_results.py`；`test_point1_artifacts.py` |
| E7 独立明文训练对照 | [`run_training_controls.py`](../../experiments/point1_compare/run_training_controls.py) | `training_control_contract.py`；同编码对照/独立浮点FedAvg | `validate_training_controls.py`、`report_training_controls.py`；`test_point1_training_controls.py` |
| E8 功能/可信假设 | [功能与信任对照](../../docs/POINT1_FUNCTION_TRUST_COMPARISON_20261018.md) | [实现审查](../../docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md) | 文档审查项，不伪装成训练或性能实验 |

## 方案实现

- **HVSA**：[`adapters/hvsa_point1.py`](../../experiments/point1_compare/adapters/hvsa_point1.py)，调用真实`User/EdgeServer/CloudServer/TrustedAuthority`，包含完整验证与消息计量。
- **GPVO正式实例化**：[`adapters/gpvo_decided.py`](../../experiments/point1_compare/adapters/gpvo_decided.py)，使用`gpvo_full.py`的群/掩码组件，按已说明的整数群与每轮新会话选择执行。
- **VLMRS正式实例化**：[`adapters/vlmrs_fullgraph.py`](../../experiments/point1_compare/adapters/vlmrs_fullgraph.py)、`vlmrs_verification.py`；平面全邻居协议，边缘数M不适用。
- **接口/代数探索**：`probe_*.py`、`ovp_full.py`、`saev_full.py`；属于接口与机制核查，不自动晋级为主表比较。
- 根目录[`baselines/`](../../baselines/)是较早开销基准；正式点一比较优先阅读本节的适配器。

## 原始数据与阅读边界

新实验默认落在`results/point1_compare/<run_id>/`。典型归档有`manifest.json`、`raw.jsonl`、`messages.csv`、`correctness.json`、`sources/`，长实验另有`inputs/`、`shards/`、`journal.json`。原始归档没有随代码审阅分支上传。

正式扫描的原始网格、输入、预热及验证者范围由contract约束；脚本名存在不能代替实跑结果。文档中的历史中断前缀单独保留，不进入完整扫描的主分母。新机器复现与原历史归档核验的区别见[复现说明](../REPRODUCING.md)。
