# 历史探索与辅助代码

这些脚本保留原路径供追溯。正式论文比较应按[研究点一](../point1/README.md)或[研究点二](../point2/README.md)的入口执行。

| 类别 | 文件/目录 | 阅读提醒 |
| --- | --- | --- |
| 早期开销基准 | [`baselines/`](../../baselines/)、`examples/exp_overhead_crypto.py` | 与正式`point1_compare/adapters/`的参数、计时、消息及密钥生命周期可能不同 |
| Fig.6风格正确性展示 | [`experiments/fig6_correctness_20261013/README.md`](../../experiments/fig6_correctness_20261013/README.md) | 80训练轮中只在2个采样轮运行完整安全链路；不能表述为80轮完整协议验证 |
| 早期同步/异步展示 | [`examples/`](../../examples/)、`main.py` | 单文件实验和作图脚本保留；部分脚本有历史固定路径 |
| Agent调度候选 | `agent_async.py`、`agent_async_secure.py`、`rule_engine.py`、`orchestrate_agent.py`、`llm_reasoner.py` | 历史调度/推理探索；不属于新checkpoint矩校准实现。可选LLM依赖未并入核心运行环境 |
| 早期复用候选 | `run_reuse_perm.py`、`run_reuse_temp.py`、`run_reuse_temp2.py`、`_diag_reuse_direction.py` | 参数与判断必须回到各自诊断，不与正式多轮数据拼接 |
| 历史边界探针 | `experiments/point1_compare/probe_*.py` | 用于接口和代数审查，不是完整竞争方案 |
| 旧自动关机辅助 | `experiments/point1_compare/finish_and_shutdown.py` | 原历史工具；分类/检查/审阅命令均不会调用它 |

`examples/exp_agent_overhead.py`的本地原版本含硬编码API凭据，未纳入审阅清单。`main`保留该历史脚本，发布版本已改为读取`HVSA_LLM_API_KEY`环境变量；本地原文件保持。原密钥仍存在于旧提交历史，需撤销更换。目录清单不包含密钥文件或运行时私钥状态。

数据集、原始日志、模型、PPT/Word、参考论文全文、临时依赖和进程状态不属于本次代码审阅快照。原工作目录均保留，未删除或搬迁。
