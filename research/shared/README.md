# 公共协议与学习组件

| 模块 | 文件 | 负责内容 |
| --- | --- | --- |
| 客户端 | [`user.py`](../../user.py) | 本地训练、量化、掩码上传；点一与点二入口复用 |
| 边缘 | [`edge.py`](../../edge.py) | 组内处理、证据与最终结果验证；点二投票/提交后状态 |
| 云 | [`cloud.py`](../../cloud.py) | 掩码聚合、组包buffer与候选checkpoint |
| 可信机构 | [`ta.py`](../../ta.py) | 掩码处理、授权、token/提交管理 |
| 密码与数值工具 | [`utils.py`](../../utils.py) | LHH、PRG、签名、量化尺度、既有补偿/权重计算 |
| 学习模型 | [`models.py`](../../models.py) | 模型、参数展平/还原与本地SGD |
| 数据 | [`datasets.py`](../../datasets.py) | MNIST/FashionMNIST/CIFAR10加载及IID/Non-IID分配 |
| 点二消息与事务 | [`point2_types.py`](../../point2_types.py)、[`commit_tx.py`](../../commit_tx.py) | 候选、验证、投票、收据与提交恢复 |
| 点一计量与归档 | [`artifacts.py`](../../experiments/point1_compare/artifacts.py)、[`round_shards.py`](../../experiments/point1_compare/round_shards.py) | 独占角色计时、逐接收者字节、逐轮不可覆盖分片 |
| 时间与环境记录 | [`watchdog.py`](../../experiments/point1_compare/watchdog.py)、[`multirun_environment.py`](../../experiments/point1_compare/multirun_environment.py) | 调用限时、只读运行条件观察 |
| Windows归档写入 | [`windows_artifact_io.py`](../../experiments/point1_compare/windows_artifact_io.py) | 计时外文件替换重试 |

单一模块可能服务两个研究点，以其调用入口确定实验归属。目录分类没有把同一实现复制成两份。

测试从`tests/`读取。点一归档测试中有一部分需要真实完成的run目录与环境变量；没有原始归档时不要把这类测试的缺失/失败解释为协议失败，也不要静默跳过后宣称全测试通过。详见[复现说明](../REPRODUCING.md)。
