# 研究点二: Agent 设计选项

> 结论先行: "Agent" 不等于必须用 LLM。对研究点二，主要有三种选择。

---

## 1. 规则型 Agent（最稳妥）

每个 Agent 根据状态和预设规则行动。

例如 Coordinator-Agent 根据:

- buffer 中组包数量
- 等待时间
- 组包陈旧度
- 预计下一个组包到达时间
- 当前掉线率

决定:

- 立即触发 checkpoint
- 继续等待
- 选择哪些组包
- 延后哪些 stale 组包

这本质上是"自治调度模块"，可以称为传统 MAS，但创新性有限。优点是稳定、可解释、容易复现。

---

## 2. 强化学习 Agent（真正的 AI Agent）

把 checkpoint 调度建模成强化学习问题:

- **状态**: buffer、staleness、延迟、掉线率、历史吞吐等
- **动作**: wait、trigger、选择组包、调整等待阈值
- **奖励**: 提高吞吐、降低延迟、减少过期包和 stale 误差
- **算法**: DQN、PPO 或 contextual bandit

示例:

```text
状态：buffer=4，平均staleness=2，预计1个包即将到达
动作：再等待1个tick
结果：获得更高吞吐，但增加等待延迟
```

不过这会明显扩大研究范围，你需要新增:

- RL 训练过程
- 环境设计
- 奖励函数
- 收敛或稳定性实验
- 与固定规则、贪心策略、随机策略比较
- 安全约束下的策略验证

而且强化学习 Agent **不能决定密码学安全事项**。它不能绕过 LHH、签名、TA 掩码或隐私阈值，必须经过 RuleEngine 硬约束。

---

## 3. 优化/控制型 Agent（更适合本项目）

可以使用:

- 上下文 bandit
- MPC 模型预测控制
- 效用函数优化
- 多目标调度
- 贝叶斯决策

它根据当前系统状态最大化:

```text
效用 = 吞吐收益 - 等待成本 - stale惩罚 - 过期惩罚 - 公平性惩罚
```

这比 LLM 和完整强化学习简单，也比固定规则更有"自适应"味道。但严格来说，它更适合叫:

> 自适应 checkpoint 调度器

不一定要叫 Agent。

---

## 推荐排序

```text
确定性自适应调度器
        >
强化学习 Coordinator-Agent
        >
LLM Coordinator-Agent
```

原因是核心是**安全聚合协议**，而不是智能决策算法。

---

## 最推荐的结构

```text
Edge-Agent：规则或轻量策略，决定本地组包 seal/wait
Coordinator-Agent：负责 checkpoint 触发和组包选择
Audit-Agent：固定规则检查 replay、异常和验证失败
RuleEngine：硬约束
TA + LHH + 签名：最终安全验证
```

其中真正值得做成 Agent 的只有 **Coordinator-Agent**。

- TA 不需要 Agent：密码学流程不应由学习策略控制
- Audit-Agent 最好也使用确定性规则

---

## 如果确实要做强化学习（最小可行版本）

> 单 Coordinator-Agent + PPO/DQN + RuleEngine 安全约束

研究问题可以写成:

> 在组包到达延迟、客户端掉线和模型陈旧度动态变化的条件下，利用强化学习自适应选择 checkpoint 触发时机和待聚合组包，以优化系统吞吐与训练时延。

但这已经不是简单"给研究点二加 Agent"，而是研究点二外面又增加了一个强化学习调度研究点。

---

## 结论

不用 LLM 完全可以，用强化学习或规则型 Agent 都可以；但对当前项目，最合理的是**规则/优化型 Coordinator-Agent**。强化学习只有在导师明确要求"智能决策算法"并且愿意增加大量实验时才值得做。