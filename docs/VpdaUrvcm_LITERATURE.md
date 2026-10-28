# V-PDA / UR-VCM 文献核查记录 (2025–2026)

> 检索日期: 2026-09-11 · 检索方式: arXiv API 元数据检索 (按提交日期倒序)
> **覆盖性声明**: arXiv API 只检索标题/摘要等元数据, 不是全文检索; 检索词命中稀疏,
> 因此「未检索到」≠「不存在」。一次委托的深度检索未返回结果, 本文档只列可核验条目。

## 1. 异步/半异步 FL 的陈旧度补偿 (最接近 UR-VCM 的邻域)

| 论文 | arXiv | 日期 | 与本工作的关系 |
|---|---|---|---|
| FedSN: A Federated Learning Framework over Heterogeneous LEO Satellite Networks | 2311.01483 | 2023-11 / 2024-10 更新 | 伪同步聚合 + "compensating model staleness"; 补偿按调度/伪同步实现, 未见范数比门控 |
| FedGSM: Efficient FL for LEO Constellations with Gradient Staleness Mitigation | 2304.08537 | 2023-04 | 异步 FL + 梯度陈旧补偿机制 |
| The Impact Analysis of Delays in Asynchronous FL with Data Heterogeneity for Edge Intelligence | 2503.04052 | 2025-03 | 延迟影响分析 + 复用延迟梯度补偿 (PSURDG); 无可信域门控 |
| FedRAW: Preserving Rare-Label Influence in Asynchronous Federated Learning | 2609.07192 | 2026-09 | 异步聚合偏置 (稀有标签影响) 的修正; 与 V-PDA 的"参与偏置"动机相近, 但作用于标签影响而非参与频率 |

**差异**: 上述工作都直接施加补偿/复用策略, 未检索到把补偿项按
`η = min(1, ρ·‖R−B‖/(‖C_avg‖+ε))` (R=聚合模型均值, B=所选组基模型样本加权均值)
做可信域缩放的做法; 也未检索到以"聚合训练增量范数"作为唯一尺度来源的补偿门控。

## 2. 服务器端可信域 / 裁剪

- 检索 `abs:"trust region" AND abs:"federated learning"` 命中多为**投毒攻击**
  类工作 (如 arXiv:2501.08002 Delphi, 2025-01), 不是防御侧补偿门控。
- 检索 `all:"aggregation" AND all:"trust ratio"` → **0 命中**。
- 本项目已引用的 FedAdam (arXiv:2003.00295) 有服务器自适应优化与信赖域步长上限
  (`gamma_tr`), FedLAW 有聚合收缩 — 属"对聚合更新整体"的尺度控制, 与"对补偿项
  单独做可信域门控"不同; 且不涉及掩码聚合下的可验证性。

## 3. 参与偏置 / 逆倾向加权 (V-PDA 邻域)

- 检索 `abs:"inverse propensity" AND abs:"federated"` 命中 3 篇, 全部属**联邦因果推断**
  的倾向得分加权 (Fed-IPW/Fed-AIPW, arXiv:2505.17961; arXiv:2404.15746;
  arXiv:2607.26287), 用于因果效应估计, **不是**用 IPW 修正联邦聚合的参与频率偏置。
- 即"用逆参与权重消除快组过采样偏置"这一动机在 FL 聚合侧检索不到成熟先例 —
  但这不构成新颖性正面证据 (我们的实验结论是负收益, 见 VPDA_REPORT)。

## 4. 可验证加权掩码聚合

- 本项目采用的 LHH (线性同态哈希) 加权关系 `∏H_k^{w_k}` 与
  `weight_source` 独立重算校验, 在本轮 arXiv 元数据检索中未返回同类条目
  (相关检索无提取结果), 无法据此判断 2025–2026 是否已有等价工作。
- 结论: 该能力可作为协议贡献保留, 但**新颖性需更充分的全文检索** (IEEE/ACM/DBLP
  全文 + 引文追踪) 才能定稿。

## 5. 新颖性评估结论 (针对 UR-VCM)

问题: "在掩码聚合协议中, 由边缘节点独立重算, 以聚合训练增量范数为尺度对
版本补偿项做可信域门控" 是否已发表?

- 本轮可核验检索**未发现**完全相同的组合。
- 最接近的三支: (a) 异步 FL 的陈旧度补偿 (FedSN / FedGSM / 2503.04052);
  (b) 服务器端信赖域/trust ratio 的自适应优化 (FedAdam / FedLAW);
  (c) 联邦因果推断中的 IPW (与聚合去偏不同)。
- **但需注意**: 由于 (i) arXiv 元数据检索覆盖有限, (ii) 委托的深度检索未返回,
  该结论只能标注为"未检索到", 不能写成"我们首次提出"。
- 命名建议 (仅用于负结果台账): V-PDA → *per-checkpoint inverse-participation
  weighted aggregation*; UR-VCM → *training-increment trust-region gated
  version compensation*。两者均**不进入论文主创新表述** (阶段 1/2 均未晋级)。

## 6. 检索词

`all:"secure aggregation" AND all:"asynchronous" AND all:"federated"`;
`all:"staleness" AND all:"compensation" AND all:"federated"`;
`all:"asynchronous federated learning" AND all:"compensation"`;
`abs:"trust region" AND abs:"federated learning"`;
`all:"aggregation" AND all:"trust ratio"`;
`abs:"inverse propensity" AND abs:"federated"`;
`abs:"semi-asynchronous" AND abs:"federated"`;
`all:"verifiable" AND all:"secure aggregation" AND all:"federated"`。
