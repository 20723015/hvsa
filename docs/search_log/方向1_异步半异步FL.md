# 方向 1 检索报告：异步/半异步联邦学习主干

检索日期 2026-09-22。用于论文第 2.5 节综述骨架与新颖性检索日志。
本报告由子智能体执行，原始检索输出见 `agent_raw/`（`arxiv_out*.txt`、`arxiv_raw*.json`）。

三轴记法：**边界**（异步发生在哪一级）/ **粒度**（什么触发聚合）/ **陈旧度**（怎么处理）。

---

## 1. 必读清单

### A. 经典奠基线

| 文献 | 标识符 | 一句话 | 边界 | 粒度 | 陈旧度 |
|---|---|---|---|---|---|
| Xie et al., *Asynchronous Federated Optimization* (FedAsync), 2019 | arXiv:1903.03934 | 服务器每收到一个本地模型就立即更新全局模型，给出近线性收敛证明 | **客户端级** | 无缓冲，来一个聚合一个 | **加权衰减** α_t=α·s(t−τ)，示例 s_a(τ)=1/(τ+1)；K 仅为收敛分析的延迟上界，非运行时缓冲 |
| Nguyen et al., *Federated Learning with Buffered Asynchronous Aggregation* (FedBuff), AISTATS 2022 | arXiv:2106.06639 | 用大小 K 的缓冲替代「来一个聚合一个」，兼顾异步吞吐与 SecAgg 兼容 | **客户端级** | **固定数量**：缓冲满 K 个客户端更新才做一次服务器步 | 核心算法是 K 个更新的**均匀平均**；§5 另加实践性缩放 s(τ)=1/(1+τ)^0.5 |
| Huba et al., *Papaya*, MLSys 2022（venue 未验证） | arXiv:2111.04877 | 生产级异步 FL 系统，覆盖近一亿设备 | **客户端级** | **固定数量**：aggregation goal K | **加权 + 硬丢弃** w_i=1/√(1+s_i)，并中止陈旧度超 maximum staleness 的客户端 |
| Chai et al., *FedAT: ... with Asynchronous Tiers*, SC21 | arXiv:2010.05958；DOI 10.1145/3458817.3476211 | 按响应延迟把客户端分成 M 个 tier，tier 内同步、tier 间异步 | **组级**（tier 是 "logical, coarse-grained training entity"，聚合单元是 tier） | 无缓冲，tier 完成一轮即触发 | **不按陈旧度**：按**更新频率**加权（慢 tier 权重更大）；无显式上限/丢弃 |
| Li et al., *FedProx*, MLSys 2020 | arXiv:1812.06127 | 可变工作量 + 近端项，处理系统与统计异质性 | **服务器轮次级**（同步屏障，非异步） | 固定轮 | **不处理** |
| Wu et al., *SAFA: a Semi-Asynchronous Protocol ...*, IEEE TC 2021 | arXiv:1910.01355；DOI 10.1109/tc.2020.2994391 | 半异步协议，缓解掉线/崩溃/陈旧度 | **服务器轮次级 / 轮内半异步** | 轮窗口 | 容忍并缓解；**具体机制未逐条验证** |

### B. 半异步 / 组级 / 分层一族（与本文同轴）

| 文献 | 标识符 | 一句话 | 边界 | 粒度 | 陈旧度 |
|---|---|---|---|---|---|
| Li et al., *FedCompass*, ICLR 2024 | arXiv:2309.14675 | 算力感知调度器给客户端分配不同本地步数，使其「作为一组几乎同时到达」再聚合 | **组级**（arrival group + `GroupAggregate`） | **组完成触发**（Ta/Tmax）；迟到者进 general buffer 并入**下一组** | **加权 + 不丢弃** `Δ̄ += st(τg−C[i].τ)·p_i·Δ_i`；理论假设陈旧度上界 τmax |
| Ma et al., *Air-FedGA*, IPDPS 2025 | arXiv:2507.05704；DOI 10.1109/IPDPS64566.2025.00051 | worker 分组，组内 AirComp 聚合，组与参数服务器异步通信 | **组级** | 无缓冲，来一个组聚合一个 | 定义**组级陈旧度** τ_t 用于收敛证明；**未发现加权或丢弃机制**（推断，未深读） |
| *FedBASA: Buffer-Aided Semi-Asynchronous FL*, ICCT 2025 | DOI 10.1109/icct67417.2025.11374013 | 缓冲多个**簇**的模型，容量达到时按陈旧度加权做全局聚合 | **组级**（簇） | **固定数量/容量触发** | **加权衰减**；未提及显式上界或补偿项 |
| Chen et al., *FedCure*, AAAI 2026 | arXiv:2511.10227；DOI 10.1609/aaai.v40i25.39176 | 分层 SAFL，用 coalition 构造 + 参与感知调度缓解参与偏差 | **组级**（coalition，云-边-端三层） | 调度规则（虚拟队列 + 贝叶斯估计），非固定 K | 处理**参与偏差**而非直接陈旧度；具体机制未验证 |
| Yu et al., *DecantFed*, 2024 | arXiv:2403.06900 | 按计算/通信延迟给客户端分层，不同层不同频率上传 | **组级**（tier） | **频率/周期性触发** | 按 tier 调整客户端学习率以应对陈旧度 |
| Zhang et al., *CSAFL*, IJCNN 2021 | arXiv:2104.08184；DOI 10.1109/ijcnn52387.2021.9533794 | 聚类式半异步框架 | **组级**（簇） | 未验证 | 未验证 |
| Yu et al., *Async-HFL*, IoTDI 2023 | arXiv:2301.06646；DOI 10.1145/3576842.3582377 | 三层 IoT 架构，**网关层与云层都做异步聚合** | **组级（两层）** | 事件驱动，无缓冲 | 未显式处理；靠 device selection 与拓扑周期调整抗 straggler |
| Wang et al., *Semi-Synchronous FL for Energy-Efficient Training*, ACM TIST 2022 | arXiv:2102.02849；DOI 10.1145/3524885 | 「周期性混合本地模型」，最小化空闲时间 | **服务器轮次级/周期** | **时间触发** | 隐含：周期长度约束陈旧度 |
| *HiFlash*, IEEE TPDS 2023 | arXiv:2301.06447；DOI 10.1109/tpds.2023.3238049 | 分层 FL + 自适应陈旧度控制 + 客户端-边缘关联 | **组级** | 未验证 | **自适应陈旧度控制**（形式未验证） |

### C. 陈旧度处理专线

| 文献 | 标识符 | 三轴要点 |
|---|---|---|
| *SC-AFL: Staleness-Controlled AFL*, IEEE TMC 2024 | DOI 10.1109/tmc.2024.3416216 | 客户端级 / 每轮动态调聚合策略 / **Lyapunov 优化把陈旧度限制在界内**（与 s_max 同族） |
| Ma et al., *FedStaleWeight*, 2024 | arXiv:2406.02877 | 客户端级 / FedBuff 式 K 缓冲 / 用平均陈旧度做**公平性重加权** |
| Liu et al., *FedASMU*, AAAI 2024 | arXiv:2312.05770；DOI 10.1609/aaai.v38i12.29297 | 客户端级 / 动态聚合 / **动态陈旧度感知混合** + 设备端自适应 |
| Lu et al., *FedPSA*, ESWA 2026 | arXiv:2602.15337 | 客户端级 / 动态动量队列 / 用**参数敏感性**度量陈旧度 |
| *FedSteer: Taming Extreme Gradient Staleness ... with Corrective Projections*, UAI 2026 | arXiv:2606.10124 | 客户端级 / 梯度缓存 / **后置纠偏**：把过期梯度投影到当前梯度子空间（与本文「后置补偿项」最同族） |
| Bai et al., *On the Push-Based Asynchronous FL: A Bias-Correction Aggregation Approach*, KDD 2026 | arXiv:2605.26162 | 去中心化 / 有界去重缓冲 / 平均保持 push-sum **纠偏** + centroid 正则 |
| Toghani & Uribe, *Unbounded Gradients in FL with Buffered Asynchronous Aggregation*, Allerton 2022 | arXiv:2210.01161；DOI 10.1109/allerton49937.2022.9929409 | 对 FedBuff 去掉梯度有界假设的收敛分析（理论必读） |
| *Data-Aware Quorum Aggregation (QFL)*, IWCMC 2026 | DOI 10.1109/iwcmc69287.2026.11579914 | 客户端级 / **数据感知 quorum 触发**（累积标签分布近似均匀才聚合，非计数非时间） |
| *Asynchronous FL on Heterogeneous Devices: A Survey*, 2021 | arXiv:2109.04269 | 综述，2.5 节起手式 |

---

## 2. 与本文最接近的 3 条

### ① FedBASA（ICCT 2025）— 三轴最接近

- **近**：聚合单元是**簇（组）**而非单客户端；**缓冲 + 容量阈值触发**（"buffer mechanism that stores models from multiple clusters and applies weighted aggregation based on staleness when capacity is reached"）；**陈旧度加权聚合**。
- **差**：① 未提**陈旧度上限 s_max**；② 未提**后置版本补偿项**；③ 触发是「缓冲容量」单一条件，未见与版本/checkpoint 语义绑定的形式化；④ 实验仅 MNIST 的 IID/Non-IID，评估面窄。
- **验证程度**：摘要级，正文未读。

### ② FedAT（SC21）— 「聚合单元是组」这一条最接近

- **近**：明确把 **tier 当作 "logical, coarse-grained training entity"**，tier 内同步、**tier 间异步**——即异步边界在组级、聚合单元是组。**本文的差异点在这里已有先例，必须在论文中正面引用并区分。**
- **差**：① **无缓冲、无阈值 K**（tier 一完成就触发）；② 陈旧度按**更新频率**加权（慢 tier 权重更大），而非按版本差加权，也**没有 s_max**；③ 无 mixed-version 补偿项；④ tier 划分是静态延迟分档，不随训练演化。
- **验证程度**：ar5iv 全文抽取（含权重公式），置信度较高。

### ③ FedCompass（ICLR 2024）— 触发 + 陈旧度两轴最接近

- **近**：**组级聚合**（arrival group + `GroupAggregate`）；触发是**组完成/最晚到达时刻**；聚合时按**陈旧度因子加权** `st(τg−C[i].τ)`；**迟到更新不丢弃**而是并入下一组——与本文「允许组包带陈旧 + 上限 s_max」同构。
- **差**：① **组的语义不同**：FedCompass 的组是「被调度器安排成几乎同时到达的**一批单个客户端**」，组内并无一次预先完成的组内聚合；本文的组包是**一组用户聚合后的结果**。② 触发靠**时间/预期到达时刻**，不是**计数阈值 K 个组包**；③ 未见对组包陈旧度的显式硬上限；④ 无版本差后置补偿项。
- **验证程度**：ar5iv 全文抽取并复核，置信度较高。

**补充最接近候选**：Air-FedGA（IPDPS 2025）——组为聚合单元、组与服务器异步，但**完全没有陈旧度处理机制**，是「组级异步但不管陈旧度」的干净对照点。

**限定表述**：在本次检索到的集合内，**未找到**同时具备「组包为聚合单元 + 计数阈值 K 触发 checkpoint + 显式 s_max 上限 + 按版本差的后置补偿项」这四要素的工作；最接近的都是三缺一或三缺二。**这不等于「该方向无人做过」**——见第 4 节。

---

## 3. 检索式清单（原文）

### 3.1 arXiv API（间隔 ≥3.5s）

```
all:"asynchronous federated learning"
all:"semi-asynchronous federated learning"
all:"semi-synchronous federated learning"
all:"staleness" AND all:"federated learning"
all:"buffered" AND all:"asynchronous federated learning"
all:"hierarchical asynchronous federated learning"
all:"group asynchronous federated learning"
all:"straggler" AND all:"federated learning"
all:"staleness-aware federated learning"
all:"hierarchical federated learning" AND all:"asynchronous"
all:"periodic aggregation" AND all:"federated learning"
all:"checkpoint" AND all:"federated learning"
all:"group-level" AND all:"federated learning"
all:"aggregation" AND all:"stale" AND all:"federated learning"
all:"client drift" AND all:"asynchronous federated learning"
all:"convergence analysis" AND all:"asynchronous federated learning"
all:"partial synchronization" AND all:"federated learning"
all:"cohort" AND all:"federated learning"
all:"cluster-level" AND all:"federated learning"
all:"model version" AND all:"federated learning" AND all:"staleness"
all:"tier" AND all:"asynchronous" AND all:"federated learning"
all:"update buffer" AND all:"federated learning"
all:"cluster-level asynchronous federated learning"
all:"hierarchical semi-asynchronous federated learning"
all:"stale updates" AND all:"partial participation" AND all:"federated learning"
all:"survey" AND all:"asynchronous federated learning"
all:"age of information" AND all:"federated learning" AND all:"aggregation"
all:"group" AND all:"synchronization" AND all:"hierarchical federated learning"
all:"staleness" AND all:"compensation" AND all:"federated learning"
all:"delayed aggregation" AND all:"federated learning"
all:"version" AND all:"compensation" AND all:"federated learning"
all:"group-based" AND all:"asynchronous" AND all:"federated learning"
all:"proximal" AND all:"asynchronous federated learning"
all:"staleness-weighted" AND all:"federated learning"
all:"staleness" AND all:"weighted aggregation" AND all:"federated"
ti:"Asynchronous Federated Optimization"
ti:"FedBuff"
ti:"Federated Learning with Buffered Asynchronous Aggregation"
ti:"FedAT"
ti:"Papaya"
ti:"Federated Optimization in Heterogeneous Networks"
ti:"FedProx"
ti:"Semi-Synchronous Federated Learning"
ti:"FedCompass"
ti:"FedAsync"
```

命中量为 0 的检索式（说明 arXiv 全字段短语检索过严）：`all:"hierarchical asynchronous federated learning"`、`all:"staleness-aware federated learning"`、`all:"cluster-level asynchronous federated learning"`、`all:"hierarchical semi-asynchronous federated learning"`、`all:"proximal" AND all:"asynchronous federated learning"`。去重后共 249 条记录。

### 3.2 Crossref

`query.bibliographic` 取值：
```
asynchronous federated learning staleness
semi-asynchronous federated learning aggregation
buffered asynchronous federated learning staleness
group asynchronous federated learning aggregation
hierarchical asynchronous federated learning edge cloud
staleness-aware aggregation federated learning
checkpoint federated learning aggregation
```
另对 24 个具体标题做逐条核验，并抓取 19 个 DOI 的 `works/{doi}` 元数据。

### 3.3 全文核验（ar5iv HTML）

```
https://ar5iv.labs.arxiv.org/html/1903.03934   (FedAsync)
https://ar5iv.labs.arxiv.org/html/2106.06639   (FedBuff)
https://ar5iv.labs.arxiv.org/html/2010.05958   (FedAT)
https://ar5iv.labs.arxiv.org/html/2111.04877   (Papaya)
https://ar5iv.labs.arxiv.org/html/2309.14675   (FedCompass)
https://ar5iv.labs.arxiv.org/html/2507.05704   (Air-FedGA)
```

---

## 4. 覆盖局限与不确定项

**未覆盖的来源**
1. **OpenAlex 全程 429**，未采用；跨库去重与引用量排序缺失。
2. **Semantic Scholar Graph API 几乎全程 429**（11 次请求仅 1 次成功）。
3. **dblp API 返回非 JSON**，venue 核验未完成。
4. **未做 Google Scholar / Scopus / Web of Science / IEEE Xplore / ACM DL 全文检索**；IEEE/ACM/Elsevier 条目**只经 Crossref/S2 元数据覆盖，未读正文**（FedBASA、SC-AFL、QFL、HiFlash 等均属此类）。
5. **未检索中文文献**（CNKI/万方）。
6. **未做引文图滚雪球**（既无前向引用检索，也无后向追踪）。**这是最大方法论缺口**：FedBuff 的后代工作散落在 IEEE 期刊与会议中，Crossref 相关性排序无法保证召回。
7. Crossref 宽泛查询噪声极大（`asynchronous federated learning staleness` 的 total-results 达 270 万），**标题中不含 async/semi-async/staleness 关键词的工作会系统性漏检**。

**未验证 / 低置信度条目**
- **venue 未验证**：FedAsync（无 journal_ref）、Papaya（MLSys 2022 为常见引用，未证实）、FedBuff（venue 来自 arXiv comment 字段）。
- **机制细节仅摘要级**：SAFA、CSAFL、FedCure、HiFlash、FedBASA 的触发与陈旧度形式。
- **推断项**：Air-FedGA「无陈旧度加权机制」是全文正则扫描的推断，非作者声明。
- **FedBuff 的 s(τ)=1/(1+τ)^0.5 与 FedAT 的频率权重公式**来自 ar5iv 模型抽取，**非人工逐字校对**，写进论文前建议回原文核对。
- **「FedProx-async 系列」未能定位为独立谱系**：FedProx 本体是同步的；异步+近端项的组合散见个性化/正则化变体，**该子线覆盖不完整，建议单独补检**。

**表述纪律**：本报告只陈述「在本次检索到的 249 条 arXiv 记录 + 若干 Crossref 记录中，未发现同时满足四要素的工作」。**未检索到 ≠ 不存在**。在补做引文图前向检索（至少覆盖 FedBuff、Papaya、FedAT、FedCompass、FedBASA 的施引文献）之前，不建议写出任何「无人做过」式断言。
