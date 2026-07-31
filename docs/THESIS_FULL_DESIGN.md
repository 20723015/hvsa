# 面向分层联邦学习的单TA可验证安全聚合及其检查点式半异步扩展

> 本文档为研究点1（HVSA）与研究点2（检查点式组级半异步）的整合完整设计文档。
> 实验部分仅给出设计框架与预期趋势，所有数值留空待回填。
> 研究点2 协议公式与 `THESIS_POINT2_STALENESS_AWARE_REUSE.md`（已定稿）一致；实验设计在其基础上扩展（如 reuse×compensation 2×2 消融），属框架性扩充而非协议改动。

---

## 目录

- 第0章 摘要
- 第1章 引言
- 第2章 相关工作
- 第3章 系统模型、威胁模型与预备知识
- 第4章 研究点一：HVSA 单TA可验证安全聚合协议
- 第5章 研究点一安全分析
- 第6章 研究点二动机：从同步到检查点式半异步
- 第7章 研究点二协议：检查点式组级半异步安全聚合
- 第8章 研究点二安全分析
- 第9章 实验框架（数值留空）
- 第10章 结论与展望

---

# 第0章 摘要

分层联邦学习（Hierarchical Federated Learning）在"云—边—端"架构下兼顾了隐私与通信效率，但仍面临两大痛点：一是客户端本地模型在上传与聚合过程中可能泄露隐私（梯度/参数泄露攻击），二是云端聚合结果可能被不可信云服务器篡改而难以被参与方检测。本文围绕**单可信权威机构（TA）约束下的轻量可验证安全聚合**展开，提出两个递进的研究点。

研究点一提出 **HVSA**（Hierarchical Verifiable Secure Aggregation），在"云—边—端—单TA"架构下，通过 PRG 掩码保护客户端参数隐私，利用线性同态哈希（LHH）与 Ed25519 签名让边缘服务器（ES）能够对云端聚合结果进行高效验证，验证通信开销与模型维度无关。

研究点二在研究点一的密码学闭环基础上，针对同步聚合的"全局等待"与"掉线训练浪费"问题，提出**检查点式组级半异步安全聚合**。通过组内 stale reuse 复用客户端掉线未上传的训练结果、以 masked group package 进入缓冲区、按 checkpoint 批量解掩码聚合，并引入 Direct-VCM 组级后置补偿，在不破坏单TA掩码与 LHH 验证链路的前提下，将跨组训练组织从"全组同步 barrier"扩展为"组包缓冲 + checkpoint 触发"的组级半异步机制。

**关键词**：联邦学习；分层聚合；安全聚合；可验证聚合；半异步；陈旧性复用

---

# 第1章 引言

## 1.1 研究背景

联邦学习（FL）允许多个客户端在不共享原始数据的前提下协同训练全局模型。然而两条研究线揭示了其脆弱性：

- **隐私泄露**：DLG 等梯度反演攻击表明，服务器可从客户端上传的明文梯度/参数中重建训练数据。
- **结果不可信**：云端作为聚合方，可能出于节省算力或恶意目的篡改聚合结果，而客户端/边缘难以检测。

分层联邦学习引入边缘服务器（ES）作为中间层，缓解了云端通信压力，但也使"掩码—聚合—验证"的信任链更复杂。

## 1.2 问题陈述

- **研究点1问题**：如何在单TA约束下，既保护客户端参数隐私，又让边缘方能高效验证云端聚合结果，且验证开销不随模型维度膨胀？
- **研究点2问题**：在研究点1的同步框架下，全组同步 barrier 导致快组等慢组、客户端掉线导致已完成训练结果被丢弃；但完全异步又会破坏"逐轮在线验证"的密码学闭环。如何在不破坏验证链路的前提下引入异步性？

## 1.3 本文贡献

本文贡献按两个研究点统一编号：

**研究点一（HVSA）：**
- 1-a 单TA分层可验证安全聚合架构：单TA即可支持掩码分发与聚合掩码计算，无需多TA/门限TA。
- 1-b 基于 LHH + Ed25519 的边缘可验证聚合：验证式 `H(W_sum^raw)·H(M_global)=∏_k H_k`，验证通信与模型维度无关。
- 1-c 验证外包到边缘：客户端零验证开销，验证由 ES 承担。

**研究点二（检查点半异步扩展）：**
- 2-a 检查点式组级半异步扩展：借鉴 FedBuff 缓冲思想提升至组包粒度，将跨组同步 barrier 扩展为 buffer+checkpoint 半异步。
- 2-b 安全聚合兼容的组内 stale reuse：复用客户端掉线未上传结果，权重 `ρ_i=ρ_max·e^{−β_c·E_i}` 随陈旧度衰减（修正 MAFSA 方向），掩码前本地混合，不破坏验证链路。
- 2-c 面向 mixed-version 组包的公开确定性后置补偿（Direct-VCM）与 checkpoint 验证：`C_k=S_k·γ_vcm·(W(t)−W(t_k))`，ES 基于已验证 raw sum 本地重构最终模型。

## 1.4 章节组织

第3章给出统一系统模型、威胁模型与密码学预备知识；第4-5章阐述研究点1；第6-8章阐述研究点2；第9章给出实验框架；第10章总结。

---

# 第2章 相关工作

## 2.1 安全聚合

掩码式安全聚合（如 Bonawitz 等）通过成对/种子掩码隐藏单个客户端更新。多TA/门限方案进一步分散信任，但引入额外协调开销。本文采用单TA以简化信任假设。

## 2.2 可验证聚合

VerifyNet、VeriFL、VLMRS、GPVO-FL 等通过同态哈希/签名/承诺实现聚合结果可验证。多数方案验证开销随客户端数或维度增长。本文将验证集中到边缘并利用 LHH 定长哈希降低开销。

## 2.3 分层联邦学习

云—边—端三层结构下，边缘聚合降低云端压力。本文在其上叠加可验证安全聚合。

## 2.4 异步与半异步联邦学习

- **FedBuff**：缓冲异步聚合，单客户端粒度。
- **FedFleet**：性能感知分组，加速异构IoT收敛（本文仅借鉴其一次性分组作初始化优化）。
- **MAFSA**：多层异步 + 掉线客户端新旧结果融合（本文借鉴其复用动机但修正权重方向并限定 stale 来源）。
- **DVAFL**（Chen et al., *"Toward Efficient and Scalable Asynchronous Federated Learning via Stragglers Version Control"*, IEEE TMC 2026）：版本控制机制（VCM）补偿陈旧更新（本文借鉴其 Direct-VCM 思想提升至组包粒度并去除降权路径）。

## 2.5 小结

| 方案 | 隐私 | 可验证 | 异步粒度 | 单TA |
|------|------|--------|----------|------|
| FedBuff | 部分 | 否 | 客户端 | — |
| FedFleet | 部分 | 否 | 分组（初始化） | — |
| MAFSA | 部分 | 否 | 客户端/层 | — |
| DVAFL | 部分 | 否 | 客户端 | — |
| HVSA(点1) | 是 | 是 | 同步 | 是 |
| 本文(点2) | 是 | 是 | 组包/checkpoint | 是 |

---

# 第3章 系统模型、威胁模型与预备知识

## 3.1 系统模型

四元架构：

- **Cloud (CS)**：云端聚合服务器，不可信。
- **Edge Server (ES)**：`e_k`，边缘聚合与验证者。
- **User (u_i^k)**：客户端，持有本地数据 `D_i^k`，样本量 `s_i^k=|D_i^k|`。
- **Trusted Authority (TA)**：单可信权威机构，分发掩码种子、计算聚合掩码。

记号总表：

| 记号 | 含义 |
|------|------|
| `t` | 全局版本（点1为同步轮次，点2扩展为 checkpoint 版本，见7.1） |
| `t_k` | 组包 `Π_k` 的训练基版本 |
| `u_i^k` | 组 `k` 中第 `i` 个客户端 |
| `s_i^k` | 客户端样本量；`S_k=Σ s_i^k` 组样本量；`S_Q=Σ S_k` checkpoint 总样本量 |
| 上横线（如 `w̄`） | 量化+掩码后的量 |
| `W(t)` | 版本 `t` 的全局模型 |
| `m_i^k` | 客户端掩码；`M_k`/`M_Q`/`M_global` 组/checkpoint/全局掩码和（下标区分） |
| `M` | 边缘组数（仅作计数，与带下标的掩码和 `M_k`/`M_Q`/`M_global` 区分） |
| `H_k` | 组包 LHH 哈希 |
| `γ_vcm` | Direct-VCM 补偿强度（唯一补偿超参） |
| `ρ_max, β_c` | stale reuse 权重上界与衰减速度 |

## 3.2 威胁模型

统一声明（标注 P1/P2 适用范围）：

- **半诚实用户与 ES**：按协议执行，但可能好奇。
- **不可信 Cloud**：可能篡改聚合结果。
  - P1：防 Cloud 篡改当轮同步聚合结果。
  - P2：防 Cloud 篡改 checkpoint **选中**的聚合结果（`W_sum^raw`）。
- **ES 非攻击者**：不将 ES 列为攻击者。
- **显式出界**：manifest 漏选/调度审查属 anti-censorship，不作主协议安全保证（P2）；全局模型为共识量，不作攻击面。
- **样本量自报**：`s_i^k` 为客户端自报，防虚报属客户端数据可信问题，出界。
- **exactly-once 语义声明**：每个客户端更新与组包携带唯一标识，Cloud/TA 对已聚合组包拒绝重放；完整 ACK 握手/原子提交/缓存并发属实现完善项，不作主协议数学前提。
- **单TA 信任边界与不合谋假设**：TA 仅持有掩码种子与在线集合，不接收 masked model（`w̄_i^k` 或 `W̄_k`）；Cloud/ES 持有 masked model 但无掩码。隐私依赖二者职责分离，故假设 **TA 不与 Cloud 或 ES 合谋**（trusted-dealer 式安全聚合标准前提）。此假设为单TA轻量架构的固有代价；门限化 TA 以消除单点信任列为第10章展望，不在本文威胁模型内。

## 3.3 预备知识（密码学原语）

- **LHH（线性同态哈希）**：NIST P-256 上构造，逐维生成元由 `H("HVSA-LHH-generator-v1"‖seed‖i)·G` 派生；加法同态 `LHH.Add(H(a),H(b))=H(a+b)`；定长输出（~32B，与维度 `d` 无关）。
- **Ed25519**：`SIG.sign/verify`，用于组包与聚合结果签名。
- **PRG 掩码**：AES-CTR over HKDF-SHA256，`PRG.mask_int(seed, user_id, round_token, dim)`，逐客户端逐轮派生。
- **量化**：`QUANTIZE_SCALE=1e6`，`model_int=(flat·1e6).astype(int64)`。

---

# 第4章 研究点一：HVSA 单TA可验证安全聚合协议

## 4.1 设计目标

在单TA下同时实现：客户端参数隐私、边缘可验证云端聚合、验证开销与维度无关、客户端零验证负担。

## 4.2 初始化与密钥分发

TA 与每个用户经 X25519 ECDH + HKDF 协商共享种子 `_ta_seed`；TA 持有所有用户掩码种子与 ES 公钥列表。

## 4.3 本地训练、量化与掩码上传

客户端 `u_i^k` 本地训练得 `w_i^k`，展平量化后加掩码：

```text
w̄_i^k(t) = Quantize(w_i^k) + m_i^k(t)
m_i^k(t) = PRG.mask_int(seed_i, u_i^k, t, d)
```

> 隐私前提：当组内（或全局）在线客户端数 `n_online ≥ 2` 时成立。若仅单个客户端在线，解掩码后的组/全局和即退化为该客户端明文模型；故协议要求 `n_min ≥ 2`（见 5.1 与 7.4）。

## 4.4 边缘聚合

ES `e_k` 求组内掩码和并计算 LHH 与签名：

```text
W̄_k(t) = Σ_i w̄_i^k(t)
H_k(t)  = LHH.Hash(pp, W̄_k(t))
σ_k     = SIG.sign(H_k(t))
```

上报 Cloud 与 TA（在线列表）。

## 4.5 云端聚合与 TA 解掩码

```text
W̄(t)      = Σ_k W̄_k(t)
M_global  = Σ_{u∈online} PRG.mask_int(...)   (TA 计算)
W_sum^raw = W̄(t) − M_global
W(t+1)    = W_sum^raw / n_online             (简单平均，n_online 为在线客户端数)
```

> 注：研究点1 采用客户端等权的简单平均（与实现一致）；样本量加权的 strict sample-weighted sum 是研究点2（7.4/7.7）引入的扩展，两点的归一化口径不同。

## 4.6 边缘验证

ES 验证云端返回的 `W_sum^raw`：

```text
H(W_sum^raw) · H(M_global) = ∏_k H_k(t)
```

成立则聚合结果可信。验证通信为定长哈希，与 `d` 无关。

## 4.7 复杂度

| 实体 | 计算 | 通信 |
|------|------|------|
| 用户 | O(d) | O(d) |
| ES | O(n_k d) | O(n_k d) |
| Cloud | O(Md) | O(Md) |
| 验证(ES) | O(d) 哈希 + O(1) 通信 | O(1) |

---

# 第5章 研究点一安全分析

## 5.1 隐私性

PRG 掩码在随机预言机模型下与随机数不可区分，Cloud/ES 无法从 `w̄_i^k` 恢复 `w_i^k`；抗 DLG 攻击。该结论以在线客户端数 `n_online ≥ n_min ≥ 2` 为前提：单客户端在线时，解掩码和即该客户端明文模型，不构成隐私保护。

## 5.2 可验证性

LHH 抗碰撞 ⇒ Cloud 篡改 `W_sum^raw` 使验证等式不成立；Ed25519 防组包伪造。

## 5.3 鲁棒性

TA 按 ES 上报在线集合动态重构 `M_global`，支持客户端掉线。

## 5.4 局限（引出研究点2）

- 全组同步 barrier：快组等慢组，墙钟时间受最慢组制约。
- 掉线训练浪费：客户端完成训练但掉线，结果被丢弃。
- 逐轮在线验证难以直接推广到完全异步。

---

# 第6章 研究点二动机：从同步到检查点式半异步

同步 HVSA 的等待与浪费问题促使引入异步性。但完全异步（来一个更新一个）下，每次全局下发都需在线验证，与单TA+LHH 的逐轮验证闭环冲突。

核心论点：**主动把异步边界收缩到 checkpoint 级**。组内仍同步封组，组间用 buffer+checkpoint 半异步；Cloud 保存组级掩码包而非明文，验证粒度为 checkpoint 选中集合。这样既不破坏掩码与 LHH 链路，又减少全局等待与掉线浪费。

---

# 第7章 研究点二协议：检查点式组级半异步安全聚合

> 本章协议公式与已定稿 `THESIS_POINT2_STALENESS_AWARE_REUSE.md` 一致。

## 7.1 记号重定义

- `t`：当前最新已提交的 **checkpoint 全局版本**（扩展点1的轮次语义）。
- `t_k`：组包 `Π_k(t_k)` 的训练基版本。mixed-version checkpoint 中通常 `t_k<t`。
- 客户端级陈旧度 `E_i=t_k−t_i`；组级陈旧度 `s_k=t−t_k`。

## 7.2 初始化与一次性性能分组

按 `d_i=t_i^train+t_i^comm`（或仅通信时延）将客户端分为 `M` 组，各绑定一个 ES。借鉴 FedFleet，仅作初始化优化，不改变协议、不作创新点。

## 7.3 组内 stale reuse

客户端至多缓存一个"已完成训练但因掉线/上传超时未被 ES 接收"的 stale 候选 `w_i^{stale,k}`（产生版本 `t_i`）。**stale 来源限定为单次未上传结果**，不含 MAFSA 掉线期间持续训练假设。恢复参与后：

```text
E_i  = t_k − t_i
ρ_i  = ρ_max · exp(−β_c · E_i),   0<ρ_max≤0.5
w_i^k = (1−ρ_i)·w_i^{new,k} + ρ_i·w_i^{stale,k}   (Ours Decayed-Stale)
```

权重方向修正 MAFSA（其 `α=e^{−βE}` 作用于 fresh，使越旧 stale 权重越大）。混合在掩码前本地完成，只上传一条 masked model。客户端级与组级以"ES 封组前是否接收"为界，上传被接收即清除 stale 候选。

## 7.4 掩码上传与组包生成

```text
w̄_i^k(t_k) = s_i^k·w_i^k + m_i^k(t_k)     (s_i^k 只乘明文，不乘掩码)
W̄_k(t_k)   = Σ_i w̄_i^k = Σ_i s_i^k·w_i^k + Σ_i m_i^k
S_k(t_k)   = Σ_i s_i^k
```

要求最小在线人数 `n_min≥2`（保护客户端级隐私）。组包：

```text
Π_k(t_k) = (k, t_k, h_base(t_k), U_online, n_k, S_k, W̄_k, H_k, σ_k)
```

其中 `σ_k` 为 ES 对最小元数据 `(k, t_k, h_base(t_k), digest(U_online), n_k, S_k, H_k)` 的 Ed25519 签名；该元数据即 7.8 Step 1 manifest 一致性检查的比对对象。

## 7.5 TA 的组包掩码支持

TA 按参与集合计算 `M_k=Σ m_i^k`，**不在组包入 buffer 时单独释放**；仅当 checkpoint 选中集合 `Q(t)` 确定后返回 `M_Q=Σ_{Π_k∈Q} M_k`。

## 7.6 Buffer 与 Checkpoint 触发

Cloud 维护组包缓冲区 `B`，每组至多一个在途包。过期判据 `s_k=t−t_k>s_max` 移除。

- 常规：`|B_legal|≥K` 时按到达顺序选最早 `K` 个。
- 超时：最早合法组包等待达 `Δ_timeout` 且 `1≤|B_legal|<K` 时，对全部合法组包保守 checkpoint。

发布 `checkpoint_manifest(t)`（含 mode、|Q|、各组包元数据）及公开摘要 `h_manifest`，先广播给 TA 与 ES。

可选调度增强（消融项，不改密码链路）：adaptive s_max、公平保底（`D_fair`）、选择策略（FIFO/Fresh-first/Fairness-aware）。

## 7.7 解掩码、Direct-VCM 补偿与全局更新

```text
W̄_Q(t)        = Σ_{Π_k∈Q} W̄_k
W_sum^raw(t)  = W̄_Q(t) − M_Q(t)               (样本量加权和)
G_k(t)        = W(t) − W(t_k)                  (t_k<t)
C_k(t)        = S_k(t_k)·γ_vcm·G_k(t)          (Direct-VCM 组级补偿)
W_sum^corr(t) = W_sum^raw(t) + Σ_{t_k<t} C_k(t)
W(t+1)        = W_sum^corr(t) / S_Q(t)         (S_Q=Σ S_k, FedAvg-form 归一化)
```

不引入解掩码前的组间连续非等权系数（避免破坏掩码+LHH）；不采用降权路径，用 s_max 硬截断 + 补偿。

## 7.8 Checkpoint 级验证与最终模型重构

- **Step 1**：manifest 一致性检查 + 各组包 Ed25519 验签。
- **Step 2**：LHH 验证 raw sum `H_sum^raw·H_mask = ∏_{Π_k∈Q} H_k`。
- **Step 3**：ES 基于已验证 raw sum，用公开量（历史模型、`S_k`、`γ_vcm`）本地确定性重构 `W(t+1)`，下发给组内客户端。Cloud 广播的 `W(t+1)` 仅作参考，规范结果以 ES 本地派生为准。

## 7.9 状态转移与 exactly-once

被选中组包移出 buffer、对应组获 `W(t+1)` 并开始下一轮；未选中组包继续等待或过期。客户端级/组级状态分界清晰，不反向生成 stale 候选；唯一 ID + 拒绝重放保证 exactly-once。

## 7.10 创新点（措辞定稿）

- **创新点1**：单TA可验证安全聚合下的检查点式组级半异步扩展。
- **创新点2**：安全聚合兼容的组内 stale reuse 机制（修正 MAFSA 方向）。
- **创新点3**：面向 mixed-version 组包的公开确定性后置补偿与 checkpoint 验证机制。

---

# 第8章 研究点二安全分析

## 8.1 篡改检测归约

- Cloud 改 `W_sum^raw` → Step 2 LHH 等式不成立。
- Cloud 替换/伪造组包 → Step 1 manifest 一致性 + 签名验证失败。
- Cloud 重放 → 唯一 ID 拒绝。
- Cloud 改补偿项 → 补偿由公开量唯一确定，ES 本地重算可发现；补偿输入（历史全局模型）为共识量，不作攻击面。

## 8.2 隐私保持

stale 混合在掩码前本地完成，ES/Cloud 仅见一条 masked model；TA 仅按参与集合求掩码，不知 `ρ_i`/`w_stale`。在 3.2 不合谋假设下，ES/Cloud 持 masked model 无法解掩码，TA 持掩码无法还原明文——职责分离保证客户端参数不被任何单一实体恢复。

## 8.3 收敛启发式讨论

本节给出一个"有界陈旧 ⇒ 有界单步偏差"的启发式刻画（非完整收敛证明，作为答辩支撑草纲，完整证明留作后续）。

**引理 8.1（单 checkpoint 偏差有界，草纲）.** 设目标函数 `F` 的梯度 `L`-Lipschitz，全局模型经 SGD 型更新，单步位移有界：`‖W(v+1)−W(v)‖ ≤ ηB`（`η` 学习率、`B` 有效梯度范数上界）。则一次 checkpoint 相对"无陈旧、无补偿的理想 FedAvg-form 聚合"的偏差可被两项有界量控制：

**(i) 客户端级陈旧偏差。** 客户端混合 `w_i=(1−ρ_i)w_new+ρ_i·w_stale`，相对全新结果的偏移 `ρ_i·(w_stale−w_new)`。由 `ρ_i≤ρ_max≤0.5` 且 `w_stale`/`w_new` 各自位移有界，客户端级偏差 `≤ ρ_max·C₁`，`C₁` 为常数。

**(ii) 组级版本失配偏差。** 组包 `Π_k(t_k)`（`t_k<t`）经 Direct-VCM 补偿 `C_k=S_k·γ_vcm·(W(t)−W(t_k))`。由 `s_k=t−t_k≤s_max` 及单步位移有界，`‖W(t)−W(t_k)‖ ≤ s_max·ηB`，故组级补偿幅度 `‖C_k‖ ≤ S_k·γ_vcm·s_max·ηB`。经 `÷S_Q` 归一化后组级权重 `S_k/S_Q` 约去，补偿引入的模型空间偏差 `‖C_k‖/S_Q ≤ γ_vcm·s_max·ηB`，与项 (i) 同量纲。

综合：单 checkpoint 总偏差

```text
‖Δ_bias‖ ≤ ρ_max·C₁  +  γ_vcm·s_max·ηB
```

即偏差被 `ρ_max`、`γ_vcm`、`s_max`、`η` 四个有界量的乘积/线性组合显式控制。增大陈旧容忍（`s_max`↑）或补偿强度（`γ_vcm`↑）线性放大偏差上界，与第9章对 `s_max`、`γ_vcm` 的敏感性消融设计相互印证。

**说明与边界.** 该刻画是确定性的单步幅度界，未覆盖：non-IID 下的客户端漂移、SGD 噪声、`ρ_i` 与数据分布的耦合。完整收敛率（如 `O(1/√T)` 邻域收敛）需在 Lipschitz/有界方差假设下另行证明，留作展望（见第10章）。本节目的是表明协议的超参（`ρ_max, γ_vcm, s_max`）为偏差提供了**可调的显式上界**，而非断言已达到某收敛率。

## 8.4 出界声明重申

manifest 漏选/审查（anti-censorship）、样本量虚报、完整 ACK 协议不作主协议安全保证。

---

# 第9章 实验框架（数值留空）

> 所有准确率/开销/时间数值留空（`___`），仅给出图表设计、指标定义与预期趋势。

## 9.1 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | ___ |
| 数据集 | MNIST、CIFAR-10 |
| 模型 | MLP、CNN |
| 聚合 | FedAvg-form |
| 层级 | 1 Cloud + M Edge + N Users |
| 密码 | LHH(P-256) / Ed25519 / PRG(AES-CTR/HKDF) / QUANTIZE_SCALE=1e6 |

## 9.2 研究点1实验（HVSA）

复用 `_tmp_研究点一实验方案.md` 结构：

- **已完成**：E2 IID 收敛（vs FedAvg）、E3 掉线鲁棒（0/10/20/30%）、E4 开销对比（8 sub）。
- **待补**：N1 Non-IID（Dirichlet α=0.5/1.0）、N2 DLG 防御（PSNR/SSIM）、N3 扩开销基线（VerifyNet/VeriFL/VLMRS/GPVO-FL）、N4 阶段开销分解、N5 验证开销 vs d、N6 MLP、N7 参数分布一致性。
- **必做最低集**：E2+N1、N2、N7、E4+N3、N5。
- **指标**：准确率、PSNR/SSIM、计算/通信开销、验证开销（随 d 恒定）。

## 9.3 研究点2实验（检查点半异步）

**五组主线**：

1. Sync-HVSA
2. Grouped-HVSA + Inter-Group Sync
3. + Buffer/Checkpoint Semi-Async
4. + Intra-Group Stale Reuse
5. + Direct-VCM Group Compensation（完整方案）

**攻击实验**（以第5组为基座，与性能基线分开）：tamper_global / tamper_manifest / drop_replace_pkg，报告 tampering detection rate。

**消融顺序**：

1. reuse-rule（none / mafsa / ours，必做，验方向修正）
2. No Compensation（γ_vcm=0）
3. Direct-VCM（γ_vcm=1.0）
4. reuse × compensation 2×2
5. ρ_max∈{0.25,0.5}、β_c∈{0.1,0.3,0.5}
6. γ_vcm∈{0.5,0.8,1.0}
7. K∈{2,3,5}
8. s_max∈{2,3,5}
9. policy（FIFO/Fresh-first/Fairness-aware）
10. adaptive s_max（off/on）

**指标**：达到目标精度墙钟时间、平均等待、最终精度、compensation gain、stale reuse gain、stale-weight stability、tampering detection rate、verification overhead、组包过期率、重复贡献率、组间公平性、有效吞吐量。

**超参表**：

| 参数 | 取值 |
|------|------|
| ρ_max | {0.25, 0.5} |
| β_c | {0.1, 0.3, 0.5} |
| γ_vcm | {0, 0.5, 0.8, 1.0} |
| K | {2, 3, 5} |
| s_max | {2, 3, 5} |
| Δ_timeout | 开关对比 |
| policy | FIFO / Fresh-first / Fairness-aware |
| D_fair | {1, 2, 3} |

## 9.4 结果占位

| 实验 | 指标 | 数值 |
|------|------|------|
| 点1 E2 收敛 | MNIST/CIFAR-10 准确率 | ___ |
| 点1 N2 DLG | PSNR/SSIM | ___ |
| 点2 组5 vs 组1 | 墙钟时间/最终精度 | ___ |
| 点2 stale reuse gain | 复用收益 | ___ |
| 点2 compensation gain | 补偿收益 | ___ |
| 点2 攻击 | detection rate | ___ |

---

# 第10章 结论与展望

本文在单TA约束下，先提出 HVSA 实现分层可验证安全聚合，再以检查点式组级半异步扩展解决同步等待与掉线浪费，同时保持掩码与 LHH 验证闭环。

未来方向：

- 双TA / 门限TA 以进一步分散信任。
- 基版本对齐的更新量混合（stale reuse 更严格变体）。
- 完整 ACK / 原子提交协议的协议化。
- anti-censorship 可审计调度。
- 完整收敛率证明：在 Lipschitz/有界方差假设下将引理 8.1 的单步偏差界推进为 `O(1/√T)` 邻域收敛分析（见 8.3）。

---

# 附录 A：设计决策记录（Design Rationale）

> 本附录保留研究点2 从多轮讨论收敛到最终协议的关键取舍理由，正文为行文简洁未展开。
> 与正文口径一致，供答辩/审稿追溯设计动机。

## A.1 为什么把异步边界收缩到 checkpoint 级

最初考虑过完全异步（组包"来一个更新一个"），但完全异步下每次全局下发都需在线验证，与单TA+LHH 的逐轮验证闭环冲突。最终选择：组内仍同步封组，组间用 buffer+checkpoint 半异步，验证粒度为 checkpoint 选中集合。一句话：**不是把验证硬塞进完全异步，而是主动收缩异步边界以保住密码学闭环。**

## A.2 stale reuse 权重方向的修正

MAFSA 原始定义 `α_i=exp(−β_c·E_i)` 作用于 fresh model，导致版本间隔 `E_i` 越大、stale 权重 `1−α_i` 反而越大。考虑到间隔越大陈旧结果与当前优化方向偏差风险越高，本文将指数衰减量改为直接控制 stale 项并加 `ρ_max≤0.5` 上界，保证新结果主导。故表述为"受 MAFSA 启发的时效性修正"而非"直接采用 MAFSA 加权公式"。

## A.3 为什么 VCM 用 S_k 而非 n_k

`W_sum^raw` 本身处在"样本量加权和"量纲中，组间补偿项必须同量纲才能与最终 `S_Q` 归一化一致。故 `C_k=S_k·γ_vcm·G_k`，而非按客户端个数 `n_k`。`n_k` 仅承担在线计数与最小在线人数约束。

## A.4 为什么不引入解掩码前的组间连续非等权系数

曾考虑 `λ_g=λ_0/(1+β_g·s_g)` 这类陈旧度加权。但一旦不同组在解掩码前后带不同连续权重，会改变 TA 掩码和与 LHH 的对应关系，破坏研究点1的验证链路。故主协议只保留解掩码后、LHH 验证后的 Direct-VCM 公开确定性补偿。

## A.5 为什么去掉降权路径

DVAFL 是"补偿+降权"双机制（straggler 既被降权又被补偿）。本文在 masked group package 约束下无法对单个组包做解掩码前的连续降权，故只保留补偿，用 `s_max` 硬截断 + 补偿替代软降权。这使系统对 `s_max` 取值更敏感，故 `s_max∈{2,3,5}` 列为必做消融。

## A.6 为什么样本量权重只乘明文不乘掩码

客户端上传 `w̄_i^k=s_i^k·w_i^k+m_i^k`。若 `s_i^k` 也乘掩码，TA 求掩码和时就需额外按样本量加权，改变单TA掩码求和方式。只乘明文可让解掩码后直接恢复 strict sample-weighted sum，同时不改变 TA 按参与集合求掩码和的链路。

## A.7 全局模型为何是共识量而非 Cloud 分发量

`W(t+1)` 由参与 checkpoint 的 ES 基于已验证 raw sum **本地独立重构**（公开确定性计算），各 ES 算得一致即为共识。Cloud 即使广播 `W(t+1)` 也仅作参考，规范结果以 ES 本地派生为准。因此 Cloud 无法在补偿/归一化阶段注入篡改——补偿输入（历史全局模型）是共识量，不作攻击面。

## A.8 显式出界的考量

下列方向有价值但会稀释主线，统一出界：完全异步、解掩码前连续陈旧度系数 `β_g`、双TA/门限TA、全局异步轨迹在线审计、anti-censorship 可审计调度、First/Second-order VCM、组级质量追踪、HGA-FL 式 bias correction。出界是为让研究点2 保持"一个能讲圆的协议扩展"而非大拼盘。

---

## 附：全文一致性自检

- [x] 研究点2 公式与 `STALENESS_AWARE_REUSE.md` 一致（stale/VCM/验证/术语）
- [x] 无 "strict FedAvg" 残留（仅组内 strict sample-weighted sum，最终 FedAvg-form）
- [x] 威胁模型 3.2 统一声明，P1/P2 适用范围标注
- [x] 记号表全文唯一
- [x] 实验数值全部留空，仅框架+预期趋势
- [x] 创新点 1.3 统一编号，两章不重复计数
