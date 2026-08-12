# 面向分层联邦学习的单 TA 可验证安全聚合及其检查点式半异步扩展

> 本文档为研究点 1（HVSA）与研究点 2（检查点式组级半异步）的整合硕士论文设计文档。
> 实验部分仅给出设计框架与预期趋势，所有数值留空待回填。
> 研究点 2 协议公式与 `THESIS_POINT2_STALENESS_AWARE_REUSE.md`（已定稿）一致；实验设计在其基础上扩展（如 reuse×compensation 2×2 消融），属框架性扩充而非协议改动。

## 论文设计与数值实现的评审边界

本文档当前是硕士论文的**理论方法与实验 workflow 框架**，评估重点是研究问题是否成立、协议流程与公式是否闭合、安全性/隐私性/可验证性分析是否正确，以及实验设计能否支撑论文结论。本文档不是最终的数值计算规范或代码实现说明。

- **论文理论方法层**：关注协议输入输出、参与实体、信任与威胁模型、公式之间的逻辑关系、密码学原语所需性质、定理假设与结论，以及研究点一和研究点二能否形成递进的论文主线。
- **数值实现层**：模型参数由浮点数映射到 LHH 或掩码运算所需整数/有限域、量化尺度、浮点误差、数据类型、模运算、溢出边界、序列化方式及精确计算流程等，属于代码与实验实现需要解决的问题，可在实现、测试和实验复现材料中具体设计。
- **理论抽象约定**：论文协议公式默认存在正确的 `Encode/Quantize` 与 `Decode/Dequantize` 映射，使模型向量进入密码学原语要求的代数域，并默认公式中的加法、标量乘法、掩码抵消与 LHH 同态运算在该抽象域中正确成立。具体采用 `float32/float64`、`int64`、大整数或某一有限域，不作为当前论文 workflow 是否成立的首要判断标准。
- **边界例外**：若某个实现选择会改变论文所依赖的理论性质，例如使同态等式本身不成立、改变安全假设或导致协议输入超出原语定义域，则需在最终实现说明中补充必要条件与验证；但在当前设计阶段，不以具体数据类型或量化误差替代理论方法本身的正确性评估。

---

## 目录

- 中文摘要
- 英文摘要（Abstract）
- 第 1 章 绪论
- 第 2 章 相关技术与预备知识
- 第 3 章 单 TA 可验证安全聚合协议 HVSA
- 第 4 章 检查点式组级半异步安全聚合扩展
- 第 5 章 总结与展望
- 参考文献
- 附录 A 设计决策记录

---

# 中文摘要

分层联邦学习（Hierarchical Federated Learning）在"云—边—端"架构下兼顾了隐私与通信效率，但仍面临两大痛点：一是客户端本地模型在上传与聚合过程中可能泄露隐私（梯度/参数泄露攻击），二是云端聚合结果可能被不可信云服务器篡改而难以被参与方检测。本文围绕**单可信权威机构（TA）约束下的轻量可验证安全聚合**展开，提出两个递进的研究点。

研究点一提出 **HVSA**（Hierarchical Verifiable Secure Aggregation），在"云—边—端—单 TA"架构下，通过 PRG 掩码保护客户端参数隐私，利用线性同态哈希（LHH）与 Ed25519 签名让边缘服务器（ES）能够对云端聚合结果进行高效验证，验证通信开销与模型维度无关。理论分析表明，在 PRG 伪随机性、n_min≥2 与 TA 不合谋假设下，HVSA 保证客户端参数隐私不可恢复；在 LHH 抗碰撞与 Ed25519 EUF-CMA 假设下，云端任何篡改均可被边缘确定性检测；同时协议对客户端掉线具有自然鲁棒性，掉线用户的掩码与更新自动成对剔除，不破坏验证等式。

研究点二在研究点一的密码学闭环基础上，针对同步聚合的"全局等待"与"掉线训练浪费"问题，提出**检查点式组级半异步安全聚合**。通过组内 stale reuse 复用客户端掉线未上传的训练结果、以 masked group package 进入缓冲区、按 checkpoint 批量解掩码聚合，并引入 Direct-VCM 组级后置补偿，在不破坏单 TA 掩码与 LHH 验证链路的前提下，将跨组训练组织从"全组同步 barrier"扩展为"组包缓冲 + checkpoint 触发"的组级半异步机制。安全分析将四类 Cloud 篡改攻击归约到 LHH 抗碰撞与 Ed25519 EUF-CMA，并证明 stale reuse 与 Direct-VCM 补偿不削弱研究点一的隐私界；在 IID 默认条件下，进一步给出当前协议的条件性单 checkpoint 偏差上界 `ρ_max·(1+ηL)·E_max·ηB + (|1−γ_vcm|+ηL)·s_max·ηB`。该上界是本文独立的偏差刻画，不直接继承参考论文的完整收敛定理。

**关键词**：联邦学习；分层聚合；安全聚合；可验证聚合；半异步；陈旧性复用

---

# Abstract

Hierarchical Federated Learning balances privacy and communication efficiency under a "cloud-edge-end" architecture, yet it still faces two major pain points: (i) client-side local models may leak privacy during upload and aggregation (gradient/parameter inversion attacks); (ii) aggregation results may be tampered with by an untrusted cloud server and such tampering is hard for participants to detect. This thesis focuses on **lightweight verifiable secure aggregation under a single Trusted Authority (TA)** and proposes two progressive research points.

Research Point 1 proposes **HVSA** (Hierarchical Verifiable Secure Aggregation). Under a "cloud-edge-end-single TA" architecture, HVSA protects client parameter privacy via PRG masking, and enables Edge Servers (ES) to efficiently verify cloud aggregation results via Linear Homomorphic Hashing (LHH) and Ed25519 signatures, with verification communication overhead independent of model dimension. Theoretical analysis shows that under PRG pseudorandomness, n_min≥2, and TA non-collusion assumptions, HVSA guarantees client parameter privacy; under LHH collision resistance and Ed25519 EUF-CMA assumptions, any cloud tampering is deterministically detectable by the edge; meanwhile, the protocol is naturally robust to client dropout — masks and updates of dropped clients are removed in pairs without breaking the verification equation.

Research Point 2, building on the cryptographic closure of Research Point 1, addresses the "global wait" and "dropout training waste" problems of synchronous aggregation by proposing **Checkpoint-based Group-level Semi-Asynchronous Secure Aggregation**. It reuses client dropout-unuploaded training results via intra-group stale reuse, buffers masked group packages, performs batch unmasking aggregation per checkpoint, and introduces Direct-VCM group-level post-compensation — extending cross-group training from a "global synchronous barrier" to a "package buffer + checkpoint trigger" group-level semi-asynchronous mechanism without breaking the single-TA masking and LHH verification chain. Security analysis reduces four classes of cloud tampering attacks to LHH collision resistance and Ed25519 EUF-CMA, and proves that stale reuse and Direct-VCM compensation do not weaken the privacy bound of Research Point 1. Under the default IID condition, the thesis gives a conditional single-checkpoint bias bound, `ρ_max·(1+ηL)·E_max·ηB + (|1−γ_vcm|+ηL)·s_max·ηB`, for the present protocol; this is an original bias characterization and does not directly inherit the complete convergence theorem of either reference paper.

**Keywords**: Federated Learning; Hierarchical Aggregation; Secure Aggregation; Verifiable Aggregation; Semi-Asynchronous; Staleness Reuse

---

# 第 1 章 绪论

## 1.1 研究背景与意义

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协同训练全局模型。然而两条研究线揭示了其脆弱性：

- **隐私泄露**：DLG 等梯度反演攻击表明，服务器可从客户端上传的明文梯度/参数中重建训练数据。
- **结果不可信**：云端作为聚合方，可能出于节省算力或恶意目的篡改聚合结果，而客户端/边缘难以检测。

分层联邦学习引入边缘服务器（ES）作为中间层，缓解了云端通信压力，但也使"掩码—聚合—验证"的信任链更复杂。在此背景下，如何在单 TA 约束下同时实现隐私保护、可验证聚合与对异构性的容忍，具有重要的理论价值与工程意义。

## 1.2 研究现状

### 1.2.1 安全聚合与可验证聚合

掩码式安全聚合（如 Bonawitz 等）通过成对/种子掩码隐藏单个客户端更新，多 TA/门限方案进一步分散信任但引入额外协调开销。VerifyNet、VeriFL、VLMRS、GPVO-FL 等通过同态哈希/签名/承诺实现聚合结果可验证，但多数方案验证开销随客户端数或维度增长，且与单 TA 架构不兼容。

### 1.2.2 分层联邦学习

云—边—端三层结构下，边缘聚合降低云端压力。FedFleet 等工作引入性能感知分组以加速异构 IoT 收敛，但通常不涉及可验证安全聚合。本文在其上叠加可验证安全聚合与单 TA 信任假设。

### 1.2.3 异步与半异步联邦学习

- **FedBuff**：缓冲异步聚合，单客户端粒度。
- **MAFSA**：多层异步 + 掉线客户端新旧结果融合，但其权重方向使越旧 stale 权重越大。
- **DVAFL**（Chen et al., IEEE TMC 2026）：版本控制机制（VCM）补偿陈旧更新，采用"补偿+降权"双机制。

完全异步与"逐轮在线验证"的密码学闭环存在根本冲突，现有工作未解决在单 TA + LHH 验证链路下引入异步性的问题。

## 1.3 本文主要贡献

本文贡献体现为**单 TA + 分层架构 + 边缘验证 + checkpoint 组级半异步**的组合与对应威胁模型，按两个研究点统一编号：

**研究点一（HVSA）：**
- 1-a 单 TA 分层可验证安全聚合架构：单 TA 即可支持掩码分发与聚合掩码计算，无需多 TA/门限 TA。
- 1-b 在单 TA 约束下首次将 LHH + Ed25519 验证外包到边缘：验证式 `H(W_sum^raw)·H(M_global)=∏_k H_k`，验证通信与模型维度无关。
- 1-c 验证外包到边缘：客户端零验证开销，验证由 ES 承担。

**研究点二（检查点半异步扩展）：**
- 2-a 在单 TA + LHH 验证链路下首次引入 checkpoint 级组级半异步：借鉴 FedBuff 缓冲思想提升至组包粒度，将跨组同步 barrier 扩展为 buffer+checkpoint 半异步。
- 2-b 安全聚合兼容的组内 stale reuse：复用客户端掉线未上传结果，权重 `ρ_i=ρ_max·e^{−β_c·E_i}` 随陈旧度衰减（修正 MAFSA 方向），掩码前本地混合，不破坏验证链路。
- 2-c 面向 mixed-version 组包的公开确定性后置补偿（Direct-VCM）与 checkpoint 验证：`C_k=S_k·γ_vcm·(W(t)−W(t_k))`，ES 基于已验证 raw sum 本地重构最终模型。

## 1.4 论文结构安排

本文共分 5 章。第 1 章为绪论，介绍研究背景、现状与本文贡献。第 2 章介绍相关技术与预备知识，包括联邦学习基础、密码学原语及安全/可验证/异步聚合综述。第 3 章给出研究点一 HVSA 协议设计、安全分析与实验评估。第 4 章给出研究点二检查点式组级半异步扩展的协议设计、安全分析与实验评估。第 5 章总结全文并展望未来工作。附录 A 记录研究点二的关键设计决策动机。

---

# 第 2 章 相关技术与预备知识

## 2.1 联邦学习基础

### 2.1.1 FedAvg 与分层联邦学习

FedAvg 由 McMahan 等于 2017 年提出，客户端在本地数据上执行若干 epoch SGD 后上传模型更新，云端按客户端样本量加权平均得到新全局模型。分层联邦学习在客户端与云端之间引入边缘服务器（ES），ES 先对组内客户端更新做边缘聚合，再由云端做跨组聚合，以降低云端通信压力。

### 2.1.2 Non-IID 与异构性挑战

实际场景中客户端数据非独立同分布（Non-IID）、设备算力与通信资源异构，导致同步 FL 中快组等待慢组、掉线客户端训练结果被丢弃等问题。本文研究点二通过组级半异步与 stale reuse 缓解上述问题。

## 2.2 密码学预备

### 2.2.1 线性同态哈希 LHH

LHH 在 NIST P-256 上构造，逐维生成元由 TA 在 Setup 阶段生成 `a_i ←$ Z_q` 并计算 `g_i = a_i·G`，公开 `{g_i}`，`a_i` 由 TA 持有且不外泄（见 3.2.2）；满足加法同态性 `LHH.Add(H(a),H(b))=H(a+b)`；定长输出（~32B，与维度 `d` 无关）。在 P-256 离散对数困难（DLP）假设下抗碰撞。

### 2.2.2 Ed25519 数字签名

Ed25519 满足 EUF-CMA（存在性不可伪造）安全性，用于组包与聚合结果签名。公钥/私钥 32B/64B，签名 64B，验证开销与消息长度线性相关。

### 2.2.3 PRG 掩码与量化

PRG 基于 AES-CTR over HKDF-SHA256 派生密钥流，`PRG.mask_int(seed, user_id, round_token, dim)` 逐客户端逐轮派生整数掩码；在随机预言机模型下与均匀随机不可区分。理论公式默认模型参数已通过正确的 `Encode/Quantize` 映射进入掩码与 LHH 所需的代数域。当前代码实例采用 `QUANTIZE_SCALE=1e6` 与 `int64` 表示；其浮点映射、溢出边界和精确模运算属于数值实现层，在代码测试与实验复现说明中处理，不作为本文理论协议的额外创新点。

## 2.3 安全聚合综述

掩码式安全聚合通过成对/种子掩码隐藏单个客户端更新。Bonawitz 等提出 SecAgg，多 TA/门限方案进一步分散信任但引入额外协调开销。本文采用单 TA 以简化信任假设，代价是依赖 TA 不合谋（trusted-dealer 式标准前提）。

## 2.4 可验证聚合综述

VerifyNet、VeriFL、VLMRS、GPVO-FL 等通过同态哈希/签名/承诺实现聚合结果可验证。多数方案验证开销随客户端数或维度增长。本文将验证集中到边缘并利用 LHH 定长哈希降低开销，使验证通信与模型维度无关。

## 2.5 异步与半异步联邦学习综述

- **FedBuff**：缓冲异步聚合，单客户端粒度。
- **FedFleet**：性能感知分组，加速异构 IoT 收敛（本文仅借鉴其一次性分组作初始化优化）。
- **MAFSA**：多层异步 + 掉线客户端新旧结果融合（本文借鉴其复用动机但修正权重方向并限定 stale 来源）。
- **DVAFL**：版本控制机制（VCM）补偿陈旧更新（本文借鉴其 Direct-VCM 思想提升至组包粒度并去除降权路径）。

本章对两篇论文的理论继承边界与研究点 2 文档一致：MAFSA 支撑分层异步、stale-result reuse 和有界陈旧度分析思路；DVAFL 支撑版本失配与 Direct-VCM 补偿思路。当前方案改变了 stale 权重方向、补偿粒度、补偿位置以及 checkpoint 参与集合，因此不直接继承两篇论文的原始公式、系统模型或完整收敛定理。本文只在 IID 默认条件下对当前协议给出独立的条件性单 checkpoint 偏差界；Dirichlet Non-IID 作为实验条件，需额外的梯度异质性假设才能进行理论外推。

## 2.6 本章小结

| 方案 | 隐私 | 可验证 | 异步粒度 | 单 TA |
|------|------|--------|----------|------|
| FedBuff | 部分 | 否 | 客户端 | — |
| FedFleet | 部分 | 否 | 分组（初始化） | — |
| MAFSA | 部分 | 否 | 客户端/层 | — |
| DVAFL | 部分 | 否 | 客户端 | — |
| HVSA（点 1） | 是 | 是 | 同步 | 是 |
| 本文（点 2） | 是 | 是 | 组包/checkpoint | 是 |

本文在单 TA + 可验证的前提下，将异步粒度从"客户端/层"扩展到"组包/checkpoint"，填补了现有工作的空白。

---

# 第 3 章 单 TA 可验证安全聚合协议 HVSA

## 3.1 引言

本章针对研究点一问题——如何在单 TA 约束下既保护客户端参数隐私、又让边缘方能高效验证云端聚合结果、且验证开销不随模型维度膨胀——提出 HVSA 协议。设计目标为：客户端参数隐私、边缘可验证云端聚合、验证开销与维度无关、客户端零验证负担。

## 3.2 系统模型与威胁模型

### 3.2.1 系统模型

四元架构：

- **Cloud (CS)**：云端聚合服务器，不可信。
- **Edge Server (ES)**：`e_k`，边缘聚合与验证者。
- **User (u_i^k)**：客户端，持有本地数据 `D_i^k`，样本量 `s_i^k=|D_i^k|`。
- **Trusted Authority (TA)**：单可信权威机构，分发掩码种子、计算聚合掩码。

记号总表：

| 记号 | 含义 |
|------|------|
| `t` | 同步轮次（点 1）；点 2 扩展为 checkpoint 版本（见 4.3.1） |
| `u_i^k` | 组 `k` 中第 `i` 个客户端 |
| `s_i^k` | 客户端样本量 |
| 上横线（如 `w̄`） | 量化+掩码后的量 |
| `W(t)` | 版本 `t` 的全局模型 |
| `m_i^k` | 客户端掩码；`M_global` 全局掩码和 |

### 3.2.2 威胁模型

- **半诚实用户与 ES**：按协议执行，但可能好奇。
- **Malicious Cloud**：可篡改聚合结果、丢弃或重放组包。点 1 防 Cloud 篡改当轮同步聚合结果。
- **ES 非攻击者**：不将 ES 列为攻击者。
- **样本量自报**：`s_i^k` 为客户端自报，防虚报属客户端数据可信问题，出界。
- **单 TA 信任边界与不合谋假设**：TA 持有掩码种子、`{a_i}`（LHH 生成元秘密值，见 3.3.1）与在线集合，不接收 masked model（`w̄_i^k` 或 `W̄_k`）；Cloud/ES 持有 masked model 但无掩码。TA 不外泄 `{a_i}` 或掩码种子；隐私依赖 TA 与 Cloud/ES 职责分离，故假设 **TA 不与 Cloud 或 ES 合谋**（trusted-dealer 式安全聚合标准前提）。此假设为单 TA 轻量架构的固有代价；门限化 TA 以消除单点信任列为第 5 章展望，不在本文威胁模型内。

## 3.3 协议设计

### 3.3.1 初始化与密钥分发

TA 与每个用户经 X25519 ECDH + HKDF 协商共享种子 `_ta_seed`；TA 持有所有用户掩码种子与 ES 公钥列表。此外，TA 在 Setup 阶段为 LHH 生成逐维生成元：对 `i = 1..d`，随机选取 `a_i ←$ Z_q`，计算 `g_i = a_i·G`，公开参数 `pp_LHH = {g_1, ..., g_d}`；秘密值 `a_i` 由 TA 持有且不外泄（见 3.2.2）。

### 3.3.2 本地训练、量化与掩码上传

客户端 `u_i^k` 本地训练得 `w_i^k`，展平量化后加掩码：

```text
w̄_i^k(t) = Quantize(w_i^k) + m_i^k(t)
m_i^k(t) = PRG.mask_int(seed_i, u_i^k, t, d)
```

> 隐私前提：当组内（或全局）在线客户端数 `n_online ≥ 2` 时成立。若仅单个客户端在线，解掩码后的组/全局和即退化为该客户端明文模型；故协议要求 `n_min ≥ 2`（见 3.4.1 与 4.3.4）。

### 3.3.3 边缘聚合

ES `e_k` 求组内掩码和并计算 LHH 与签名：

```text
W̄_k(t) = Σ_i w̄_i^k(t)
H_k(t)  = LHH.Hash(pp, W̄_k(t))
σ_k     = SIG.sign(H_k(t))
```

上报 Cloud 与 TA（在线列表）。

### 3.3.4 云端聚合与 TA 解掩码

```text
W̄(t)      = Σ_k W̄_k(t)
M_global  = Σ_{u∈online} PRG.mask_int(...)   (TA 计算)
W_sum^raw = W̄(t) − M_global
W(t+1)    = W_sum^raw / n_online             (简单平均，n_online 为在线客户端数)
```

> 注：点 1 采用等权平均作为可验证聚合的简化基线，点 2 升级样本量加权以匹配 FedAvg-form；两点口径差异在 4.5 实验中通过分阶段对比处理。

### 3.3.5 边缘验证

ES 验证云端返回的 `W_sum^raw`：

```text
H(W_sum^raw) · H(M_global) = ∏_k H_k(t)
```

成立则聚合结果可信。验证通信为定长哈希，与 `d` 无关。

### 3.3.6 协议执行伪代码

本节给出 HVSA 单轮同步执行的五阶段伪代码，作为 3.3.1–3.3.5 各步骤的紧凑汇总。各实体的内部状态变量记号与第 2 章一致：用户 `u_i^k` 持有与 TA 协商的种子 `s_i^k`、本地模型 `w_i^k`、量化掩码向量 `w̄_i^k`；ES `e_k` 持有签名私钥 `sk_k`、LHH 公共参数 `pp_LHH`；TA 持有所有用户种子 `{s_i^k}` 与 ES 公钥列表 `{pk_k}`。

#### Phase 1: 系统初始化

```text
Setup(dim d, edge 数 M, 各 edge 用户数 N_k):
    TA:
        (pk_TA, sk_TA) ← KA.gen()                      # X25519
        pp_LHH ← LHH.Gen(d)                            # P-256 上 dim 个生成元
        es_pub_keys ← {}
    for k = 1..M:
        (pk_k, sk_k) ← SIG.gen()                       # Ed25519
        TA.es_pub_keys[e_k.id] ← pk_k
        ES[e_k.id].sk_k ← sk_k
        ES[e_k.id].pp_LHH ← pp_LHH
    for k = 1..M, for i = 1..N_k:
        (pk_i^k, sk_i^k) ← KA.gen()
        s_i^k ← KA.agree(sk_TA, pk_i^k)               # 双方一致 32B 种子
        TA._user_secrets[u_i^k.id] ← s_i^k
        User[u_i^k].seed ← KA.agree(sk_i^k, pk_TA)
    publish: pk_TA, pp_LHH, {pk_k}, {pk_i^k}
```

#### Phase 2: 本地训练、量化与掩码上传

```text
LocalTrainAndUpload(u_i^k, t, W(t), D_i^k):
    w_i^k ← SGD_train(W(t), D_i^k, local_epochs, η)  # 明文本地模型
    w_int  ← Quantize(w_i^k)                          # flat · 1e6, int64
    m_i^k(t) ← PRG.mask_int(seed=s_i^k, user_id=u_i^k.id,
                            round_token=t, dim=d)     # AES-CTR over HKDF
    w̄_i^k(t) ← w_int + m_i^k(t)
    send (u_i^k.id, t, w̄_i^k(t)) → e_k                # 单条掩码向量
```

#### Phase 3: 边缘聚合、LHH 哈希与签名

```text
EdgeAggregateAndForward(e_k, t):
    U_online^k(t) ← { u_i^k : e_k 收到 (u_i^k.id, t, w̄_i^k(t)) }
    if |U_online^k(t)| < n_min:
        skip                                       # 不形成组包 (P2)
    W̄_k(t) ← Σ_{u∈U_online^k(t)} w̄_u(t)            # int64 加法
    H_k(t)  ← LHH.Hash(pp_LHH, W̄_k(t))             # P-256 点
    meta_k  ← (k, t, h_base(t), digest(U_online^k(t)),
                n_k(t), S_k(t), H_k(t))
    σ_k ← SIG.sign(pickle(meta_k), sk_k)           # Ed25519
    send (e_k.id, t, W̄_k(t), H_k(t), σ_k) → CS
    send (e_k.id, t, U_online^k(t)) → TA           # 在线列表 (用于 M_global)
```

#### Phase 4: TA 聚合掩码与云端解掩码

```text
TAGlobalMaskAndUnmask(t, TA, CS):
    # 4a: TA 收集所有 ES 上报的在线列表
    U_online(t) ← ⋃_{k=1..M} U_online^k(t)         # 全局在线集合
    if |U_online(t)| == 0:
        return FAIL                                 # 本轮无可用更新

    # 4b: TA 计算全局掩码 (按在线集合, 不按全注册集合)
    M_global(t) ← Σ_{u∈U_online(t)} PRG.mask_int(
                          TA._user_secrets[u.id], u.id, t, d)
    H_mask(t)   ← LHH.Hash(pp_LHH, M_global(t))
    send (t, M_global(t), |U_online(t)|) → CS

    # 4c: Cloud 解掩码 + 平均
    W̄(t)      ← Σ_{k=1..M} W̄_k(t)                  # int64
    W_sum^raw ← W̄(t) − M_global(t)                  # 整数明文和
    W(t+1)    ← W_sum^raw / |U_online(t)|           # 等权平均 (点1)
    broadcast (W_sum^raw, H_mask(t), { (k, H_k(t), σ_k) }) → ES
```

#### Phase 5: 边缘验证（含失败处理）

```text
EdgeVerify(e_k, t, W_sum^raw, H_mask(t), { (j, H_j(t), σ_j) }):
    # 5a: 签名验证 + RHS 聚合
    aggregated_rhs ← null
    for (j, H_j, σ_j) in {ES 元数据列表}:
        meta_j ← (j, t, h_base(t), digest(U_online^j(t)),
                   n_j(t), S_j(t), H_j)
        if not SIG.verify(pickle(meta_j), σ_j, TA.es_pub_keys[e_j.id]):
            return REJECT("ES 签名无效: " + e_j.id)
        aggregated_rhs ← LHH.Add(aggregated_rhs, H_j)

    # 5b: LHH 同态验证
    H_sum_raw ← LHH.Hash(pp_LHH, W_sum^raw)
    LHS ← LHH.Add(H_sum_raw, H_mask(t))
    if LHS ≠ aggregated_rhs:
        return REJECT("LHH 一致性等式不成立; Cloud 篡改嫌疑")

    # 5c: 通过 → 接受 W(t+1)
    return ACCEPT(W(t+1))
```

#### Dropout 处理子协议

HVSA 对客户端掉线的处理完全由 ES 上报在线集合驱动，TA 不需要预先知道哪些用户会在线：

```text
DropoutResilience(t):
    # 正常路径
    for k = 1..M:
        ES[e_k] 在 Phase 3 收集实际在线集合 U_online^k(t)
        # 掉线用户 u_d 的 (w̄_d, m_d) 不进入 W̄_k(t)
        # TA 在 Phase 4a 收到 U_online^k(t) 后, 只对在线用户求 M_global
        # ⇒ 掉线用户的掩码 m_d 也不出现在 M_global(t)
        # ⇒ W_sum^raw = Σ_k (Σ_{u∈online} w_u + m_u) − Σ_{u∈online} m_u
        #            = Σ_{u∈online} w_u                          (干净)

    # 关键不变量:
    #   ∀ u ∈ U_online(t): m_u 同时出现在 W̄(t) 与 M_global(t) 中
    #   ∀ u ∉ U_online(t): m_u 不出现在 W̄(t) 也不出现在 M_global(t)
    # ⇒ 验证等式 H_sum^raw · H_mask = ∏_k H_k 对任意 dropout 集合 D ⊆ U 仍成立
    #   (Theorem 3.3 形式化证明)
```

Dropout 处理不需要单独的密码学步骤，只需 ES→TA 的在线列表上报即可。该性质是 HVSA 单 TA + 集合化掩码求和结构的直接结果。

### 3.3.7 复杂度分析

| 实体 | 计算 | 通信 |
|------|------|------|
| 用户 | O(d) | O(d) |
| ES | O(n_k d) | O(n_k d) |
| Cloud | O(Md) | O(Md) |
| 验证(ES) | O(d) 哈希 + O(1) 通信 | O(1) |

## 3.4 安全分析

本章给出 HVSA 在第 3.2 节威胁模型下的三条核心定理：隐私性（Theorem 3.1）、验证可靠性（Theorem 3.2）与掉线鲁棒性（Theorem 3.3）。所有定理均以"安全参数 λ、可忽略函数 negl(λ)、概率多项式时间（PPT）敌手"的标准密码学框架表述；归约目标为本协议直接调用的原语（PRG、LHH、Ed25519、KA）。设掩码域宽度 `R = MASK_HIGH − MASK_LOW ≈ 2·10^9`，模 2^64 整数算术。

### 3.4.1 隐私性

> **Theorem 3.1（simulation-based 客户端参数隐私）.** 设 `n_online(t) ≥ n_min ≥ 2`，PRG 在随机预言机模型下为 CSPRNG，KA（X25519+HKDF-SHA256）在 CDH 假设下安全，TA 不与 Cloud 或 ES 合谋（见 3.2.2），且至多 `n_min−1` 个客户端与 Cloud 合谋。则存在 PPT 模拟器 `S`，给定 `(W_sum^raw, U_online(t), 公开元数据)`，输出模拟视图 `View_S`，使得任意 PPT 敌手 `A` 区分真实视图 `View_A` 与 `View_S` 的优势 `≤ negl(λ)`。

证明思路：构造模拟器 `S`，其仅知聚合结果 `W_sum^raw` 与在线集合，生成与真实协议计算不可区分的模拟视图；合谋阈值保证至少一个诚实客户端的掩码不可区分。

**Lemma 3.1（PRG 伪随机性）.** 对任意 PPT 区分器 `D`，以未协商种子 `s_i^k` 为视角，`{PRG.mask_int(s_i^k, u_i^k, t, d)}` 与均匀随机向量 `U ← Z_R^d` 的区分优势满足

```text
| Pr[D(m_i^k(t)) = 1] − Pr[D(U) = 1] | ≤ negl(λ).
```

证明：PRG 基于 HKDF-SHA256 派生密钥 + AES-CTR 生成密钥流，再经拒绝采样映射到 `Z_R`。在随机预言机模型下，HKDF 与 AES 均被视为随机预言机/理想密码，密钥流与均匀随机不可区分；拒绝采样在 `R | 2^64` 余项上接受概率 `1 − (2^64 mod R)/2^64 > 1 − R/2^64 > 1 − 2^−33`，引入的统计偏差 `≤ 2^−33`，可忽略。□

**Lemma 3.2（单客户端在线时隐私退化）.** 若 `|U_online(t)| = 1`，则存在 PPT 算法以概率 1 恢复该在线客户端的明文 `w_i^k`。

证明：由 3.3.4，`W_sum^raw = W̄(t) − M_global(t) = w̄_i^k − m_i^k = w_i^k`。Cloud 在 Phase 4c 直接获得 `W_sum^raw`，即该客户端明文。□

**Theorem 3.1 证明.** 构造模拟器 `S`：
- 输入：`W_sum^raw`, `U_online(t)`, `{n_k(t)}`, 公开参数 `pp`。
- 对每个在线客户端 `u_i^k`，独立均匀随机选取 `r_i^k ← Z_R^d`，设模拟掩码上传 `w̃̄_i^k = r_i^k`。
- 对每组 `k`，计算模拟组 LHH 哈希 `H̃_k = LHH.Hash(Σ_{i∈group_k} r_i^k)`。
- 由验证等式反推模拟全局掩码哈希 `H̃_mask = Σ_k H̃_k − LHH.Hash(W_sum^raw)`，输出模拟视图 `View_S = {w̃̄_i^k, H̃_k, H̃_mask, U_online(t)}`。

注意敌手 `(CS, ES)` 在真实协议中不直接接收 `M_global`（TA 仅释放 `H_mask = LHH.Hash(M_global)`），故 `S` 无需模拟 `M_global` 本身。由 Lemma 3.1，PRG 输出 `{m_i^k}` 与均匀随机 `{r_i^k}` 不可区分，故 `View_S` 与真实协议视图 `View_A` 计算不可区分。

**合谋阈值讨论**：`n_min ≥ 2` 保证至多 `n_min−1` 个客户端可与 Cloud 合谋而不破坏隐私——剩余至少一个诚实客户端的掩码在 `M_global` 中保持随机性。若 `n_min−1` 以上客户端与 Cloud 合谋，则诚客户端掩码不足，`M_global` 可被完全抵消，隐私退化为 Lemma 3.2 情形。□

### 3.4.2 可验证性

> **Theorem 3.2（验证可靠性）.** 设 LHH 在 P-256 上抗碰撞（Lemma 3.3），Ed25519 满足 EUF-CMA（Lemma 3.4）。若 Cloud 在 Phase 4c 输出的 `W_sum^raw' ≠ W_sum^raw`（真实明文和）使得至少一个 ES 在 Phase 5 验证通过，则可构造 PPT 算法以不可忽略概率找到 LHH 碰撞或伪造 ES 签名。

**Lemma 3.3（LHH 抗碰撞）.** 设 `H: Z^d → G`（`G` 为 P-256 群）是 LHH 同态哈希函数。在 P-256 离散对数困难（DLP）假设下，对任意 PPT 算法 `A`，

```text
Pr[ (x, x') ← A(pp_LHH): x ≠ x' ∧ LHH.Hash(pp_LHH, x) = LHH.Hash(pp_LHH, x') ] ≤ negl(λ).
```

证明：LHH 形式为 `H(x) = Σ_{i=1..d} g_i·x_i`（加性群运算），其中 `g_i = a_i·G`，`a_i ←$ Z_q` 由 TA 在 Setup 阶段生成并保密（3.3.1）。若存在 `x ≠ x'` 使 `H(x) = H(x')`，令 `δ = x − x' ≠ 0`，则 `Σ g_i·δ_i = O`，即 `(Σ δ_i·a_i)·G = O`，从而 `Σ δ_i·a_i ≡ 0 (mod q)`。敌手仅知 `g_i = a_i·G` 而不知 `a_i`，由 DLP 假设，找到非零 `δ` 使 `Σ δ_i·a_i ≡ 0 (mod q)` 的概率 `≤ negl(λ)`。否则，可将敌手归约为 DLP 求解器：给定挑战 `(G, aG)`，设 `g_1 = aG`，其余 `g_i` 随机生成，则敌手输出的 `δ` 满足 `a·δ_1 ≡ −Σ_{i≥2} δ_i·a_i (mod q)`，可解出 `a`，与 DLP 矛盾。□

**Lemma 3.4（Ed25519 EUF-CMA）.** 在 Ed25519 标准假设下，任意 PPT 敌手即使可访问签名预言机，也不能以不可忽略概率对未签名消息 `m` 输出通过验证的签名 `(m, σ)`。

证明：标准 Ed25519 安全性结果，略。□

**Theorem 3.2 证明.** 设 Cloud 输出 `W_sum^raw' ≠ W_sum^raw` 使 ES 验证通过。考虑验证等式：

```text
LHH.Hash(W_sum^raw') + H_mask(t) = Π_k H_k(t).                    (*)
```

真实等式（按协议执行）：

```text
LHH.Hash(W_sum^raw) + H_mask(t) = Π_k H_k(t).                    (**)
```

由 (**) 减 (*) 得 `LHH.Hash(W_sum^raw) − LHH.Hash(W_sum^raw') = 0`，即 `LHH.Hash(W_sum^raw) = LHH.Hash(W_sum^raw')`。由 `W_sum^raw' ≠ W_sum^raw`，这给出 LHH 碰撞，与 Lemma 3.3 矛盾。故 Cloud 无法在 LHH 验证下篡改 `W_sum^raw` 而不被检测。

至于组包伪造：若 Cloud 替换某 `H_k(t)` 为 `H_k'(t)` 但仍声称来自 ES `e_k`，需对应签名 `σ_k'` 通过 `SIG.verify(meta_k', σ_k', pk_k)`。由 `meta_k' ≠ meta_k`（因 `H_k' ≠ H_k`），`σ_k'` 是对未签名消息 `meta_k'` 的有效签名，与 Lemma 3.4 矛盾。□

### 3.4.3 掉线鲁棒性

> **Theorem 3.3（掉线鲁棒性）.** 设 `D(t) ⊆ U` 为任意掉线集合，`U_online(t) = U \ D(t)`，且 `|U_online(t)| ≥ n_min ≥ 2`。TA 在 Phase 4a 收到所有 ES 上报的 `{U_online^k(t)}` 后重构 `M_global(t)`，则：
> (a) 协议仍正确执行，最终输出 `W(t+1) = (1/|U_online(t)|) Σ_{u∈online} w_u`；
> (b) 验证等式 `H_sum^raw · H_mask = Π_k H_k` 对该 dropout 集合仍成立；
> (c) 掉线用户的更新 `w_d`（`d ∈ D(t)`）不进入 `W_sum^raw`，也不影响 `M_global` 的计算。

证明：关键不变量：对任意用户 `u`，

```text
m_u(t) 同时出现在 W̄(t) 中   ⇔   u ∈ U_online(t)   ⇔   m_u(t) 出现在 M_global(t) 中.
```

(⇒) 由 Phase 3，`W̄_k(t) = Σ_{u∈U_online^k(t)} w̄_u(t) = Σ_{u∈online} (w_u + m_u)`；由 Phase 4，`M_global(t) = Σ_{u∈U_online(t)} m_u`，其中 `U_online(t) = ⋃_k U_online^k(t)`。故

```text
W_sum^raw = W̄(t) − M_global(t)
          = Σ_{u∈online} w_u + Σ_{u∈online} m_u − Σ_{u∈online} m_u
          = Σ_{u∈online} w_u,
```

即掉线用户的 `w_d` 不出现，`m_d` 也不出现。这给出 (a) 与 (c)。

(b) 由 LHH 同态性，
```text
LHH.Hash(W̄(t)) = LHH.Hash(Σ_k W̄_k(t)) = LHH.Add_k(LHH.Hash(W̄_k(t))) = LHH.Add_k(H_k(t)) = Π_k H_k(t),
LHH.Hash(W_sum^raw) + LHH.Hash(M_global) = LHH.Hash(W_sum^raw + M_global) = LHH.Hash(W̄(t)) = Π_k H_k(t),
```
即 `H_sum^raw · H_mask = Π_k H_k` 成立，与 `D(t)` 的具体选择无关。□

### 3.4.4 局限性分析

Theorem 3.1–3.3 在同步聚合语义下完整，但留下三个工程性问题：

- **全组同步 barrier**：所有 ES 必须在 Phase 4c 前到齐，墙钟时间受最慢 ES 制约。
- **掉线训练浪费**：客户端完成 Phase 2 训练但在 Phase 3 截止前掉线，结果被丢弃（虽然 Theorem 3.3 保证协议正确性，但训练算力浪费）。
- **逐轮在线验证难以推广到完全异步**：每次全局下发都需 TA 在线求 `M_global`，与"来一个更新一个"的纯异步模式冲突。

研究点 2 通过把异步边界收缩到 checkpoint 级，在不破坏 Theorem 3.1–3.3 链路的前提下引入组级半异步（见第 4 章）。

## 3.5 实验评估

### 3.5.1 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | ___ |
| 数据集 | MNIST、CIFAR-10 |
| 模型 | MLP、CNN |
| 聚合 | FedAvg-form |
| 层级 | 1 Cloud + M Edge + N Users |
| 密码 | LHH(P-256) / Ed25519 / PRG(AES-CTR/HKDF) / QUANTIZE_SCALE=1e6 |

### 3.5.2 IID 收敛性 E2

在不同客户端数与 edge 数配置下，对比 HVSA 与 FedAvg 在 IID 数据划分下的全局模型收敛曲线。预期 HVSA 收敛精度与 FedAvg 持平，验证掩码+LHH 不破坏聚合正确性。

- **指标**：MNIST/CIFAR-10 上的测试准确率随轮次变化曲线、达到目标精度所需轮次。
- **数值**：___（留空待回填）

### 3.5.3 Non-IID 鲁棒性 N1

在 Dirichlet α=0.5/1.0 划分下重复 E2，验证 HVSA 在 Non-IID 下的鲁棒性。

- **指标**：不同 α 下的最终准确率与收敛轮次。
- **数值**：___（留空待回填）

### 3.5.4 DLG 防御 N2

对客户端上传的 `w̄_i^k` 与解掩码后的 `w_i^k` 分别执行 DLG 梯度反演攻击，对比重建图像与原始图像的 PSNR/SSIM。

- **指标**：PSNR（越高越接近原始）、SSIM（越高越相似）。
- **预期**：masked model 上 DLG 失败（PSNR≈0、SSIM≈0），明文上 DLG 成功。
- **数值**：___（留空待回填）

### 3.5.5 参数分布一致性 N7

统计客户端上传 `w̄_i^k` 在各维度的取值分布，验证掩码后向量服从均匀分布、与明文分布显著不同。

- **指标**：各维度直方图、KL 散度。
- **数值**：___（留空待回填）

### 3.5.6 开销对比与验证开销 E4+N3+N5

**(a) 总开销对比 E4**：与不加密 FedAvg、差分隐私方案、其他安全聚合方案对比 HVSA 的计算/通信总开销。

**(b) 扩展开销基线 N3**：补入 VerifyNet、VeriFL、VLMRS、GPVO-FL 等可验证聚合基线的开销对比。

**(c) 验证开销随维度变化 N5**：固定客户端数，扫描模型维度 `d`，验证 HVSA 验证通信开销与 `d` 无关（定长 LHH 哈希）。

- **指标**：用户/ES/Cloud 计算时间、通信字节数、验证通信字节数 vs `d`。
- **预期**：HVSA 验证通信为定长 32B，与 `d` 无关；基线方案验证开销随 `d` 线性增长。
- **数值**：___（留空待回填）

## 3.6 本章小结

本章提出 HVSA——单 TA 分层可验证安全聚合协议。协议通过 PRG 掩码保护客户端参数隐私（Theorem 3.1），通过 LHH 同态哈希与 Ed25519 签名让 ES 以定长通信开销验证云端聚合结果（Theorem 3.2），并通过"在线集合驱动的掩码求和"实现对客户端掉线的自然鲁棒性（Theorem 3.3）。实验框架覆盖 IID/Non-IID 收敛、DLG 防御、参数分布一致性、开销对比与验证开销维度无关性五个方面。HVSA 的同步聚合语义留下"全组同步 barrier"与"掉线训练浪费"两个工程问题，下一章将以检查点式组级半异步扩展解决。

---

# 第 4 章 检查点式组级半异步安全聚合扩展

## 4.1 引言

第 3.4.4 节指出 HVSA 的同步语义留下三个工程问题：

1. **(i) 全组同步 barrier**——所有 ES 必须在 Phase 4c 前到齐，墙钟时间受最慢 ES 制约；
2. **(ii) 掉线训练浪费**——客户端完成 Phase 2 训练但未赶上封组的更新被直接丢弃（Theorem 3.3 仅保证协议正确，不回收训练算力）；
3. **(iii) 逐轮在线验证难以推广到完全异步**——每次全局下发都需 TA 在线求 `M_global`，与"来一个更新一个"的纯异步模式冲突。

一个直接的思路是引入完全异步（FedBuff/MAFSA/DVAFL 式），但问题 (iii) 表明该路线与单 TA + LHH 的逐轮验证闭环**根本冲突**：验证粒度必须与掩码聚合粒度一致，完全异步下不存在可验证的"聚合结果"。换言之，不能把现有异步方案直接搬进本架构，必须先回答"异步边界放哪里验证链路才不破"。

**核心论点：主动把异步边界收缩到 checkpoint 级。** 组内仍同步封组，组间用 buffer+checkpoint 半异步；Cloud 保存组级掩码包而非明文，验证粒度收缩到 checkpoint 选中集合。这一洞见同时给出三个问题的解：

- **问题 (i) → buffer+checkpoint 半异步（创新点 1）**：跨组同步 barrier 扩展为"组包缓冲 + checkpoint 触发"，验证粒度收缩到 checkpoint 选中集合。
- **问题 (ii) → 组内 stale reuse（创新点 2）**：复用掉线客户端未上传的训练结果；为不破坏单 TA 掩码链路，混合必须在掩码前由客户端本地完成（4.3.3）。
- **问题 (iii) → checkpoint 级验证粒度**：验证对象从"单轮全局和"变为"checkpoint 选中集合的 raw sum"，TA 在线求掩码的频率从"逐更新"降到"每 checkpoint 一次"（4.3.5 按选中集合 `Q(t)` 释放 `M_Q`），仍保持 LHH 定长验证。

异步边界收缩本身引入一个新后果：选中集合中组包基于不同基版本 `W(t_k)`（mixed-version），需要 **Direct-VCM 组级后置补偿（创新点 3）** 在解掩码后以公开确定性方式修正，并维持 checkpoint 级验证。因此三个机制不是并列拼装，而是同一架构决策（checkpoint 级异步边界）依次引发的三个必要组件（见附录 A 的组织逻辑）。

本章创新点（措辞定稿）：

- **创新点 1**：单 TA 可验证安全聚合下的检查点式组级半异步扩展。
- **创新点 2**：安全聚合兼容的组内 stale reuse 机制（修正 MAFSA 方向）。
- **创新点 3**：面向 mixed-version 组包的公开确定性后置补偿与 checkpoint 验证机制。

## 4.2 系统模型扩展与威胁模型

### 4.2.1 系统模型扩展

继承第 3 章四元架构（Cloud / ES / User / TA），新增组包缓冲区与 checkpoint 触发机制。Cloud 维护组包缓冲区 `B`，每组至多一个在途包；按 checkpoint 触发规则（4.3.6）选择集合 `Q(t)` 进行批量解掩码与聚合。

### 4.2.2 威胁模型扩展

继承 3.2.2 全部假设，并扩展：

- **Malicious Cloud（点 2 适用）**：可篡改聚合结果、丢弃或重放组包。防 Cloud 篡改 checkpoint **选中**的聚合结果（`W_sum^raw`）。
- **安全目标与显式出界（点 2）**：主协议只提供 checkpoint 选中聚合结果的完整性检测；检测到攻击或不一致即丢弃本次 checkpoint，不推进全局版本。它不保证恶意 Cloud 下的训练可用性，也不提供 anti-censorship 保证；manifest 漏选、候选组包静默丢弃和调度审查均不属于主协议安全目标。全局模型由 ES 基于已验证结果本地确定性重构并作为共识量维护，不把 Cloud 分发的模型作为规范结果。
- **exactly-once 语义声明**：每个客户端更新与组包携带唯一标识，Cloud/TA 对已聚合组包拒绝重放；完整 ACK 握手/原子提交/缓存并发属实现完善项，不作主协议数学前提。本文进一步假设 `Aggregated_IDs` 由 TA 维护且仅追加，ES 对 `package_token` 的联合签名写入 TA 维护的 append-only log（TA 可信，见 3.2.2），以保证重放检测确定性。
- **单 TA 不合谋假设**：同 3.2.2，TA 不与 Cloud 或 ES 合谋。

## 4.3 协议设计

### 4.3.1 记号重定义

- `t`：当前最新已提交的 **checkpoint 全局版本**（扩展点 1 的轮次语义）。
- `t_k`：组包 `Π_k(t_k)` 的训练基版本。mixed-version checkpoint 中通常 `t_k<t`。
- 客户端级陈旧度 `E_i=t_k−t_i`；组级陈旧度 `s_k=t−t_k`。
- `S_k(t_k)`：组包 `Π_k` 的样本量和；`S_Q=Σ_{Π_k∈Q} S_k` checkpoint 总样本量。
- `M_k`/`M_Q`：组/checkpoint 掩码和（与点 1 `M_global` 区分）。
- `M`：边缘组数（仅作计数）。
- `γ_vcm`：Direct-VCM 补偿强度（唯一补偿超参）。
- `ρ_max, β_c`：stale reuse 权重上界与衰减速度。

### 4.3.2 初始化与一次性性能分组

按 `d_i=t_i^train+t_i^comm`（或仅通信时延）将客户端分为 `M` 组，各绑定一个 ES。借鉴 FedFleet，仅作初始化优化，不改变协议、不作创新点。

### 4.3.3 组内 stale reuse

本节针对 4.1 的问题 (ii)——**掉线训练浪费**。在同步 HVSA 中，客户端完成 Phase 2 训练但未赶上封组的更新被直接丢弃；本节以组内 stale reuse 复用该结果。关键约束是**安全聚合兼容**：混合必须发生在掩码之前、由客户端本地完成，且只上传一条 masked model，否则会在掩码层引入客户端间依赖、破坏单 TA 掩码与 LHH 验证链路（隐私保持见 Theorem 4.2 步骤 1）。这正是 stale reuse 与 MAFSA 等非安全设定下复用机制的本质差异——后者不承担"掩码前混合 + 只上传一条掩码模型"这一密码学约束。

客户端至多缓存一个"已完成训练但因掉线/上传超时未被 ES 接收"的 stale 候选 `w_i^{stale,k}`（产生版本 `t_i`）。**stale 来源唯一：仅客户端本地产生于单次未上传结果**，不含 MAFSA 掉线期间持续训练假设，也不由组包过期生成（4.3.9 已移除该路径）。恢复参与后：

```text
E_i  = t_k − t_i
ρ_i  = ρ_max · exp(−β_c · E_i),   0<ρ_max≤0.5
w_i^k = (1−ρ_i)·w_i^{new,k} + ρ_i·w_i^{stale,k}   (Ours Decayed-Stale)
```

权重方向修正 MAFSA（其 `α=e^{−βE}` 作用于 fresh，使越旧 stale 权重越大）。混合在掩码前本地完成，只上传一条 masked model。客户端级与组级以"ES 封组前是否接收"为界，上传被接收即清除 stale 候选。

### 4.3.4 掩码上传与组包生成

```text
w̄_i^k(t_k) = s_i^k·w_i^k + m_i^k(t_k)     (s_i^k 只乘明文，不乘掩码)
W̄_k(t_k)   = Σ_i w̄_i^k = Σ_i s_i^k·w_i^k + Σ_i m_i^k
S_k(t_k)   = Σ_i s_i^k
```

要求最小在线人数 `n_min≥2`（保护客户端级隐私）。组包：

```text
Π_k(t_k) = (k, t_k, h_base(t_k), U_online, n_k, S_k, W̄_k, H_k, σ_k)
```

其中 `σ_k` 为 ES 对最小元数据 `(k, t_k, h_base(t_k), digest(U_online), n_k, S_k, H_k)` 的 Ed25519 签名；该元数据即 4.3.8 Step 1 manifest 一致性检查的比对对象。

### 4.3.5 TA 的组包掩码支持

TA 按参与集合计算 `M_k=Σ m_i^k`，**不在组包入 buffer 时单独释放**；仅当 checkpoint 选中集合 `Q(t)` 确定后返回 `M_Q=Σ_{Π_k∈Q} M_k`。

### 4.3.6 Buffer 与 Checkpoint 触发

Cloud 维护组包缓冲区 `B`，每组至多一个在途包。过期判据 `s_k=t−t_k>s_max` 移除。

- 常规：`|B_legal|≥K` 时按到达顺序选最早 `K` 个。
- 超时：最早合法组包等待达 `Δ_timeout` 且 `1≤|B_legal|<K` 时，对全部合法组包保守 checkpoint。

发布 `checkpoint_manifest(t)`（含 mode、|Q|、各组包元数据）及公开摘要 `h_manifest`，先广播给 TA 与 ES。

可选调度增强（消融项，不改密码链路）：adaptive s_max、公平保底（`D_fair`）、选择策略（FIFO/Fresh-first/Fairness-aware）。

### 4.3.7 解掩码、Direct-VCM 补偿与全局更新

```text
W̄_Q(t)        = Σ_{Π_k∈Q} W̄_k
W_sum^raw(t)  = W̄_Q(t) − M_Q(t)               (样本量加权和)
G_k(t)        = W(t) − W(t_k)                  (t_k<t)
C_k(t)        = S_k(t_k)·γ_vcm·G_k(t)          (Direct-VCM 组级补偿)
W_sum^corr(t) = W_sum^raw(t) + Σ_{t_k<t} C_k(t)
W(t+1)        = W_sum^corr(t) / S_Q(t)         (S_Q=Σ S_k, FedAvg-form 归一化)
```

不引入解掩码前的组间连续非等权系数（避免破坏掩码+LHH）；不采用降权路径，用 s_max 硬截断 + 补偿。

### 4.3.8 Checkpoint 级验证与最终模型重构

本节给出 checkpoint 级验证与最终重构的完整伪代码。验证分为三步：manifest 一致性、LHH 等式、ES 本地确定性重构。三步任一失败则 checkpoint 终止并回滚，组包按 4.3.9 状态转移规则处理。

#### Step 1：manifest 一致性检查 + Ed25519 验签

Cloud 发布 `checkpoint_manifest(t)` 后，TA 与每个 ES（其组包在 Q 中）独立执行：

```text
VerifyManifest(checkpoint_manifest(t), Q, B):
  // 输入：checkpoint_manifest(t) = (t, mode, |Q|, {(k, t_k, h_base, U_online, n_k, S_k, H_k, σ_k)}_{Π_k∈Q})
  //       B = Cloud 维护的组包缓冲区
  accepted ← ∅
  for each Π_k ∈ Q:
    1a. // —— manifest 一致性 ——
        从 B 中按 (k, t_k, h_base) 检索到唯一组包 Π_k^B
        若检索失败（B 中无对应组包 / 多条匹配）→ reject Π_k
        逐字段比对 manifest 元数据与 Π_k^B 的元数据：
          U_online、n_k、S_k、H_k 必须完全一致
        若任一字段不一致 → reject Π_k            // tamper_manifest
    1b. // —— Ed25519 验签 ——
        pk_ES_k ← 从 PKI 获取 ES_k 的 Ed25519 公钥
        meta_k ← (k, t_k, h_base(t_k), digest(U_online), n_k, S_k, H_k)
        ok ← Verify(pk_ES_k, meta_k, σ_k)
        若 ok = false → reject Π_k                // 伪造组包
    1c. // —— 时效性检查 ——
        若 t − t_k > s_max → reject Π_k           // 超期组包，4.3.6 过期规则
        accepted ← accepted ∪ {Π_k^B}
  若 |accepted| < |Q| → checkpoint 失败，回滚
  返回 accepted
```

`reject Π_k` 的语义：该组包不进入本轮聚合，但其状态转移由 4.3.9 处理（区分"manifest 不一致"导致的丢弃 vs "超期"导致的过期）。manifest 一致性检查是 tamper_manifest 与 drop_replace_pkg 两类攻击的第一道防线（见 4.4.1 Theorem 4.1）。

#### Step 2：LHH 等式验证

TA 释放 `M_Q` 后，ES 对每个组包本地计算 `H_k`（已知 `W̄_k` 与 `U_online`），并对 raw sum 计算 `H_sum^raw`：

```text
VerifyLHH(accepted, W_sum^raw, H_mask):
  // 输入：accepted = Step 1 通过的组包集合
  //       W_sum^raw = Σ_{Π_k∈accepted} (W̄_k − M_k)   (M_k 来自 TA 释放的 M_Q 分解)
  //       H_mask    = TA 公布的掩码聚合哈希
  H_prod ← identity()
  for each Π_k ∈ accepted:
    H_k ← LHH_Commit(W̄_k, U_online, pk_TA)        // 本地重算
    H_prod ← H_prod · H_k
  if H_sum^raw · H_mask == H_prod:
    return PASS
  else:
    return FAIL                                    // tamper_raw_sum
```

LHH 等式失败 ⇒ Cloud 篡改了 `W_sum^raw` 或某组包的 `W̄_k`（见 Theorem 4.1 归约 (i)）。

#### Step 3：ES 本地确定性重构 W(t+1)

ES 基于**已验证的 raw sum + 全开公开量**本地派生 `W(t+1)`，不信任 Cloud 广播值：

```text
ReconstructW(t+1, accepted, W_sum^raw, public_state):
  // public_state = (W(t), {(t_k, S_k) : Π_k ∈ accepted}, γ_vcm, S_Q)
  // —— Direct-VCM 补偿 ——
  W_sum^corr ← W_sum^raw
  for each Π_k ∈ accepted:
    G_k ← W(t) − W(t_k)                            // 历史全局模型差
    C_k ← S_k(t_k) · γ_vcm · G_k                   // 组级补偿
    W_sum^corr ← W_sum^corr + C_k
  // —— FedAvg-form 归一化 ——
  S_Q ← Σ_{Π_k∈accepted} S_k(t_k)
  W(t+1) ← W_sum^corr / S_Q
  // —— 下发 ——
  broadcast_to_group(W(t+1))                       // 仅下发本组客户端
  return W(t+1)
```

`public_state` 中所有量均为公开共识量：`W(t)`、`W(t_k)` 为历史全局模型（ES 持有），`S_k(t_k)` 在组包元数据中（已通过 manifest 一致性），`γ_vcm` 为协议超参。因此**补偿项 `C_k` 完全由公开量确定**，Cloud 无法注入一个独立的恶意补偿值；ES 本地派生的 `W(t+1)` 是规范结果，Cloud 广播的模型仅作参考。若系统同时接收 Cloud 广播结果，可将其与 ES 本地派生结果做确定性一致性检查，但该检查不构成新的 LHH 验证对象。

### 4.3.9 状态转移与 exactly-once

被选中组包移出 buffer、对应组获 `W(t+1)` 并开始下一轮；未选中组包继续等待或过期。客户端级/组级状态分界清晰，不反向生成 stale 候选；唯一 ID + 拒绝重放保证 exactly-once。

#### 唯一 ID 生成规则

每个组包 `Π_k(t_k)` 在入 buffer 时由 ES 分配全局唯一的 `package_token`：

```text
package_token = f"{edge_id}:{base_version}:{nonce}"
// edge_id      = ES 的全局唯一标识（PKI 注册）
// base_version = h_base(t_k)，本组本轮训练所基于的全局模型版本摘要
// nonce        = ES 维护的单调递增计数器，每次封组 +1（持久化）
```

`package_token` 作为组包元数据的一部分写入 manifest 与 buffer 索引。`base_version` 嵌入 ID 保证"基于同一全局模型版本的同组重复封组"不会被误认为重放；`nonce` 保证同一 ES 在同一 `base_version` 下的多次封组可区分。

#### Cloud 已聚合组包 ID 集合与去重逻辑

Cloud 维护一个持久化的已聚合组包 ID 集合 `Aggregated_IDs`，每次 checkpoint 完成后追加本轮选中组包的 `package_token`。组包入 buffer 与 checkpoint 选中时各做一次去重检查：

```text
CheckExactlyOnce(Π_k, buffer_phase, Aggregated_IDs):
  // buffer_phase ∈ {"on_enqueue", "on_select"}
  if Π_k.package_token ∈ Aggregated_IDs:
    return REJECT_REPLAY                            // 已聚合过，拒绝重放
  if buffer_phase == "on_enqueue":
    // 同 (edge_id, base_version) 在 buffer 中不应已存在
    // （ES 单组在途规则，4.3.6）
    for Π_k' ∈ B:
      if Π_k'.edge_id == Π_k.edge_id and Π_k'.base_version == Π_k.base_version:
        return REJECT_DUPLICATE_INFLIGHT
  return ACCEPT
```

#### Checkpoint 完成后的状态转移

```text
FinalizeCheckpoint(accepted, B, Aggregated_IDs, group_state):
  // —— 选中组包 ——
  for each Π_k ∈ accepted:
    B.remove(Π_k)
    Aggregated_IDs.add(Π_k.package_token)
    group_state[k].W_current ← W(t+1)               // 该组进入下一轮
    group_state[k].stale_candidate ← null            // 清除 stale 候选
  // —— 未选中组包 ——
  for each Π_k ∈ B \ accepted:
    if t − t_k > s_max:
      B.remove(Π_k)                                 // 过期丢弃
    else:
      keep Π_k in B                                 // 继续等待下一 checkpoint
```

`exactly-once` 由三重保证：(a) `package_token` 唯一性防止同一组包被聚合两次；(b) `Aggregated_IDs` 持久化防止跨 checkpoint 重放；(c) ES 单组在途规则 + `nonce` 单调性防止 ES 在同一 `base_version` 下重复封组。上述机制保证 stale reuse 不破坏 `exactly-once` 的组级语义（见 Theorem 4.2）。

## 4.4 安全分析

### 4.4.1 篡改检测归约

本节将第 4.3 节协议对 Cloud 四类篡改/重放攻击的“未检测却被接受”概率归约到第 3 章的密码原语安全性（LHH 抗碰撞、Ed25519 EUF-CMA）与本协议新引入的机制（manifest 一致性、唯一 ID 拒绝重放）。检测失败的协议语义是丢弃本次 checkpoint，不推进全局版本；本节不把 Cloud 的拒绝服务或候选组包漏选纳入安全保证。

#### 威胁模型与攻击面

Cloud 是 malicious（可篡改）实体，被允许执行以下四类篡改攻击：

| 攻击 | 操作 | 协议检测点 |
|------|------|------------|
| **tamper_raw_sum** | 篡改 `W_sum^raw` | 4.3.8 Step 2 LHH 等式 |
| **tamper_manifest** | 篡改组包元数据 `U_online/n_k/S_k/H_k` | 4.3.8 Step 1a manifest 一致性 + 1b Ed25519 |
| **drop_replace_pkg** | 丢弃合法组包并用伪造/重放组包替换 | 4.3.8 Step 1a + 1b |
| **replay** | 重放已聚合组包 | 4.3.9 `Aggregated_IDs` 去重 |

补偿项 `C_k` 由公开量唯一确定（4.3.8 Step 3），不是 Cloud 可自由提交的密码学输入。ES 在通过 `raw sum` 验证后本地重算 `W_sum^corr` 和 `W(t+1)`；若实验中同时接收 Cloud 广播的结果，可将不一致记录为确定性的重构不一致，但这不意味着 `W_sum^corr` 获得了独立的 LHH 证明。因此主定理的密码学对象是 `W_sum^raw`，补偿篡改属于公开确定性重构检查。

#### 主定理

**定理 4.1（Checkpoint Integrity Detection）.** *在以下假设下：*
- *(A1) LHH 是抗碰撞的哈希函数族（Lemma 3.3）；*
- *(A2) Ed25519 满足 EUF-CMA 安全性（Lemma 3.4）；*
- *(A3) PKI 中 ES 公钥绑定正确，且 ES 私钥未被攻破；*
- *(A4) `Aggregated_IDs` 由 TA 维护且仅追加（ES 联合签名，见 4.2.2），TA 可信（3.2.2）；*
- *(A5) TA 诚实执行掩码释放，不与 Cloud 合谋（3.2.2 假设）；*
- *(A6) `W(t)`、`W(t_k)` 及其版本/hash 由共识或已提交版本历史维护，参与验证的 ES 使用同一规范历史。*

*对上述四类 Cloud 篡改攻击，协议 4.3.8 + 4.3.9 使攻击者构造一个“未被检测却被接受”的 checkpoint 的成功概率关于安全参数 `λ` 是可忽略的；具体地，每类攻击的未检测成功概率被归约到对应原语的攻击优势。若检查失败，协议丢弃本次 checkpoint，不推进全局版本。该定理只讨论选中聚合结果的完整性，不讨论 Cloud 的拒绝服务、候选组包漏选或 anti-censorship。*

**证明.** 对四类攻击分别归约。

##### 归约 (i)：tamper_raw_sum → LHH 抗碰撞

设敌手 `A_raw` 以非可忽略优势 `ε_raw(λ)` 通过篡改 `W_sum^raw` 使 4.3.8 Step 2 LHH 等式仍成立。构造归约 `B_coll` 利用 `A_raw` 攻破 LHH 抗碰撞。

`B_coll` 挑战者给出 LHH 公钥 `pk_LHH`，`B_coll` 充当 `A_global` 的协议环境：正常生成所有组包 `{Π_k}`，按 4.3.7 计算 `W_sum^raw`（诚实的 raw sum，记 `W_sum^raw*`），并令 `H_sum^raw* = LHH_Commit(W_sum^raw*, U_global)`、`H_mask` 由 TA 诚实释放。`A_global` 输出篡改值 `W_sum^raw' ≠ W_sum^raw*`，要求其满足：

```text
H_sum^raw' · H_mask == ∏_{Π_k∈Q} H_k
```

其中 `H_sum^raw' = LHH_Commit(W_sum^raw', U_global)`。由诚实 raw sum 满足 `H_sum^raw* · H_mask == ∏ H_k`（协议正确性），两式相减得：

```text
H_sum^raw' · H_mask == H_sum^raw* · H_mask
⇒  H_sum^raw' == H_sum^raw*        (LHH 群上 H_mask 可消去)
```

即 `LHH_Commit(W_sum^raw', U_global) == LHH_Commit(W_sum^raw*, U_global)` 而 `W_sum^raw' ≠ W_sum^raw*`。这给出 LHH 的一个碰撞对 `(W_sum^raw', W_sum^raw*)`。`B_coll` 输出该对，挑战者验证通过即攻破 LHH 抗碰撞。

由 LHH 抗碰撞假设，存在该碰撞的概率 `≤ Adv_coll^{LHH}(λ)`，故：

```text
ε_raw(λ) ≤ Adv_coll^{LHH}(λ)
```

可忽略。**检测成功**：Step 2 LHH 等式失败 ⇒ checkpoint 被拒绝。

##### 归约 (ii)：tamper_manifest → Ed25519 EUF-CMA + manifest 一致性

设敌手 `A_manifest` 以非可忽略优势 `ε_manifest(λ)` 篡改组包元数据 `meta_k = (k, t_k, h_base, U_online, n_k, S_k, H_k)` 使 4.3.8 Step 1a manifest 一致性通过且 1b Ed25519 验签通过。

分两种情形：

**情形 (ii-a)**：`A_manifest` 篡改 buffer 中组包 `Π_k^B` 的元数据（与 ES 签名时的元数据不一致）。此时 manifest 一致性检查 1a 比较 manifest 中字段与 `Π_k^B` 中字段，至少一处不一致 ⇒ 立即 reject。`A_manifest` 若要绕过 1a，必须使 `Π_k^B` 的字段与 manifest 一致，即同步篡改 `Π_k^B` 的明文元数据；但 `Π_k^B` 的签名 `σ_k` 仍是对原始 `meta_k` 的签名。1b Ed25519 验签 `Verify(pk_ES_k, meta_k^B', σ_k)` 必然失败 ⇒ reject。

**情形 (ii-b)**：`A_manifest` 同时篡改 manifest、`Π_k^B` 元数据、签名 `σ_k`，使 1a 通过且 1b 通过。此时 `σ_k'` 是对篡改后 `meta_k'` 的有效签名，但 `A_manifest` 不持有 ES 私钥（A3）。构造归约 `B_euf`：以 `A_manifest` 为子程序攻破 Ed25519 EUF-CMA。`B_euf` 充当签名预言机：诚实签发所有 ES 在协议中产生的 `meta_k`；`A_manifest` 输出 `(meta_k', σ_k')` 其中 `meta_k'` 未被查询过（因 `meta_k' ≠ meta_k`，否则未篡改）。`B_euf` 输出该对作为 EUF-CMA 伪造。

由 Ed25519 EUF-CMA 假设：

```text
ε_manifest(λ) ≤ Adv_euf^{Ed25519}(λ)
```

可忽略。**检测成功**：1a 或 1b 任一失败 ⇒ checkpoint 被拒绝。

##### 归约 (iii)：drop_replace_pkg → Ed25519 EUF-CMA + manifest 一致性

设敌手 `A_drop` 丢弃合法组包 `Π_k*` 并以伪造或重放组包 `Π_k'` 替换。

**伪造子情形**：`Π_k'` 的签名 `σ_k'` 必须对 `meta_k'` 有效。归约同 (ii-b)，`A_drop` 不持有 ES 私钥 ⇒ 必须伪造 Ed25519 签名 ⇒ 归约到 `Adv_euf^{Ed25519}(λ)`。

**重放子情形**：`Π_k'` 来自历史 checkpoint（已聚合）。两种检测路径：
- 若 `Π_k'` 来自同一 `base_version` 的历史 checkpoint，其 `package_token` 已在 `Aggregated_IDs` 中 ⇒ 4.3.9 `CheckExactlyOnce` 在 `on_select` 阶段 reject。
- 若 `Π_k'` 来自不同 `base_version`，其 `h_base(t_k') ≠ h_base(t_k*)` ⇒ 4.3.8 Step 1a manifest 一致性检查（`h_base` 字段不匹配）reject。

由 (A4) `Aggregated_IDs` 不可篡改，重放子情形被确定性检测；伪造子情形被归约到 EUF-CMA。综合：

```text
ε_drop(λ) ≤ Adv_euf^{Ed25519}(λ) + 0
```

**检测成功**：1a 或 4.3.9 任一失败 ⇒ checkpoint 被拒绝。

##### 归约 (iv)：replay → 唯一 ID 拒绝

设敌手 `A_replay` 在 `t' > t` 的 checkpoint 中重放 `Π_k(t)`（已在 `t` 聚合）。由 4.3.9 `package_token` = `(edge_id, base_version, nonce)`，且 `nonce` 单调递增、`Aggregated_IDs` 仅追加，`Π_k(t).package_token ∈ Aggregated_IDs` 永久成立。

4.3.9 `CheckExactlyOnce` 在 `on_select` 阶段执行：

```text
if Π_k.package_token ∈ Aggregated_IDs:
    return REJECT_REPLAY
```

由 (A4) `Aggregated_IDs` 由 TA 维护且仅追加（TA 可信，3.2.2），该检查确定性拒绝重放：

```text
ε_replay(λ) = 0
```

**检测成功**：4.3.9 reject ⇒ checkpoint 中不含重放组包。

#### 综合与界

敌手可选择上述攻击中的任意一种或组合。无需假设攻击相互独立，由 union bound：

```text
ε_checkpoint(λ) ≤ Adv_coll^{LHH}(λ) + Adv_euf^{Ed25519}(λ) + Adv_euf^{Ed25519}(λ) + 0
                 = Adv_coll^{LHH}(λ) + 2·Adv_euf^{Ed25519}(λ)
```

在 A1-A6 假设下，右侧各项关于 `λ` 可忽略，故 `ε_checkpoint(λ)` 可忽略，定理 4.1 成立。□

**附注 4.1（compensation tampering）.** `C_k` 由 `Q(t)`、共识维护的历史模型、`S_k` 和公开超参数唯一确定，不是 Cloud 可以自由提交的独立密码学输入。ES 在 `raw sum` 验证通过后本地重算 `W_sum^corr` 与 `W(t+1)`；若同时收到 Cloud 广播结果，可将不一致记录为确定性的重构检查失败。该检查不依赖新的密码原语，也不意味着 `W_sum^corr` 获得独立的 LHH 证明。

### 4.4.2 隐私保持

本节证明研究点 2 的检查点式组级半异步扩展在 3.2.2 假设下不破坏研究点 1 HVSA 的隐私性，即 stale reuse 与组级补偿机制不向任何单一实体泄露客户端明文参数。

#### 隐私性扩展定义

研究点 1 的隐私性（Theorem 3.1）已保证：在 n_min≥2 与 PRG 伪随机性下，TA 持掩码、ES/Cloud 持 masked model 均无法恢复任一客户端明文 `w_i`。研究点 2 引入两个新机制——组内 stale reuse（4.3.3）与 Direct-VCM 补偿（4.3.7）——需证明它们不削弱该保证。

#### 主定理

**定理 4.2（Privacy Preservation）.** *在以下假设下：*
- *(B1) 研究点 1 Theorem 3.1 的全部假设成立（PRG 伪随机性、n_min≥2、3.2.2 不合谋假设）；*
- *(B2) stale reuse 在客户端本地掩码前完成（4.3.3：`w_i^k = (1−ρ_i)w_i^{new} + ρ_i·w_i^{stale}` 在生成掩码 `m_i^k` 之前计算）；*
- *(B3) Direct-VCM 补偿项 `C_k = S_k·γ_vcm·(W(t) − W(t_k))` 仅依赖公开共识量（历史全局模型、组包元数据、协议超参），不依赖任一客户端的明文 `w_i` 或掩码 `m_i^k`；*
- *(B4) TA 仅按 checkpoint 选中集合 Q 释放 `M_Q = Σ_{Π_k∈Q} M_k`，不释放单组 `M_k`。*

*研究点 2 协议在 3.2.2 不合谋假设下保持客户端明文参数隐私性：TA、ES、Cloud 任一单一实体无法以非可忽略优势恢复任一客户端的 `w_i^{new}` 或 `w_i^{stale}`。*

**证明.** 分两步：先证 stale reuse 不泄露，再证 Direct-VCM 补偿不泄露。

##### 步骤 1：stale reuse 不泄露明文

由 (B2)，stale reuse 是客户端本地操作：

```text
w_i^k = (1−ρ_i)·w_i^{new} + ρ_i·w_i^{stale}        (客户端本地)
m_i^k = PRG(seed_i^k)                              (客户端本地)
w̄_i^k = s_i^k·w_i^k + m_i^k                        (上传到 ES)
```

外部实体（ES、Cloud、TA）仅观察到 `w̄_i^k`，不观察 `w_i^{new}`、`w_i^{stale}`、`ρ_i` 或 `w_i^k` 本身。stale reuse 改变的是"被掩码的明文内容"（从 `w_i^{new}` 变为 `w_i^k`），但**掩码 `m_i^k` 仍由 PRG 生成且不依赖 `w_i^k`**（掩码种子 `seed_i^k` 与明文独立）。

形式化地，研究点 1 Theorem 3.1 的归约证明中，敌手的视角为 `{w̄_i^k, seed_i^k 的承诺}` 等。在研究点 2 中，`w̄_i^k = s_i^k·w_i^k + m_i^k` 的形式与 Theorem 3.1 完全一致（仅 `w_i^{new}` 被 `w_i^k` 替换），掩码 `m_i^k` 仍是 PRG 输出。由 (B1) PRG 伪随机性，敌手无法区分 `m_i^k` 与均匀随机；由 (B1) n_min≥2，组内掩码和 `Σ m_i^k` 不可被任一客户端减去自身掩码恢复他客户端明文。

故 stale reuse 的引入**不改变 Theorem 3.1 的归约链条**，敌手恢复 `w_i^k` 的优势仍受 `Adv_{PRG}(λ)` 界。又 `w_i^k` 是 `w_i^{new}` 与 `w_i^{stale}` 的凸组合（`(1−ρ_i)· + ρ_i·`，`ρ_i ∈ [0, 0.5]`），恢复 `w_i^k` 不等价于恢复 `w_i^{new}` 或 `w_i^{stale}` 单独——stale reuse 实际上**降低了**单次上传泄露的明文新鲜度，但**不增加**明文可恢复性。

##### 步骤 2：Direct-VCM 补偿不泄露明文

由 (B3)，补偿项 `C_k = S_k·γ_vcm·(W(t) − W(t_k))` 的输入全部为公开共识量：
- `S_k(t_k)`：组包元数据（已通过 4.3.8 Step 1a manifest 一致性公开）；
- `γ_vcm`：协议超参（公开）；
- `W(t)`、`W(t_k)`：历史全局模型（所有 ES、Cloud、TA 均持有）。

`C_k` 的计算在 4.3.8 Step 3 `ReconstructW` 中由 ES 本地完成，**不引入任何新的客户端级密文或明文泄露通道**。补偿操作发生在 raw sum 验证之后（4.3.8 Step 2 PASS 之后才进入 Step 3），即**在掩码已被验证、raw sum 已公开的时刻**——此时 raw sum `W_sum^raw` 的隐私性已由 Theorem 3.1 在"聚合层"保证（n_min≥2 ⇒ 单客户端明文不可从 raw sum 中分离）。

形式化地，敌手在研究点 2 中的额外视角（相对研究点 1）仅为 `{C_k}_{Π_k∈Q}` 与 `W(t+1)`。但：
- `C_k` 由公开量确定性派生，不引入新随机性也不暴露客户端密文；
- `W(t+1) = (W_sum^raw + Σ C_k) / S_Q`，其中 `W_sum^raw` 的隐私性已被 Theorem 3.1 保证，`Σ C_k` 与 `S_Q` 为公开量。

故 Direct-VCM 补偿**不增加**敌手恢复任一客户端明文的优势。

##### 步骤 3：组级聚合不削弱客户端级隐私

研究点 2 的组级聚合 `W_sum^raw = Σ_{Π_k∈Q} (W̄_k − M_k)` 是研究点 1 单组聚合 `W_sum = Σ_i (w̄_i − m_i)` 的多层推广。由 (B4)，TA 仅释放 `M_Q = Σ_{Π_k∈Q} M_k`，不释放单组 `M_k`——这保证跨组掩码和仍以"组级最小单元"释放，与 Theorem 3.1 中 TA 释放 `M_global` 而不释放单客户端 `m_i` 的语义对偶。

形式化地，将 Theorem 3.1 中的"客户端"替换为"组"，"n_min≥2 客户端"替换为"|Q|≥2 组"，"m_i"替换为"M_k"，归约链条完全平移：在 |Q|≥2 下，TA 持 `M_Q` 无法分离单组 `M_k`，ES/Cloud 持 `W_sum^raw` 无法分离单组 `W̄_k − M_k`。结合步骤 1（组内 stale reuse 不泄露）与 Theorem 3.1（组内 n_min≥2 保护），客户端级明文 `w_i` 在组内与组间双重保护下不被恢复。

##### 综合

由步骤 1、2、3，研究点 2 的全部新增机制（stale reuse、Direct-VCM 补偿、组级聚合）均不引入新的明文泄露通道，且不削弱 Theorem 3.1 的归约界。故在 (B1)-(B4) 假设下，敌手恢复任一客户端 `w_i^{new}` 或 `w_i^{stale}` 的优势仍受 Theorem 3.1 的界：

```text
ε_priv^Point2(λ) ≤ Adv_{PRG}(λ) + Adv_coll^{LHH}(λ)
```

可忽略。定理 4.2 成立。□

**附注 4.2（不合谋假设的必要性）.** 定理 4.2 在 3.2.2 不合谋假设下成立。若 TA 与 Cloud 合谋，TA 持 `M_Q` 与 Cloud 持 `W_sum^raw` 仍无法恢复单客户端明文（因 |Q|≥2 与 n_min≥2 双重保护）；但若 TA 与单个 ES 合谋，TA 持 `M_Q`、ES 持本组 `W̄_k` 与组内全部掩码种子（ES 在 PKI 中签发组包但不持种子），仍不足恢复——因种子在客户端。真正的合谋威胁是 TA 与全部 n_min 客户端中的 n_min−1 个合谋（研究点 1 同假设），此情形已被 Theorem 3.1 排除。

### 4.4.3 收敛性分析

本节只给出与 `MAFSA`/`DVAFL` 分析思路对齐的、针对当前协议的一步偏差刻画，不直接继承两篇参考论文的收敛定理。原因是当前协议同时改变了 stale 权重方向、补偿粒度（客户端级变为组级）、补偿位置（LHH 验证后）和参与集合（checkpoint 选中集合）。因此，本节结论应写成“条件性偏差上界”，而不是完整收敛定理。

#### 假设与记号

以下假设贯穿本节；其中 IID 是 Point 2 的默认理论条件，Dirichlet Non-IID 作为实验条件单独报告：

| 假设 | 内容 |
|------|------|
| **(C1) 局部更新映射 Lipschitz** | IID 默认条件下，一步本地更新映射 `T_i(w)` 满足 `‖T_i(w)−T_i(w')‖≤(1+ηL)‖w−w'‖`；例如 `T_i(w)=w−η∇F_i(w)` 且 `∇F_i` 为 `L`-Lipschitz |
| **(C2) 单步位移有界** | 全局模型经 SGD 型更新，单步位移满足 `‖W(v+1) − W(v)‖ ≤ ηB`，`η` 为学习率，`B` 为有效梯度范数上界 |
| **(C3) 陈旧上界** | 客户端陈旧度 `E_i = t_k − t_i ≤ E_max`，组级陈旧度 `s_k = t − t_k ≤ s_max` |
| **(C4) 混合系数有界** | `ρ_i ≤ ρ_max ≤ 0.5`（4.3.3 stale reuse） |
| **(C5) 补偿系数有界** | `γ_vcm` 为协议超参，固定常数 |
| **(C6) 组样本量有界** | `S_k(t_k)>0`，且 checkpoint 使用真实样本量加权；不需要把 `S_max/S_min` 人为引入偏差界 |

记理想 FedAvg-form 聚合（无陈旧、无补偿）结果为 `W^*(t+1)`，本协议实际结果为 `W(t+1)`。单 checkpoint 偏差定义为 `Δ_bias = W(t+1) − W^*(t+1)`。

#### 条件性界与推导说明

**条件性界 4.1（单 checkpoint 偏差上界，非完整收敛定理）.** *在 (C1)-(C6) 假设下，并把理想结果定义为“同一 checkpoint 选中客户端、同一真实样本量权重、所有组均从 `W(t)` 出发且不使用 stale/VCM”的 FedAvg-form 结果，则一次 checkpoint 的偏差满足：*

```text
‖Δ_bias‖ ≤ ρ_max · (1 + ηL) · E_max · ηB
              + (|1 − γ_vcm| + ηL) · s_max · ηB
```

*第一项对应客户端级 stale reuse；第二项对应组级 mixed-version 结果在 Direct-VCM 后的残差。由于实际 checkpoint 是按 `S_k` 做加权平均，分子中的组样本量和会被 `S_Q` 归一化，不应额外留下 `S_max/S_min` 因子。*

在 IID 和一步梯度更新下，`γ_vcm≈1` 时，第二项可由版本位移的一阶项抵消到 `ηL·s_max·ηB` 量级；若要把这种“补偿确实降低偏差”升级为严格结论，还需要额外的线性化/梯度同向性假设。

**说明.** 偏差 `Δ_bias` 由两个来源叠加：客户端级 stale reuse 引入的偏差 `Δ_stale`，与组级 mixed-version 结果在 Direct-VCM 后的残差 `Δ_vcm`。下面给出条件性上界；它用于说明 `ρ_max`、`E_max`、`γ_vcm` 与 `s_max` 如何进入误差控制，不声称已经得到参考论文的完整多步收敛率。

```text
‖Δ_bias‖ ≤ ‖Δ_stale‖ + ‖Δ_vcm‖
```

分别界两项。

##### 步骤 (i)：客户端级陈旧偏差 `‖Δ_stale‖` 的界

理想聚合中客户端 `i` 贡献 `w_i^{new,k}`（基于 `W(t_k)` 的本地一步训练结果）。本协议中客户端混合：

```text
w_i^k = (1−ρ_i)·w_i^{new,k} + ρ_i·w_i^{stale,k}
```

其中 `w_i^{stale,k}` 是基于更旧全局模型 `W(t_i)`（`t_i < t_k`）的训练结果。客户端级偏差为：

```text
δ_i = w_i^k − w_i^{new,k} = ρ_i·(w_i^{stale,k} − w_i^{new,k})
```

由 (C2)，`w_i^{new,k}` 是从 `W(t_k)` 出发的一步 SGD 结果，`w_i^{stale,k}` 是从 `W(t_i)` 出发的一步 SGD 结果。在 (C1) `L`-Lipschitz 假设下，SGD 一步更新 `w = w' − η∇F(w')` 是 `1 + ηL`-Lipschitz 的（关于初始点 `w'`）：

```text
‖w_i^{new,k} − w_i^{stale,k}‖ ≤ (1 + ηL)·‖W(t_k) − W(t_i)‖
```

由 (C2) 单步位移有界 `ηB` 与 (C3) `t_k − t_i ≤ E_max`：

```text
‖W(t_k) − W(t_i)‖ ≤ (t_k − t_i)·ηB ≤ E_max·ηB
```

故：

```text
‖w_i^{new,k} − w_i^{stale,k}‖ ≤ (1 + ηL)·E_max·ηB
```

由 (C4) `ρ_i ≤ ρ_max ≤ 0.5`：

```text
‖δ_i‖ = ρ_i·‖w_i^{stale,k} − w_i^{new,k}‖ ≤ ρ_max·(1 + ηL)·E_max·ηB
```

客户端级偏差在 raw sum 中按真实样本量累加，归一化后的结果为：

```text
Δ_stale^{norm} = (1/S_Q) · Σ_{Π_k∈Q} Σ_{i∈U_online,k} s_i^k · δ_i
```

因为 `Σ s_i^k = S_Q`，直接得到：

```text
‖Δ_stale^{norm}‖ ≤ ρ_max·(1 + ηL)·E_max·ηB
```

##### 步骤 (ii)：组级版本失配偏差 `‖Δ_vcm‖` 的界

理想聚合中所有组包基于同一全局模型 `W(t)`（即 `t_k = t`）。本协议中组包 `Π_k(t_k)` 基于 `W(t_k)`，`t_k < t`，需 Direct-VCM 补偿 `C_k = S_k·γ_vcm·(W(t) − W(t_k))`。

对一步更新映射 `T_k(w)=w−η∇F_k(w)`，令 `G_k=W(t)−W(t_k)`。组包结果与从 `W(t)` 出发的理想结果之差，在加入 Direct-VCM 后为：

```text
res_k = T_k(W(t_k)) + γ_vcm G_k − T_k(W(t))
```

由 `T_k(w)=w−η∇F_k(w)` 和 (C1)：

```text
res_k = (γ_vcm−1)G_k − η(∇F_k(W(t_k))−∇F_k(W(t)))
‖res_k‖ ≤ (|1−γ_vcm| + ηL)·‖G_k‖
```

再由 (C2) 与 (C3) `s_k = t − t_k ≤ s_max`：

```text
‖G_k‖ = ‖W(t) − W(t_k)‖ ≤ s_k·ηB ≤ s_max·ηB
‖res_k‖ ≤ (|1−γ_vcm| + ηL)·s_max·ηB
```

组级残差在 raw sum 中按 `S_k` 累加，归一化后仍由最大组级残差控制：

```text
Δ_vcm^{norm} = (1/S_Q) · Σ_{Π_k∈Q} S_k · res_k
‖Δ_vcm^{norm}‖ ≤ (|1−γ_vcm| + ηL)·s_max·ηB
```

##### 综合

由步骤 (i) 与 (ii) 的归一化结果：

```text
‖Δ_bias‖ ≤ ‖Δ_stale^{norm}‖ + ‖Δ_vcm^{norm}‖
           ≤ ρ_max·(1+ηL)·E_max·ηB
             + (|1−γ_vcm|+ηL)·s_max·ηB
```

该界是本文在 IID/一步更新条件下的条件性单 checkpoint 偏差上界，不是直接来自 `MAFSA` 或 `DVAFL` 的定理。对 Dirichlet Non-IID，需额外引入局部梯度异质性或 bounded gradient dissimilarity 项；本文将其作为实验验证，不把 IID 上界无条件外推到 Non-IID。

#### 推论：多 checkpoint 累积偏差界

**推论 4.1（T 个 checkpoint 的保守累积界，非收敛率定理）.** *在条件性界 4.1 的假设下，若进一步把每个 checkpoint 的偏差按幅度逐步累加，则：*

```text
‖W(T) − W^*(T)‖ ≤ T·[ρ_max·(1+ηL)·E_max·ηB
                    + (|1−γ_vcm|+ηL)·s_max·ηB]
```

条件性界 4.1 与推论 4.1 是**确定性的单步/保守多步幅度界**，未覆盖：
- non-IID 下的客户端漂移（需更强假设如 bounded gradient dissimilarity）；
- SGD 噪声的方差（需有界方差假设 `E[‖∇F − ∇f_i‖²] ≤ σ²`）；
- `ρ_i` 与数据分布的耦合（stale reuse 偏向数据稳定的客户端）。

完整收敛率（如 `O(1/√T)` 邻域收敛）需在当前协议上重新建立，并补充有界方差、独立鞅差、局部梯度异质性与 checkpoint 选择过程的假设，不能直接引用 `MAFSA` 或 `DVAFL` 的收敛率。这里的结论只说明 `ρ_max`、`E_max`、`γ_vcm` 与 `s_max` 如何进入可调的偏差控制项；实际 `γ_vcm` 是否带来精度和收敛收益，交由 4.5 的对照和消融实验验证。

### 4.4.4 出界声明重申

manifest 漏选/审查（anti-censorship）、样本量虚报、完整 ACK 协议不作主协议安全保证。

## 4.5 实验评估

### 4.5.1 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | ___ |
| 数据集 | MNIST、CIFAR-10 |
| 模型 | MLP、CNN |
| 聚合 | FedAvg-form |
| 层级 | 1 Cloud + M Edge + N Users |
| 密码 | LHH(P-256) / Ed25519 / PRG(AES-CTR/HKDF) / QUANTIZE_SCALE=1e6 |
| 种子数 | ≥3（含置信区间） |

### 4.5.2 五组主线对比

为分离各机制的贡献，设计五组递进主线：

1. **Sync-HVSA**：研究点 1 同步聚合基线。
2. **Grouped-HVSA + Inter-Group Sync**：引入一次性分组与组间同步。
3. **+ Buffer/Checkpoint Semi-Async**：引入组包缓冲与 checkpoint 触发。
4. **+ Intra-Group Stale Reuse**：引入组内 stale reuse。
5. **+ Direct-VCM Group Compensation**：完整方案（含组级补偿）。

五组采用**逐一叠加**设计，其**增量增益**是各机制必要性的直接度量：G3−G2 衡量 checkpoint 半异步、G4−G3 衡量 stale reuse、G5−G4 衡量 Direct-VCM。该设计同时作为对"机制拼凑"质疑的经验检验——若某步增量为负或近零，则表明该机制非必要，协议应裁剪之。增量需结合 4.1 的因果链解释：VCM 的增益应主要来自 mixed-version（仅在半异步下出现），stale reuse 的增益应主要来自半异步时间线下增多的"未赶上封组"场景，而非叠加式堆砌。

- **指标**：达到目标精度的墙钟时间、平均等待时间、最终精度、compensation gain、stale reuse gain、stale-weight stability、verification overhead、组包过期率、重复贡献率、组间公平性、有效吞吐量。
- **数值**：___（留空待回填）

### 4.5.2' 竞争性基线对比

为定位本协议在分层安全聚合与半异步 FL 中的相对性能，引入以下基线（基线数据优先取自原文报告，本实验在相同数据集/模型下复现关键指标作为参照）：

| 基线 | 特点 | 对齐维度 |
|------|------|---------|
| **FedAvg / HFL** | 同步分层聚合，无安全/验证 | 同步聚合上界 |
| **FedBuff** | 缓冲异步，单客户端粒度 | 异步基线 |
| **MAFSA** | 多层异步 + stale 融合（权重方向为 `α=e^{-βE}` 作用于 fresh） | stale reuse 方向对比 |
| **DVAFL** | VCM 版本控制补偿（Chen et al., IEEE TMC 2026） | 补偿机制对比 |

- **对比指标**：达到目标精度墙钟时间、最终精度、stale reuse gain、compensation gain。
- **预期**：HVSA 系列在验证开销上优于所有基线（因 LHH 定长验证）；stale reuse 方向修正（Ours Decayed-Stale）应优于 MAFSA 方向；完整方案（Group 5）在异构性下墙钟时间优于同步基线。
- **数值**：___（留空待回填）

### 4.5.3 攻击实验

以第 5 组（完整方案）为基座，与性能基线分开，验证 Theorem 4.1 的篡改检测能力：

- **tamper_raw_sum**：Cloud 篡改 `W_sum^raw`，预期 LHH 等式失败。
- **tamper_manifest**：Cloud 篡改组包元数据，预期 manifest 一致性或 Ed25519 验签失败。
- **drop_replace_pkg**：Cloud 丢弃合法组包并替换为伪造/重放组包，预期 manifest 一致性或签名验证失败。
- **replay**：Cloud 重放已消费的组包，预期 `Aggregated_IDs` 确定性拒绝。

- **指标**：tampering detection rate（应接近 100%）。
- **数值**：___（留空待回填）

### 4.5.4 消融实验

**机制交互检验（协同证据）**：为验证 4.1 的因果链（三个机制是同一架构决策的展开而非并列拼装），先于逐参数消融设置两组交互实验：

- **I1：VCM × checkpoint 开关**。在 checkpoint 关闭（inter-group sync，全部组包 `t_k=t`）下 VCM 补偿 `C_k=S_k·γ_vcm·(W(t)−W(t_k))≡0`，无 mixed-version 可补；开启 checkpoint 后 VCM 才可能产生增益。预期：VCM 增益≈0（checkpoint off）→ 显著为正（checkpoint on），证明补偿被架构内生需要。
- **I2：reuse × buffer 开关**。stale reuse 在半异步时间线下有更多"未赶上封组"场景可复用；同步下复用机会少。预期：reuse 增益在 buffer on 时显著大于 buffer off，证明复用与该架构的时间线强耦合。

I1/I2 的"开关 × 机制"交互增量是回应机制拼凑质疑的核心证据：若某机制在开关两端增益一致，则其与架构无耦合，需重新定位。

**消融顺序**：

1. **reuse-rule**（none / mafsa / ours，必做，验方向修正）
2. **No Compensation**（γ_vcm=0）
3. **Direct-VCM**（γ_vcm=1.0）
4. **reuse × compensation 2×2**（与 I1/I2 互补：2×2 检验两机制间的联合效应，I1/I2 检验各机制与架构的耦合）
5. **ρ_max**∈{0.25, 0.5}、**β_c**∈{0.1, 0.3, 0.5}
6. **γ_vcm**∈{0.5, 0.8, 1.0}
7. **K**∈{2, 3, 5}
8. **s_max**∈{2, 3, 5}
9. **policy**（FIFO / Fresh-first / Fairness-aware）
10. **adaptive s_max**（off / on）

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

- **数值**：___（留空待回填）

## 4.6 本章小结

本章提出研究点二——检查点式组级半异步安全聚合扩展。在研究点一 HVSA 的密码学闭环基础上，通过组内 stale reuse（修正 MAFSA 权重方向）、组包缓冲与 checkpoint 触发、Direct-VCM 组级后置补偿三个机制，将跨组训练组织从"全组同步 barrier"扩展为"组包缓冲 + checkpoint 触发"的组级半异步机制，同时保持单 TA 掩码与 LHH 验证链路不被破坏。安全分析证明：对 checkpoint 选中聚合结果的完整性检测可归约到 LHH 抗碰撞或 Ed25519 EUF-CMA；检测失败即丢弃本次 checkpoint，不推进全局版本，也不保证恶意 Cloud 下的训练可用性或 anti-censorship（Theorem 4.1）。stale reuse 与 Direct-VCM 补偿不削弱研究点一的隐私界（Theorem 4.2）；在 IID 默认条件下，给出当前协议的条件性单 checkpoint 偏差上界 `ρ_max·(1+ηL)·E_max·ηB + (|1−γ_vcm|+ηL)·s_max·ηB`（条件性界 4.1），多 checkpoint 仅给出保守的线性累积幅度界（推论 4.1），不把 MAFSA 或 DVAFL 的完整收敛率直接继承为本文定理。实验框架覆盖五组主线对比、攻击实验与十项消融，为协议性能与安全性提供系统评估。

---

# 第 5 章 总结与展望

## 5.1 全文总结

本文在单 TA 约束下，先提出 HVSA 实现分层可验证安全聚合，再以检查点式组级半异步扩展解决同步等待与掉线浪费，同时保持掩码与 LHH 验证闭环。具体地：

**研究点一（HVSA）**：在"云—边—端—单 TA"架构下，通过 PRG 掩码保护客户端参数隐私（Theorem 3.1，归约 PRG + n_min≥2），利用 LHH 与 Ed25519 让 ES 以定长通信开销验证云端聚合结果（Theorem 3.2，归约 LHH 抗碰撞 + Ed25519 EUF-CMA），并通过"在线集合驱动的掩码求和"实现对客户端掉线的自然鲁棒性（Theorem 3.3）。验证通信与模型维度无关，客户端零验证负担。

**研究点二（检查点半异步扩展）**：在研究点一密码学闭环基础上，通过组内 stale reuse（修正 MAFSA 方向）、组包缓冲与 checkpoint 触发、Direct-VCM 组级后置补偿三个机制，将跨组训练组织从"全组同步 barrier"扩展为"组包缓冲 + checkpoint 触发"的组级半异步机制。安全分析证明：对 checkpoint 选中聚合结果的完整性检测可归约到 LHH 抗碰撞或 Ed25519 EUF-CMA；检测失败即丢弃本次 checkpoint，不推进全局版本，也不保证恶意 Cloud 下的训练可用性或 anti-censorship（Theorem 4.1，`ε_checkpoint ≤ Adv_coll^LHH + 2·Adv_euf^Ed25519`）。stale reuse 与 Direct-VCM 补偿不削弱研究点一隐私界（Theorem 4.2）；在 IID 默认条件下，给出当前协议的条件性单 checkpoint 偏差上界 `ρ_max·(1+ηL)·E_max·ηB + (|1−γ_vcm|+ηL)·s_max·ηB`（条件性界 4.1）与保守的线性累积幅度界（推论 4.1），不把 MAFSA 或 DVAFL 的完整收敛率直接继承为本文定理。

## 5.2 未来工作展望

- **双 TA / 门限 TA**：以进一步分散信任，消除单点信任假设。
- **基版本对齐的更新量混合**：stale reuse 更严格变体，对 stale 与 fresh 的基版本对齐做更强约束。
- **完整 ACK / 原子提交协议的协议化**：将 exactly-once 语义从"声明"升级为完整协议。
- **anti-censorship 可审计调度**：将 manifest 漏选/调度审查纳入可审计安全保证。
- **完整收敛率证明**：在 Lipschitz/有界方差/独立鞅差假设下，将条件性界 4.1 的单步偏差刻画推进为当前协议的 `O(1/√T)` 邻域收敛分析（见 4.4.3）。

---

# 参考文献

1. Shiwen Zhang et al. “MAFSA: A Multi-Layer Asynchronous Federated Learning with Staleness-Awareness in Edge Computing.” *Engineering Applications of Artificial Intelligence*, 2026. DOI: `10.1016/j.engappai.2026.114369`.
2. Chuyi Chen, Yanchao Zhao, Zhe Zhang, Wenzhong Li, and Jie Wu. “Toward Efficient and Scalable Asynchronous Federated Learning via Stragglers Version Control.” *IEEE Transactions on Mobile Computing*, 2026. DOI: `10.1109/TMC.2025.3609568`.
3. Lam M. Nguyen et al. “Federated Learning with Buffered Asynchronous Aggregation.” *AISTATS*, 2022.
4. Jiaming Xu and Shen Wang. “FedFleet: A Hierarchical Federated Learning Framework for Faster Convergence in Heterogeneous IoT Systems.” *IEEE Internet of Things Journal*, 2025.

---

# 附录 A 设计决策记录

> 本附录保留研究点 2 从多轮讨论收敛到最终协议的关键取舍理由，正文为行文简洁未展开。
> 与正文口径一致，供答辩/审稿追溯设计动机。
>
> **组织逻辑**：所有取舍归因于三层——A.1 **约束层**（单 TA 掩码 + LHH 验证链路施加的硬约束，回答"为什么不能直接借鉴现有异步方案"）、A.2 **架构层**（checkpoint 级异步边界的确定）、A.3 **组件层**（各机制在约束内的实现取舍）。约束层在先：它决定了哪些方案根本不可行，架构层与组件层均在其允许的空间内取值。

## A.1 约束层：单 TA 掩码 + LHH 验证链路施加的设计约束

主协议的一切设计受制于一条硬约束：**解掩码前不得引入任何破坏 TA 掩码和与 LHH 对应关系的变换**。以下四项取舍均由该约束直接推出，也是"为什么不能把 FedBuff/MAFSA/DVAFL 直接搬进本架构"的正面回答。

### A.1.1 为什么不引入解掩码前的组间连续非等权系数

曾考虑 `λ_g=λ_0/(1+β_g·s_g)` 这类陈旧度加权。但一旦不同组在解掩码前后带不同连续权重，会改变 TA 掩码和与 LHH 的对应关系，破坏研究点 1 的验证链路。故主协议只保留解掩码后、LHH 验证后的 Direct-VCM 公开确定性补偿。

### A.1.2 为什么去掉降权路径

DVAFL 是"补偿+降权"双机制（straggler 既被降权又被补偿）。本文在 masked group package 约束下无法对单个组包做解掩码前的连续降权，故只保留补偿，用 `s_max` 硬截断 + 补偿替代软降权。这使系统对 `s_max` 取值更敏感，故 `s_max∈{2,3,5}` 列为必做消融。

### A.1.3 为什么样本量权重只乘明文不乘掩码

客户端上传 `w̄_i^k=s_i^k·w_i^k+m_i^k`。若 `s_i^k` 也乘掩码，TA 求掩码和时就需额外按样本量加权，改变单 TA 掩码求和方式。只乘明文可让解掩码后直接恢复 strict sample-weighted sum，同时不改变 TA 按参与集合求掩码和的链路。

### A.1.4 全局模型为何是共识量而非 Cloud 分发量

`W(t+1)` 由参与 checkpoint 的 ES 基于已验证 raw sum **本地独立重构**（公开确定性计算），各 ES 算得一致即为共识。Cloud 即使广播 `W(t+1)` 也仅作参考，规范结果以 ES 本地派生为准。因此 Cloud 无法在补偿/归一化阶段注入篡改——补偿输入（历史全局模型）是共识量，不作攻击面。

## A.2 架构层：为什么把异步边界收缩到 checkpoint 级

最初考虑过完全异步（组包"来一个更新一个"），但完全异步下每次全局下发都需在线验证，与单 TA + LHH 的逐轮验证闭环冲突。最终选择：组内仍同步封组，组间用 buffer+checkpoint 半异步，验证粒度为 checkpoint 选中集合。一句话：**不是把验证硬塞进完全异步，而是主动收缩异步边界以保住密码学闭环。** 该架构决策同时引出三个必要组件：checkpoint 触发本身（4.3.6）、mixed-version 的补偿（4.3.7）与掉线浪费的复用（4.3.3），故三者不是并列拼装而是同一决策的展开。

## A.3 组件层：各机制的关键实现取舍

### A.3.1 stale reuse 权重方向的修正

MAFSA 原始定义 `α_i=exp(−β_c·E_i)` 作用于 fresh model，导致版本间隔 `E_i` 越大、stale 权重 `1−α_i` 反而越大。考虑到间隔越大陈旧结果与当前优化方向偏差风险越高，本文将指数衰减量改为直接控制 stale 项并加 `ρ_max≤0.5` 上界，保证新结果主导。故表述为"受 MAFSA 启发的时效性修正"而非"直接采用 MAFSA 加权公式"。

### A.3.2 为什么 VCM 用 S_k 而非 n_k

`W_sum^raw` 本身处在"样本量加权和"量纲中，组间补偿项必须同量纲才能与最终 `S_Q` 归一化一致。故 `C_k=S_k·γ_vcm·G_k`，而非按客户端个数 `n_k`。`n_k` 仅承担在线计数与最小在线人数约束。

## A.4 显式出界的考量

下列方向有价值但会稀释主线，统一出界：完全异步、解掩码前连续陈旧度系数 `β_g`、双 TA/门限 TA、全局异步轨迹在线审计、anti-censorship 可审计调度、First/Second-order VCM、组级质量追踪、HGA-FL 式 bias correction。出界是为让研究点 2 保持"一个能讲圆的协议扩展"而非大拼盘。

---

## 附：全文一致性自检

- [x] 研究点 2 公式与 `STALENESS_AWARE_REUSE.md` 一致（stale/VCM/验证/术语）
- [x] 无 "strict FedAvg" 残留（仅组内 strict sample-weighted sum，最终 FedAvg-form）
- [x] 威胁模型 3.2.2 / 4.2.2 统一声明，P1/P2 适用范围标注
- [x] 记号表全文唯一
- [x] 实验数值全部留空，仅框架+预期趋势
- [x] 创新点 1.3 统一编号，第 3、4 章不重复计数
- [x] 定理按章重编号：Theorem 3.1-3.3（点 1）/ Theorem 4.1-4.2 + 条件性界 4.1 + 推论 4.1（点 2）
- [x] 6 章结构：绪论 / 相关技术 / HVSA / 检查点半异步 / 总结展望 / 参考文献 + 附录 A
- [x] 4.1 采用"3.4.4 三局限 → 约束 → checkpoint 洞见 → 三机制映射"因果链；4.3.3 显式锚定局限 (ii) 与安全聚合兼容
- [x] 附录 A 重排为 A.1 约束层 / A.2 架构层 / A.3 组件层 / A.4 出界，与 4.1 的组织逻辑呼应
- [x] 4.5.4 增设 I1/I2 机制交互检验（VCM×checkpoint、reuse×buffer），`run_all_ablations.py` 已同步
- [x] 三机制定位为"同一架构决策的展开"而非并列拼装（4.1 末段 + A.2 末段）
