# 面向分层联邦学习的单 TA 可验证安全聚合及其检查点式半异步扩展

> 本文档为研究点 1（HVSA）与研究点 2（检查点式组级半异步）的整合硕士论文设计文档。
> 实验部分仅给出设计框架与预期趋势，所有数值留空待回填。
> 研究点 2 协议公式与 `THESIS_POINT2_STALENESS_AWARE_REUSE.md`（已定稿）一致；实验设计在其基础上扩展（如 reuse×compensation 2×2 消融），属框架性扩充而非协议改动。
> **2026-09-14 修订（第五轮，finalize 闭环 P0 修复）**：⑰ finalize 由"直接写 ledger"改为**可验证**——校验 TA 签名并核对持久化 `prepared_checkpoints`（完整 receipt 字节 + token 元数据 + `result_digest`），伪造或未经 prepare 的 receipt 无法落地；同一 receipt 重复 finalize **幂等成功**（返回 `already`，不重复消费），同版本不同 receipt 冲突破坏摘要一致性时才拒绝；同版本换候选时新 prepare 原子替换记录、旧 receipt 随之作废；⑱ 提交改为 **stage → TA finalize → publish → notify** 四阶段事务（新增 `commit_tx.py`）——stage 只读校验并把变更写入持久化 journal，Cloud 不再出现"已提交而 TA 未落地/TA 已落地而 ES 未推进"的静默分叉；finalize/publish/notify 均幂等，阶段边界崩溃可重放收敛（含"仅通知部分 ES"分叉）；⑲ `confirm_receipt` 核对完整 receipt（签名 + prepare 记录 + 已 finalize），`--ta-state-path` 持久化 TA ledger；补伪造 finalize、幂等/冲突、prepare 替换、四阶段崩溃恢复测试（`tests/test_receipt_flow.py`）。
>
> **2026-09-13 修订（第四轮，独立验收 P0/P1 修复）**：⑬ 结果摘要绑定——vote/receipt 绑定 ES 实际验证的 `raw sum`/`M_Q`/`final model` 摘要，Cloud 提交与 ES 推进双重核对，取诚实 receipt 后偷换结果被拒；⑭ TA 授权改 prepare/finalize 两阶段——提交失败不推进 ledger 与权威状态，重复 finalize 由主键约束拒绝；⑮ ES 推进前增加 TA ledger 确认（`confirm_receipt`）；⑯ `config_hash` 只哈希科学配置（产物参数单列）；`--deterministic` 修复 warn_only 覆盖为严格模式；补真实 `s_max` 负例与结果绑定回归，`pytest.ini` 限定收集范围。
>
> **2026-09-13 修订（第三轮，验收缺口补全）**：⑨ 重放防护改为 TA 权威 ledger（SQLite 仅追加：token 注册/消费 + checkpoint 版本）驱动的授权流——ES 签名 vote → TA 验票与约束核验 → 原子落地并签发 `CheckpointReceipt` → Cloud 凭 receipt 提交、ES 凭 receipt 推进（4.2.2/4.3.9）；⑩ verify-on 组合边界 fail-fast（`vcm_mode` 白名单、`agg_shrink`/`server_opt` 不兼容、`--attack` 强制 `--verify`）；⑪ 可复现性强化：异配置结果非零退出、NPZ/JSON 原子替换、`--deterministic` 严格确定性、元数据含配置模式版本与 CUDA/cuDNN/线程数；⑫ RA-CMA 统一降为可选后处理组件。
>
> **2026-09-13 修订（第二轮，10 周计划第 1-3 周落地）**：④ 主问题统一为"跨组同步等待、mixed-version 偏差、恶意 Cloud 下的聚合策略完整性"（4.1），stale reuse 彻底退出主目标；⑤ 条件性偏差界 4.1 删除 stale 项（移入附录 A.5 备选分析），新增 σ/Aσ 策略条件正确性小节；⑥ 全文清除"首次"（待检索日志支持后再以"据本文检索所知"恢复）、"根本冲突"、"整体验证轻量"表述，exactly-once 降级为"至多一次纳入的重放防护"；⑦ 协议新增版本化 `ProtocolConfig` 与 `policy_config_digest` manifest 绑定（4.3.6/4.3.10），TA/ES 独立复算落地（声明值只作审计）；⑧ 实验基建（config_hash/运行元数据/--output-dir/--run-id/--force）与策略绑定攻击族注入（4.5.3）已实现。

## 论文设计与数值实现的评审边界

本文档当前是硕士论文的**理论方法与实验 workflow 框架**，评估重点是研究问题是否成立、协议流程与公式是否闭合、安全性/隐私性/可验证性分析是否正确，以及实验设计能否支撑论文结论。本文档不是最终的数值计算规范或代码实现说明。

- **论文理论方法层**：关注协议输入输出、参与实体、信任与威胁模型、公式之间的逻辑关系、密码学原语所需性质、定理假设与结论，以及研究点一和研究点二能否形成递进的论文主线。
- **数值实现层**：模型参数由浮点数映射到 LHH 或掩码运算所需整数/有限域、量化尺度、浮点误差、数据类型、模运算、溢出边界、序列化方式及精确计算流程等，属于代码与实验实现需要解决的问题，可在实现、测试和实验复现材料中具体设计。
- **理论抽象约定**：论文协议公式默认存在正确的 `Encode/Quantize` 映射，使模型向量进入密码学原语要求的代数域，并默认公式中的加法、标量乘法、掩码抵消、聚合后归一化与 LHH 同态运算在该抽象域中正确成立。具体采用 `float32/float64`、`int64`、大整数或某一有限域，不作为当前论文 workflow 是否成立的首要判断标准。
- **边界例外**：若某个实现选择会改变论文所依赖的理论性质，例如使同态等式本身不成立、改变安全假设或导致协议输入超出原语定义域，则需在最终实现说明中补充必要条件与验证；但在当前设计阶段，不以具体数据类型或量化误差替代理论方法本身的正确性评估。
- **多参与方与抗合谋评审边界**：本文面向每轮或每个 checkpoint 有多个（`N` 个）客户端进入同一已发布聚合结果的场景，不用人为缩减成二参与方场景来替代协议的实际参与集合。允许 Cloud 与全部 ES 合谋，并允许 Cloud 与至多 `c≤N−2` 个客户端合谋；隐私成立的客户端级阈值为 `N−c≥2`。研究点一取 `N=N_online(t)`，研究点二取 `N=N_Q(t)=Σ_{Π_k∈Q(t)}n_k(t_k)`。`n_min≥2` 只是封组/发布的准入下界，不等同于合谋阈值；单组超时 checkpoint 只要满足客户端级阈值即为合法，不要求 `|Q|≥2`。
- **理想安全信道约定**：协议实体之间的信息交互抽象为百分百可靠的安全信道，不考虑消息在传输中的窃听、失真、替换、伪造或发送方冒充。TA 为完全可信实体，因此 ES 收到的 TA 输出按模型定义即为真实、完整且来源可信；实验代码可以用进程内对象或统一 `bundle` 模拟该交付，不因缺少传输层签名或认证信道代码而构成协议缺陷。CS/Cloud 仍是不可信实体，安全信道只能保证“收到的确为 Cloud 所发送”，不能保证 Cloud 生成内容正确，因此 Cloud 提供的 `raw sum`、聚合声明及其他协议指定对象仍须依据可信 TA/ES 证据验证。
- **共识值与隐私边界**：每轮全局模型、历史全局模型及协议公开超参数默认为共识值，不讨论缓存丢失或传输失真。隐私目标只保护单个客户端参数；ES/组级聚合结果与 CS/全局聚合结果允许公开。Cloud 为完成解掩码必须知道所选集合的聚合掩码 `M_global` 或 `M_Q`，该聚合掩码本身不是秘密。

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

分层联邦学习（Hierarchical Federated Learning）在"云—边—端"架构下兼顾了隐私与通信效率，但仍面临两大痛点：一是客户端本地模型在上传与聚合过程中可能泄露隐私（梯度/参数泄露攻击），二是云端聚合结果可能被不可信云服务器篡改而难以被参与方检测。本文围绕**单可信权威机构（TA）约束下、以定长验证通信为特征的可验证安全聚合**展开，提出两个递进的研究点。

研究点一提出 **HVSA**（Hierarchical Verifiable Secure Aggregation），在"云—边—端—单 TA"架构下，通过 PRG 掩码保护客户端参数隐私，利用线性同态哈希（LHH）与 Ed25519 签名让边缘服务器（ES）能够对云端聚合结果进行高效验证，验证通信开销与模型维度无关。理论分析表明，在 PRG 伪随机性、TA 永不合谋且实际在线参与数满足 `N_online(t)−c≥2` 的假设下，即使 Cloud 与全部 ES 及至多 `c≤N_online(t)−2` 个客户端合谋，也不能从允许公开的聚合结果中恢复任一剩余客户端参数；在 LHH 抗碰撞与 Ed25519 EUF-CMA 假设下，云端任何篡改均可被边缘检测；同时协议对客户端掉线具有自然鲁棒性，掉线用户的掩码与更新自动成对剔除，不破坏验证等式。

研究点二在研究点一的密码学闭环基础上，针对同步聚合的"全局等待"与"掉线训练浪费"问题，提出**检查点式组级半异步安全聚合**。通过组包缓冲与 checkpoint 触发、面向公开确定性聚合策略的策略绑定可验证加权聚合（V-PDA/σ 权重与补偿参数均由边缘独立复算，恶意云端不能自行选择权重）、masked group package、checkpoint 批量聚合与 Direct-VCM 后置补偿，将跨组训练扩展为组级半异步机制。安全分析将 Cloud 篡改攻击归约到 LHH 抗碰撞与 Ed25519 EUF-CMA，并证明权重伪造/改标在策略绑定下被确定性拒绝；客户端隐私按实际 checkpoint 客户端数满足 `N_Q−c≥2` 判定，组/全局聚合允许公开。在 IID、一步更新与耦合相同随机性的条件下，本文给出单 checkpoint 偏差上界，并以 `e_{t+1}≤L_Te_t+ε_t` 刻画条件性多步传播，不直接继承参考论文的完整收敛定理。

**关键词**：联邦学习；分层聚合；安全聚合；可验证聚合；半异步；陈旧性复用

---

# Abstract

Hierarchical Federated Learning balances privacy and communication efficiency under a "cloud-edge-end" architecture, yet it still faces two major pain points: (i) client-side local models may leak privacy during upload and aggregation (gradient/parameter inversion attacks); (ii) aggregation results may be tampered with by an untrusted cloud server and such tampering is hard for participants to detect. This thesis focuses on **verifiable secure aggregation under a single Trusted Authority (TA), characterized by fixed-length verification communication**, and proposes two progressive research points.

Research Point 1 proposes **HVSA** (Hierarchical Verifiable Secure Aggregation). Under a "cloud-edge-end-single TA" architecture, HVSA protects client parameter privacy via PRG masking, and enables Edge Servers (ES) to efficiently verify cloud aggregation results via Linear Homomorphic Hashing (LHH) and Ed25519 signatures, with verification communication overhead independent of model dimension. Under PRG pseudorandomness, a fully trusted non-colluding TA, and `N_online(t)-c≥2`, the privacy result allows the Cloud to collude with all ESs and up to `c≤N_online(t)-2` clients while protecting each remaining client's parameters; aggregate results may be public. Under LHH collision resistance and Ed25519 EUF-CMA, cloud tampering with an accepted result is detectable; meanwhile, masks and updates of dropped clients are removed in pairs without breaking the verification equation.

Research Point 2 extends this cryptographic closure with group-package buffering and checkpoint triggering, policy-bound verifiable weighted aggregation (V-PDA/staleness weights and compensation parameters are independently recomputed at the edge, so a malicious cloud cannot choose weights itself), masked group packages, checkpoint aggregation, and Direct-VCM post-compensation. Cloud tampering is reduced to LHH collision resistance and Ed25519 EUF-CMA, and weight forgery or relabeling is deterministically rejected under policy binding. Client privacy is evaluated on the actual checkpoint population under `N_Q-c≥2`, while group and global aggregates may be public. Under IID one-step updates with coupled identical randomness, the thesis gives a conditional single-checkpoint bias bound and models multi-checkpoint propagation as `e_{t+1}≤L_Te_t+ε_t`; it does not directly inherit a complete convergence theorem from either reference paper.

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

完全异步与"逐轮在线验证"的密码学闭环难以直接兼容（验证粒度需与掩码聚合粒度一致），现有工作未给出在单 TA + LHH 验证链路下引入异步性的方案。

## 1.3 本文主要贡献

本文贡献体现为**单 TA + 分层架构 + 边缘验证 + checkpoint 组级半异步**的组合与对应威胁模型，按两个研究点统一编号：

**研究点一（HVSA）：**
- 1-a 单 TA 分层可验证安全聚合架构：单 TA 即可支持掩码分发与聚合掩码计算，无需多 TA/门限 TA。
- 1-b 在单 TA 约束下将 LHH + Ed25519 验证外包到边缘：验证式 `H(W_sum^raw)·H(M_global)=∏_k H_k`，验证通信与模型维度无关（新颖性表述以检索日志为前提，见 1.2 与第 2 章）。
- 1-c 验证外包到边缘：客户端零验证开销，验证由 ES 承担。

**研究点二（检查点半异步扩展）：**
- 2-a 在单 TA + LHH 验证链路下引入 checkpoint 级组级半异步：借鉴 FedBuff 缓冲思想提升至组包粒度，将跨组同步 barrier 扩展为 buffer+checkpoint 半异步；安全聚合兼容的组内 stale reuse（掩码前本地混合，修正 MAFSA 权重方向）作为其客户端级组件保留，不单列创新点（新颖性表述以检索日志为前提）。
- 2-b 策略绑定的可验证加权聚合：checkpoint 场景下面向公开确定性聚合策略（V-PDA 逆参与权重、σ 陈旧度权重）的策略绑定、边缘独立复算与加权 LHH 验证——恶意云端不能自行选择权重或补偿策略参数，一切可采纳权重来源均须可由 ES 依据共识量独立复算。
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

LHH 在 NIST P-256 上构造，逐维生成元由公开种子经 **hash-to-curve（try-and-increment）** 确定性派生：`g_i = HashToCurve(pp_seed, i)`，其中 `pp_seed` 为公开参数，`g_i` 的离散对数（相对基点 `G`）在随机预言机模型下对包括 TA 在内的所有实体不可知（见 3.3.1）；不需要 TA 秘密持有逐维标量，也不需要存储 `O(d)` 个点。定义 `H(m) = ∏_{i=1}^d g_i^{m_i}`，满足线性同态性（对**向量加法**同态）：`H(a)·H(b) = H(a+b)`。正文统一采用该乘法记号，与 VeriFL/VPFLI 文献口径一致；其 P-256 实现中该群运算为椭圆曲线点加法，即代码中的 `LHH.Mul`，`LHH.Hash` 按 `Σ m_i·G_i` 逐项点累加。定长输出（~32B，与维度 `d` 无关）。在 P-256 离散对数困难（DLP）假设下抗碰撞。

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

本章针对研究点一问题——如何在单 TA 约束下既保护客户端参数隐私、又让边缘方能高效验证云端聚合结果、且验证通信开销不随模型维度膨胀——提出 HVSA 协议。设计目标为：客户端参数隐私、边缘可验证云端聚合、验证**通信**开销与维度无关（LHH 计算本身仍为 O(d)，定长的是验证所传输的哈希）、客户端零验证负担。

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
- **ES 合谋边界**：允许全部 ES 与 Cloud 共享其协议视图；该合谋不改变只保护客户端个人参数、允许聚合结果公开的隐私定义。
- **样本量自报**：`s_i^k` 为客户端自报，防虚报属客户端数据可信问题，出界。
- **最大客户端合谋边界**：对包含实际参与数 `N` 的同一已发布聚合，Cloud 可与至多 `c≤N−2` 个客户端合谋，即始终保留 `N−c≥2` 个非合谋客户端。研究点一的 `N=N_online(t)`，研究点二的 `N=N_Q(t)`。`n_min≥2` 只用于准入检查，不替代按实际参与数定义的合谋阈值。
- **安全信道**：所有实体间通信均通过理想安全信道，消息不会在传输中失真、替换、伪造或被错误归因。TA 完全可信，故 ES 接收的 `M_global/M_Q/H_mask` 等 TA 材料按定义可信；Cloud 提供的聚合结果和声明仍因 Cloud 不可信而必须执行协议规定的验证。
- **单 TA 信任边界**：TA 完全可信且永不参与任何合谋。TA 持有掩码种子与在线集合，不接收 masked model（`w̄_i^k` 或 `W̄_k`），也不外泄单客户端种子/掩码。LHH 逐维生成元由公开种子 hash-to-curve 派生（3.3.1），其离散对数对 TA 同样不可知，因此 TA 不需要再秘密持有 `{a_i}`。TA 向 Cloud 释放完成聚合所必需的聚合掩码 `M_global/M_Q`；释放聚合掩码符合协议设计，不等于释放单客户端掩码。

## 3.3 协议设计

### 3.3.1 初始化与密钥分发

TA 与每个用户经 X25519 ECDH + HKDF 协商共享种子 `_ta_seed`；TA 持有所有用户掩码种子与 ES 公钥列表。此外，LHH 在 Setup 阶段由公开种子经 hash-to-curve 派生逐维生成元：对 `i = 1..d`，`g_i = HashToCurve(pp_seed, i)`（try-and-increment 使 `g_i` 落在 P-256 上），公开参数 `pp_LHH = {pp_seed}`；`g_i` 的离散对数（相对基点 `G`）在随机预言机模型下不可知，因此不存在需要 TA 秘密持有的逐维标量 `a_i`，也无需存储 `O(d)` 个点（见 2.2.1）。

### 3.3.2 本地训练、量化与掩码上传

客户端 `u_i^k` 本地训练得 `w_i^k`，展平量化后加掩码：

```text
w̄_i^k(t) = Quantize(w_i^k) + m_i^k(t)
m_i^k(t) = PRG.mask_int(seed_i, u_i^k, t, d)
```

> 隐私前提：对某一实际发布的聚合结果，若其包含 `N` 个客户端且有 `c` 个客户端与 Cloud 合谋，则须满足 `N−c≥2`。若只剩一个非合谋客户端，其参数可由公开聚合和合谋客户端贡献相减得到。`n_min≥2` 是无合谋时排除单客户端退化的准入下界，并非一般合谋阈值（见 3.4.1 与 4.3.4）。

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
```

> 注：点 1 采用等权平均作为可验证聚合的简化基线，点 2 升级样本量加权以匹配 FedAvg-form；两点口径差异在 4.5 实验中通过分阶段对比处理。

### 3.3.5 边缘验证

ES 验证云端返回的 `W_sum^raw`：

```text
H(W_sum^raw) · H(M_global) = ∏_k H_k(t)
```

成立则聚合结果可信。验证通信为定长哈希，与 `d` 无关。

验证通过后，ES 直接计算并提交共识模型：

```text
W(t+1) = W_sum^raw / n_online
```

这里是对已验证明文和的等权平均，不引入额外的数值恢复步骤。

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

    # 4c: Cloud 按协议使用聚合掩码恢复明文和（尚不生成规范全局模型）
    W̄(t)      ← Σ_{k=1..M} W̄_k(t)                  # int64
    W_sum^raw ← W̄(t) − M_global(t)                  # 整数明文和
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
        aggregated_rhs ← LHH.Mul(aggregated_rhs, H_j)

    # 5b: LHH 同态验证
    H_sum_raw ← LHH.Hash(pp_LHH, W_sum^raw)
    LHS ← LHH.Mul(H_sum_raw, H_mask(t))
    if LHS ≠ aggregated_rhs:
        return REJECT("LHH 一致性等式不成立; Cloud 篡改嫌疑")

    # 5c: 通过后由 ES 直接平均，并将确定性结果确认为下一轮共识模型
    W(t+1) ← W_sum^raw / |U_online(t)|
    commit_consensus(W(t+1))
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

注意口径：**与维度无关的只是验证通信量**（LHH 输出为定长 ~32B 哈希）；ES 侧的 LHH 计算本身仍是 O(d)（对 `W_sum^raw` 逐维哈希）。本文所有"验证开销与维度无关"的表述均指验证通信开销，不得写成"验证计算开销与维度无关"。

## 3.4 安全分析

本章给出 HVSA 在第 3.2 节威胁模型下的三条核心定理：隐私性（Theorem 3.1）、验证可靠性（Theorem 3.2）与掉线鲁棒性（Theorem 3.3）。所有定理均以"安全参数 λ、可忽略函数 negl(λ)、概率多项式时间（PPT）敌手"的标准密码学框架表述；归约目标为本协议直接调用的原语（PRG、LHH、Ed25519、KA）。设掩码域宽度 `R = MASK_HIGH − MASK_LOW ≈ 2·10^9`，模 2^64 整数算术。

### 3.4.1 隐私性

> **Theorem 3.1（聚合泄露下的客户端参数隐私）.** 令 `N=N_online(t)=|U_online(t)|` 为本轮实际进入已发布聚合的客户端数。设 PRG 为安全 CSPRNG，KA（X25519+HKDF-SHA256）在 CDH 假设下安全，TA 完全可信且永不合谋；允许 Cloud 与全部 ES 以及任意 `c≤N−2` 个客户端合谋，即非合谋集合 `H` 满足 `|H|=N−c≥2`。则敌手的联合视图除泄露公开元数据、聚合掩码 `M_global`、全局明文和 `W_sum^raw`、合谋客户端输入，以及由此可推出的非合谋客户端参数总和 `Σ_{i∈H}w_i` 外，不能以非可忽略优势恢复任一指定非合谋客户端的个人参数。

等价地，在固定上述允许泄露量的条件下，存在 PPT 模拟器生成 Cloud、全部 ES 与合谋客户端的联合视图，使之与真实视图计算不可区分。该结论不隐藏 ES/组级聚合结果或 CS/全局聚合结果；它只保护个人参数无法从至少两个非合谋客户端的聚合中被唯一分离。

**Lemma 3.1（PRG 伪随机性）.** 对任意 PPT 区分器 `D`，以未协商种子 `s_i^k` 为视角，`{PRG.mask_int(s_i^k, u_i^k, t, d)}` 与均匀随机向量 `U ← Z_R^d` 的区分优势满足

```text
| Pr[D(m_i^k(t)) = 1] − Pr[D(U) = 1] | ≤ negl(λ).
```

证明：PRG 基于 HKDF-SHA256 派生密钥 + AES-CTR 生成密钥流，再经拒绝采样映射到 `Z_R`。在随机预言机模型下，HKDF 与 AES 均被视为随机预言机/理想密码，密钥流与均匀随机不可区分；拒绝采样在 `R | 2^64` 余项上接受概率 `1 − (2^64 mod R)/2^64 > 1 − R/2^64 > 1 − 2^−33`，引入的统计偏差 `≤ 2^−33`，可忽略。□

**Lemma 3.2（单客户端在线时隐私退化）.** 若 `|U_online(t)| = 1`，则存在 PPT 算法以概率 1 恢复该在线客户端的明文 `w_i^k`。

证明：由 3.3.4，`W_sum^raw = W̄(t) − M_global(t) = w̄_i^k − m_i^k = w_i^k`。Cloud 在 Phase 4c 直接获得 `W_sum^raw`，即该客户端明文。更一般地，当 Cloud 与 `N−1` 个客户端合谋时，可从公开总和减去这些合谋客户端贡献，恢复唯一非合谋客户端参数。□

**Theorem 3.1 证明.** 敌手联合视图包括所有客户端 masked uploads、各 ES 聚合及签名、公开元数据、Cloud 从 TA 收到的 `M_global`、`H_mask=H(M_global)`、`W_sum^raw`，以及合谋客户端主动公开的参数、种子和掩码。移除合谋客户端贡献后，敌手可得到非合谋集合的明文参数和 `Σ_{i∈H}w_i`，这是隐私定义明确允许的聚合泄露。

对 `|H|≥2`，模拟器以允许泄露量为输入：先完整复现合谋客户端视图；再为前 `|H|−1` 个非合谋客户端选择伪随机掩码/一致的 masked share，并用最后一个非合谋客户端的 share 吸收总和约束，使全部非合谋 share 的和同时与公开 `M_global` 和 `W_sum^raw` 一致；最后按这些 share 确定性计算组聚合、LHH 值与签名证据。由 PRG 伪随机性与 KA 安全性，模拟 share 与真实 share 计算不可区分；LHH 和签名只是既定 share 的确定性/可模拟派生量。因此，敌手至多知道非合谋客户端的聚合，无法唯一确定其中任一人的个人参数。

若 `N−c=1`，敌手可由 `W_sum^raw` 减去全部合谋客户端贡献，直接得到唯一剩余客户端参数，隐私以概率 1 失效。因此最大合谋数恰为 `c≤N−2`。这里阈值取决于实际参与数 `N_online(t)`，而不是配置常量 `n_min`。□

### 3.4.2 可验证性

> **Theorem 3.2（验证可靠性）.** 设 LHH 在 P-256 上抗碰撞（Lemma 3.3），Ed25519 满足 EUF-CMA（Lemma 3.4）。若 Cloud 在 Phase 4c 输出的 `W_sum^raw' ≠ W_sum^raw`（真实明文和）使得至少一个 ES 在 Phase 5 验证通过，则可构造 PPT 算法以不可忽略概率找到 LHH 碰撞或伪造 ES 签名。

**Lemma 3.3（LHH 抗碰撞）.** 设 `H: Z^d → G`（`G` 为 P-256 群）是 LHH 同态哈希函数。在 P-256 离散对数困难（DLP）假设下，对任意 PPT 算法 `A`，

```text
Pr[ (x, x') ← A(pp_LHH): x ≠ x' ∧ LHH.Hash(pp_LHH, x) = LHH.Hash(pp_LHH, x') ] ≤ negl(λ).
```

证明：LHH 形式为 `H(x) = Σ_{i=1..d} g_i·x_i`（加性群运算），其中 `g_i = HashToCurve(pp_seed, i)` 由公开种子 hash-to-curve 派生，其离散对数 `a_i = dlog(g_i)` 在随机预言机（ROM）模型下对所有实体不可知（3.3.1）。若存在 `x ≠ x'` 使 `H(x) = H(x')`，令 `δ = x − x' ≠ 0`，则 `Σ g_i·δ_i = O`，即 `Σ δ_i·a_i ≡ 0 (mod q)`——敌手在不掌握任何 `a_i` 的前提下找到一组非平凡线性关系。由 DLP 假设，这在计算上不可行：否则可经 ROM 可编程性将 DLP 挑战 `(G, aG)` 嵌入某个 `g_j = aG`（其余 `g_i` 由归约按已知标量生成），敌手输出的 `δ` 满足 `a·δ_j ≡ −Σ_{i≠j} δ_i·a_i (mod q)`，在 `δ_j ≠ 0` 时可解出 `a`，与 DLP 矛盾。故存在碰撞的概率 `≤ negl(λ)`，抗碰撞性依赖 **DLP + 随机预言机**（而非 TA 秘密持有标量）。□

**Lemma 3.4（Ed25519 EUF-CMA）.** 在 Ed25519 标准假设下，任意 PPT 敌手即使可访问签名预言机，也不能以不可忽略概率对未签名消息 `m` 输出通过验证的签名 `(m, σ)`。

证明：标准 Ed25519 安全性结果，略。□

**Theorem 3.2 证明.** 设 Cloud 输出 `W_sum^raw' ≠ W_sum^raw` 使 ES 验证通过。考虑验证等式：

```text
H_sum^raw' · H_mask(t) = Π_k H_k(t).                    (*)
```

真实等式（按协议执行）：

```text
H_sum^raw · H_mask(t) = Π_k H_k(t).                    (**)
```

在素数阶群上由 (**) 与 (*) 约去 `H_mask(t)`（群运算可逆）得 `H_sum^raw = H_sum^raw'`，即 `LHH.Hash(W_sum^raw) = LHH.Hash(W_sum^raw')`。由 `W_sum^raw' ≠ W_sum^raw`，这给出 LHH 碰撞，与 Lemma 3.3 矛盾。故 Cloud 无法在 LHH 验证下篡改 `W_sum^raw` 而不被检测。

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
LHH.Hash(W̄(t)) = LHH.Hash(Σ_k W̄_k(t)) = ∏_k LHH.Hash(W̄_k(t)) = ∏_k H_k(t),
LHH.Hash(W_sum^raw) · LHH.Hash(M_global) = LHH.Hash(W_sum^raw + M_global) = LHH.Hash(W̄(t)) = ∏_k H_k(t),
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
| 硬件 | AMD Ryzen 7 7745HX (8 核/16 线程) + 31.2 GB RAM + NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, compute capability 8.9) |
| 软件环境 | Windows 10.0.26100 / Python 3.9.15 / PyTorch 2.7.0+cu118 / CUDA 11.8 / cuDNN 9.1.0 / torch 线程数 8 |
| 数据集 | MNIST、FashionMNIST、CIFAR-10 |
| 模型 | CNN（MNIST/FashionMNIST 用 CNN_MNIST，`d`=582,026；CIFAR-10 用 CNN_CIFAR10，`d`=906,890）；安全证据批次另用 Tiny（`d`=7,850，全密码学 verify 端到端） |
| 聚合 | FedAvg-form |
| 层级 | 1 Cloud + M Edge + N Users |
| 密码 | LHH(P-256) / Ed25519 / PRG(AES-CTR/HKDF) / QUANTIZE_SCALE=1e6 |

### 3.5.2 IID 收敛性 E2

在不同客户端数与 edge 数配置下，对比 HVSA 与 FedAvg 在 IID 数据划分下的全局模型收敛曲线。预期 HVSA 收敛精度与 FedAvg 持平，验证掩码+LHH 不破坏聚合正确性。

- **指标**：MNIST/CIFAR-10 上的测试准确率随轮次变化曲线、达到目标精度所需轮次。
- **数值**：正式批次（`results_thesis_core`，5 seeds，`--deterministic`，固定拓扑 5 ES × 4 用户/组）同步臂 `G2` 对 `G1` FedAvg 的同 seed 配对差：

  | 数据集 | 分布 | FedAvg (G1) | 同步 HVSA (G2) | 配对差 |
  |---|---|---|---|---|
  | MNIST | IID | 91.81 ± 0.37 | 91.66 ± 0.60 | −0.15 |
  | FashionMNIST | IID | 70.19 ± 0.35 | 69.94 ± 1.19 | −0.25 |
  | CIFAR-10 | IID | 64.46 ± 0.56 | 64.51 ± 0.77 | +0.05 |
  | MNIST | NI (α=0.5) | 92.24 ± 0.88 | 92.28 ± 0.73 | +0.04 |
  | FashionMNIST | NI (α=0.5) | 70.36 ± 1.03 | 69.98 ± 1.54 | −0.38 |
  | CIFAR-10 | NI (α=0.5) | 63.73 ± 0.68 | 63.03 ± 0.55 | −0.70 |

  **结论**：六个单元配对差最大 `0.70 pp`、其余 ≤ `0.38 pp`，方向无系统性偏差，验证 PRG 掩码 + LHH 聚合层在精度上中性（这是"不破坏聚合正确性"最直接的证据）。
  **边界**：正式批次固定 5 ES × 4 用户的拓扑，"不同客户端数与 edge 数"的扫描未做；`达到目标精度所需轮次`因各臂统一取 20 个 checkpoint 而未单列，需另行补充。
  **口径提示**：NI 行同时属于 3.5.3，此处列出是为与 IID 对照。

### 3.5.3 Non-IID 鲁棒性 N1

在 Dirichlet α=0.5/1.0 划分下重复 E2，验证 HVSA 在 Non-IID 下的鲁棒性。

- **指标**：不同 α 下的最终准确率与收敛轮次。
- **数值**：补充批次 `results_n1_alpha10`（30 runs = G1/G2 × 3 数据集 × 5 seeds，`--noniid --alpha 1.0`，`--deterministic`，代码与正式批次同 `code_digest`）已补齐 **α=1.0**。

  **同步臂 `G2` vs `FedAvg`（G1）在各 α 下的配对差**：

  | 数据集 | α=0.5 配对差 | α=1.0 配对差 |
  |---|---|---|
  | MNIST | +0.04 | −0.31 |
  | FashionMNIST | −0.38 | −0.94 |
  | CIFAR-10 | −0.71 | +0.04 |

  **两个 α 的绝对精度（mean ± sample std，n=5）**：

  | 数据集 | 臂 | α=0.5 | α=1.0 | 配对差（1.0−0.5） | 95% CI | 正种子 |
  |---|---|---|---|---|---|---|
  | MNIST | G1 | 92.24 ± 0.88 | 92.20 ± 0.53 | −0.04 | [−1.07, +0.99] | 3/5 |
  | MNIST | G2 | 92.28 ± 0.73 | 91.89 ± 0.57 | −0.39 | [−1.40, +0.62] | 2/5 |
  | FashionMNIST | G1 | 70.36 ± 1.03 | 70.43 ± 0.99 | +0.07 | [−1.57, +1.70] | 2/5 |
  | FashionMNIST | G2 | 69.98 ± 1.54 | 69.49 ± 1.15 | −0.49 | [−2.41, +1.43] | 2/5 |
  | CIFAR-10 | G1 | 63.73 ± 0.68 | 64.15 ± 0.51 | +0.42 | [−0.68, +1.52] | 3/5 |
  | CIFAR-10 | G2 | 63.03 ± 0.55 | 64.19 ± 0.83 | +1.17 | [−0.25, +2.58] | 4/5 |

  **结论**：①两个 α 下 `G2−G1` 配对差全部落在 `±1 pp` 内且无系统性方向，**掩码+LHH 聚合层在两种异构强度下都保持精度中性**；②**α 从 0.5 变到 1.0 本身对精度无显著影响**（全部 |Δ| ≤ 1.17 pp、95% CI 均跨 0），即本模拟器在该样本规模（每用户约 350 条 / 20 用户）下对 Dirichlet α 的这两档取值不敏感——**这是结论的鲁棒性证据，同时也意味着 α 不宜作为正文的敏感性卖点**。
  **边界**：①仅覆盖 α∈{0.5, 1.0}，更极端划分（α≤0.1）未测；②该结论只覆盖"聚合层不破坏精度"，不涉及半异步（半异步臂见第 4 章）；③α 不敏感是本模拟器样本规模下的实测结果，不应外推为"任意 α 都无影响"。

### 3.5.4 DLG 防御 N2

对客户端上传的 `w̄_i^k` 与解掩码后的 `w_i^k` 分别执行 DLG 梯度反演攻击，对比重建图像与原始图像的 PSNR/SSIM。

- **指标**：PSNR（越高越接近原始）、SSIM（越高越相似）。
- **预期**：masked model 上 DLG 失败（PSNR≈0、SSIM≈0），明文上 DLG 成功。
- **数值**：**已完成**（`exp_n2_dlg_inversion.py`）。攻击按最强设定：攻击者已知模型结构与全局参数、单样本（batch=1）、虚样本与虚标签联合优化、多次重启取损失最小者；损失为归一化梯度 MSE（DLG, Zhu et al. NeurIPS 2019 原配方）。对照组唯一差别是攻击目标——明文组用真实本地更新 `Δ`，掩码组用客户端实际上传量可算出的 `Δ + mask`（`PRG.mask_int`，取值域 ±10⁹）。

  | 模型 | 参数量 | 明文组 PSNR / SSIM | 掩码组 PSNR / SSIM |
  |---|---|---|---|
  | Tiny（线性） | 7,850 | **20.51 ± 0.26 dB / 0.9459 ± 0.0104** | 8.69 ± 1.85 dB / **0.0087 ± 0.0034** |
  | MLP | 199,210 | **18.92 ± 0.79 dB / 0.6432 ± 0.1842** | 8.69 ± 1.85 dB / **0.0087 ± 0.0034** |
  | CNN_MNIST（主矩阵模型） | 582,026 | ≤9.3 dB / ≤0.09（未重建，见下） | 6.64 dB / ~0.01 |

  **结论**：在 Tiny 与 MLP 上，**明文组 DLG 成功（SSIM 0.946 / 0.643）、掩码组完全失败（SSIM 0.009）**，3.5.4 预期成立。

  **掩码组的失败是"零信息"而非"弱重建"**，有两条独立证据：①反演损失在**全部重启中恒为同一常数**（`1.27×10⁻⁴`，即目标归一化后的基线值），说明梯度匹配项对虚样本**不携带任何梯度信号**，优化器无从下手；②**掩码组在 Tiny 与 MLP 上给出完全相同的数值**（`8.69 ± 1.85 dB / 0.0087 ± 0.0034`）——反演结果只等于随机初值，与模型、数据均无关，即掩码后上传量中不含可被该攻击利用的信息。机制上：`‖mask‖ / ‖Δ‖ ≈ 5×10¹⁰`（掩码 ±10⁹ 逐维、581,996 维），目标被均匀掩码完全主导。

  **口径边界**：①`CNN_MNIST`（主矩阵模型的 `d=582,026`）上**明文组在本实现中未重建**（多种配置下 PSNR ≤9.3 dB、SSIM ≤0.09），故该模型上无法给出明文/掩码的对照——DLG 原文的成功实例在更小模型（LeNet 级）上，本实验在 Tiny/MLP 上复现了该量级，但主矩阵 CNN 上的明文反演需另行调优或更多算力；②本实验检验的是**掩码对梯度反演的抵抗**，与 3.5.5 的分布层面检验互补，二者共同支撑 Theorem 3.1 的经验侧；③攻击为最强设定（已知全部公开信息、单样本），是对隐私的下界攻击而非上界。

### 3.5.5 参数分布一致性 N7

统计客户端上传 `w̄_i^k` 在各维度的取值分布，验证掩码后向量服从均匀分布、与明文分布显著不同。

- **指标**：各维度直方图、KL 散度。
- **数值**：**已完成**。以 monkeypatch 捕获真实 run 中每个 `(user, package_token)` 的明文量化向量 `w` 与掩码 `mask`（`PRG.mask_int`，取值域 `[−10⁹, 10⁹)`），并取两者之和作为实际上传向量 `w̄`。样本：MNIST 运行（CNN_MNIST，`d=582,026`）34 次上传，合计 **19,788,884 维**（= 34 × 582,026，与 `comm_bytes_by_type.upload / 8` 精确一致，互为校验）。

  | 量 | mean | std | 5% / 95% | **KL vs 均匀 (bits)** | 逐 bin 最大偏差 |
  |---|---|---|---|---|---|
  | 明文 `w` | −2,830 | 6.60×10⁶ | −9.975×10⁶ / 9.986×10⁶ | **4.8973** | 0.48299 |
  | 掩码 `mask` | 3.14×10⁴ | 5.774×10⁸ | −9.002×10⁸ / 9.001×10⁸ | **0.0000** | 0.00008 |
  | 上传 `w̄` | 2.85×10⁴ | 5.775×10⁸ | −9.002×10⁸ / 9.001×10⁸ | **0.0002** | 0.00135 |

  （60 个等宽 bin；均匀分布的理论 bin 概率 `1/60 = 0.01667`。）

  **结论**：上传向量 `w̄` 的分布与均匀分布的 KL 散度仅 **0.0002 bits**、逐 bin 最大偏差 0.00135，即**统计上不可区分于均匀分布**；明文分布的 KL 达 **4.8973 bits**、逐 bin 最大偏差 0.483，两者**显著不同**（差 4 个数量级）。**3.5.5 的预期成立。**

  **机制侧补充**：`std(mask)/std(w) = 87.5` 倍——掩码量级高出明文约两个数量级，故 `w̄ ≈ mask`，明文的边际贡献被压到测量噪声以下。这给出 Theorem 3.1 的信息论直觉：单次上传若不解除掩码，其中关于 `w` 的信息量可忽略。

  **口径边界**：①本实验验证的是**分布层面的隐藏**（掩码实现是否忠实均匀），**不是** DLG 梯度反演下的可重建性——后者见 3.5.4（未执行）；②复现脚本已入库：`exp_n7_masked_distribution.py`（运行时 monkeypatch 捕获明文/掩码，不改动协议代码，故不影响已冻结批次指纹）；③样本来自单次 MNIST run 的 34 次上传，跨数据集/跨轮次的分布稳定性未单独检验（`PRG.mask_int` 的输出仅依赖 seed 与 token，与模型、数据无关，故该缺口风险低，但正文应如实标注样本范围）。

### 3.5.6 开销对比与验证开销 E4+N3+N5

**(a) 总开销对比 E4**：与不加密 FedAvg、差分隐私方案、其他安全聚合方案对比 HVSA 的计算/通信总开销。

**(b) 扩展开销基线 N3**：补入 VerifyNet、VeriFL、VLMRS、GPVO-FL 等可验证聚合基线的开销对比。

**(c) 验证开销随维度变化 N5**：固定客户端数，扫描模型维度 `d`，验证 HVSA 验证通信开销与 `d` 无关（定长 LHH 哈希）。

- **指标**：用户/ES/Cloud 计算时间、通信字节数、验证通信字节数 vs `d`。
- **预期**：HVSA 验证通信为定长 32B，与 `d` 无关；基线方案验证开销随 `d` 线性增长。
- **数值**：**(a) E4 已完成（实测，见下）；(b) N3 部分完成；(c) N5 通信侧已建立。**

  **(a) E4 跨方案掩码层开销实测**（`exp_overhead_crypto.py`，按 `docs/OVERHEAD_EXPERIMENT.md` 冻结规格；四个方案实现取自仓库内 `baselines/`，median of 5 次）。

  *Exp-A：固定 `d`=10,000，扫描 `N`*

  | 方案 | N=500 | N=1000 | N=2000 | 用户计算随 N | 云侧入流随 N |
  |---|---|---|---|---|---|
  | VeriFL（Bonawitz 掩码层，全对 P2P） | 10,133 ms | 20,289 ms | 40,593 ms | **O(N) 线性** | 35.9 → 326.5 MB（**O(N²)** 主导） |
  | VLMRS（Flamingo 图邻居） | 1,228 ms | 1,146 ms | 1,292 ms | ~平坦（度 ~3log₂N） | 20.9 → 84.4 MB（**O(Nd)**） |
  | GPVO-FL（分组 + 验证外包） | 40.8 ms | 41.0 ms | 40.9 ms | **O(1)** | 0.381 MB **恒定** |
  | **HVSA（本方案）** | **0.16 ms** | **0.16 ms** | **0.16 ms** | **O(1)** | **0.191 MB 恒定** |

  *Exp-B：固定 `N`=1000，扫描 `d`*

  | 方案 | d=10k | d=100k | 云侧入流 d=10k → 100k |
  |---|---|---|---|
  | VeriFL | 20,289 ms | 20,727 ms | 102.3 → 473.1 MB |
  | VLMRS | 1,146 ms | 1,171 ms | 42.0 → 399.1 MB |
  | GPVO-FL | 41.0 ms | 43.2 ms | 0.381 → 3.815 MB |
  | **HVSA** | **0.16 ms** | **1.27 ms** | **0.191 → 1.907 MB** |

  **结论**：`N`=2000、`d`=10k 时，HVSA 用户计算 **0.16 ms** vs VeriFL **40,593 ms**（**约 2.5×10⁵ 倍**），云侧入流 **0.191 MB** vs **326.5 MB**（**约 1.7×10³ 倍**）；`N`=1000、`d`=100k 时分别为 **1.63×10⁴ 倍**与 **248 倍**。**HVSA 是四者中唯一在 `N` 维度上用户计算与云侧入流双双为 O(1) 的方案**——因为单 TA 掩码不需要任何用户间 P2P，且云侧只接收 M=5 个组向量。GPVO-FL 同为 O(1)，但其云侧入流为 HVSA 的 2 倍（多一份掩码/验证向量）。VLMRS 的用户侧靠图邻居压到 ~平坦，但**云侧仍收 N 份用户更新，入流仍是 O(Nd)**。

  > ⚠ **口径边界（引用时必须随表标注）**：①本基准范围 = **掩码与云侧入流量**，**不含 LHH、签名与双线性配对验证**，因此它回答的是"掩码层谁更省"，不是"完整方案谁更省"；②**VeriFL 曲线是其安全聚合掩码路径，不是完整 VeriFL 验证流程**；③VLMRS 云侧为 O(Nd) 主导，**不是** O(log N)；④HVSA 云侧为 O(Md)，与 N 无关但随 d 线性。**这四条来自冻结规格的 caption checklist，正文与图注均须保留。**

  **(b) N3 部分完成**：`baselines/` 内已实现并纳入上表的有 **VeriFL、VLMRS、GPVO-FL** 三篇可验证聚合基线；**VerifyNet 未实现**（四篇中唯一缺项），其开销数据只能引原文，或需另行实现——正文应如实区分"本环境实测（前三者）"与"引自原文（VerifyNet）"，不得混列。

  **方法说明（影响可辩护性，须在正文交代）**：上表基线为**本仓库实现**，按各方案原文的掩码层口径复现，非其官方代码。若审稿质疑实现忠实度，退路是同时给出各基线原文报告值与"引自原文、非本环境复现"的标注（见 4.5.2' 的口径纪律）。

  **(c) N5 验证通信维度无关性**。按 `main_point2._CommAccount` 的实现公式拆分（拓扑 5 ES × 4 用户/组，K=3）：

  | 组成 | 是否随 `d` | 每 checkpoint 字节 |
  |---|---|---|
  | package 内 `H_k`(33B) + `σ_k`(64B) + online 列表(48B×4) | **否** | 3 × 289 = 867 B |
  | manifest 内逐组包元数据 `(33+64+48)×|Q|` | **否** | 3 × 145 = 435 B |
  | vote(256B×5) + receipt(320B) + notify(128B×6) | **否** | 2,368 B |
  | **验证元数据合计** | **否** | **3,670 B** |
  | 模型载荷（upload 8B/维 + package 8B/维 + manifest 16B/维） | 是 | `40·d + 96` B |

  | 模型 | `d` | 载荷/checkpoint (`40d+96`) | 验证元数据占比 |
  |---|---|---|---|
  | Tiny（安全证据批次） | 7,850 | 314,096 B | 1.155% |
  | CNN_MNIST（主矩阵 MNIST/FashionMNIST） | 582,026 | 23,281,136 B | 0.016% |
  | CNN_CIFAR10（主矩阵 CIFAR-10） | 906,890 | 36,275,696 B | 0.010% |

  **结论**：验证元数据**定长 `3,670 B/checkpoint`、与 `d` 无关**，且随 `d` 增大其占比单调下降（Tiny `1.155%` → CNN_MNIST `0.016%` → CNN_CIFAR10 `0.010%`）——**创新点 1-b 的定长验证通信主张成立**。实测交叉验证：主矩阵两个真实 run（FashionMNIST 与 CIFAR-10）的分类型占比**完全相同**（upload `72.6%` / package `19.6%` / manifest `7.8%`），证明各项均随 `d` 线性、占比与 `d` 无关；维度另由 upload 字节数反推校验（`8·d·371` 精确整除，`d` 分别为 `582,026` / `906,890`）。

  **计算侧（已有实测，`exp_security_overhead.py`）**：LHH.Hash 成本 `≈188 μs/维`（线性），外推全维 `d=582,026` 单次 Hash `≈109.5 s`；Ed25519 签名 `27.6 μs` / 验签 `78.8 μs`；PRG 掩码 `0.01 μs/维`。即 **LHH 计算为 O(d)、验证通信为 O(1)**——这正是 3.4.2 与本章小结的表述口径。

  > ⚠ **口径警告（写作必须遵守）**：`comm_bytes_by_type` 的 `manifest` / `package` 两个桶**混装了模型载荷与验证元数据**（`manifest` 里的 `16B×d` 是 Step 2 需交给 ES 的 `W_sum^raw`，属载荷而非验证开销）。因此**直接画"manifest 字节数 vs d"会得到随 d 线性增长的曲线，看似推翻定长主张**——那是把载荷误当验证开销。正文必须报告上表的**元数据子集**，并显式说明拆分依据。

  **(a) E4 / (b) N3 未执行**：与不加密 FedAvg、DP 方案、VerifyNet/VeriFL/VLMRS/GPVO-FL 的开销对比需按各基线原文口径复现或引用，当前未做。
  **其他口径提示**：主矩阵各臂为 verify-off，其验证耗时列（`0.038–0.071 s/ckpt`）只是提交事务开销，**不构成"验证开销"证据**，不得据此宣称"验证零开销"；verify-on 实测见 4.5.3（Tiny 口径 `10.46–11.99 s/checkpoint`）。
  **待补**：`comm_counts_by_type` 未落盘，上表由实现公式推导 + 缩放关系实测验证得出；若要给出逐 `d` 的实测曲线，需先落盘分类型计数并补一次 d-sweep（主矩阵的 `d=582,026` 与 `d=906,890` 两个真实维度已覆盖）。

## 3.6 本章小结

本章提出 HVSA——单 TA 分层可验证安全聚合协议。协议通过 PRG 掩码保护客户端参数隐私（Theorem 3.1），通过 LHH 同态哈希与 Ed25519 签名让 ES 以定长**通信**开销验证云端聚合结果（Theorem 3.2；LHH 计算仍为 O(d)），并通过"在线集合驱动的掩码求和"实现对客户端掉线的自然鲁棒性（Theorem 3.3）。实验框架覆盖 IID/Non-IID 收敛、DLG 防御、参数分布一致性、开销对比与验证通信开销维度无关性五个方面。HVSA 的同步聚合语义留下"全组同步 barrier"与"掉线训练浪费"两个工程问题，下一章将以检查点式组级半异步扩展解决。

---

# 第 4 章 检查点式组级半异步安全聚合扩展

## 4.1 引言

本章的主问题定义为三点：**跨组同步等待、mixed-version 偏差、恶意 Cloud 下的聚合策略完整性**。第 3.4.4 节指出 HVSA 的同步语义留下的工程现象可归并为：

1. **(i) 跨组同步等待**——全组同步 barrier 要求所有 ES 在 Phase 4c 前到齐，墙钟时间受最慢组制约；
2. **(ii) mixed-version 偏差**——半异步下同一 checkpoint 的选中组包基于不同基版本 `W(t_k)`（`t_k<t`），直接平均引入版本失配偏差；
3. **(iii) 恶意 Cloud 下的聚合策略完整性**——一旦聚合从等权扩展到加权（V-PDA/σ）并引入补偿，权重与补偿策略若可由 Cloud 声明，恶意 Cloud 即可自行选择权重、重分配各组贡献，且不受密码学检测（见下）。

此外，客户端完成训练但未赶上封组的更新被直接丢弃（掉线训练浪费，3.4.4 (ii)）作为附带现象处理：由安全聚合兼容的组内 stale reuse 组件以掩码前本地混合的方式回收（4.3.3），其收益边界如实呈现（4.5.4、附录 A.3.3），不承担本章主目标。逐轮在线验证与完全异步的兼容性问题（3.4.4 (iii)）则是架构选择的出发点：验证粒度需与掩码聚合粒度一致，因此本文不采用"来一个更新一个"的完全异步，而是主动收缩异步边界（见下）。

**核心论点：主动把异步边界收缩到 checkpoint 级。** 组内仍同步封组，组间用 buffer+checkpoint 半异步；Cloud 保存组级掩码包而非明文，验证粒度收缩到 checkpoint 选中集合。这一架构决策给出三个主问题的解：

- **问题 (i) → buffer+checkpoint 半异步（创新点 1）**：跨组同步 barrier 扩展为"组包缓冲 + checkpoint 触发"，TA 在线求掩码的频率从"逐更新"降到"每 checkpoint 一次"（4.3.5 按选中集合 `Q(t)` 释放 `M_Q`），仍保持 LHH 定长验证。
- **问题 (ii) → Direct-VCM 组级后置补偿（创新点 3，支撑机制 B）**：mixed-version 组包在解掩码后以公开确定性方式补偿 `C_k=S_k·γ_vcm·(W(t)−W(t_k))`，ES 基于已验证 raw sum 本地重构最终模型（4.3.7）。
- **问题 (iii) → 策略绑定的可验证加权聚合（创新点 2，支撑机制 A）**：整数加权 LHH `∏_k H_k^{w_k}` 本身只是线性同态的直接推广，且 LHH 等式对任意整数权重都成立——权重的正确性不能靠 LHH 保证，只能靠"公开确定性策略规则 + 来源声明绑定 + ES/TA 独立复算"保证：**恶意云端不能自行选择权重**（4.3.10）。

三个创新点不是并列拼装，而是同一架构决策（checkpoint 级异步边界）依次引发的必要组件：半异步产生 mixed-version，mixed-version 需要补偿，加权与补偿则要求策略绑定（见附录 A 的组织逻辑）。

本章创新点（措辞定稿）：

- **创新点 1（核心贡献）**：单 TA 与 LHH 验证约束下的 checkpoint 组级半异步安全聚合协议。
- **创新点 2（支撑机制 A）**：策略来源绑定、TA/ES 独立复算与 fail-closed 验证——面向公开确定性聚合策略（V-PDA/σ），恶意云端不能自行选择聚合权重或补偿策略参数（α/γ/mode），一切可采纳权重类均须可由边缘依据共识量独立复算（4.3.10）。
- **创新点 3（支撑机制 B）**：面向 mixed-version 组包的公开确定性后置补偿（Direct-VCM）与 checkpoint 验证机制。

固定 σ 与 Aσ 仅作为可选策略实例和消融呈现，不列为独立创新；stale reuse 只保留为负结果与特殊场景边界（附录 A.3.3、A.5）。

> 措辞纪律：创新点 2 的新颖性**不在**"整数加权 LHH 能乘权重"（`∏ H_k^{w_k}` 是线性同态的直接推广，不足以立住）；可立住的表述是策略绑定三元组（策略绑定、边缘独立复算、可验证加权聚合）。新颖性声明一律以检索日志与全文比较为前提（1.2、第 2 章综述），未经支持不写"首次"。

## 4.2 系统模型扩展与威胁模型

### 4.2.1 系统模型扩展

继承第 3 章四元架构（Cloud / ES / User / TA），新增组包缓冲区与 checkpoint 触发机制。Cloud 维护组包缓冲区 `B`，每组至多一个在途包；按 checkpoint 触发规则（4.3.6）选择集合 `Q(t)` 进行批量解掩码与聚合。

### 4.2.2 威胁模型扩展

继承 3.2.2 全部假设，并扩展：

- **Malicious Cloud（点 2 适用）**：可篡改聚合结果、丢弃或重放组包。防 Cloud 篡改 checkpoint **选中**的聚合结果（`W_sum^raw`）。
- **策略声明不可信（点 2 适用）**：Cloud 不得自行选择聚合权重或补偿策略参数。checkpoint 中的一切加权量（V-PDA 逆参与权重、σ 陈旧度整数权重）与补偿策略参数（衰减系数 `α_k`、Aσ 的 `σ*` 档位、`γ_vcm`、`vcm_mode`、momentum）要么由 ES 依据共识量与本地协议配置独立复算比对，要么声明值与配置不符即拒绝（4.3.10）。ES 的全部策略配置由 TA/本地管理员在初始化时预置并本地持有，不由 Cloud 下发。特别是 Cloud 不得把伪造权重改标为其他权重来源以绕过复算，也不得把预定的权重来源降级为等权（`none`）或在合法来源间择优——权重来源本身由 ES 本地协议配置唯一确定（4.3.10 来源绑定），每个可采纳权重来源都必须有一条 ES 可复算的公开规则；凡配置要求衰减而度量不可复算或所需共识历史/α 被省略，一律 fail-closed 拒绝。
- **安全目标与显式出界（点 2）**：主协议只提供 checkpoint 选中聚合结果的完整性检测；检测到攻击或不一致即丢弃本次 checkpoint，不推进全局版本。它不保证恶意 Cloud 下的训练可用性，也不提供 anti-censorship 保证；manifest 漏选、候选组包静默丢弃和调度审查均不属于主协议安全目标。全局模型由 ES 基于已验证结果本地确定性重构并作为共识量维护，不把 Cloud 分发的模型作为规范结果。
- **至多一次纳入（重放防护）声明**：每个组包携带唯一且被 ES 签名绑定的 `package_token`，ES 封包后即向 TA 注册该 token。TA 以仅追加的持久化 ledger（SQLite 主键约束）为**唯一权威状态**，记录 token 注册、token 消费与 checkpoint 版本；Cloud 仅持非权威缓存，可重启或被篡改而不影响权威状态。checkpoint 提交需经权威授权流（prepare/finalize 两阶段）：全部 ES 对候选投票（签名 vote，绑定 `result_digest` = 已验证的 raw sum/M_Q/final model 摘要）→ TA 验票、核验 token 已注册且未消费、核验调度约束与版本连续性 → 签发 `CheckpointReceipt`（**此时不写 ledger**）→ Cloud 凭有效 receipt 提交 → 提交成功后 TA 才原子落地 ledger（版本 + token 消费）；ES 见有效 receipt、结果摘要一致且 ledger 已确认才推进版本与策略状态（4.3.9）。本协议提供的是基于唯一 token、TA 权威 ledger 与 receipt 授权的**重放防护（至多一次纳入）**，不声称完整分布式 exactly-once——完整 ACK 与原子提交留作展望（5.2）；ACK/缓存并发属于实现层，不作数学前提。
- **单 TA 信任假设**：同 3.2.2，TA 完全可信且永不参与任何合谋。

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

要求最小在线人数 `n_min≥2`，用于无合谋时排除单客户端组包；一般隐私阈值仍按整个 checkpoint 的实际客户端数 `N_Q−c≥2` 判定。组包：

```text
Π_k(t_k) = (package_token, k, t_k, h_base(t_k), U_online, n_k, S_k, W̄_k, H_k, σ_k)
```

其中 `σ_k` 为 ES 对最小元数据 `(package_token, k, t_k, h_base(t_k), digest(U_online), n_k, S_k, H_k)` 的 Ed25519 签名；该元数据即 4.3.8 Step 1 manifest 一致性检查的比对对象。绑定 token 可阻止 Cloud 为合法 payload 改名后绕过重放检查。

### 4.3.5 TA 的组包掩码支持

TA 按参与集合计算 `M_k=Σ m_i^k`，**不在组包入 buffer 时单独释放**；仅当 checkpoint 选中集合 `Q(t)` 确定后返回 `M_Q=Σ_{Π_k∈Q} M_k`。

### 4.3.6 Buffer 与 Checkpoint 触发

Cloud 维护组包缓冲区 `B`，每组至多一个在途包。过期判据 `s_k=t−t_k>s_max` 移除。

- 常规：`|B_legal|≥K` 时按到达顺序选最早 `K` 个。
- 超时：最早合法组包等待达 `Δ_timeout` 且 `1≤|B_legal|<K` 时，对全部合法组包保守 checkpoint。

发布 `checkpoint_manifest(t)`（含 mode、|Q|、各组包元数据、**协议配置摘要 `policy_config_digest`** 与规范 checkpoint 版本）及公开摘要 `h_manifest`，先广播给 TA 与 ES。

`policy_config_digest` 是版本化 `ProtocolConfig`（绑定 `K`、`s_max`、权重来源、σ 策略及参数、`γ_vcm` 与 VCM mode、momentum、V-PDA 参数、配置版本号）规范 JSON 序列化的 SHA-256。配置在系统初始化时由 TA/本地管理员预置并同步持有于 TA 与全部 ES，不由 Cloud 下发；ES 校验 manifest 声明的摘要与其本地配置一致（缺失或错配即拒绝，fail-closed），使"云端另选策略"在配置层面即被排除（4.3.10）；TA 同样依据该配置与本地参与历史独立复算权重，云端声明值只作审计。

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
  // 输入：checkpoint_manifest(t) = (t, mode, |Q|, {(package_token, k, t_k, h_base, U_online, n_k, S_k, H_k, σ_k)}_{Π_k∈Q})
  //       B = Cloud 维护的组包缓冲区
  accepted ← ∅
  for each Π_k ∈ Q:
    1a. // —— manifest 一致性 ——
        从 B 中按 package_token 检索到唯一组包 Π_k^B
        若检索失败（B 中无对应组包 / 多条匹配）→ reject Π_k
        逐字段比对 manifest 元数据与 Π_k^B 的元数据：
          package_token、k、t_k、h_base、U_online、n_k、S_k、H_k 必须完全一致
        若任一字段不一致 → reject Π_k            // tamper_manifest
    1b. // —— Ed25519 验签 ——
        pk_ES_k ← 从 PKI 获取 ES_k 的 Ed25519 公钥
        meta_k ← (package_token, k, t_k, h_base(t_k), digest(U_online), n_k, S_k, H_k)
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

### 4.3.9 状态转移与重放防护（至多一次纳入）

被选中组包移出 buffer、对应组获 `W(t+1)` 并开始下一轮；未选中组包继续等待或过期。客户端级/组级状态分界清晰，不反向生成 stale 候选；唯一 ID + 拒绝重放提供"至多一次纳入"的重放防护。

#### 唯一 ID 生成规则

每个组包 `Π_k(t_k)` 在入 buffer 时由 ES 分配全局唯一的 `package_token`：

```text
package_token = f"{edge_id}:{base_version}:{nonce}"
// edge_id      = ES 的全局唯一标识（PKI 注册）
// base_version = h_base(t_k)，本组本轮训练所基于的全局模型版本摘要
// nonce        = ES 维护的单调递增计数器，每次封组 +1（持久化）
```

`package_token` 作为组包元数据的一部分写入 manifest 与 buffer 索引。`base_version` 嵌入 ID 保证"基于同一全局模型版本的同组重复封组"不会被误认为重放；`nonce` 保证同一 ES 在同一 `base_version` 下的多次封组可区分。

#### TA 权威 ledger 与授权流（token 注册 → vote → receipt → 提交 → 推进）

重放防护的权威状态完全置于 TA：TA 以仅追加的持久化 ledger（SQLite `PRIMARY KEY`/`UNIQUE` 约束）记录四类事实——(i) `registered_tokens`（ES 封包时注册的合法 token）、(ii) `consumed_tokens`（已随终结 checkpoint 消费的 token）、(iii) `checkpoints`（`version` 主键 + `receipt_id` 唯一）、(iv) `prepared_checkpoints`（prepare 阶段原子保存的完整 receipt 字节 + token 元数据 + `result_digest`，按版本唯一）。Cloud 的 buffer/`aggregated_ids` 只是非权威缓存：重启、清空或篡改均不影响权威状态，也不能使任何 token 被二次消费。

授权分 **prepare / finalize 两阶段**，并叠加以 stage 为起点的提交事务（P0 修复）：

```text
PrepareCheckpoint(Q, {vote_e}, v, trigger_mode, config) -> receipt | REJECT
  // 1. vote 完整性: 覆盖全部 ES、验签、passed、
  //    v / token_digest / config_digest / result_digest 一致
  // 2. 调度约束: normal ⇒ |Q| = K; timeout ⇒ 1 ≤ |Q| < K;
  //    ∀Π_k ∈ Q: 0 ≤ v − t_k ≤ s_max
  // 3. token 状态: 全部已注册 (ES 封包登记) 且未被消费 (ledger 权威)
  // 4. 版本连续性: v == ledger 下一个未用版本号
  // 5. 返回 receipt = SIG_TA(receipt_id, v, token_digest,
  //                           tokens, trigger_mode, result_digest),
  //    并原子写入持久化 prepared_checkpoints: 完整 receipt 字节 +
  //    token 元数据 (token→edge_id/base_version) + result_digest
  //    —— prepare 仍不消费 token、不推进版本

FinalizeCheckpoint(receipt) -> committed | already | REJECT  // Cloud 提交成功后调用
  // 1. 校验 receipt 的 TA 签名 —— 伪造 receipt 直接拒绝
  // 2. 核对同 receipt_id 的 prepare 记录, 摘要逐字节一致
  //    —— 未经 prepare 签发 / 内容被改的 receipt 无法落地
  // 3. 幂等: 该记录已 finalize ⇒ 返回 already (重放不重复消费)
  // 4. 原子事务: 写 checkpoints(v, receipt_id, token_digest) +
  //    consumed_tokens(全部 token, 带真实 edge_id/base_version) +
  //    标记 prepare 记录 finalized; 任一约束冲突 ⇒ 整体回滚
```

提交事务 **stage → TA finalize → publish → notify**（`commit_tx.run_commit_transaction`）：stage 只做只读校验（receipt 签名/版本/token/结果摘要、buffer 原包仍在）并把"提交成功要应用的变更"写入持久化 journal，**不触碰 Cloud 权威状态**；TA finalize 成功（ledger 落地）后才 publish（消费 buffer、版本 +1、写模型历史）并 notify（TA 与各 ES 幂等推进共识状态）。TA finalize 被协议拒绝时丢弃 stage，Cloud/TA/ES 三者均停在原版本 —— 旧实现"Cloud 已本地提交而 TA 未落地"的分叉不再出现；finalize/publish/notify 三阶段幂等，进程内异常可在 journal 上重放收敛。

**结果摘要绑定 `result_digest`**（P0-1）：`result_digest = H(v, token_digest, raw sum, M_Q, final model)`。ES 在投票时把"自己实际验证过的聚合结果"钉进 vote；TA 要求全部 vote 的 `result_digest` 一致并写入 receipt；Cloud 提交时按提交内容重算并与 receipt 比对（不一致即拒绝）；ES 收到 commit 通知时同样要求 receipt 的 `result_digest` 等于本地已验证候选的摘要、通知模型与已验证模型一致，且 receipt 已在 ledger 落地（`confirm_receipt`，核对**完整 receipt**：签名 + prepare 记录 + 已 finalize，而非只查 `receipt_id`）才推进。因此**取得诚实 receipt 后再替换 `raw sum`/`final model` 会在 commit 层与 ES 侧双重被拒**。

两阶段语义带来三条明确结论：(a) **提交失败（commit 守卫拒绝、finalize 核对失败、异常中断）不推进任何权威状态** —— token 未被消费、版本未推进，可安全重选重试，不存在"更新被授权但未聚合"的悬空消费；(b) **至多一次消费且可安全重放** —— 同一 receipt 重复 finalize 幂等返回 `already`；对同一版本提交两份不同 receipt 则因与 prepare 记录摘要不符被确定性拒绝；同版本换候选时新 prepare 替换记录、旧 receipt 随之作废；(c) **提交事务可恢复** —— stage journal 与三阶段幂等使"finalize 成功但 publish/notify 未完成"（含"仅通知部分 ES 后崩溃"的 Cloud v+1 / ES v 分叉）可在进程内重放收敛。需要说明的是跨进程恢复还要求 ES/TA 共识状态持久化（本实现的 ES 状态是内存态），故进程重启后遗留 journal 一律 fail-closed 丢弃；完整跨节点 ACK/原子提交仍在出界范围（5.2）。

ES 收到 commit 通知时**只有 receipt 验签通过、版本严格等于本地已提交版本 +1、结果摘要与本地已验证候选一致、且 receipt 已在 TA ledger 落地**，才推进参与计数、版本历史与 Aσ 策略状态——重放通知、跳版本通知与结果调包均被拒绝；通知版本 ≤ 本地已提交版本时幂等跳过（崩溃恢复重发通知不重复计数）。

#### 唯一 ID 生成规则

每个组包 `Π_k(t_k)` 在入 buffer 时由 ES 分配全局唯一的 `package_token`：

```text
package_token = f"{edge_id}:{base_version}:{nonce}"
// edge_id      = ES 的全局唯一标识（PKI 注册）
// base_version = h_base(t_k)，本组本轮训练所基于的全局模型版本摘要
// nonce        = ES 维护的单调递增计数器，每次封组 +1（持久化）
```

`package_token` 作为组包元数据的一部分写入 manifest 与 buffer 索引，并在封包时经 `register_token` 登记入 TA ledger。`base_version` 嵌入 ID 保证"基于同一全局模型版本的同组重复封组"不会被误认为重放；`nonce` 保证同一 ES 在同一 `base_version` 下的多次封组可区分。

#### Cloud 侧非权威缓存与去重逻辑

Cloud 维护只读镜像 `aggregated_ids` 以加速筛选（非权威，可重启重建）。组包入 buffer 与 checkpoint 选中时做本地去重检查：

```text
CheckExactlyOnce(Π_k, buffer_phase, aggregated_ids):
  // buffer_phase ∈ {"on_enqueue", "on_select"}
  if Π_k.package_token ∈ aggregated_ids:
    return REJECT_REPLAY                            // 本地缓存命中，快速拒绝
  if buffer_phase == "on_enqueue":
    // 同 (edge_id, base_version) 在 buffer 中不应已存在
    // （ES 单组在途规则，4.3.6）
    for Π_k' ∈ B:
      if Π_k'.edge_id == Π_k.edge_id and Π_k'.base_version == Π_k.base_version:
        return REJECT_DUPLICATE_INFLIGHT
  return ACCEPT
```

该本地检查只作快速路径；**权威拒绝由 TA ledger 在授权时给出**（缓存被清空或篡改不构成绕过）。

### 4.3.10 策略绑定的可验证加权聚合

第 3 章主协议是等权聚合；研究点 2 的实验主线需要两类非等权扩展：**V-PDA 逆参与权重**（对高参与组降权，缓解参与偏差）与 **σ 陈旧度权重**（对陈旧组包降权或衰减其补偿项）。本节给出把这两类权重纳入验证链路的协议语义，其核心不是"整数加权 LHH 能乘权重"——`∏_k H_k^{w_k}` 只是线性同态的直接推广——而是**恶意云端不能自行选择权重**。

#### 掩码域整数化与加权验证等式

EC 群标量乘只支持整数标量，故浮点权重须整数化后进入掩码域（共用实现，云端与 ES 同一函数）：

```text
w_k = round(α_k · 2^16)                     (整数化, w_k > 0)
M_Q^w  = Σ_{Π_k∈Q} w_k · M_k                (TA 按声明权重构造加权掩码和)
W̄_Q^w = Σ_{Π_k∈Q} w_k · W̄_k                (云端加权聚合)
W_sum^raw = W̄_Q^w − M_Q^w = Σ w_k·(样本量加权和)_k
```

LHH 同态验证等式相应变为：

```text
H(W_sum^raw) · H(M_Q^w) = ∏_{Π_k∈Q} H_k^{w_k}
```

**关键观察：该等式对任意正整数权重 `{w_k}` 都成立**（对每个 k，`H(w_k·W̄_k) = H_k^{w_k}` 由线性同态直接给出）。因此 LHH 等式约束的是"raw sum 与声明权重的掩码算术一致"，**不约束权重本身的取值**。若权重可由 Cloud 自由声明，Cloud 即可任意重分配各组贡献——这不是密码学可检测的攻击，必须由策略层消除。

#### 权重来源与 ES 独立复算

协议要求每个 checkpoint 声明 `weight_source ∈ {none, vpda, staleness}`，且**每个可采纳来源都必须对应一条 ES 可独立复算的公开确定性规则**：

| 来源 | 公开规则（ES 复算输入） | 输入可信性 |
|------|------------------------|-----------|
| `none` | 等权 `w_k=1` | — |
| `vpda` | `w_k = round(2^16·clip((mean_j(A_j+1)/(A_k+1))^κ, w_min, w_max))`，均值在全部组上取 | 参与历史 `A_k` 由各 ES 从每个成功提交的 checkpoint 选中集合独立累计（commit 通知驱动），Cloud 无法伪造 |
| `staleness` | `α_k` 由陈旧度度量与协议超参 `σ` 决定：`version` 度量 `α_k=(v−r_k+1)^{−σ}`；`rel2` 度量 `α_k=exp(−σ·‖W(v)−W(r_k)‖/‖W(v)−W(0)‖)` | `v` 与 `r_k` 为共识版本（`r_k` 由组包签名绑定，`v` 由 ES 本地已提交版本计数绑定），历史模型为共识值 |

`wait` 度量（按云端缓冲等待时间衰减）依赖 Cloud 的调度状态，**不存在 ES 可复算的公开规则**，故不可作为策略绑定的权重来源；开启验证时一律拒绝（fail-closed）。

**来源绑定**：采用哪个权重来源不是云端可选项——它由协议配置（`participation_debias`、`sigma_target`、`σ`）唯一确定，并以 `expected_weight_source` 形式在 ES 本地持有（与云端同源接线）。ES 验证时**首先**核对 `declared_source == expected_source`，不符即拒绝；来源核对通过后，按该唯一规则独立复算 `{w_k}` 并与声明值逐项比对，同时按同一公开规则复算补偿系数 `{α_k}`（固定 `σ` 或 Aσ 的 `σ*` 档位）并与声明值比对。**策略配置的预置方**：ES 的全部策略配置（权重来源、σ 及其度量、γ/mode/momentum、Aσ 对齐阈值）在系统初始化阶段由 TA/本地管理员预置并本地持有，不属于任何 Cloud 可下发的协议消息——策略"来自哪"因此是配置问题而非传输问题，其正确性不依赖对云端声明的任何信任。由此：

- **来源降级/择优**（把预定的 `vpda`/`staleness` 降级为 `none` 静默退回等权、或在两条合法规则之间声明更有利的一条）：来源核对第一步即拒绝——这类攻击不伪造任何数值、只改变"本应执行的策略"，仅靠数值复算拦不住，必须由来源绑定关闭；
- **伪造权重**（任选 `{w_k}`）：与该唯一来源的复算值不匹配 → 拒绝；
- **改标攻击**（把伪造的 V-PDA 权重声明为 `staleness`，或反之）：来源核对即拒（声明来源 ≠ 期望来源）；数值层面两条规则的复算值也不同，改标不提供任何绕过路径；
- **策略参数篡改**（`γ_vcm`/`vcm_mode`/momentum/`σ*`）：这些量为 ES 预置的协议配置，声明值与配置不符即拒绝；`α_k` 与 `σ*` 一律由 ES 复算，云端声明值仅用于比对。

**复算失败的 fail-closed 语义**：协议严格区分"合法策略不产生 α"（σ=0 等权补偿）与"策略不可复算"（`wait` 度量含云端调度状态、rel2 所需共识历史缺失）两种情形，二者不共用任何返回值/消息形态。凡 ES 本地配置要求衰减而度量不可复算，或 bundle 缺少补偿所需共识历史（含 `W(v)` 本身——省略该条目等价于静默关闭 VCM 补偿），ES 一律确定性拒绝：云端**省略 α 或历史条目**与伪造它们同样是攻击，不能把衰减策略静默关闭。

#### 版本与历史的绑定

σ 权重与 VCM 补偿都以 checkpoint 版本 `v` 和基版本 `r_k` 为输入，必须钉死：`r_k` 已由组包签名的 `base_hash` 绑定到共识历史；`v` 由 ES 本地维护的已提交版本计数（每次 commit 通知推进）与本地已提交模型历史绑定——bundle 声明的 `checkpoint_version` 必须等于 ES 本地计数，声明的 `W(v)` 必须与本地历史一致，Cloud 谎报版本以配合伪造权重或补偿值即被拒绝。

#### 提交语义与状态落地

ES 在验证阶段对策略量只做**试探性复算与比对**；Aσ 的方向探测状态（EMA/探测计数/档位锁定）仅在 checkpoint 成功提交后由 commit 通知落地。验证失败或未提交的候选不推进任何一侧（云端与 ES）的策略状态，保证双方状态序列一致，不因攻击候选漂移。

#### 安全语义小结

策略绑定不引入新密码学原语，也不改变 Theorem 4.1 的归约对象（`W_sum^raw` 仍由 LHH 等式保护）。它把"权重与补偿策略"从 Cloud 的自由声明空间移入公开确定性规则空间：对可复算来源，伪造/改标成功当且仅当伪造值等于公开规则的复算值（此时二者无差别，不构成攻击），未检测概率为 0；对不可复算来源（`wait`），协议直接拒绝，fail-closed。代码实现见 `edge.py::p2_verify_checkpoint`（复算与比对）、`utils.py`（共享公开规则：`vpda_weights`/`staleness_alpha`/`staleness_int_weights`/`AlignmentSigmaPolicy`）。

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

**附注 4.2（weight/policy tampering，策略绑定）.** 加权路径（4.3.10）下，LHH 验证等式 `H(W_sum^raw)·H(M_Q^w)=∏_k H_k^{w_k}` 对**任意**正整数权重 `{w_k}` 成立——权重本身不是 LHH 的约束对象。因此"Cloud 伪造权重 / 把伪造权重改标为其他来源 / 篡改补偿策略参数"不由 Theorem 4.1 的归约覆盖，而由 4.3.10 的策略绑定机制覆盖，分两层：**来源绑定**——本 run 的权重来源由 ES 本地协议配置唯一确定（`expected_weight_source`；全部策略配置由 TA/本地管理员预置，不经 Cloud 下发），声明值仅作一致性核对，把预定来源降级为等权 `none` 或在合法来源间择优于来源核对即被确定性拒绝（未检测概率 0，数值无需伪造即可检出）；**数值绑定**——每个可采纳权重来源对应一条 ES 可依据共识量独立复算的公开确定性规则，权重当且仅当等于该规则的复算值才被接受（此时与诚实值无差别）；补偿系数 `α_k` 与 `σ*` 档位同理由公开规则复算比对，且复算失败（不可复算度量、共识历史缺失）与"合法不产生 α"严格区分——云端省略 α 或省略 `W(v)` 等历史条目与伪造它们同样被确定性拒绝，衰减策略不能被静默关闭；`γ_vcm`/`vcm_mode`/momentum 为 ES 预置协议配置，声明不符即拒绝。依赖的前提是 (A6) 版本与历史由共识维护（4.3.10 的版本绑定把 `v`、`W(v)` 钉到 ES 本地已提交状态），以及参与历史 `A_k` 由 ES 从 commit 通知独立累计。`wait` 等不可复算度量不在可采纳来源集合内，开启验证时 fail-closed 拒绝。

### 4.4.2 隐私保持

研究点二仍只保护客户端个人参数；ES/组级聚合与 CS/checkpoint 聚合允许公开。令

```text
N_Q(t) = Σ_{Π_k(t_k)∈Q(t)} n_k(t_k)
```

为本次已发布 checkpoint 实际包含的客户端总数。TA 完全可信且永不合谋；Cloud 可与全部 ES 及至多 `c≤N_Q(t)−2` 个客户端合谋，因此非合谋客户端数满足 `N_Q(t)−c≥2`。

**定理 4.2（客户端参数隐私保持）.** 若研究点一 Theorem 3.1 的原语假设成立，stale reuse 在客户端本地且掩码前完成，Direct-VCM 仅依赖历史全局模型、`S_k` 和公开超参数等共识值，则研究点二相对研究点一不增加客户端个人参数泄露：敌手即使联合 Cloud、全部 ES 和至多 `N_Q(t)−2` 个客户端，也只能得到允许公开的组/全局聚合及合谋客户端输入，不能以非可忽略优势恢复任一指定非合谋客户端的 `w_i^{new}`、`w_i^{stale}` 或其本地混合结果。

**证明.** stale reuse 只把被掩码明文从 `w_i^{new}` 改为 `(1−ρ_i)w_i^{new}+ρ_iw_i^{stale}`，不改变独立 PRG 掩码的分布；Direct-VCM 和最终归一化只对已允许公开的 checkpoint 聚合与共识值作确定性变换，不产生新的客户端级观测。把合谋客户端贡献从 checkpoint 聚合中扣除后，敌手至多获得 `N_Q(t)−c≥2` 个非合谋客户端的加权和，Theorem 3.1 的模拟论证直接适用于该客户端集合。

该证明不需要将“客户端”替换成“组”，也不要求 `|Q(t)|≥2`。超时 checkpoint 即使只含一个组，只要该组所含实际客户端仍满足 `N_Q(t)−c≥2`，个人参数隐私仍成立；反之，即使有多个组，只要最终只剩一个非合谋客户端，其参数仍可由聚合差分恢复。阈值的基本单位始终是进入同一已发布聚合的客户端，而不是组。□

### 4.4.3 收敛性分析

本节只给出与 `MAFSA`/`DVAFL` 分析思路对齐的、针对当前主协议的一步偏差刻画，不直接继承两篇参考论文的收敛定理。原因是主协议改变了补偿粒度（客户端级变为组级）、补偿位置（LHH 验证后）和参与集合（checkpoint 选中集合）；stale reuse 不在主协议主线中（附录 A.3.3），其客户端级偏差项单独移入附录 A.5 作备选机制分析。因此，本节结论应写成“条件性偏差上界”，而不是完整收敛定理。

#### 假设与记号

以下假设贯穿本节；其中 IID 是 Point 2 的默认理论条件，Dirichlet Non-IID 作为实验条件单独报告：

| 假设 | 内容 |
|------|------|
| **(C1) 局部更新映射 Lipschitz** | IID 默认条件下，一步本地更新映射 `T_i(w)` 满足 `‖T_i(w)−T_i(w')‖≤(1+ηL)‖w−w'‖`；例如 `T_i(w)=w−η∇F_i(w)` 且 `∇F_i` 为 `L`-Lipschitz |
| **(C2) 单步位移有界** | 全局模型经 SGD 型更新，单步位移满足 `‖W(v+1) − W(v)‖ ≤ ηB`，`η` 为学习率，`B` 为有效梯度范数上界 |
| **(C3) 陈旧上界** | 客户端陈旧度 `E_i = t_k − t_i ≤ E_max`，组级陈旧度 `s_k = t − t_k ≤ s_max` |
| **(C4) 混合系数有界**（仅附录 A.5 备选分析使用） | `ρ_i ≤ ρ_max ≤ 0.5`；主协议不含 stale reuse，核心偏差界不使用 (C4) |
| **(C5) 补偿系数有界** | `γ_vcm` 为协议超参，固定常数 |
| **(C6) 组样本量有界** | `S_k(t_k)>0`，且 checkpoint 使用真实样本量加权；不需要把 `S_max/S_min` 人为引入偏差界 |
| **(C7) 随机性耦合** | 比较 stale 与 fresh 本地更新时，采用确定性本地更新，或对两条比较轨迹耦合使用完全相同的随机样本/随机种子；否则需在界中另加 SGD 方差项 |

记理想 FedAvg-form 聚合（无陈旧、无补偿）结果为 `W^*(t+1)`，本协议实际结果为 `W(t+1)`。单 checkpoint 偏差定义为 `Δ_bias = W(t+1) − W^*(t+1)`。

#### 条件性界与推导说明

**条件性界 4.1（单 checkpoint 偏差上界，非完整收敛定理）.** *在 (C1)-(C3)、(C5)-(C7) 假设下，并把理想结果定义为“同一 checkpoint 选中客户端、同一真实样本量权重、所有组均从 `W(t)` 出发、无版本失配”的 FedAvg-form 结果，则一次 checkpoint 的偏差满足：*

```text
‖Δ_bias‖ ≤ (|1 − γ_vcm| + ηL) · s_max · ηB
```

*该项对应组级 mixed-version 结果在 Direct-VCM 后的残差；客户端级 stale reuse 项（`ρ_max·(1+ηL)·E_max·ηB`）仅当 reuse 作为可选组件启用时才出现，见附录 A.5，不进入主协议的核心偏差界。由于实际 checkpoint 是按 `S_k` 做加权平均，分子中的组样本量和会被 `S_Q` 归一化，不应额外留下 `S_max/S_min` 因子。*

在 IID 和一步梯度更新下，`γ_vcm≈1` 时，第二项可由版本位移的一阶项抵消到 `ηL·s_max·ηB` 量级；若要把这种“补偿确实降低偏差”升级为严格结论，还需要额外的线性化/梯度同向性假设。

**说明.** 主协议下偏差 `Δ_bias` 只由组级 mixed-version 结果在 Direct-VCM 后的残差 `Δ_vcm` 构成；客户端级 stale reuse 偏差 `Δ_stale` 不在主线（附录 A.3.3），其推导移入附录 A.5 备选分析。本界用于说明 `γ_vcm` 与 `s_max` 如何进入误差控制，不声称已经得到参考论文的完整多步收敛率。

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

由步骤 (ii) 的归一化结果，主协议的条件性单 checkpoint 偏差上界为：

```text
‖Δ_bias‖ ≤ ‖Δ_vcm^{norm}‖ ≤ (|1−γ_vcm|+ηL)·s_max·ηB
```

该界是本文在 IID/一步更新条件下的条件性单 checkpoint 偏差上界，不是直接来自 `MAFSA` 或 `DVAFL` 的定理；若同时启用 stale reuse 组件，总界再加上附录 A.5 的客户端级项。对 Dirichlet Non-IID，需额外引入局部梯度异质性或 bounded gradient dissimilarity 项；本文将其作为实验验证，不把 IID 上界无条件外推到 Non-IID。

#### σ/Aσ 的策略条件正确性

固定 σ 与 Aσ 是聚合策略的可选实例（4.3.10），对其偏差刻画采用**策略条件**口径：

- **比较对象**：与 σ/Aσ 同归一化权重的当前版本理想聚合 `W_ref = Σ_k ŵ_k·T_k(W(t)) / Σ_k ŵ_k`（`ŵ_k` 为协议归一化权重，`T_k` 为一步更新映射）。在该口径下，σ 衰减与 Aσ 档位选择只经由 Direct-VCM 残差项（条件性界 4.1）影响偏差，不引入新的无偏性破坏；σ 的实际收益（方差稳定化）由 4.5 对照实验验证。
- **与标准 FedAvg 比较**：若改以等权 FedAvg-form 为比较对象，需在界中额外加入权重分布偏移项 `‖Δ_weight‖ = O(‖ŵ − s_k/S_Q‖_1 · max_k ‖T_k(W(t)) − W(t)‖)`，即陈旧度降权相对等权的贡献重分配误差；本文不把该口径与条件性界 4.1 混用。
- **Aσ**：仅在给定已提交策略状态（`σ*` 已按 4.3.10 公开规则锁定）的条件下分析——锁定后 Aσ 等价于固定 `σ*` 的策略实例，偏差界与固定 σ 情形相同；探测期采用保守 `σ=σ_fallback` 同理。本文不宣称 Aσ 的完整收敛率，其档位决策本身的可验证性由 4.3.10 覆盖。

#### 条件性多 checkpoint 误差传播

单 checkpoint 偏差不能在没有更新映射稳定性假设时直接线性相加。令 `e_t=‖W(t)−W^*(t)‖`，若理想 checkpoint 更新映射对输入模型是 `L_T`-Lipschitz，且第 `t` 次 checkpoint 由 mixed-version/VCM 引入的局部扰动为 `ε_t`（启用 reuse 组件时另含附录 A.5 的客户端级项），则

```text
e_{t+1} ≤ L_T·e_t + ε_t,
e_T ≤ L_T^T·e_0 + Σ_{j=0}^{T−1} L_T^{T−1−j}·ε_j.
```

只有在额外满足非扩张条件 `L_T≤1`、初始误差 `e_0=0` 且各步 `ε_j≤ε` 时，才能进一步得到 `e_T≤Tε`；若 `L_T>1`，误差可能被放大，不能宣称无条件线性累积。条件性界 4.1 与上述递推均未覆盖：
- non-IID 下的客户端漂移（需更强假设如 bounded gradient dissimilarity）；
- SGD 噪声的方差（需有界方差假设 `E[‖∇F − ∇f_i‖²] ≤ σ²`）；
- `ρ_i` 与数据分布的耦合（stale reuse 偏向数据稳定的客户端）。

完整收敛率（如 `O(1/√T)` 邻域收敛）需在当前协议上重新建立，并补充有界方差、独立鞅差、局部梯度异质性与 checkpoint 选择过程的假设，不能直接引用 `MAFSA` 或 `DVAFL` 的收敛率。这里的结论只说明 `γ_vcm` 与 `s_max`（启用 reuse 组件时另有 `ρ_max`、`E_max`，见附录 A.5）如何进入可调的偏差控制项；实际 `γ_vcm`/σ 是否带来精度和收敛收益，交由 4.5 的对照和消融实验验证。

### 4.4.4 出界声明重申

manifest 漏选/审查（anti-censorship）、样本量虚报、完整 ACK 协议不作主协议安全保证。

## 4.5 实验评估

### 4.5.1 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | AMD Ryzen 7 7745HX (8 核/16 线程) + 31.2 GB RAM + NVIDIA GeForce RTX 4070 Laptop GPU (8 GB, compute capability 8.9) |
| 软件环境 | Windows 10.0.26100 / Python 3.9.15 / PyTorch 2.7.0+cu118 / CUDA 11.8 / cuDNN 9.1.0 / torch 线程数 8 |
| 数据集 | MNIST、FashionMNIST、CIFAR-10 |
| 模型 | CNN（MNIST/FashionMNIST 用 CNN_MNIST，`d`=582,026；CIFAR-10 用 CNN_CIFAR10，`d`=906,890）；安全证据批次另用 Tiny（`d`=7,850，全密码学 verify 端到端） |
| 聚合 | FedAvg-form |
| 层级 | 1 Cloud + M Edge + N Users |
| 密码 | LHH(P-256) / Ed25519 / PRG(AES-CTR/HKDF) / QUANTIZE_SCALE=1e6 |
| 种子数 | ≥3（含置信区间） |

### 4.5.2 五组主线对比

为分离各机制的贡献，设计五组递进主线（与既有实验矩阵的阶梯一致）：

1. **Sync-HVSA**：研究点 1 同步聚合基线。
2. **Grouped-HVSA + Inter-Group Sync（G2）**：引入一次性分组与组间同步。
3. **+ Buffer/Checkpoint Semi-Async（G3，半异步①）**：引入组包缓冲与 checkpoint 触发。
4. **+ Direct-VCM（G3v，③，主方法）**：引入 mixed-version 组级后置补偿。
5. **+ σ 陈旧度幂衰减（G5，仅消融）**：引入补偿端陈旧度衰减。**σ 不进入主线**——正式批次显示其增量在 IID 下显著为负、在 Non-IID 下无显著收益（见 4.5.4），故降为消融项，主线止于 G3v。

stale reuse **不进入主线**：受控实验表明其在暂时断连场景无稳定正收益（其最有利场合 0/3 全负，三 regime 复现），仅以消融与被否定替代设计呈现（4.5.4、附录 A.3.3）。

四组（G2/G3/G3v 主线 + G5 消融）采用**逐一叠加**设计，其**增量增益**是各机制必要性的直接度量。**口径纪律（写作时必须遵守）**：

- 每个机制的增益只相对其**直接前驱**报告并标注口径：checkpoint 半异步 = `G3−G2`；Direct-VCM = `G3v−G3`；σ = `G5−G3v`（仅消融）。
- **必须标注三口径**，三者在 G3−G2 上结论不同，混用即失真：
  - **等轮次**（各臂同为 20 个 checkpoint，取末次精度）：G2 用 K=5、G3 家族用 K=3，故 G2 的实际计算量约为 G3 家族的 **1.66 倍**（ΣS_Q：126,840 vs 76,370），此口径下"同步占优"是**预期行为而非缺陷**，只可作附录对照。
  - **等计算量**（目标 = 两臂 ΣS_Q 的较小者，ΣS_Q 取自 `checkpoint_Ss` 累计）：回答"同算力下谁更准"。
  - **等时间**（预算 T = 同 seed 的 G3 末次 tick）：回答"同模拟时间内谁更准"，是吞吐口径。
- **主结果用等计算量 + 等时间双列**（两者一致支持主方法），等轮次降附录并注明每轮聚合量不同。
- 正式批次参照值（`results_thesis_core`，5 seeds，同 seed 配对差，`--deterministic`）：
  - **VCM 相对 G3**：IID 三单元 `+2.64 / +3.53 / +1.27 pp`（均 5/5 正，95% CI 不跨 0，稳定提升）；Non-IID `+0.08 / −0.31 / +1.27 pp`（2/5、2/5、4/5 正，均不显著 → 表述为"趋势"或"无提升"）。
  - **主方法 G3v 相对同步 G2**：等计算量 IID 三单元 `+2.39 / +3.93 / +0.65 pp`（均 5/5 正，显著）；等时间 IID 三单元 `+4.62 / +9.51 / +2.24 pp`（均 5/5 正，显著）。Non-IID 各单元方向为正但均不显著。
  - **σ 相对 G3v**：IID `−0.92 / −0.53 / −0.14 pp`（0/5、2/5、1/5 正；MNIST 显著为负）；Non-IID `+2.54 / +1.04 / +0.54 pp`（4/5、2/5、5/5 正，**均不显著**）。→ σ 按分布启用也不成立，故整体降为消融。
  - **VCM 的无陈旧参照（等计算量）**：四个配置计算量相同（224 次采纳用户更新 / 60 组包），`s_max=0` 拒绝陈旧 vs `s_max=3` 容忍陈旧——VCM 在 IID 追回 `84% / 81%` 的精度损失且节省 `34%` 模拟 tick；Non-IID 仅追回 `2% / −13%`。
- **严禁**把单机制增量标成"VCM+σ"联合收益；联合效果须以 `G5−G3` 净差另行报告（正式批次：`+1.72 / +2.62 / +2.99 / +0.74 / +1.13 / +1.81 pp`，前五项 5/5 显著）。
- 报告均值时必须同时给出配对 std 与正/负种子数，不隐藏方差、不筛选 seed；Non-IID 下不显著的结果表述为"趋势"而非"增益"。

- **指标**：达到目标精度的墙钟时间、平均等待时间、最终精度、compensation gain（按 `G3v−G3` 口径）、σ gain（按 `G5−G3v` 口径）、verification overhead、组包过期率、重复贡献率、组间公平性、有效吞吐量。
- **数值**：正式批次 `results_thesis_core`（5 seeds，`--deterministic`，固定拓扑 5 ES × 4 用户/组，20 个 checkpoint）。逐值可复现已由 `verify-repeat` 验证（同 seed 同配置 `max|Δacc| = 0`，退出码 0）。

  **(a) 各臂终点精度（等轮次口径，mean ± sample std，n=5）**

  | 数据集 | 分布 | G2 (K=5) | G3 (半异步①) | **G3v (主方法)** | G5 (σ 消融) | G1 FedAvg |
  |---|---|---|---|---|---|---|
  | MNIST | IID | 91.66 ± 0.60 | 88.16 ± 0.55 | **90.80 ± 0.88** | 89.88 ± 0.77 | 91.81 ± 0.37 |
  | MNIST | NI | 92.28 ± 0.73 | 87.11 ± 1.94 | **87.19 ± 3.13** | 89.72 ± 1.74 | 92.24 ± 0.88 |
  | FashionMNIST | IID | 69.94 ± 1.19 | 65.39 ± 1.98 | **68.91 ± 1.13** | 68.38 ± 0.89 | 70.19 ± 0.35 |
  | FashionMNIST | NI | 69.98 ± 1.54 | 63.48 ± 2.01 | **63.18 ± 5.56** | 64.22 ± 4.71 | 70.36 ± 1.03 |
  | CIFAR-10 | IID | 64.51 ± 0.77 | 61.56 ± 0.61 | **62.83 ± 0.77** | 62.69 ± 0.78 | 64.46 ± 0.56 |
  | CIFAR-10 | NI | 63.03 ± 0.55 | 58.24 ± 1.11 | **59.51 ± 1.45** | 60.05 ± 1.10 | 63.73 ± 0.68 |

  > **本口径不得作为主结果**：G2 计算量约为 G3 家族的 1.66 倍（ΣS_Q 126,840 vs 76,370），"同步占优"是预期行为。仅作附录对照。

  **(b) 等计算量配对差（目标 = 两臂 ΣS_Q 的较小者；95% CI，t=2.776；正种子数）**

  | 比较 | MNIST IID | MNIST NI | FashionMNIST IID | FashionMNIST NI | CIFAR-10 IID | CIFAR-10 NI |
  |---|---|---|---|---|---|---|
  | G3−G2 | −0.25 [−0.90,+0.40] 2/5 | +0.17 [−2.44,+2.77] 3/5 | +0.40 [−0.63,+1.43] 3/5 | +2.04 [−4.38,+8.46] 3/5 | **−0.62 [−0.89,−0.35] 0/5 显著下降** | −2.09 [−4.82,+0.63] 1/5 |
  | G3v−G3 | **+2.64 [+1.88,+3.40] 5/5 显著提升** | +0.08 [−3.78,+3.94] 2/5 | **+3.53 [+0.93,+6.12] 5/5 显著提升** | −0.31 [−7.70,+7.09] 2/5 | **+1.27 [+0.99,+1.55] 5/5 显著提升** | +1.27 [−0.32,+2.87] 4/5 |
  | G3v−G2 | **+2.39 [+1.58,+3.20] 5/5 显著提升** | +0.24 [−2.70,+3.19] 3/5 | **+3.93 [+1.10,+6.76] 5/5 显著提升** | +1.73 [−6.17,+9.64] 4/5 | **+0.65 [+0.40,+0.90] 5/5 显著提升** | −0.82 [−3.21,+1.57] 2/5 |

  **读法**：纯半异步（G3）在同算力下与同步（G2）无显著差异（MNIST/Fashion 四单元 |Δ| ≤ 2.04 均不显著，仅 CIFAR IID 显著为负 −0.62）——即引入半异步**不付精度代价**；而 VCM 把它推成显著为正，**主方法 G3v 在三个 IID 单元同算力下显著优于同步基线（5/5 种子）**。

  **(c) 等时间配对差（预算 T = 同 seed 的 G3 末次 tick，基准 G2；95% CI）**

  | 臂 | MNIST IID | MNIST NI | FashionMNIST IID | FashionMNIST NI | CIFAR-10 IID | CIFAR-10 NI |
  |---|---|---|---|---|---|---|
  | G3 | +1.98 [−0.05,+4.01] 4/5 | +3.55 [−3.60,+10.69] 3/5 | +5.99 [−1.91,+13.89] 5/5 | +3.99 [−1.82,+9.80] 4/5 | +0.97 [−0.56,+2.51] 4/5 | **−0.87 [−1.70,−0.04] 1/5 显著下降** |
  | G3v | **+4.62 [+2.05,+7.19] 5/5 显著提升** | +3.63 [−2.48,+9.73] 4/5 | **+9.51 [+2.35,+16.68] 5/5 显著提升** | +3.68 [−8.02,+15.39] 3/5 | **+2.24 [+0.51,+3.97] 5/5 显著提升** | +0.40 [−1.67,+2.47] 3/5 |

  **读法**：同模拟时间内半异步家族整体占优（G3v 三个 IID 单元 5/5 显著），收益来自同时间内完成更多聚合（G3/G3v 到 20 个 checkpoint 用 51 tick，G2 需 100 tick）。

  **(d) σ 消融（G5−G3v，等时间口径；与等计算量口径一致，因 G3/G3v/G5 的 ΣS_Q 序列相同）**

  | MNIST IID | MNIST NI | FashionMNIST IID | FashionMNIST NI | CIFAR-10 IID | CIFAR-10 NI |
  |---|---|---|---|---|---|
  | **−0.92 [−1.22,−0.62] 0/5 显著下降** | +2.54 [−0.87,+5.94] 4/5 | −0.53 [−1.51,+0.44] 2/5 | +1.04 [−2.62,+4.71] 2/5 | −0.14 [−0.36,+0.09] 1/5 | +0.54 [−0.11,+1.19] 5/5 |

  **读法**：σ 在 IID 下一致为负（MNIST 显著），Non-IID 下方向一致为正但**无一单元达显著**——故 σ 不进入主线（见 4.5.4）。

  **(e) 其余指标的落盘位置**：`平均等待 / compensation gain / 组包过期率 / 重复贡献率 / 组间公平性 / 有效吞吐量` 均逐 run 记录在 `point2_data_*.npz` 与 `point2_meta_*.json`（字段 `avg_wait`、`comp_gain`、`expired_buffer_count`、`duplicate_contributions`、`fairness_var`、`checkpoint_Ss` 等），统计汇总见 `results_thesis_core/stats/thesis_stats_report.md` 第 5 节。`达到目标精度的墙钟时间`因各臂统一取 20 个 checkpoint 而未按目标精度截取，需另行补充；`verification overhead` 见 4.5.3（主矩阵为 verify-off，其验证列不构成验证开销证据）。

### 4.5.2' 竞争性基线对比

为定位本协议在分层安全聚合与半异步 FL 中的相对性能，引入以下基线（基线数据优先取自原文报告，本实验在相同数据集/模型下复现关键指标作为参照）：

| 基线 | 特点 | 对齐维度 |
|------|------|---------|
| **FedAvg / HFL** | 同步分层聚合，无安全/验证 | 同步聚合上界 |
| **FedBuff** | 缓冲异步，单客户端粒度 | 异步基线 |
| **MAFSA** | 多层异步 + stale 融合（权重方向为 `α=e^{-βE}` 作用于 fresh） | stale reuse 方向对比 |
| **DVAFL** | VCM 版本控制补偿（Chen et al., IEEE TMC 2026） | 补偿机制对比 |

- **对比指标**：达到目标精度墙钟时间、最终精度、compensation gain、σ gain（口径同 4.5.2）。
- **预期**：HVSA 系列在验证开销上优于所有基线（因 LHH 定长验证）；主方法（G3v = G3+VCM）在异构性下墙钟时间优于同步基线；stale reuse 方向修正（Ours Decayed-Stale）相对 MAFSA 方向仅作消融对照（主线已不含 reuse，见 4.5.2）。
- **数值**：**未执行（明确缺口）**。四个基线均未在本实验框架内复现：FedAvg 可由 `G1` 直接充当同步上界（其数据见 4.5.2(a)），但 FedBuff 需逐客户端粒度的异步实现、MAFSA 需 `α=e^{-βE}` 权重方向、DVAFL 需其 TMC2026 式(15)–(18) 的矩校准补偿——**本模拟器的最小聚合单元是组包而非客户端，因此 FedBuff 与 MAFSA 无法在本框架内忠实复现**（见 `docs/THESIS_CORE_PIPELINE.md` 对 GFB 的口径边界说明：GFB 只检验 checkpoint 阈值，不等价于 FedBuff 复现，不能作为竞争基线）。
  **可行的替代方案**：①DVAFL 的矩校准方向已在本框架内以 `--vcm-mode moment1/moment2` 实现并测过（结果见 4.5.4(3)：两模式在 NI 下均显著劣于 Direct-VCM，且属不可验证模式），可作为"DVAFL 方向不适配本约束"的对照证据引用；②若必须给出竞争性基线数值，应改为引用各基线原文在相同数据集/模型下的报告值并明确标注"引自原文、非同环境复现"，不得与本实验数据混列。
  **写作纪律**：在补跑或引用完成前，正文不得出现"优于 FedBuff/MAFSA/DVAFL"的定量断言。

### 4.5.3 攻击实验

以**主方法 G3v**（半异步① + Direct-VCM；不含已降级的 σ 组件）为基座，与性能基线分开，验证 Theorem 4.1 的篡改检测能力。攻击批次的配置即 `--vcm --gamma-vcm 1.0`（无 `--staleness-sigma`），故其基座与主线主方法一致：

- **tamper_raw_sum**：Cloud 篡改 `W_sum^raw`，预期 LHH 等式失败。
- **tamper_manifest**：Cloud 篡改组包元数据，预期 manifest 一致性或 Ed25519 验签失败。
- **drop_replace_pkg**：Cloud 丢弃合法组包并替换为伪造/重放组包，预期 manifest 一致性或签名验证失败。
- **replay**：Cloud 重放已消费的组包，预期 `Aggregated_IDs` 确定性拒绝。

策略绑定攻击（对应附注 4.2 与 4.3.10，需在加权路径 + verify on 下验证）：

- **weight_forge**：Cloud 任选正整数权重提交（σ-agg/V-PDA 路径），预期 ES 独立复算不匹配拒绝。
- **weight_relabel**：Cloud 把伪造的 V-PDA 权重改标为 `staleness`（或反之），预期按声明来源复算仍不匹配拒绝。
- **source_downgrade**：Cloud 把预定权重来源（V-PDA/staleness-agg）降级为等权 `none`，或在合法来源间择优声明（数值无需伪造），预期来源绑定核对确定性拒绝（4.3.10）。
- **alpha_or_sigma_tamper**：Cloud 篡改 `alphas`（如全 1 关闭衰减）或谎报 Aσ 锁定档位，预期 ES 复算 `α/σ*` 不匹配拒绝；另验证 γ/mode/版本错位声明的拒绝。
- **wait_metric_fail_closed**：`wait` 度量（不可复算）作为权重/衰减策略提交，预期验证模式下确定性拒绝——包括云端**省略 alphas** 的变体（不可复算与"合法不产生 α"严格区分，省略不能把衰减策略静默关闭）。

- **指标**：tampering detection rate（应接近 100%）、误拒率（clean control 应为 0）、检测阶段、失败后版本与策略状态是否推进。
- **注入实现**：`main_point2.py --attack` 已覆盖全部攻击族（主四类 + 策略绑定五类 `weight_forge/weight_relabel/source_downgrade/alpha_tamper/history_missing`）；配置不适用的攻击（如等权路径伪造权重、全新鲜包下 α 无差别）在注入时记录"不适用"或以区分性注入值处理，报告中以 clean control 补齐。每类攻击 100 个随机化单元与 Tiny 5 种子端到端属于第 8 周实验执行。
- **既有单测证据**：`tests/test_vpda.py`（V-PDA 伪造权重拒绝）、`tests/test_policy_binding.py`（σ 权重复算、双向改标拒绝、来源降级拒绝、α/σ* 篡改拒绝、γ/momentum/版本/配置摘要拒绝、wait fail-closed、TA 独立复算审计）、`_verify_p03.py`（四类主攻击 100% 检出）。
- **攻击实验数值（正式批次 `results_thesis_core_attack`，seeds 42–46，`--units 100`，Tiny 模型 dim=7850，MNIST，5 ES × 4 用户，verify-on）**：

  | 类别 | 成员攻击 | 检出 / 已尝试 | 拒绝率 | 攻击被采纳 | 检出阶段分布 |
  |---|---|---|---|---|---|
  | raw_sum | `tamper_global` | 633 / 633 | **100.0%** | 0 | `lhh`: 633 |
  | manifest_replay | `tamper_manifest` + `replay` | 865 / 865 | **100.0%** | 0 | `signature`: 384, `replay_exactly_once`: 481 |
  | weight_source | `weight_forge` / `weight_relabel` / `source_downgrade` | 909 / 909 | **100.0%** | 0 | `policy_weights`: 303, `policy_source`: 606 |
  | comp_param | `alpha_tamper` | 633 / 633 | **100.0%** | 0 | `policy_alpha`: 633 |
  | clean control | （无攻击） | 误拒 0 / 500 | **误拒率 0.0%** | — | — |

  **合计 3040 个适用攻击单元全部检出，500 次 clean 尝试零误拒。** 五个 seed 各自独立达标（每类拒绝率 100%、clean 误拒率 0%），判定由 `run_attack_evidence.py` 的验收口径自动给出。

  **状态未推进的证据**：每个被拒候选的 `AttackOutcome` 均记录 `state_before` / `state_after`（`cloud_version`、`ledger_version`、`es_committed`、`adopted_packages`），被拒时四者完全相同——即拒绝不仅记了标记，且**未推进版本、未消费 token、未被采纳**。

  **验证开销（verify-on，Tiny dim=7850 口径）**：每 checkpoint 验证耗时（含 ES 验证 + vote + TA 授权 + 提交事务）——`tamper_global` `11.05–11.34 s`、`replay` `10.46–11.47 s`、clean `11.28–11.99 s`；提前在签名/策略层拒绝的攻击族耗时仅 `0.0016–0.0042 s`（未进入 LHH 验证）。通信量：`135.4–147.5 KiB/单元`（拒绝路径），`968–984 KiB/单元`（含完整聚合与提交的路径）。
  **口径边界**：①主矩阵各臂为 verify-off，其验证耗时列**不构成验证开销证据**，验证开销以上表为准；②LHH 计算为 O(d)，上表为 dim=7850 的小模型口径，外推到 CNN/MLP 主矩阵模型需按维度缩放，定长通信（32B LHH 哈希）与 d 无关的结论须由 N5 维度扫描（`exp_security_overhead.py`）另行给出，**尚未执行**；③`drop_replace_pkg` 与 `history_missing` 两个成员攻击未纳入本批次（前者由 `tamper_manifest`/`replay` 覆盖同类检测层，后者仅在 `--participation-debias ipw` 下有区分性），如需完整覆盖须补跑。

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

- **数值**：正式批次 `results_thesis_core` 及其补充批次（均 5 seeds、`--deterministic`）。**已完成项给数值，未执行项明确标注缺口，不填推测值。**

  **(1) I1：VCM × checkpoint 开关（已完成，20 run 逐值确认）**

  判据是"无 mixed-version 可补时 VCM 恒为 0"。用 `s_max=0`（拒绝一切陈旧组包，故全部组包 `r_k=v`）构造该端点，得四单元 × 5 seeds 共 20 对比较：**G3v 与 G3 的 `acc_history` 与 `checkpoint_ticks` 全部逐值相同，`max|Δacc| = 0.000e+00`**；对照正常配置 `s_max=3` 下两者显著不同（MNIST IID `+2.64 pp`，5/5 种子）。→ **VCM 增益确实由 checkpoint 机制内生引出，且无副作用**（排除"隐式正则化"解释）。

  **(2) 无陈旧参照与 VCM 追回率（已完成，等计算量口径）**

  四个配置计算量完全相同（224 次采纳用户更新 / 60 组包），差别仅在是否容忍陈旧与是否补偿：

  | 数据集 | 分布 | s_max=0 无补偿 | s_max=0 +VCM | s_max=3 无补偿 | s_max=3 +VCM | tick(0→3) |
  |---|---|---|---|---|---|---|
  | MNIST | IID | 91.28 | 91.28 | 88.16 | 90.80 | 87 → 57 |
  | MNIST | NI | 90.45 | 90.45 | 87.11 | 87.19 | 87 → 57 |
  | FashionMNIST | IID | 69.74 | 69.74 | 65.39 | 68.91 | 87 → 57 |
  | FashionMNIST | NI | 65.81 | 65.81 | 63.48 | 63.18 | 87 → 57 |

  | 数据集 | 分布 | 容忍陈旧的损失 | VCM 追回 | 追回比例 | tick 节省 |
  |---|---|---|---|---|---|
  | MNIST | IID | +3.13 | +2.64 | **84%** | 34% |
  | FashionMNIST | IID | +4.35 | +3.53 | **81%** | 34% |
  | MNIST | NI | +3.34 | +0.08 | **2%** | 34% |
  | FashionMNIST | NI | +2.33 | −0.31 | **−13%** | 34% |

  **结论**：IID 下"容忍陈旧 + VCM"以 **34% 更少的模拟 tick** 换回 **81–84%** 的精度损失，是划算交易；**Non-IID 下 VCM 追回 2% / −13%，等于无效，此时正确策略是收紧 `s_max`（拒绝陈旧）而非补偿**（MNIST NI：90.45 vs 87.19，差 3.26 pp）。机制解释：VCM 补偿的是陈旧组包相对当前全局模型的漂移方向，IID 下该方向可信，NI 下各组数据分布差异大使方向不可靠。→ **Fashion NI 的失效不是调参问题，而是方法适用条件**。

  **(3) VCM 模式消融（已完成，NI 单元，相对基线 G3v）**

  | 数据集 | 模式 | acc | std | Δ vs G3v | 95% CI | 正种子 |
  |---|---|---|---|---|---|---|
  | MNIST | `span` / `clip` | 与 `full` **逐值相同** | — | 0 | — | — |
  | MNIST | `moment1` | 89.89 | 2.95 | +2.70 | [−1.14, +6.54] | 4/5 |
  | MNIST | `moment2` | 70.68 | **35.19** | −16.51 | [−57.91, +24.89] | 2/5 |
  | MNIST | `curve` | 73.55 | 8.14 | **−13.63** | [−20.94, −6.33] | 0/5 |
  | FashionMNIST | `moment1` | 60.53 | 7.23 | −2.65 | [−6.44, +1.14] | 1/5 |
  | FashionMNIST | `moment2` | 39.78 | **27.82** | −23.40 | [−55.98, +9.18] | 0/5 |
  | FashionMNIST | `curve` | 54.08 | 4.10 | **−9.10** | [−12.93, −5.26] | 0/5 |

  `span` / `clip` 为**结构性空操作**（独立诊断脚本实测：陈旧度 `v−r_k` 恒为 0 或 1，故 `span` 除 1 无效；补偿范数仅占 raw sum 的 **0.58%**，远低于 `clip` 的 10% 阈值，18 个观测中 0 次触发）。`curve` 可降 Fashion NI 方差（5.56 → 4.10）但代价是精度 **−9.10 pp**（0/5，显著），属方差换精度而非修复。
  **附带发现**：`moment1` 在 MNIST NI 上优于可验证的 `full`（+2.70 pp，不显著），而 `moment1` 恰是不可验证模式（ES 无法独立复算，属 verify-off 研究模式）——即 **"ES 可独立复算"这一约束本身有可测代价**。

  **(4) checkpoint 阈值 K 消融（已完成，等计算量口径）**

  K=1（GFB）在 20 个 checkpoint 时仅采纳 73 次用户更新（K=3 为 218），计算量不足，故补跑 `--max-checkpoints 60` 对齐到 218：

  | 数据集 | 分布 | K=1 (GFB) | K=3 (G3) | 配对差 | 95% CI | K=1 tick | K=3 tick | 判定 |
  |---|---|---|---|---|---|---|---|---|
  | FashionMNIST | IID | 67.61 ± 1.06 | 65.39 ± 1.98 | +2.22 | [−0.64, +5.08] | 70 | 57 | 不显著 |
  | FashionMNIST | NI | 58.00 ± 7.12 | 63.48 ± 2.01 | −5.48 | [−15.77, +4.80] | 70 | 57 | 不显著 |
  | CIFAR-10 | IID | 54.16 ± 0.40 | 61.56 ± 0.61 | **−7.39** | [−8.49, −6.29] | 70 | 57 | **显著下降** |
  | CIFAR-10 | NI | 46.13 ± 3.78 | 58.24 ± 1.11 | **−12.12** | [−17.05, −7.18] | 70 | 57 | **显著下降** |

  **K=1 被 K=3 支配**：同算力下 CIFAR 两个单元显著更差，Fashion 无显著优势，且**达到同等计算量需 70 个模拟 tick（K=3 只需 57，慢 23%）**——同时排除了"精度换速度"的可能。规律随任务难度变化：Fashion（MLP，简单）K=1 略优（不显著），CIFAR（CNN，难）K=1 显著更差，即**任务越难，每次聚合越需要足够样本量**。
  **口径说明**：该消融在 20 个 checkpoint 下曾显示 −16.75 pp，但那是计算量仅 1/3 的混淆结果；等计算量后真实差距为 −5.69 pp（全格均值）。**正文不得引用 20 checkpoint 口径的差值。**

  **(5) σ 组件消融（已完成）**：见 4.5.2(d)。IID 一致为负（MNIST 显著），NI 方向为正但无一显著 → σ 不进入主线。σ 度量（version/rel2）跨数据集复测与 Aσ 自适应档位**未执行**。

  **(6) 参数敏感性扫描（已完成，基座 = G3v，MNIST，IID+NI 各 5 seeds，每次只动一个参数）**

  | 参数设置 | IID acc | Δ vs 基座 | NI acc | Δ vs 基座 |
  |---|---|---|---|---|
  | γ_vcm=0.5 | 89.34 ± 0.58 | **−1.46 [−2.00,−0.91] 0/5 显著下降** | 89.38 ± 2.16 | +2.20 [−1.41,+5.80] 4/5 |
  | γ_vcm=0.8 | 90.18 ± 0.84 | **−0.62 [−0.73,−0.51] 0/5 显著下降** | 89.37 ± 1.91 | +2.18 [−0.79,+5.15] 4/5 |
  | K=2（等计算量，34 ckpt） | 76.87 ± 13.54 | −11.96 [−29.88,+5.96] 2/5 | 82.81 ± 10.60 | −4.30 [−15.93,+7.33] 1/5 |
  | s_max=2 | 90.80 ± 0.88 | **0.00（逐值相同）** | 87.19 ± 3.13 | **0.00（逐值相同）** |
  | s_max=5 | 90.80 ± 0.88 | **0.00（逐值相同）** | 87.19 ± 3.13 | **0.00（逐值相同）** |
  | policy=fresh_first | 90.69 ± 0.72 | −0.11 [−0.67,+0.45] 1/5 | 87.34 ± 3.63 | +0.15 [−1.60,+1.90] 1/5 |
  | policy=fair_fresh_first | 90.76 ± 0.71 | −0.04 [−0.42,+0.34] 1/5 | 87.19 ± 3.58 | +0.00 [−1.40,+1.40] 1/5 |

  （基座 G3v：K=3、s_max=3、γ_vcm=1.0、policy=fifo。参照点：γ_vcm=0 即 G3，IID 88.16 / NI 87.11。）

  四条结论：

  1. **γ_vcm 在 IID 下单调递增且 γ=1.0 最优**：0.5 与 0.8 都**显著**低于基座（CI 不跨 0，0/5 正），说明"全额补偿"是有依据的选择而非调出来的魔法数。**NI 下相反**：0.5/0.8 比 1.0 高约 2.2 pp（4/5 正，不显著），与 4.5.4(2) 的"NI 下 VCM 无效甚至过补偿"一致——两处独立指向同一机制。
  2. **K=3 的选取依据是收敛稳定性，不是均值精度**。K=2 在**等计算量**下（`--max-checkpoints 34`，采纳组包 68 = 基座的 68）均值下降不显著，但**逐 seed 呈双峰**：IID 为 71.05 / 70.97 / 90.10 / 60.49 / 91.75，即部分 seed 正常收敛、个别 seed 崩塌（对照 K=3 基座 IID 全部落在 85–92，std 1.88 vs K=2 的 13.54）。机制：K 决定每次聚合的最少组数，K≤2 时单次聚合仅含 2 组数据、更新方差过大而易发散。结合 (4) 的 K=1 被支配，**K=3 是稳定性与吞吐的折中点**。
     **口径提示**：K=2 在默认 20 ckpt 下显示 −16.50 pp，但那是计算量仅 59% 的混淆结果；正文**不得引用该数**。
  3. **s_max 在本模拟器中的有效域退化为 {0} vs {≥1}**：`s_max=2` 与 `s_max=5` 的 `acc_history` 与基座 `s_max=3` **逐值完全相同（5/5 seeds、两个分布）**，因为实测陈旧度 `v−r_k` 恒为 0 或 1（见 (3) 的诊断），故任何 `s_max ≥ 1` 都等价。`s_max=0`（拒绝一切陈旧）是唯一有区分度的端点：精度更高（IID 91.28 / NI 90.45，见 (2)）但需 **87 个模拟 tick 而非 57（多 34%）**。→ **`s_max` 不是可调的性能旋钮，而是"是否容忍陈旧"的开关**；正文应如此表述，不应给出"s_max 敏感性曲线"。
  4. **selection policy 对精度无显著影响**（两变体 |Δ| ≤ 0.15 pp、均 1/5 正），说明主方法不依赖选包策略调优——这是一个鲁棒性证据。

  **(7) 未执行项（明确缺口）**：**ρ_max / β_c 未扫**——这两个参数仅在 `--reuse-rule ours/mafsa` 下生效，而 reuse 已退出主线（4.5.2、附录 A.3.3），对主方法 G3v 扫描它们测的是无效参数，故不跑；若为附录 A.3.3 补 reuse 方向敏感性，须另设带 `--reuse-rule ours` 的批次。此外 I2（reuse × buffer 开关交互）、`D_fair` / `Δ_timeout` 超参、adaptive s_max（代码中无对应开关）均未执行。这些需另行补跑，**不得以推测值填入正文**。

## 4.6 本章小结

本章提出研究点二——检查点式组级半异步安全聚合扩展。在研究点一 HVSA 的密码学闭环基础上，通过组包缓冲与 checkpoint 触发、策略绑定的可验证加权聚合（V-PDA/σ 权重与补偿参数由 ES 独立复算，恶意云端不能自行选择权重）、Direct-VCM 组级后置补偿三个机制，将跨组训练组织从"全组同步 barrier"扩展为"组包缓冲 + checkpoint 触发"的组级半异步机制，同时保持单 TA 掩码与 LHH 验证链路不被破坏；安全聚合兼容的组内 stale reuse 作为客户端级组件保留，其精度收益边界由实验如实刻画（附注：多轮受控实验表明其在暂时断连下无稳定正收益，不入创新点）。安全分析证明：对 checkpoint 选中聚合结果的完整性检测可归约到 LHH 抗碰撞或 Ed25519 EUF-CMA（Theorem 4.1）；权重伪造与改标由策略绑定的公开规则复算确定性拒绝（附注 4.2）；检测失败即丢弃本次 checkpoint，不推进全局版本，也不保证恶意 Cloud 下的训练可用性或 anti-censorship。在 `N_Q−c≥2` 下，加权聚合与 Direct-VCM 补偿不削弱客户端个人参数隐私（Theorem 4.2）；在 IID 与耦合相同随机性的条件下，给出条件性单 checkpoint 偏差上界，并以 `e_{t+1}≤L_Te_t+ε_t` 描述多 checkpoint 传播；只有 `L_T≤1` 等额外条件下才得到线性界，不把参考论文的完整收敛率直接继承为本文定理。

---

# 第 5 章 总结与展望

## 5.1 全文总结

本文在单 TA 约束下，先提出 HVSA 实现分层可验证安全聚合，再以检查点式组级半异步扩展解决同步等待与掉线浪费，同时保持掩码与 LHH 验证闭环。具体地：

**研究点一（HVSA）**：在"云—边—端—单 TA"架构下，通过 PRG 掩码保护客户端个人参数（Theorem 3.1：TA 永不合谋，Cloud 可与全部 ES 及 `c≤N_online−2` 个客户端合谋），利用 LHH 与 Ed25519 让 ES 以定长通信开销验证云端聚合结果，并通过"在线集合驱动的掩码求和"实现掉线鲁棒性。Cloud 知道聚合掩码和全局聚合结果符合协议隐私边界；ES 验证通过后直接用 `W_sum^raw/N_online` 得到共识模型。

**研究点二（检查点半异步扩展）**：在研究点一密码学闭环基础上，通过组包缓冲/checkpoint 触发、策略绑定的可验证加权聚合（V-PDA/σ 权重与补偿参数由 ES 依据共识量独立复算，恶意云端不能自行选择权重）与 Direct-VCM 组级后置补偿实现半异步扩展；安全聚合兼容的组内 stale reuse 作为客户端级组件保留，其收益边界由实验如实刻画。`package_token` 被 ES 签名绑定，`Aggregated_IDs` 是 TA 维护的仅追加共识状态。客户端隐私按 `N_Q−c≥2` 判定，不要求 `|Q|≥2`；组/全局聚合允许公开。在 IID、一步更新和耦合相同随机性下给出单 checkpoint 偏差上界；多步只给出 `e_{t+1}≤L_Te_t+ε_t` 的条件性传播式。

## 5.2 未来工作展望

- **双 TA / 门限 TA**：以进一步分散信任，消除单点信任假设。
- **基版本对齐的更新量混合**：stale reuse 更严格变体，对 stale 与 fresh 的基版本对齐做更强约束。
- **完整 ACK / 原子提交协议的协议化**：将"至多一次纳入"的重放防护升级为完整分布式 exactly-once 协议。
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

最初考虑过完全异步（组包"来一个更新一个"），但完全异步下每次全局下发都需在线验证，与单 TA + LHH 的逐轮验证闭环难以直接兼容（验证粒度需与掩码聚合粒度一致）。最终选择：组内仍同步封组，组间用 buffer+checkpoint 半异步，验证粒度为 checkpoint 选中集合。一句话：**不是把验证硬塞进完全异步，而是主动收缩异步边界以保住密码学闭环。** 该架构决策同时引出三个必要组件：checkpoint 触发本身（4.3.6）、mixed-version 的补偿（4.3.7）与掉线浪费的复用（4.3.3），故三者不是并列拼装而是同一决策的展开。

## A.3 组件层：各机制的关键实现取舍

### A.3.1 stale reuse 权重方向的修正

MAFSA 原始定义 `α_i=exp(−β_c·E_i)` 作用于 fresh model，导致版本间隔 `E_i` 越大、stale 权重 `1−α_i` 反而越大。考虑到间隔越大陈旧结果与当前优化方向偏差风险越高，本文将指数衰减量改为直接控制 stale 项并加 `ρ_max≤0.5` 上界，保证新结果主导。故表述为"受 MAFSA 启发的时效性修正"而非"直接采用 MAFSA 加权公式"。

### A.3.2 为什么 VCM 用 S_k 而非 n_k

`W_sum^raw` 本身处在"样本量加权和"量纲中，组间补偿项必须同量纲才能与最终 `S_Q` 归一化一致。故 `C_k=S_k·γ_vcm·G_k`，而非按客户端个数 `n_k`。`n_k` 仅承担在线计数与最小在线人数约束。

### A.3.3 为什么 stale reuse 降级为组件、策略绑定上位为创新点 2

**reuse 降级**：多轮受控实验（同 seed 配对、控制失败率）表明，组内 stale reuse 在其最有利的场合（同失败率下 reuse on/off）0/3 全负；突发离线、极端 Non-IID、MAFSA 忠实掉线回收语义、delta 重锚定等变体均无稳定正收益。机理是协议本身已吸收暂时断连（K-of-E 替换 + `S_Q` 重归一化 + VCM 重锚定），复用只剩注入陈旧偏置；唯一有价值的形态是"永久丢失 + 参与跌破封组下限"的旁路回收（+18~+50pp 恢复参与可行性），属系统级机制，不入主线创新点。因此 4.1 的创新点 2 不再是 reuse；reuse 以安全聚合兼容组件 + 被否定替代设计呈现。

**策略绑定上位**：把等权聚合扩展为加权（V-PDA/σ）后出现一个真问题——LHH 验证等式 `∏_k H_k^{w_k}` 对任意正整数权重成立，权重本身不受密码学约束；若允许 Cloud 声明权重，恶意 Cloud 即可任意重分配各组贡献，甚至把伪造权重改标为不做独立复算的来源绕过检查。`∏ H_k^{w_k}` 只是线性同态的直接推广，不足以构成创新；可立住的贡献是"恶意云端不能自行选择权重"——面向公开确定性聚合策略的策略绑定、边缘独立复算与可验证加权安全聚合（4.3.10、附注 4.2）。

## A.4 显式出界的考量

下列方向有价值但会稀释主线，统一出界：完全异步、解掩码前连续陈旧度系数 `β_g`、双 TA/门限 TA、全局异步轨迹在线审计、anti-censorship 可审计调度、First/Second-order VCM、组级质量追踪、HGA-FL 式 bias correction。出界是为让研究点 2 保持"一个能讲圆的协议扩展"而非大拼盘。

---

## A.5 备选机制分析：stale reuse 的偏差项

> stale reuse 已不入主协议主线（A.3.3）。本节保留其客户端级偏差项的推导，仅在 reuse 作为可选组件启用时，将其并入总界：
> `‖Δ_bias‖ ≤ ρ_max·(1+ηL)·E_max·ηB + (|1−γ_vcm|+ηL)·s_max·ηB`。
> 该分析使用假设 (C1)-(C4)（4.4.3 假设表），其余假设与正文一致。

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
---

## 附：全文一致性自检

- [x] 研究点 2 公式与 `STALENESS_AWARE_REUSE.md` 一致（stale/VCM/验证/术语）
- [x] 无 "strict FedAvg" 残留（仅组内 strict sample-weighted sum，最终 FedAvg-form）
- [x] 威胁模型 3.2.2 / 4.2.2 统一声明，P1/P2 适用范围标注
- [x] 记号表全文唯一
- [x] 实验数值全部留空，仅框架+预期趋势
- [x] 创新点 1.3 统一编号，第 3、4 章不重复计数
- [x] 定理按章重编号：Theorem 3.1-3.3（点 1）/ Theorem 4.1-4.2 + 条件性界 4.1 + 条件性多步传播式（点 2）
- [x] 6 章结构：绪论 / 相关技术 / HVSA / 检查点半异步 / 总结展望 / 参考文献 + 附录 A
- [x] 4.1 采用"3.4.4 三局限 → 约束 → checkpoint 洞见 → 三机制映射"因果链；4.3.3 显式锚定局限 (ii) 与安全聚合兼容
- [x] 附录 A 重排为 A.1 约束层 / A.2 架构层 / A.3 组件层 / A.4 出界，与 4.1 的组织逻辑呼应
- [x] 4.5.4 增设 I1/I2 机制交互检验（VCM×checkpoint、reuse×buffer），`run_all_ablations.py` 已同步
- [x] 三机制定位为"同一架构决策的展开"而非并列拼装（4.1 末段 + A.2 末段）
- [x] 验证开销口径统一为"验证**通信**开销与维度无关"（LHH 计算仍为 O(d)，见 3.3.7 口径注）
- [x] 实验增益口径纪律：VCM 相对 G3、σ 相对 G3v 分别报告，禁止把 VCM 单机制增益标成 "VCM+σ" 联合收益（4.5.2）
- [x] 创新点 2 = 策略绑定的可验证加权聚合（4.3.10 + 附注 4.2 + A.3.3），stale reuse 降级为组件（4.1、A.3.3），协议实现与强表述对齐：`edge.py` 独立复算覆盖全部可采纳权重来源，且来源绑定闭合（来源由协议配置唯一确定，降级为 none / 合法来源间择优于来源核对即拒，`tests/test_policy_binding.py` 含降级攻击回归）
- [x] 主问题统一为"同步等待/mixed-version 偏差/策略完整性"（4.1）；核心偏差界 4.1 不含 stale 项（A.5 备选分析）；"首次/根本冲突/整体验证轻量"已清除，exactly-once 降级为重放防护
- [x] 协议配置绑定：版本化 `ProtocolConfig` + manifest 携带 `policy_config_digest`（4.3.6），ES fail-closed 校验、TA/ES 独立复算权重（云端声明值只作审计）；实验基建 config_hash/运行元数据与策略绑定攻击族注入（`main_point2.py --attack weight_forge|weight_relabel|source_downgrade|alpha_tamper|history_missing`）
- [x] 权威授权流（prepare/finalize 两阶段 + stage/finalize/publish/notify 提交事务）：TA 仅追加且可持久化 ledger（SQLite 主键；`prepared_checkpoints` 保存完整 receipt 字节 + token 元数据 + `result_digest`）+ ES 签名 vote（绑定 `result_digest`）+ `CheckpointReceipt`；授权不消费、提交成功才落地；Cloud 先 stage（只读校验 + journal）再凭 receipt 且结果摘要一致才 publish、ES 凭 receipt＋摘要＋ledger 完整确认且版本严格 +1 才推进（4.2.2、4.3.9）；finalize 幂等（同 receipt 重放返回 `already`，不同 receipt 冲突拒绝）；负例覆盖 token 未注册/已消费、K/timeout/真实 s_max 违约、版本跳变、vote 篡改、Cloud 重启重放、receipt 篡改、伪造或未经 prepare 的 finalize、诚实 receipt 后偷换 raw sum/final model、提交失败不推进 ledger、finalize 幂等与冲突、prepare 替换、stage→finalize→publish→notify 四阶段崩溃恢复、prepared 记录与 TA 密钥跨重启持久化（`tests/test_receipt_flow.py`）
- [x] 可复现性：科学配置（数据/种子/模型/训练/协议/攻击）与产物配置（run_id/output_dir/tag/force/ta-state-path）分离，`config_hash` 只含科学配置；异配置非零退出；产物原子替换；`--deterministic` 为严格模式（`use_deterministic_algorithms(True)`，无 warn_only 覆盖）；`pytest.ini` 限定 `testpaths=tests`
- [x] verify-on 组合边界 fail-fast 与可复现性：非法组合启动即失败；异配置结果非零退出；产物原子替换；`--deterministic` 严格模式；元数据含 config_mode_version/CUDA/cuDNN/threads/commandline
