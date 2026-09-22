# 方向 3 检索报告：分层（云—边—端）可验证安全聚合

检索日期 2026-09-22。本报告是论文**创新点 1 的坐标系**，并含对「定长验证通信」新颖性的关键发现。
由子智能体执行，原始输出见 `agent_raw/`（`arxiv_raw*.json`、`crossref_raw.json`、`dois_raw.json`）。

四维记法：**[层级结构] | [验证通信随 d 与 N 的伸缩] | [验证原语] | [信任模型]**
**【据】**＝有摘要/正文原文依据；**【未验证】**＝仅题名或题录可查。

---

## ⚠ 最重要的一条结论（先读）

**「验证通信/验证材料与模型维度 d 无关」已被多篇工作主张，本文不能把它当作全新性质。**

| 出处 | 原文主张 | 设定差异 |
|---|---|---|
| arXiv:2304.03841（**e-SeaFL, 2023**） | 服务器广播的证明为**单个群元素 `x_t` + 标量 `w_t`**，用户一次等式检验完成验证——**与 d 无关**；**且比本文更早（2023）** | 扁平 + assisting nodes；**服务器侧承诺流量随 N 线性**（64 B × N） |
| arXiv:2509.05162（Bottoni et al., 2025） | **"The size of each authenticator is constant and does not depend on the column length d"**（0.28 KB） | 扁平单聚合器；每 column 一个认证子，**开销随 column 数 m 线性**；需一次性可信初始化 |
| arXiv:2603.10692（Qin et al., 2026） | **"zero additional communication overhead"**、**"zero per-round communication overhead"**；并批评既有方案 **"proofs proportional to model dimensionality"** | cross-silo 扁平；**非密码学审计**；无边缘层 |
| arXiv:2603.19949（TAPAS, 2026） | **"server-side communication that is independent of L"** | 双服务器；验证的是**输入合法性**而非聚合正确性 |
| arXiv:2511.12936（VTSAFL, 2025） | 功能密钥与部分解密结果**规模为常数** | 单层；门限 |
| arXiv:2605.24054（Zhou, 2026，**已撤稿**） | "constant verification tag size" | 双服务器非合谋；**不要引用** |

**对本文的影响**：本文必须把差异点收敛到 **「1 云 + M 边 + N 端、由边缘服务器验证云端聚合结果、单 TA、PRG 掩码组包」这一组合**，而**不能**主张「定长验证通信」本身是新的。其中 e-SeaFL（2023）最早、Qin et al.（2026）主张最强（零额外通信），**两者都必须在论文中正面引用并区分**。

---

## 1. 必读清单（18 条）

### 1.1 直接命中：分层结构 + 聚合验证

**① Fan, Chen, Jiang, Yu — "Practical Multi-Key Secure Aggregation With Homomorphic Hash Verification for Cross-Silo Federated Learning"**，IEEE TMC 2026，DOI `10.1109/tmc.2026.3679423`
把**同态哈希**用作多密钥安全聚合的聚合结果验证手段（题名层面可确认）。
层级 **未验证**（题名指向 cross-silo 扁平）| 验证通信伸缩 **未验证** | 原语：同态哈希 **【据·题名】** | 信任模型：多密钥 **【据·题名】**，是否门限/单 TA **未验证**
> **「同态哈希用于聚合验证」这条线上最接近的期刊工作，必须拿到全文核对其验证通信是否与 d 无关。**

**② Liao, Chang, Chiu, Chen — "Privacy-Preserving Verification of Secure Aggregation for Hierarchical Peer-to-Peer Federated Learning"**，IEEE ICC 2026，DOI `10.1109/icc59461.2026.11587871`
**分层 P2P** 联邦学习下安全聚合的隐私保护验证。
层级：分层 **【据·题名】** | 其余三维 **未验证**
> **题名同时含「hierarchical + verification of secure aggregation」，是与本课题最直接的结构性邻居，正文细节务必取到。**

**③ Niu, Kong, Chen, Zhou — "A Homomorphic MAC-based verifiable secure aggregation for federated learning in cloud–edge AIoT"**，Computer Communications 2025，DOI `10.1016/j.comcom.2025.108271`
**云—边 AIoT** 场景下基于**同态 MAC** 的可验证安全聚合。
层级：云 + 边 **【据·题名】** | 原语：同态 MAC **【据·题名】** | 其余 **未验证**

**④ Wu et al. — "Nautilus: A Verifiable Hierarchical Federated Learning Framework for Vehicular-Edge-Cloud Systems"**，arXiv:2606.23017（2026-06，v4 2026-07）| **正文已核**
车—边—云三层 FL，用 ZKP 验证**调度公平性与客户端压缩指令的执行合规性**。
层级：三层 **【据·正文】** | 验证通信伸缩：**无 d 相关表述**；验证约 **2 ms/proof**，电路 <10k 约束，**未给 big-O**，且验证对象是**调度约束**而非聚合结果正确性 **【据·正文】** | 原语：ZKP（上链）**【据·正文】** | 信任模型：**无 TA/门限描述**；乐观验证 + 每轮随机抽检 5% **【据·正文】**
> 三层 FL 里做可验证的最接近实例，但**验证的是调度而非聚合**，且无 d 无关性的显式主张。

**⑤ Xu et al. — "Client-Verifiable and Efficient Federated Unlearning in Low-Altitude Wireless Networks"（VerFU）**，arXiv:2603.29688（2026-03）| **正文已核**
用**线性同态哈希（LHH）+ 承诺**让客户端验证服务器端「遗忘」确实移除了自己的贡献。
层级：端—UAV 服务器（**两层，无边缘层**）**【据·正文】** | 验证通信：单设备与服务器间每轮 **<1 KB**；**未给 d 的 big-O**；LHH 密钥生成写作 `LHH.Gen(κ,d)`，即**与 d 相关** **【据·正文】** | 原语：**LHH + 承诺** **【据·正文】** | 信任模型：服务器不可信；不依赖可信第三方 **【据·正文】**
> **「LHH 用于 FL 验证」目前唯一检索到的 arXiv 命中，但验证的是 unlearning 而非聚合，且无边缘验证层。**

### 1.2 扁平结构下的可验证聚合（验证通信伸缩的主战场）

| # | 文献 | 标识符 | 四维要点 |
|---|---|---|---|
| ⑥ | **VTSAFL**（Wang, Han, Meng） | arXiv:2511.12936 | 单层多客户端；**「功能密钥与部分解密结果规模为常数」【据·摘要】**；门限多客户端函数加密 + 部分解密验证；**门限信任，无单一 TA** |
| ⑦ | **Repurposing Backdoors for Good**（Qin et al.） | arXiv:2603.10692 | cross-silo 扁平，**无云边端**；**「零额外通信开销」**，并批评既有方案 "proofs proportional to model dimensionality incur huge computational and communication burdens" **【据·正文·原文引用】**；内在证明（后门 + 灾难性遗忘），**兼容 SA**；**不依赖可信第三方**；代价：**通信为 0 但验证靠本地推理/统计审计，非密码学可证明** |
| ⑧ | **Verifiability and Privacy in FL through Context-Hiding Multi-Key Homomorphic Authenticators**（Bottoni et al.） | arXiv:2509.05162 | **扁平：客户端 + 单聚合器，无边缘层**；**"size of each authenticator is constant and does not depend on the column length d"（0.28 KB）**，但每 column 一个认证子、**随 column 数 m 线性**【据·正文·原文引用】；上下文隐藏身份基多密钥线性同态认证子（BLS12-381）；多密钥；**初始化需一个可信实体**；验证计算 1M 参数约 **10 ms/客户端** |
| ⑨ | **e-SeaFL**（Behnia et al.） | arXiv:2304.03841（2023-04, v6 2024-11） | 单服务器 + **assisting nodes**（非分层）；**证明为单个群元素 `x_t` + 标量 `w_t`，与 d 无关**；**服务器侧承诺聚合带宽 64 B × N，随 N 线性**（200–1000 用户 → 12.8–64 KB）**【据·正文】**；认证 Pedersen 向量承诺（APVC）；恶意但非合谋单服务器 |
| ⑩ | **TAPAS**（Karthikeyan, Polychroniadou） | arXiv:2603.19949 | 双服务器；**"server-side communication that is independent of L"**【据·摘要·原文引用】；格（LWE/SIS）+ 格上 ZKP；**无可信初始化** |
| ⑪ | **TAPFed**（Xu et al.） | arXiv:2501.05053 / DOI 10.1109/tdsc.2024.3350206 | **多个去中心化聚合器**，无云边端三层；门限函数加密；**门限 + 容忍若干恶意聚合器** |
| ⑫ | **VerifyNet** | DOI 10.1109/tifs.2019.2929409（引 784） | **四维未验证**（取不到摘要）。经典基线；「验证计算与通信均随模型维度线性」属领域共识说法，但**本次未取得原文证据，不作断言** |
| ⑬ | **VeriFL** | DOI 10.1109/tifs.2020.3043139（引 308） | **四维未验证**。经典基线（同态 MAC 路线） |
| ⑭ | **VerSA** | DOI 10.1109/tdsc.2021.3126323（引 116） | **四维未验证**。**注意：被多篇 "Comments on VERSA"（`10.1109/tdsc.2023.3253082`、`10.1109/tdsc.2023.3272338`）质疑，引用前需核对这些评论** |
| ⑮ | **VOSA** | DOI 10.1109/tdsc.2022.3226508（引 59） | **四维未验证**。**同被质疑（`10.1109/tdsc.2024.3352170` "On the Security of VOSA"）** |

### 1.3 分层安全聚合（无验证或弱验证）——本文结构的对照系

| # | 文献 | 标识符 | 四维要点 |
|---|---|---|---|
| ⑯ | **H-SecCoGC**（Weng, Xiao, Skoglund） | arXiv:2601.17995 / DOI 10.1109/icc59461.2026.11587757 | HFL（客户端—中继/边—服务器）；**无验证机制**；编码 + 隐私噪声（信息论安全）；半诚实服务器/中继，**无 TA 概念** |
| ⑰ | **SHIELD**（Siriwardhana et al.） | DOI 10.1109/tdsc.2024.3472869（引 18） | **多层 HFL，在设备与服务器之间的层上做聚合**；**不是密码学验证**，是中毒攻击下的鲁棒聚合；**无 TA** |
| ⑱ | **Optimal Communication and Key Rate Region for Hierarchical SA with User Collusion**（Zhang et al.） | arXiv:2410.14035（2024-10, v6 2026-01） | **三层（server—relays—users）**；**无验证**；给出**通信与密钥率最优区域**；**无 TA**，显式建模用户合谋门限。同族：arXiv:2503.04564、2507.14768、2511.20117、2603.19705、2604.12429、2604.26344、2601.12853 —— 均为**信息论 HSA 的率区/门限刻画，不含验证**，是「分层 + 门限信任模型」最完整的理论坐标系 |

### 1.4 补充清单

| 工作 | 出处 | 一句话 | 四维要点 |
|---|---|---|---|
| HierFAVG（Liu et al.） | arXiv:1905.06641 | 云边端分层 FL 的范式原点 | 1云+M边+N端；**无任何密码学**；无 TA |
| Hi-SAFE（Joo et al.） | arXiv:2511.18887 / DOI 10.1109/jiot.2026.3714526 | 面向 sign-SGD 的轻量分层 SA | 分层；无聚合验证 |
| OPSA（Guan et al.） | DOI 10.1109/tdsc.2025.3565640 | TEE 支撑的一次性可验证 SA | 四维未验证 |
| FedAggVerify（Kanakri, King） | DOI 10.1109/tdsc.2026.3717223 | 公开可验证 + 所有权保护 | 四维未验证 |
| VeSAFL（Bouamama et al.） | DOI 10.1109/tdsc.2025.3589160 | 可验证 SA | 四维未验证 |
| SVTAgg（Liu et al.） | DOI 10.1109/tdsc.2026.3707153 | **edge-assisted** FL 的安全/可验证/可追溯聚合 | 层级含边缘；其余未验证 |
| **VSAFL**（He et al.） | DOI 10.1109/jiot.2024.3449705 | **可验证 + 多对数（polylog）开销** | 四维未验证——**「验证开销」维度的关键待核项** |
| MRNIVA（Zhu et al.） | DOI 10.1109/tdsc.2024.3380669 | 非交互可验证聚合 | 四维未验证 |
| LVSA（Li et al.） | DOI 10.1016/j.neucom.2025.130712 | 轻量可验证 SA | 四维未验证 |
| Lightweight and Verifiable SA for Multi-dimensional Data（Wu et al.） | DOI 10.1016/j.comnet.2023.110079 | **边缘增强 IoT** 多维数据可验证 SA | 层级含边缘；其余未验证 |
| **V-MHESA**（Park, Chi） | DOI 10.3390/math13223687 | 用**随机抽取的模型参数**生成验证令牌 | 单服务器；令牌对应随机子集 → **验证通信可 sublinear**；半诚实威胁模型 **【据·Crossref 摘要】** |
| Peng, Shen, Gao, Wang | DOI 10.3390/e25081125 | 掩码梯度**做哈希**以验证聚合结果 | 单服务器；哈希验证 **【据·Crossref 摘要】** |
| **Elliptic Curve Multiset Hash**（Maitin-Shepard et al.） | arXiv:1601.06502 | **LHH 原语**：椭圆曲线多重集哈希，摘要定长、可增量/并行 | **原语级参考** |
| **On-the-fly verification of rateless erasure codes**（Krohn et al.） | DOI 10.1109/secpri.2004.1301326（引 213） | **同态哈希（LHH）的原始出处**（网络编码场景） | **原语级参考** |
| ByzSecAgg（Jahani-Nezhad et al.） | arXiv:2302.09913 | 编码计算 + **向量承诺**的拜占庭鲁棒 SA | 承诺原语 |
| AHSecAgg and TSKG（Zhang et al.） | arXiv:2312.04937 | 加法同态掩码的轻量 SA | 扁平；无验证 |

---

## 2. 与本文最接近的 3 条

### 第 1 近：Bottoni et al.（arXiv:2509.05162，多密钥线性同态认证子）

- **近在哪**：同样用**线性同态**原语（LHH 家族）让**验证方只收一份定长验证材料**即可核对聚合结果；明确写出 **"size of each authenticator is constant and does not depend on the column length d"（0.28 KB）**；验证计算 O(1) 级。
- **差在哪**：① **架构扁平**——客户端 + 单聚合器，**没有云/边/端三层，没有边缘服务器做验证**；② 验证方是**客户端**，不是边缘；③ **一个 column 一个认证子，开销随 column 数 m 线性**，严格说不是「全局 O(1)」；④ **需要一次性可信实体做初始化**——与本文「单 TA」部分重合但用途不同；⑤ 无 PRG 掩码式 SA 的组包/解掩码流程。
- **对「验证通信与 d 无关」的影响**：**是正面证据**——该性质在扁平单聚合器设定下已被主张并给出常数（0.28 KB）。**本文不能把「定长验证通信」本身当作全新性质**，只能主张**在 1 云 + M 边 + N 端、由边缘验证云端的分层设定下**实现。

### 第 2 近：e-SeaFL（arXiv:2304.03841，2023）

- **近在哪**：服务器广播的**证明就是一个群元素 `x_t` 加一个标量 `w_t`**，用户一次等式检验（`g^{w_t·ρ} = x_t`）完成验证——**验证通信与 d 无关**；单轮交互；单服务器。**且比本文更早（2023）。**
- **差在哪**：① **无云边端分层**，验证方是**用户**而非边缘；② 服务器侧**承诺聚合带宽随 N 线性**（64 B × N），而本文的卖点正是**云侧入流量只与 M 有关、与 N 无关**——**这是可写进对比表的硬差异**；③ 依赖 assisting nodes 与**非合谋服务器**假设；④ **没有「边缘聚合组包 → 云解掩码」这一层**。
- **对「验证通信与 d 无关」的影响**：**是正面证据，且更早**。本文必须把差异点收敛到**分层拓扑 + 边缘侧验证 + 云侧入流量与 N 解耦**。

### 第 3 近：Qin et al.（arXiv:2603.10692，内在证明）

- **近在哪**：直接针对「验证通信随 d 增长」这一痛点，给出**最强主张——验证的额外通信为零**。
- **差在哪**：① **cross-silo 扁平结构，无云边端**；② 验证是**随机化单验证者审计 + 本地推理**，**不是密码学可验证**（无 LHH/承诺的可证明性）；③ 信号会「自然衰减」，是**概率性/一次性质检**而非每轮可验证；④ 与 SA 的兼容靠「不额外发包」。
- **对「验证通信与 d 无关」的影响**：**这是最需要正面回应的竞争者**——它已把「零额外验证通信」写进标题级主张。本文的差异化必须落在**可证明安全（LHH + Ed25519）+ 分层结构 + 边缘为验证者 + 单 TA**，而不是「通信更小」。

**两位亚军（结构最近但细节未取得）**：
- **Fan et al., IEEE TMC 2026**（`10.1109/tmc.2026.3679423`）：题名即「**homomorphic hash verification**」——**同态哈希验证在 FL 中的期刊级工作，必须取全文核对验证通信伸缩**。
- **Liao et al., IEEE ICC 2026**（`10.1109/icc59461.2026.11587871`）：题名即「**Hierarchical … Verification of Secure Aggregation**」——**分层 + 可验证聚合的结构性直系邻居，必须取全文**。

---

## 3. 检索式清单（原文）

**arXiv API（50 条，间隔 3.5s）**
```
all:"hierarchical federated learning" AND all:"secure aggregation"
all:"client-edge-cloud" AND all:"federated learning"
all:"verifiable aggregation" AND all:"federated learning"
all:"homomorphic hash" AND all:aggregation
all:"secure aggregation" AND all:"edge server"
all:"verifiable secure aggregation"
all:"hierarchical federated learning" AND all:"privacy-preserving"
all:"threshold" AND all:"secure aggregation" AND all:"federated learning"
all:"homomorphic hash"
all:"linear homomorphic" AND all:"verifiable"
all:"hierarchical secure aggregation"
all:"verification outsourcing" AND all:"federated"
all:"verifiable federated learning" AND all:"edge"
all:"aggregation integrity" AND all:"edge server"
all:"multi-server" AND all:"secure aggregation" AND all:"verifiable"
all:"proof of aggregation"
all:"homomorphic commitment" AND all:"federated learning"
all:"privacy-preserving aggregation" AND all:"verification"
abs:"homomorphic hash" AND abs:"federated"
abs:"linear homomorphic hash"
abs:"verifiable" AND abs:"aggregation" AND abs:"edge servers"
all:"secure aggregation" AND all:"hierarchical" AND all:"verification"
all:"aggregate signature" AND all:"federated learning"
abs:"publicly verifiable" AND abs:"federated learning" AND abs:"aggregation"
abs:"verification" AND abs:"model dimension" AND abs:"secure aggregation"
all:"edge-assisted federated learning" AND all:"verifiable"
all:"verifiable aggregation"
all:"aggregation verification"
all:"hierarchical federated learning" AND all:"verification"
all:"secure aggregation" AND all:"cloud server"
all:"federated learning" AND all:"homomorphic hash"
all:"verifiable" AND all:"model aggregation" AND all:"edge"
all:"secure aggregation" AND all:"trusted authority"
all:"federated learning" AND all:"proof of correct"
all:"secure aggregation" AND all:"communication overhead" AND all:"verification"
all:"privacy-preserving federated learning" AND all:"bilinear pairing"
all:"peer-to-peer" AND all:"secure aggregation" AND all:"hierarchical"
all:"multi-key" AND all:"secure aggregation" AND all:"homomorphic"
all:"vector commitment" AND all:"secure aggregation"
all:"traceable" AND all:"model aggregation" AND all:"edge-assisted"
au:Siriwardhana AND all:"federated"
all:"SHIELD" AND all:"hierarchical federated learning"
all:"verifiable" AND all:"aggregation" AND all:"trusted third party"
all:"hierarchical federated learning" AND all:"secure aggregation" AND all:"edge servers"
all:"outsourced verification"
all:"verify the aggregation"
all:"two-tier" AND all:"secure aggregation"
all:"edge servers" AND all:"secure aggregation" AND all:"verifiable"
all:"hierarchical" AND all:"aggregation" AND all:"tamper"
all:"trusted execution environment" AND all:"secure aggregation" AND all:"verification"
```

**Crossref（`filter=from-pub-date:2022-01-01`）**
```
hierarchical federated learning secure aggregation
verifiable aggregation edge servers federated learning
homomorphic hash federated learning verification
client-edge-cloud federated learning privacy preserving aggregation
verifiable secure aggregation constant size proof
threshold secure aggregation multiple aggregators federated learning
aggregation result verification cloud edge collaborative
linear homomorphic hash aggregation integrity verification
VerifyNet Secure and Verifiable Federated Learning
VeriFL Communication-Efficient and Fast Verifiable Aggregation for Federated Learning
Practical Secure Aggregation for Privacy-Preserving Machine Learning
On-the-fly verification of rateless erasure codes for efficient content distribution
Secure Single-Server Aggregation with (Poly)Logarithmic Overhead
ELSA Secure Aggregation for Federated Learning with Malicious Actors
EIFFeL Ensuring Integrity for Federated Learning
RoFL Robustness of Secure Federated Learning
Verifiable Privacy-Preserving Aggregation in Federated Learning with Edge Computing
Secure and Verifiable Federated Learning against poisoning attacks hierarchical
```

**OpenAIRE**（`title=` 精确题名回查 22 条，取到摘要者：SHIELD、RFLPA、Secure group aggregation、V-MHESA、Entropy 2023）
**arXiv 网页检索 UI**（2 条，均 0 结果）

> 检索式中**未出现任何本文机制专有描述**（无 P-256、无 Ed25519、无「组包」、无 M/N 符号、无「定长 32B」），符合约束。

---

## 4. 覆盖局限与不确定处

**未能覆盖的来源**
1. **IEEE Xplore 正文与摘要全部取不到**（WebFetch 返回空页/重定向）。因此 **TMC 2026 同态哈希验证**、**ICC 2026 分层 P2P 验证**、TDSC 系列（OPSA / FedAggVerify / VeSAFL / SVTAgg / PriVeriFL / VSAFL / ClusterGuard）**只有题录级信息，四维标「未验证」**。
2. **Elsevier（Computer Communications / Computer Networks / JNCA / Information Sciences）正文不可达**，云边同态 MAC 一文只有题名。
3. **OpenAlex、Semantic Scholar、DBLP 全部 429 / 机器人拦截**；DuckDuckGo、Bing 不可用。**非 arXiv、非 Crossref 的灰色文献、技术报告、学位论文未系统覆盖。**
4. **未覆盖**：Google Scholar、Scopus、Web of Science、CNKI/万方、IACR ePrint、USENIX/NDSS 论文集（仅通过 Crossref 题录间接命中 ELSA/EIFFeL/RoFL）。

**方法学不确定处**
5. arXiv API 的 `all:"..."` 短语检索**行为不稳定**：同一短语在不同布尔组合下命中数从 0 跳到 40，**存在漏检风险**。例如 `all:"secure aggregation" AND all:"hierarchical" AND all:"verification"` 返回 0，但明显存在同时含这三个词的论文。**建议用 IEEE Xplore / Scopus 的字段检索复核。**
6. 多条 arXiv 记录的 `published` 为 2026 年 1–8 月，**arXiv ID 形如 26xx 的条目未经同行评审确认**，其中 `2605.24054` **已确认撤稿**。
7. **VERSA / VOSA 存在多篇 "Comments on…" 安全质疑**，**引用这两个经典基线前必须读这些评论**，否则对比表会被审稿人攻击。

**结论（不含「无人做过」的断言）**
本次检索**确有多篇工作主张验证材料/验证通信与模型维度无关或为常数**，其中 **Bottoni et al.（arXiv:2509.05162）给出最清晰的「认证子尺寸与 d 无关」表述，e-SeaFL（arXiv:2304.03841）给出最早的单群元素证明，Qin et al.（arXiv:2603.10692）给出最强的「零额外通信」主张**。**未检索到**在「1 云 + M 边 + N 端、由边缘服务器验证云端聚合结果、单 TA、PRG 掩码组包」这一组合下主张该性质的工作——但由于 IEEE/Elsevier 正文与多个数据库不可达，**该判断的置信度中等，不足以支撑「本文首次」的表述**。

**建议优先取得全文的两篇**：`10.1109/tmc.2026.3679423`（同态哈希验证）与 `10.1109/icc59461.2026.11587871`（分层可验证 SA）——**这两篇的正文细节直接决定创新点 1 的表述边界**。
