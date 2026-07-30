# 研究点2最终版：单TA约束下的检查点式组级半异步安全聚合

## 1. 最终定位

研究点2最终收敛为一条尽量克制、但协议闭环完整的主线：

> 在研究点1的单TA可验证安全聚合框架下，以组内 `stale reuse` 和检查点式组间半异步为主线，同时借鉴 `FedFleet` 的一次性性能分组作为初始化工程优化，使客户端能够先在本地安全复用陈旧训练结果，边缘组更新再以 `masked group package` 的形式进入缓冲区，并在后续检查点按批次解掩码、聚合和验证，从而在不破坏原有掩码与 `LHH` 验证链路的前提下，减少训练浪费和全局同步等待。

这一定义有四个关键词：

- `单TA`
- `组内 stale reuse`
- `masked group package + buffer + checkpoint`
- `每个 checkpoint 仍可由 ES 验证云端本次聚合结果`

本章不再主张：

- 组间“来一个更新一个”的完全异步
- 组间连续陈旧度权重 `beta_g`
- 全局异步轨迹的逐次在线审计
- 双TA或 threshold TA 扩展

---

## 2. 已经定下来的口径

### 2.1 系统口径

- 沿用研究点1的 `Cloud + Edge + User + 单TA` 结构
- 上传和聚合对象仍然是模型参数，而不是单步梯度
- 组内仍然走现有 `HVSA` 的掩码、签名、`LHH` 验证链路
- 验证主体仍然是 `ES`，即 edge-centric verification
- 威胁模型只防范不可信 `CS/Cloud` 对 checkpoint 选中聚合结果的篡改，不把 `ES` 作为攻击者
- 因此密码学验证只需覆盖 `CS \to ES` 交付的解掩码后 `raw sum`；`ES` 验证通过后，再执行公开确定性的组级补偿与 `FedAvg` 归一化
- 本章自洽性主要由密码学验证等式和模型训练/补偿公式保证，ACK、原子提交和缓存并发控制等属于实现完善项，不作为主协议数学成立的额外前提

### 2.2 异步口径

- 组内不是异步，而是每组独立完成一轮本地训练后封组
- 组间不再“所有组都到齐再聚合”，而是采用 `buffer + checkpoint` 半异步
- Cloud 不保存明文组模型，而是保存组级掩码包
- 未被本次 checkpoint 选中的组包继续留在 buffer 中等待，不立即丢弃

### 2.3 组内 stale reuse 口径

- `stale reuse` 只处理客户端自身掉线或上传超时造成的未上传本地训练结果，不处理组包等待、组包过期或 checkpoint 未选中
- 被复用的对象是客户端已经完成训练、但因自身掉线或上传超时而未被 `ES` 接收的明文结果 `w_i^{\mathit{stale},k}`，不是上一次已成功聚合的模型，也不是旧窗口中的 masked upload
- 一旦客户端更新已在封组前被 `ES` 确认接收，该客户端结果便不再属于客户端级 `stale reuse`；组包能否形成及其后续生命周期统一交由组级机制处理
- 客户端恢复参与后，先在本地完成新结果与 stale 结果的混合，再量化、加掩码并只上传一条混合后的 masked model
- TA 不需要知道 `\rho_i`、`w_i^{\mathit{stale},k}` 或客户端缓存内容
- 除 `stale reuse` 混合模型、组包和 checkpoint 选择集外，其余核心记号尽量沿用 `HVSA`，例如 `u_i^k`、`U_k`、`U_{\mathit{online}}^k(t_k)`、`\overline{w}_i^k(t_k)`、`\overline{W}_k(t_k)`、`H_k(t_k)`、`W_{\mathit{sum}}^{\mathit{raw}}(t)` 等

### 2.4 权重与组间补偿口径

为保证 strict `FedAvg` 与掩码、`LHH` 验证链路同时成立，本章将“客户端本地混合”“样本量加权平均”和“组间补偿”明确区分：

- 客户端本地 stale 结果权重 `\rho_i`，满足 `0 \le \rho_i \le \rho_{\max} \le 0.5`
- 客户端样本量 `s_i^k` 对应的组内 strict `FedAvg` 加权
- 组级总样本量 `S_k(t_k)` 对应的 checkpoint 聚合权重
- checkpoint 级总样本量 `S_Q(t)` 对应的最终 strict `FedAvg` 归一化
- 在 `raw sum` 完成 `LHH` 验证后，引入 `DVAFL` 的 `Direct-VCM` 风格公开确定性组间补偿

本章主协议中 **不引入解掩码前的组间连续非等权系数**。也就是说，不再把

`lambda_g = lambda_0 / (1 + beta_g * s_g)`

作为主协议的一部分。因为一旦不同组在解掩码前后带不同连续权重，就会把现有 `HVSA` 的掩码与 `LHH` 关系搞复杂，不利于把第四章讲圆。

本章只保留一种主线补偿：

```text
C_k(t) = S_k(t_k) · γ_vcm · (W(t) - W(t_k))
```

其中：

- `t` 表示当前 checkpoint 对应的最新已提交全局版本
- `t_k` 表示组包 `\Pi_k(t_k)` 的 `base version`
- `s_i^k = |D_i^k|` 表示客户端 `u_i^k` 的本地样本量
- `S_k(t_k) = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} s_i^k` 表示组包 `\Pi_k(t_k)` 的总样本量
- `n_k(t_k)` 仍表示在线客户端个数，只用于在线集合规模、最小在线人数阈值与元数据绑定
- `S_k(t_k)` 与 `W(t)-W(t_k)` 是结构量，不是额外超参数
- `γ_vcm` 是主协议中唯一保留的补偿强度超参数

需要特别说明的是：

- `\rho_i` 只在客户端本地、掩码之前生效，并随客户端级陈旧度增大而指数衰减
- 客户端样本量权重只作用在明文模型项上，不作用在掩码项上
- Cloud 与 TA 在解掩码阶段只处理“样本量加权和 + 掩码和”，不引入额外连续非等权系数
- 组间补偿只发生在 `raw sum` 已被 `LHH` 验证之后
- 最终执行顺序固定为“恢复 `raw sum` → `LHH` 验证 → ES 计算 Direct-VCM 风格补偿 → 除以 `S_Q(t)` 得到新全局模型”，不要求对补偿后的模型再增加一套密码学验证

### 2.5 最小公平约束

为避免快组持续挤占慢组机会，本章固定两条最小公平规则：

- 每个组在其上一个组包被消费或过期前，不能再产生新的待处理组包
- 正常情况下，每次 checkpoint 从 buffer 中按到达顺序选择最早的 `K` 个合法组包；当训练后期或高掉线阶段长期达不到 `K` 时，允许超时触发，对全部合法组包做保守 checkpoint（详见 Sec.4.4）

这样做的目的，是避免“快组无限刷屏、慢组长期饿死”。

在此基础上，本文允许在**不改变掩码、TA 解掩码与 `LHH` 验证链路**的前提下，引入三类仅作用于 `Q(t)` 形成阶段的缓冲调度层增强：

- `adaptive s_max`：根据上一阶段过期率动态调节组包最大陈旧度阈值
- 简单公平性约束：对长期未被选中的组包设置保底优先级
- 缓冲选择策略对比：在 `FIFO`、`Fresh-first` 与 `Fairness-aware Fresh-first` 之间做实验比较

它们都只改变“哪些合法组包进入本次 checkpoint”的调度逻辑，不改变组包内部结构、TA 返回的聚合掩码和 `M_Q(t)`、也不改变 `ES` 对 `raw sum` 的验证方式。因此，这三项适合作为**调度层增强或消融项**，而不单独上升为新的核心协议创新点。

---

## 3. 为什么这条链路是闭合的

| 模块 | 最终选择 | 为什么能闭合 |
|------|------|------|
| 组内 stale reuse | 客户端本地新旧模型混合 | 只改变明文上传内容，不改变 TA 按参与集合求掩码 |
| 权重与补偿 | 解掩码前仅保留 strict `FedAvg` 的样本量加权和，解掩码后仅保留 `Direct-VCM` 公开补偿 | 不需要让掩码跟着复杂连续非等权系数一起变化 |
| 掩码 | 用户仍只上传一条掩码参数向量 | 与研究点1一致，TA 仍可按参与集合求掩码和 |
| 验证 | “签名验证 + `raw sum` 的 `LHH` 同态关系验证 + ES 基于已验证 raw sum 本地重构最终模型” | `ES` 仍能对每个 checkpoint 的聚合结果做验证 |
| 异步 | 改为 `buffer + checkpoint` 半异步 | 避开“完全异步下每次全局下发都在线验证”的冲突 |

一句话说就是：

> 这版方案不是把验证硬塞进完全异步，而是主动把异步边界收缩到“checkpoint 级”，从而保住现有密码学闭环。

---

## 4. 核心协议

### 4.1 初始化与提前分组

系统在训练开始前根据客户端训练与通信能力做一次提前分组。推荐画像指标为：

```text
d_i = t_i^train + t_i^comm
```

如果当前实验中只能可靠测到通信时延，也可退化为：

```text
d_i = t_i^comm
```

Cloud 按 `d_i` 将客户端划分为 `M` 个组：

```text
U_1, U_2, ..., U_M
```

每个组 `U_k` 绑定一个边缘服务器 `e_k`。分组只在初始化时执行一次，目标是让组内设备性能更接近，减少组内等待。

这里需要明确其定位：

> 提前分组借鉴自 `FedFleet` 的性能感知分组思想，仅作为初始化阶段的训练组织优化，用于降低组内同步等待并平衡各组组包产出速率；它不改变单TA掩码与验证协议，也不作为本章创新点。

---

### 4.2 组内本地训练、stale reuse 混合、掩码上传与组包生成

为尽量与 `HVSA` 保持一致，本节沿用其”轮次 `t`、边缘索引 `k`、掩码上横线”记号。第三章中 `t` 表示同步通信轮次，本章将其扩展为版本语义：`t` 表示当前最新已提交的 **checkpoint 全局版本**，`t_k` 表示组包 `\Pi_k(t_k)` 生成时的 **训练基版本**。在 mixed-version checkpoint 中，通常会出现 `t_k < t`。设当前已发布的全局模型为：

```text
W(t)
```

当边缘服务器 `e_k` 被允许开始下一轮本地训练时，它将当前全局模型 `W(t)` 广播给组内客户端集合 `U_k`，本轮训练基版本定格为 `t_k = t`。实际在线并参与训练的客户端集合记为：

```text
U_{\mathit{online}}^k(t_k) \subseteq U_k
```

同时，记客户端 `u_i^k` 的本地样本量为：

```text
s_i^k = |D_i^k|
```

由于本地数据划分在训练开始前已固定，`s_i^k` 视为静态结构量。

每个在线客户端 `u_i^k \in U_{\mathit{online}}^k(t_k)` 基于同一个 `W(t_k)` 完成本地训练，先得到本轮新模型：

```text
w_i^{\mathit{new},k}
```

同时，客户端至多维护一个**因自身上传失败而未被 `ES` 接收**的 stale 候选 `w_i^{\mathit{stale},k}`，以及该结果的产生版本 `t_i`。该候选仅在以下事件中产生：客户端已经完成一轮本地训练，但因客户端自身通信掉线或上传超时，未能在本次封组截止时间前把结果交付给 `ES`。若客户端没有这类未上传结果，则视为“无 stale 结果可复用”。

这里以“`ES` 是否在封组前成功接收该客户端更新”为客户端级与组级状态的分界：

- 若上传失败，则客户端缓存本次新训练结果及其产生版本，作为下一次恢复参与时的 stale 候选
- 若上传成功并被 `ES` 确认接收，则该贡献进入组级处理链路；即使随后因在线人数不足未能封成合法组包，或组包等待、被 checkpoint 选中、最终过期，都不再由客户端 `stale reuse` 处理
- 同一客户端始终只保留一个 stale 候选；若连续多次上传失败，以最近一次尚未聚合的新训练结果替换更旧的候选，避免重复混合和重复贡献

当客户端恢复参与并重新得到本轮新模型后，先计算 stale 候选与当前组训练基版本之间的间隔：

```text
E_i = t_k - t_i
```

本文令 `\rho_i` 直接表示 stale 结果的权重，并采用带上界的指数衰减：

```text
\rho_i(E_i) = \rho_{\max} \cdot exp(-\beta_c \cdot E_i),
0 < \rho_{\max} \le 0.5, \quad \beta_c > 0
```

于是客户端真正用于上传的本地明文模型定义为：

```text
如果无 stale 候选:
    w_i^k = w_i^{\mathit{new},k}

否则:
    w_i^k = (1 - \rho_i) \cdot w_i^{\mathit{new},k}
           + \rho_i \cdot w_i^{\mathit{stale},k}
```

该定义保证 stale 结果越旧，其影响越小；同时 `\rho_{\max} \le 0.5` 保证当前新训练结果始终不弱于 stale 结果。stale 候选的作用是减少客户端自身掉线造成的训练浪费，而不是替代当前模型或补偿组包版本失配。

#### 与 MAFSA 权重定义的关系

本文借鉴 `MAFSA` 对掉线客户端新旧训练结果进行融合的基本动机，但不直接照搬其权重方向。`MAFSA` 的原始定义为：

```text
\alpha_i(E_i) = exp(-\beta_c \cdot E_i)

w_i^{\mathit{MAFSA}}
    = \alpha_i \cdot w_i^{\mathit{new}}
      + (1 - \alpha_i) \cdot w_i^{\mathit{stale}}
```

按该公式，随着 `E_i` 增大，当前新结果的权重 `\alpha_i` 减小，而 stale 结果的权重 `1-\alpha_i` 增大。考虑到版本间隔越大时 stale 结果与当前优化方向的偏差风险通常更高，本文将指数衰减量改为直接控制 stale 结果，并增加 `\rho_{\max}` 上界。因此，本章应表述为“**受 MAFSA stale-result reuse 启发的安全聚合兼容及时效性修正**”，而不是“直接采用 MAFSA 的加权公式”。

随后客户端对该混合结果加掩码上传：

```text
\overline{w}_i^k(t_k) = s_i^k \cdot w_i^k + m_i^k(t_k)
```

这里要明确一点：

> 被复用的是客户端因自身掉线或上传超时而未被 `ES` 接收的明文训练结果 `w_i^{\mathit{stale},k}`，不是已成功聚合的历史模型，也不是旧轮次里的 masked upload。旧掩码消息不能被直接重放。

这一步的 strict `FedAvg` 语义也需要特别说明：

> 客户端样本量权重 `s_i^k` 只乘在明文模型 `w_i^k` 上，不乘在掩码项 `m_i^k(t_k)` 上。这样一来，解掩码后可直接恢复 strict `FedAvg` 所需的样本量加权模型和，同时不改变 TA 对掩码和的求和方式。

`e_k` 汇总得到本轮组级掩码和：

```text
\overline{W}_k(t_k) = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} \overline{w}_i^k(t_k)
                    = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} s_i^k \cdot w_i^k
                      + \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} m_i^k(t_k)
```

为避免某个可恢复组包退化为单客户端明文结果，本文进一步要求组包生成满足最小在线人数下界 `n_{\min}`。当

```text
|U_{\mathit{online}}^k(t_k)| < n_{\min}
```

时，`e_k` 放弃本次封装——既不生成 `\Pi_k(t_k)`，也不将任何数据送入 buffer。该组需在后续时刻重新发起训练组织。本文默认取 `n_{\min} \ge 2`，其作用是保护**客户端级参数隐私**，而不是隐藏组级聚合结果。

同时定义本轮组包对应的总样本量：

```text
S_k(t_k) = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} s_i^k
```

对于满足上述下界的合法组包，`ES` 继续计算：

```text
H_k(t_k) = \mathsf{LHH.Hash}(pp_{\mathit{LHH}}, \overline{W}_k(t_k))
```

然后对如下最小元数据签名：

```text
(k, t_k, h_{\mathit{base}}(t_k), digest(U_{\mathit{online}}^k(t_k)), n_k(t_k), S_k(t_k), H_k(t_k))
```

其中：

- `h_{\mathit{base}}(t_k) = \mathsf{Hash}(W(t_k))`
- `n_k(t_k) = |U_{\mathit{online}}^k(t_k)|`
- `S_k(t_k) = \sum s_i^k`
- `digest(U_{\mathit{online}}^k(t_k))` 表示对在线参与集合的摘要

这一步完成后，形成一个组级掩码包：

```text
\Pi_k(t_k) = (k, t_k, h_{\mathit{base}}(t_k), U_{\mathit{online}}^k(t_k), n_k(t_k), S_k(t_k), \overline{W}_k(t_k), H_k(t_k), \sigma_k(t_k))
```

这里最关键的一点是：

> buffer 中保存的是 `\Pi_k(t_k)` 这样的组级掩码包，而不是明文组模型。

客户端 stale 候选的状态更新以“上传是否成功进入组包链路”为准，而不等待 checkpoint 结果：

```text
若本次新结果因客户端掉线或上传超时而未被 ES 接收:
    w_i^{\mathit{stale},k} \gets w_i^{\mathit{new},k},\quad t_i \gets t_k

若本次混合结果已在封组前被 ES 确认接收:
    clear(w_i^{\mathit{stale},k}, t_i)
```

组包形成后的等待、选择、验证和过期属于组级状态，不反向生成客户端 stale 候选。这样可避免同一贡献同时受到客户端级复用和组级 `Direct-VCM` 的双重补偿，也不需要把 checkpoint 生命周期同步回每个客户端。

---

### 4.3 TA 的组包掩码支持

对任意组包 `\Pi_k(t_k)`，TA 基于与其完全一致的参与集合 `U_{\mathit{online}}^k(t_k)` 可在内部确定对应的组包掩码和：

```text
M_k(t_k) = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} m_i^k(t_k)
```

注意这里 **不需要** 将 `s_i^k` 乘到掩码项上。原因在于客户端上传的是

```text
\overline{w}_i^k(t_k) = s_i^k \cdot w_i^k + m_i^k(t_k)
```

因此在组内求和后，有：

```text
\overline{W}_k(t_k) - M_k(t_k)
    = \sum_{u_i^k \in U_{\mathit{online}}^k(t_k)} s_i^k \cdot w_i^k
```

即解掩码后恢复出的正是 strict `FedAvg` 所需的样本量加权模型和。这一点保证了本章从“客户端本地加权”过渡到“组包级 strict `FedAvg` 聚合”时，不会破坏单TA掩码链路。

但为避免 Cloud 同时持有单组明文恢复所需的两部分证据，TA **不在组包进入 buffer 时向 Cloud 单独释放** `M_k(t_k)`。

当某次 checkpoint 选中集合 `Q(t)` 被确定后，TA 仅针对本次选中的组包集合返回聚合掩码和：

```text
M_Q(t) = \sum_{\Pi_k(t_k) \in Q(t)} M_k(t_k)
```

因此，Cloud 在 checkpoint 阶段只能恢复本次选中集合的 `raw sum`，而不能提前恢复任意单个组包的明文组模型。

---

### 4.4 Buffer 与 Checkpoint 触发

Cloud 维护一个待处理组包缓冲区（借鉴 `FedBuff`~\cite{FedBuff} 的 buffered aggregation 设计，并将其粒度从单客户端提升到组包）：

```text
B = {\Pi_k(t_k)}
```

每个组最多只能在 buffer 中保留一个尚未消费的组包。

设当前最新已提交并广播的全局模型版本为 `W(t)`，checkpoint 触发阈值为 `K`。这里的 `K` 仅用于控制组间半异步调度粒度，而**不作为隐私阈值**；本文的隐私目标是保护单个客户端参数不泄露，而不额外隐藏组级聚合结果。对任意组包 `\Pi_k(t_k)`，其中 `t_k` 表示该组包的 `base version`。由于 buffer 中可能同时存在基于旧版本模型训练得到的组包，因此先定义其陈旧度为：

```text
s_k = t - t_k
```

若：

```text
s_k > s_{\max}
```

则该组包被判为过期并从 buffer 中移除。

过滤过期组包后，记当前合法候选集合为 `B_{\mathit{legal}}`（由于 `n_{\min}` 检查已在组包封装阶段完成，buffer 内不再重复校验）。若：

```text
|B_{\mathit{legal}}| >= K
```

则触发一个新的 checkpoint。Cloud 按到达顺序选择最早的 `K` 个合法组包，记为：

```text
Q(t) \subseteq B_{\mathit{legal}},
\quad
Q(t) = {\Pi_{k_1}(t_{k_1}), \Pi_{k_2}(t_{k_2}), ..., \Pi_{k_K}(t_{k_K})}
```

为避免训练后期或高掉线阶段长期达不到 `K` 而导致系统停滞，本文再引入一个轻量的等待时限 `\Delta_{\mathit{timeout}}`。若 buffer 中最早合法组包的等待时间已达到 `\Delta_{\mathit{timeout}}`，但当前合法组包数仍满足：

```text
1 <= |B_{\mathit{legal}}| < K
```

则允许对当前全部合法组包触发一次保守 checkpoint，此时取 `Q(t) = B_{\mathit{legal}}`。为保持记号简洁，后文将常规触发和超时触发得到的选中集合统一记为 `Q(t)`。

在 `Q(t)` 确定后，Cloud 需先发布一个本次 checkpoint 的声明对象：

```text
\mathit{checkpoint\_manifest}(t)
    = (t, \mathit{mode}(t), |Q(t)|,
       {(k, t_k, h_{\mathit{base}}(t_k), n_k(t_k), S_k(t_k),
         digest(U_{\mathit{online}}^k(t_k)), H_k(t_k))}
          _{\Pi_k(t_k) \in Q(t)})
```

其中 `\mathit{mode}(t)` 用于标记本次 checkpoint 属于“常规触发”还是“超时触发”。进一步地，可对其计算公开摘要：

```text
h_{\mathit{manifest}}(t) = \mathsf{Hash}(\mathit{checkpoint\_manifest}(t))
```

Cloud 必须先将 `\mathit{checkpoint\_manifest}(t)` 广播给 TA 和参与本次验证的 `ES`，再由 TA 按该声明对象中的选中组包集合计算 `M_Q(t)`。这样一来，“已公布的 checkpoint 选中集合 `Q(t)`”在协议中就不再是抽象概念，而是一个带有明确元数据绑定的可验证载体。

本次 checkpoint 只围绕 `Q(t)` 进行聚合、解掩码、补偿和验证。

未被选中的组包继续保留在 buffer 中等待后续 checkpoint，不立即作废。

若论文需要进一步增强第四章的调度表达而又不希望破坏密码学闭环，可在上述主协议之上叠加如下三种**可选调度增强**：

**1. 自适应 `s_max`**

设本阶段进入缓冲区的陈旧组包总数为 `N_{\mathit{late}}(t)`，其中因

```text
s_k > s_{\max}(t)
```

而被判定为过期并移除的组包数为 `N_{\mathit{expired}}(t)`。定义过期压力为：

```text
P_{\mathit{exp}}(t)
    = N_{\mathit{expired}}(t) / max(N_{\mathit{late}}(t), 1)
```

则下一阶段的最大陈旧度阈值可自适应更新为：

```text
s_{\max}(t+1)
    = clip(s_{\max}(t) + \eta_s \cdot (P_{\mathit{exp}}(t) - P_{\mathit{tar}}),
           s_{\min}, s_{\hi})
```

其中，`P_{\mathit{tar}}` 为目标过期率，`\eta_s` 为调节步长，`s_{\min}` 与 `s_{\hi}` 分别为上下界。其直观含义为：

- 若上一阶段有大量迟到组包因超期被丢弃，则 `s_{\max}` 适当增大
- 若几乎没有组包过期，则 `s_{\max}` 适当减小，以减少不必要的陈旧融合

**2. 简单公平性约束**

为避免某些慢组长期被 buffer 排队机制压后，可为每个合法候选组包维护一个连续延后计数器：

```text
c_k^{\mathit{defer}}(t)
```

它表示组 `k` 自进入合法候选集合后，已经连续多少次 checkpoint 未被选中。给定公平阈值 `D_{\mathit{fair}}` 后，可采用如下硬约束：

- 先过滤全部过期组包，仅保留 `B_{\mathit{legal}}`
- 若某个候选组包满足 `c_k^{\mathit{defer}}(t) >= D_{\mathit{fair}}`，则它在本次 `Q(t)` 中获得保底优先级
- 剩余名额再按既定缓冲策略填充

这样做的目的，不是改变样本量权重或版本补偿公式，而只是避免“新鲜组包持续霸占缓冲名额、慢组永远没有机会进入 checkpoint”。

**3. 缓冲选择策略对比**

在过滤过期组包之后，Cloud 可对 `Q(t)` 的形成规则做如下实验性对比：

- `FIFO`：按到达顺序选择最早的 `K` 个合法组包
- `Fresh-first`：按陈旧度从小到大选择最“新鲜”的 `K` 个合法组包
- `Fairness-aware Fresh-first`：先满足 `c_k^{\mathit{defer}}(t) >= D_{\mathit{fair}}` 的保底候选，再按陈旧度从小到大填满剩余名额

需要强调的是，这些策略的差别只发生在 `Q(t)` 被确定之前。一旦 `Q(t)` 确定，后续 `checkpoint\_manifest(t)`、`TA` 计算 `M_Q(t)`、Cloud 恢复 `W_{\mathit{sum}}^{\mathit{raw}}(t)` 以及 `ES` 的 `LHH` 验证过程完全不变。

---

### 4.5 Checkpoint 聚合、解掩码、组间补偿与全局更新

对于本次选中的组包集合 `Q(t)`，Cloud 先聚合其掩码和：

```text
\overline{W}_Q(t) = \sum_{\Pi_k(t_k) \in Q(t)} \overline{W}_k(t_k)
```

TA 仅对本次选中组包集合返回聚合掩码和：

```text
M_Q(t) = \sum_{\Pi_k(t_k) \in Q(t)} M_k(t_k)
```

随后恢复 checkpoint 的 `raw sum`。为了与 `HVSA` 的记号区分，本章将该量记为：

```text
W_{\mathit{sum}}^{\mathit{raw}}(t) = \overline{W}_Q(t) - M_Q(t)
```

此时 `W_{\mathit{sum}}^{\mathit{raw}}(t)` 的语义已不再是“客户端等权求和”，而是：

> 本次 checkpoint 中所有被接纳客户端本地模型的**样本量加权总和**。

定义本次 checkpoint 参与的总样本量为：

```text
S_Q(t) = \sum_{\Pi_k(t_k) \in Q(t)} S_k(t_k)
```

对于本次选中的任意 stale 组包 `\Pi_k(t_k) \in Q(t)`，若 `t_k < t`，则定义其公开确定性的版本偏差为：

```text
G_k(t) = W(t) - W(t_k)
```

借鉴 `DVAFL` 中 `Direct-VCM` 的思路，并将其从“单客户端更新修正”提升到“组明文和修正”，定义组级后置补偿项为：

```text
C_k(t) = S_k(t_k) · γ_{\mathit{vcm}} · G_k(t), \quad \Pi_k(t_k) \in Q(t),\ t_k < t
```

之所以采用 `S_k(t_k)` 而不是 `n_k(t_k)`，是因为 `W_{\mathit{sum}}^{\mathit{raw}}(t)` 本身处在“样本量加权和”的量纲中，组间补偿项也必须保持相同量纲，才能与 strict `FedAvg` 的最终归一化一致。

于是，本次 checkpoint 的补偿后明文和为：

```text
W_{\mathit{sum}}^{\mathit{corr}}(t) = W_{\mathit{sum}}^{\mathit{raw}}(t)
                                      + \sum_{\Pi_k(t_k) \in Q(t),\ t_k < t} C_k(t)
```

最终生成的新全局模型为：

```text
W(t+1) = W_{\mathit{sum}}^{\mathit{corr}}(t) / S_Q(t)
```

若所有在线客户端样本量近似相同，则有 `S_k(t_k) \propto n_k(t_k)`，此时本章 strict `FedAvg` 口径会自然退化为前一版按客户端个数平均的写法。

也就是说，本章最终版保留的是”strict `FedAvg` 样本量加权平均 + 公开确定性组间补偿”的语义：

- 先在客户端本地完成 `stale reuse` 混合
- 再在组内形成“样本量加权明文模型和 + 普通掩码和”
- 再在 checkpoint 上恢复并验证样本量加权 `raw sum`
- 最后仅对 stale 组包附加 `Direct-VCM` 风格的公开组级补偿

这里需要特别强调：

- `S_k(t_k)` 与 `G_k(t)` 是结构量，不属于可自由替换的超参数
- `n_k(t_k)` 仍然只承担在线客户端计数与最小在线人数约束语义
- 主协议中唯一保留的补偿强度超参数是 `γ_{\mathit{vcm}}`
- `First-order / Second-order VCM` 仅作为扩展对比，不进入主协议

---

### 4.6 Checkpoint 级验证与最终模型重构

本次 checkpoint 的验证仍由参与本次 checkpoint 的 `ES` 执行。本章的密码学验证对象是唯一的：**解掩码后的 `raw sum`**，即 `W_{\mathit{sum}}^{\mathit{raw}}(t)`。补偿后的最终模型 `W(t+1)` 由已验证的 `raw sum` 和公开确定性补偿规则派生得到，ES 本地独立完成重构，不构成额外的密码学验证步骤。

验证与重构链路分为三步。

**Step 1: manifest 一致性检查与签名验证。** 参与本次 checkpoint 的 `ES` 先检查自己收到的 `\mathit{checkpoint\_manifest}(t)` 是否与本地可见的组包证据一致：即 manifest 中声明的每个组包元数据 `(k, t_k, h_{\mathit{base}}(t_k), n_k(t_k), S_k(t_k), digest(U_{\mathit{online}}^k(t_k)), H_k(t_k))` 是否与对应 `\Pi_k(t_k)` 完全匹配。若不一致，直接判定本次 checkpoint 声明无效。然后，`ES` 对每个被选中组包的签名 `\sigma_k(t_k)` 执行 Ed25519 验签。

**Step 2: LHH 验证 `raw sum`。** Cloud 与 TA 基于 `\mathit{checkpoint\_manifest}(t)` 中声明的选中集合 `Q(t)` 分别提供 `W_{\mathit{sum}}^{\mathit{raw}}(t)` 和 `M_Q(t)`。`ES` 对两者做 LHH 哈希：

```text
H_{\mathit{sum}}^{\mathit{raw}}(t)
    = \mathsf{LHH.Hash}(pp_{\mathit{LHH}}, W_{\mathit{sum}}^{\mathit{raw}}(t))
\qquad
H_{\mathit{mask}}(t)
    = \mathsf{LHH.Hash}(pp_{\mathit{LHH}}, M_Q(t))
```

然后检查与 `HVSA` 同风格的 LHH 一致性关系：

```text
H_{\mathit{sum}}^{\mathit{raw}}(t) \cdot H_{\mathit{mask}}(t)
    = \prod_{\Pi_k(t_k) \in Q(t)} H_k(t_k)
```

这与第三章的核心验证式 `H_{\mathit{sum}}(t) \cdot H_{\mathit{mask}}(t) = \prod_{k=1}^{M} H_k(t)` 保持同一记号逻辑；区别仅在于右端不再是”本轮所有 `ES`”，而是”本次 checkpoint 选中的组包集合”。当上述关系成立时：

> Cloud 本次 checkpoint 所解掩码得到的 `W_{\mathit{sum}}^{\mathit{raw}}(t)`，确实来自 `Q(t)` 中这些组包的掩码聚合结果。

**Step 3: ES 基于已验证 `raw sum` 本地重构 `W(t+1)`。** Step 2 通过后，`ES` 持有一个已被密码学证实的 `W_{\mathit{sum}}^{\mathit{raw}}(t)`。此后补偿与最终模型的重构是纯公开确定性计算，不涉及新的密码学操作。`ES` 利用本地已有的历史全局模型 `W(t), W(t_k)`、manifest 中公开的 `S_k(t_k)`、以及公开补偿强度 `γ_{\mathit{vcm}}`，独立计算：

```text
C_{\mathit{total}}(t)
    = \sum_{\Pi_k(t_k) \in Q(t),\ t_k < t}
      S_k(t_k) · γ_{\mathit{vcm}} · (W(t) - W(t_k))

W_{\mathit{sum}}^{\mathit{corr}}(t)
    = W_{\mathit{sum}}^{\mathit{raw}}(t) + C_{\mathit{total}}(t)

W(t+1) = W_{\mathit{sum}}^{\mathit{corr}}(t) / S_Q(t)
\qquad\text{其中}\qquad
S_Q(t) = \sum_{\Pi_k(t_k) \in Q(t)} S_k(t_k)
```

各 `ES` 将本地重构得到的 `W(t+1)` 作为本次 checkpoint 的可信最终模型，下发给各自组内的客户端用于下一轮本地训练。若 Cloud 也广播了一个 `W(t+1)`，`ES` 可对其做哈希一致性检查（例如 `\mathsf{Hash}(W_{\mathit{ES}}) = \mathsf{Hash}(W_{\mathit{Cloud}})`），但这是可选的传输完整性比对，不是密码学验证步骤——因为 Cloud 此时已不再是需要被防范的攻击面：`raw sum` 的正确性已被 Step 2 的 LHH 等式判定，补偿和平均的每一步都可以由 `ES` 用公开量独立复现。

本章的安全边界由此收口为：

> **密码学验证只发生在 `CS \to ES` 的 `raw sum` 交付阶段。** Cloud 可作恶的窗口仅限于篡改 `\mathit{checkpoint\_manifest}(t)` 中声明的 `Q(t)` 元数据（漏选、替换、伪造组包）或篡改 `W_{\mathit{sum}}^{\mathit{raw}}(t)` 的值。这两类攻击分别被 Step 1 的 manifest 一致性检查 + 签名验证、以及 Step 2 的 LHH 同态关系验证捕获。补偿项 `C_{\mathit{total}}(t)` 不是 Cloud 可自由篡改的独立攻击面——它由 `Q(t)`、历史全局模型、`S_k(t_k)`、`γ_{\mathit{vcm}}` 这些已公开量唯一确定，Cloud 即使本地也能算，但无法在不被发现的情况下用与公开规则不一致的值冒充。

相对地，本文**不额外处理** `Q(t)` 形成之前 Cloud 对候选组包的静默漏选、调度审查或 buffer 到达顺序操控问题。此类问题属于”可审计调度 / anti-censorship”范畴，不作为本章主协议的安全保证对象。

需要明确区分"训练模型下发"和"验证材料可见性"两个概念。未被前序 checkpoint 消费的组不能据此提前开始下一轮训练——训练更新权限仅在组包被选中并成功消费后授予。但当其旧组包在后续 checkpoint 中被选中时，其所属 ES 仍可获得所需的历史全局模型版本 `W(t), W(t_k)`，作为组级补偿与最终模型重构的公开输入。两者不冲突：前者是训练调度权限，后者是验证计算所需的公开数据可见性。

---

### 4.7 Checkpoint 完成后的状态转移

对于本次被选中的组包：

- 它们从 buffer 中移除
- 对应的组接收新全局模型 `W(t+1)`
- 这些组随后才能开始基于 `W(t+1)` 的下一轮本地训练

对于未被选中的组包：

- 它们继续保留在 buffer 中
- 对应组暂不生成新的待处理组包
- 它们等待后续 checkpoint 被消费，或因超过 `s_max` 而过期

这一规则的意义在于：

- 避免一个组同时持有多个在途组包
- 避免旧组包与新组包并发打架
- 让版本号、掩码与验证关系始终清楚

---

## 5. 两个直观例子

### 5.1 三个 ES、每个 ES 三个客户端的组包生成例子

设系统有三个边缘组：

- `ES1: {u11, u12, u13}`
- `ES2: {u21, u22, u23}`
- `ES3: {u31, u32, u33}`

当前全局模型为 `W(0)`。

若三组都完成本地训练并上传，则会形成三个组包：

```text
\Pi_1(0), \Pi_2(0), \Pi_3(0)
```

其中例如 `ES1` 的组包对应（为简化说明，假设各客户端样本量相同，`s_i^k` 已吸收到 `w` 记号中）：

```text
\overline{W}_1(0) = \overline{w}_{11}^1(0) + \overline{w}_{12}^1(0) + \overline{w}_{13}^1(0)
M_1(0)             = m_{11}^1(0) + m_{12}^1(0) + m_{13}^1(0)
```

`ES1` 给出：

```text
H_1(0) = \mathsf{LHH.Hash}(pp_{\mathit{LHH}}, \overline{W}_1(0))
```

为说明单个组包内部的掩码结构，可记其对应组掩码和为：

```text
H_{\mathit{mask}}^1(0) = \mathsf{LHH.Hash}(pp_{\mathit{LHH}}, M_1(0))
```

这里的 `M_1(0)` 仅用于解释单组包的内部构成；在正式协议中，TA 不会在组包进入 `buffer` 时单独向 Cloud 释放它，而只会在 checkpoint 触发后返回选中集合的聚合掩码和 `M_Q(t)`。

若仅从单组包内部关系看，对应的明文和可写为 `W_1^{\mathit{sum}}(0) = \overline{W}_1(0) - M_1(0)`，并满足：

```text
H_{\mathit{sum}}^1(0) \cdot H_{\mathit{mask}}^1(0) = H_1(0)
```

这说明本章的最小验证单元已经不是“整轮所有组同时在线”，而是“一个被 checkpoint 选中的固定组包集合”。

### 5.2 五个组、`K = 3` 的 mixed-version checkpoint 与后置补偿例子

设共有五个组，buffer 阈值 `K = 3`。

第一阶段，先到达的是：

```text
\Pi_1(0), \Pi_2(0), \Pi_3(0)
```

此时触发 checkpoint 1，选中集合为：

```text
Q(0) = {\Pi_1(0), \Pi_2(0), \Pi_3(0)}
```

Cloud 聚合：

```text
\overline{W}_Q(0) = \overline{W}_1(0) + \overline{W}_2(0) + \overline{W}_3(0)
M_Q(0)            = M_1(0) + M_2(0) + M_3(0)
W_{\mathit{sum}}^{\mathit{raw}}(0) = \overline{W}_Q(0) - M_Q(0)
W_{\mathit{sum}}^{\mathit{corr}}(0) = W_{\mathit{sum}}^{\mathit{raw}}(0)
W(1)              = W_{\mathit{sum}}^{\mathit{corr}}(0) / (S_1(0) + S_2(0) + S_3(0))
```

由 `ES1, ES2, ES3` 验证 checkpoint 1 后，三组接收 `W(1)` 并允许开始下一轮本地训练。

与此同时，较慢的两个组后来才提交：

```text
\Pi_4(0), \Pi_5(0)
```

它们先留在 buffer 中等待。

如果此后 `ES1` 又基于 `W(1)` 完成了一轮训练并形成：

```text
\Pi_1(1)
```

那么 buffer 变为：

```text
B = {\Pi_4(0), \Pi_5(0), \Pi_1(1)}
```

这时当前最新已提交全局版本为 `W(1)`，于是触发 checkpoint 2，选中集合为：

```text
Q(1) = {\Pi_4(0), \Pi_5(0), \Pi_1(1)}
```

注意这里三个组包的 `base version` 并不相同：

- `\Pi_4(0)` 和 `\Pi_5(0)` 基于 `W(0)`
- `\Pi_1(1)` 基于 `W(1)`

本章在这里不再直接做“mixed-version 组包的原样平均”，而是先恢复 `raw sum`，再对 stale 组包做 `Direct-VCM` 风格的组级后置补偿：

```text
\overline{W}_Q(1) = \overline{W}_4(0) + \overline{W}_5(0) + \overline{W}_1(1)
M_Q(1)            = M_4(0) + M_5(0) + M_1(1)
W_{\mathit{sum}}^{\mathit{raw}}(1) = \overline{W}_Q(1) - M_Q(1)
```

其中：

```text
G_4(1) = W(1) - W(0)
G_5(1) = W(1) - W(0)
G_1(1) = 0
```

于是补偿项为：

```text
C_4(1) = S_4(0) · γ_vcm · (W(1) - W(0))
C_5(1) = S_5(0) · γ_vcm · (W(1) - W(0))
C_1(1) = 0
```

因此：

```text
W_{\mathit{sum}}^{\mathit{corr}}(1)
    = W_{\mathit{sum}}^{\mathit{raw}}(1) + C_4(1) + C_5(1)

W(2) = W_{\mathit{sum}}^{\mathit{corr}}(1) / (S_4(0) + S_5(0) + S_1(1))
```

随后由 `ES4, ES5, ES1` 先对 `W_{\mathit{sum}}^{\mathit{raw}}(1)` 执行 `LHH` 验证，再基于公开补偿规则重算 `W(2)`。

这个例子对应的正是我们讨论过的关键情形：

> `buffer = {\Pi_4(0), \Pi_5(0), \Pi_1(1)}` 时，`LHH` 仍然直接验证 `raw sum`，而 mixed-version 组包之间的版本失配则由公开确定性的组级后置补偿来缓解。

---

## 6. 组内 stale reuse 正式机制及其兼容性

本章将组内 `stale reuse` 作为正式机制，而不再把它放在“可选增强”位置。它承担的是客户端级掉线/回归处理职责，与组间 `buffer + checkpoint` 半异步并不冲突：

- 组内 `stale reuse` 解决的是“客户端已完成训练，但因自身掉线或上传超时而未被 `ES` 接收的结果不要白白浪费”
- 组间 `buffer + checkpoint` 解决的是“不同边缘组到达时间不一致，Cloud 不必全局 barrier”

这两层机制正好分工：

- 客户端层处理“人掉了再回来怎么办”
- 组间层处理“组到得快慢不一致怎么办”

二者以 `ES` 在封组前是否成功接收客户端更新为边界。一旦更新被 `ES` 确认接收，组包形成、等待和过期等后续状态只由组级机制管理；即使因在线人数不足未形成合法组包，也不触发客户端级 stale reuse。

还需要进一步说明的是，本文中的“陈旧性”实际上来自两个不同维度：

- 客户端级陈旧性：由 `E_i = t_k - t_i` 描述，反映客户端自身掉线造成的未上传训练结果与当前组训练基版本之间的间隔
- 组级版本陈旧性：由 `t - t_k` 描述，反映组包 `\Pi_k(t_k)` 的 `base version` 与当前 checkpoint 全局版本之间的失配

这两个维度的问题在本文中被刻意分开处理：客户端级陈旧性通过本地 `stale reuse` 混合缓解，且 stale 权重随 `E_i` 指数衰减；组级版本陈旧性通过 checkpoint 后置的 `Direct-VCM` 补偿缓解。本文不追求用一套统一补偿项对这两个维度同时做精确校正，也不对同一个失败事件实施双重补偿，而是优先保持掩码、`LHH` 与公开确定性链路的闭合性。

组内 `stale reuse` 之所以适合保留成一个正式点，核心原因就是它不破坏现有密码学框架：

- 它发生在客户端本地、掩码之前
- TA 仍然只按实际参与集合计算掩码和
- Cloud 仍然只做组包求和、解掩码、公开补偿和 strict `FedAvg` 归一化
- `ES` 仍然先做”签名验证 + LHH 关系验证”，再按公开规则本地重构最终模型并下发给所属客户端

如果将组内 `stale reuse` 纳入正式方案，则第四章的结构可以稳定表述为：

> 提前分组 + 组内 stale reuse + 组间 buffer/checkpoint 半异步 + checkpoint 级 ES 验证。

---

## 7. 本章真正的创新点

### 创新点1：单TA可验证安全聚合下的检查点式组级半异步扩展

借鉴 `FedBuff` 的 buffered asynchronous aggregation 思想，并将其从单客户端异步提交提升到组包粒度，在不改变研究点1单TA掩码与 `LHH` 验证闭环的前提下，将跨组训练组织方式从”全组同步 barrier”扩展为”组包缓冲 + checkpoint 触发”的组级半异步机制。

### 创新点2：安全聚合兼容的组内 stale reuse 机制

受 `MAFSA` 的掉线客户端新旧结果复用思路启发，本文将复用操作下沉到客户端本地，仅复用因客户端自身掉线或上传超时而未被 `ES` 接收的训练结果；同时对其权重方向作时效性修正，使 stale 结果权重 `\rho_i=\rho_{\max}e^{-\beta_cE_i}` 随版本间隔衰减，并限制 `\rho_{\max}\le 0.5`。客户端在掩码前完成新旧结果混合且只上传一条混合后的 masked model，因而不会向 ES 暴露两个明文模型，也不会破坏 TA 解掩码与 `LHH` 验证关系。

### 创新点3：面向 mixed-version 组包的公开确定性后置补偿与 checkpoint 验证机制

不在 Cloud 侧保存明文组模型，而是保存带有签名证据、参与集合摘要、`base model hash` 与组级总样本量 `S_k(t_k)` 的组级掩码包；对 mixed-version checkpoint，先用 `LHH` 直接验证 `raw sum`，补偿项由 `Q(t)`、历史全局模型、`S_k(t_k)`、`γ_{\mathit{vcm}}` 这些已公开量唯一确定，ES 基于已验证的 `raw sum` 本地独立重构 `W(t+1)` 并下发给所属客户端，从而将密码学验证集中在 `CS \to ES` 的 `raw sum` 交付阶段。

---

## 8. 实验方案

建议实验主线保留六组，既能体现组间半异步，也能把组内 `stale reuse` 与组间补偿单独打出来：

1. `Sync-HVSA`
2. `Grouped-HVSA + Inter-Group Sync`
3. `Grouped-HVSA + Buffer/Checkpoint Semi-Async`
4. `Grouped-HVSA + Buffer/Checkpoint Semi-Async + Intra-Group Stale Reuse`
5. `Grouped-HVSA + Buffer/Checkpoint Semi-Async + Intra-Group Stale Reuse + Direct-VCM Group Compensation`
6. `Grouped-HVSA + Buffer/Checkpoint Semi-Async + Intra-Group Stale Reuse + Direct-VCM Group Compensation + Tampering Attack`

攻击实验只做三类即可：

- 篡改本次 checkpoint 输出的全局模型
- 篡改 `checkpoint_manifest(t)`，即篡改已公布的 checkpoint 选中集合或其中组包元数据
- 在 `checkpoint_manifest(t)` 已声明的选中集合内部漏聚合某个组包，或错误替换其中某个组包

建议指标：

- 达到目标精度所需墙钟时间
- 平均等待时间
- 最终精度
- compensation gain / mixed-version 组包补偿收益
- stale reuse gain / 客户端掉线未上传结果的复用收益
- stale-weight stability / 不同 stale 权重方向下的精度、收敛速度与训练波动
- tampering detection rate
- verification overhead

### 8.1 参数空间与消融顺序

协议中涉及的超参数分属三个独立模块：

**Buffer / Checkpoint 参数（调度层）：**

| 参数 | 含义 | 消融 |
|------|------|------|
| `K` | checkpoint 触发阈值 | 固定 `K = 3`，不做消融 |
| `s_max` | 最大陈旧度（过期阈值） | `s_max ∈ {2, 3, 5}` 敏感性测试 |
| `Δ_timeout` | 超时触发时限 | 可做开关对比，不做网格搜索 |
| `policy` | 缓冲选择策略 | `FIFO / Fresh-first / Fairness-aware Fresh-first` |
| `D_fair` | 公平保底阈值 | `D_fair ∈ {1, 2, 3}`，仅 fairness-aware 策略启用 |
| `adaptive s_max` | 是否启用自适应过期阈值 | `off / on` 开关对比；启用时固定 `P_tar, \eta_s, s_min, s_hi` |

**VCM 补偿参数（补偿层）：**

| 参数 | 含义 | 消融 |
|------|------|------|
| `γ_vcm` | Direct-VCM 补偿强度 | `γ_vcm ∈ {0, 0.5, 0.8, 1.0}` |
| `VCM mode` | Direct / First / Second | 仅在扩展对比时考虑，不进入主协议 |
| `λ` | First/Second-order 辅助参数 | 不进入主协议 |

**客户端 stale reuse 参数（客户端层）：**

| 参数 | 含义 | 消融 |
|------|------|------|
| `\rho_{\max}` | stale 结果的最大权重 | `{0.25, 0.5}`，主协议要求 `\rho_{\max}\le 0.5` |
| `\beta_c` | stale 权重随客户端级陈旧度衰减的速度 | `{0.1, 0.3, 0.5}` |
| `reuse rule` | 新旧结果权重方向 | `No Reuse / MAFSA Original / Ours Decayed-Stale` |

其中三种权重规则固定为：

```text
No Reuse:
    w_i^k = w_i^{new,k}

MAFSA Original:
    \alpha_i = exp(-\beta_c E_i)
    w_i^k = \alpha_i w_i^{new,k} + (1-\alpha_i)w_i^{stale,k}

Ours Decayed-Stale:
    \rho_i = \rho_{\max} exp(-\beta_c E_i)
    w_i^k = (1-\rho_i)w_i^{new,k} + \rho_i w_i^{stale,k}
```

该消融用于回答两个独立问题：复用客户端掉线造成的未上传结果是否优于直接丢弃，以及 stale 结果“越旧权重越大”还是“越旧权重越小”更有利于准确率、收敛速度和稳定性。正式主协议采用 `Ours Decayed-Stale`。

建议消融实验顺序：

1. `stale reuse rule`（`No Reuse / MAFSA Original / Ours Decayed-Stale`，必做）
2. `No Compensation`（`γ_vcm = 0`）
3. `Direct-VCM`（`γ_vcm = 1.0`）
4. `\rho_{\max}, \beta_c sensitivity`（客户端 stale reuse 参数）
5. `γ_vcm sensitivity`（`0.5, 0.8, 1.0`）
6. `s_max sensitivity`（`2, 3, 5`）
7. `policy comparison`（`FIFO / Fresh-first / Fairness-aware Fresh-first`）
8. `adaptive s_max`（`off / on`）
9. `Direct vs First vs Second`（工作量不足时可不做）

---

## 9. 明确不写进主协议的内容

为了保证第四章边界清楚，下列内容统一不纳入最终主线：

- 完全异步的“来一个组就更新一次”
- 解掩码前的组间连续非等权陈旧度系数 `beta_g`
- 自适应 `s_max` 不作为主协议必选组件，只作为调度层增强或消融项
- 公平性约束不扩展为复杂的 anti-censorship / 可审计调度协议，只保留简单保底规则
- 除 `Direct-VCM` 外，其余 `VCM` 变体（`First-order / Second-order`）不进入主协议
- `λ` 不进入主协议（仅 `First / Second-order VCM` 扩展对比时需要）
- 组级质量追踪 `m̂_k`
- 双TA或 threshold TA
- 最终/检查点全局审计日志系统
- HGA-FL 风格 bias correction

这样做不是因为这些方向没价值，而是因为：

> 它们会把第四章从“一个能讲圆的协议扩展”重新拖回“大拼盘”。

---

## 10. 参考文献

1. Teng Lei et al. *HVSA: A Lightweight and Verifiable Secure Aggregation Service for Hierarchical Federated Learning*. Manuscript, 2026.
2. Jiaming Xu and Shen Wang. "FedFleet: A Hierarchical Federated Learning Framework for Faster Convergence in Heterogeneous IoT Systems." *IEEE Internet of Things Journal*, 2025.
3. Shiwen Zhang et al. "MAFSA: A Multi-Layer Asynchronous Federated Learning with Staleness-Awareness in Edge Computing." *Engineering Applications of Artificial Intelligence*, 2026.
4. Lam M. Nguyen, Jianyu Wang, Qing Ling, Sabato Marco Siniscalchi, Parikshit Goswami, and Vladimir Matyas. "Federated Learning with Buffered Asynchronous Aggregation." In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2022.
5. Chuyi Chen, Yanchao Zhao, Zhe Zhang, Wenzhong Li, and Jie Wu. "Toward Efficient and Scalable Asynchronous Federated Learning via Stragglers Version Control." *IEEE Transactions on Mobile Computing*, 2026.
