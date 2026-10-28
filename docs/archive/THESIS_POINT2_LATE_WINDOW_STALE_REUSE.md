# 研究点2聚焦版：单TA约束下的组级 Late-Window Stale Reuse 半异步安全聚合

## 1. 最终聚焦方向

研究点2不再以“我也做异步”作为主线，也不再继续扩展双TA口径，而是回到与你研究点1小论文一致的 **单TA HVSA** 叙事，聚焦一个更清晰、也更像递进工作的核心问题：

> 在研究点1的单TA可验证安全聚合框架下，如何让掉线或迟到客户端已经完成的训练结果不被直接丢弃，而能够在不破坏原有掩码流程和组级验证闭环的前提下，被安全复用到后续训练中。

一句话定义可写为：

> 本文在单TA可验证安全聚合框架基础上，提出一种支持迟到训练结果安全复用的组级半异步联邦学习方法。该方法通过提前分组与组内定时同步保持原有掩码与验证闭环，同时为错过当前聚合窗口的客户端设计 late-window 重入机制，使其本地训练结果在重新掩码、重新验证后以组级陈旧更新的形式参与后续全局融合，从而减少掉线导致的训练浪费并提升异构环境下的训练鲁棒性。

从拼接来源上看，这一版可直接定位为安全约束下的 `FedFleet + MAFSA`：

- `FedFleet` 提供“提前分组 + 组内同步 + 组间半异步”的训练组织骨架
- `MAFSA` 提供“迟到/掉线训练结果值得复用”的问题意识
- 你的研究点2则把它们重构到 **单TA、edge-centric verification、参数平均** 的 HVSA 框架中

## 2. 为什么这样聚焦

这个方向最适合作为研究点2，原因有五个：

- 它和研究点1强关联。研究点1解决的是“如何安全聚合并验证”，研究点2自然延伸为“在不破坏这套掩码流程和组级验证机制的前提下，如何处理异构、迟到和掉线”。
- 它的问题意识足够明确。相比泛泛地做“半异步”或“偏置校正”，掉线客户端训练结果被直接丢弃是一个更具体、也更容易论证价值的问题。
- 它不会和研究点1的贡献重叠。研究点1卖点仍是轻量 masking 与 edge-centric verification；研究点2卖点变成 delayed result safe reuse，而不是重复强调“分组隐私保护”或“验证外包”。
- 它和单TA口径更一致。研究点1小论文本身就是单TA系统模型，研究点2沿用同一系统设定，叙事更顺，评审也更容易接受。
- 它的工作量更可控。相比“分组 + 偏置校正 + 检查点审计 + 双TA容灾”这种大拼盘，当前方向更容易做成一章边界清楚、实验可落地的研究内容。

## 3. 口径确认

### 3.1 研究主线

研究点2的主线固定为：

`提前分组 + 组内定时同步安全聚合 + late-window stale reuse + 云端组级半异步融合`

### 3.2 语义口径

本章继续沿用研究点1与现有实验主线的基本语义：

- 上传和聚合的对象始终是 **模型参数向量及其平均结果**
- 不切换成客户端级梯度回灌
- 不切换成增量式异步优化

### 3.3 验证口径

本章主动收缩验证主张：

- 保留的是 **窗口/组级验证**
- 不再主张“每次异步产生的新全局模型在下发时都可沿用研究点1的在线验证语义”

若后续需要更强的全局完整性保证，可将其扩展为最终审计或检查点审计，但这不作为本章主线。

### 3.4 为什么回到单TA

研究点2回到单TA，不是因为双TA一定错误，而是因为在你当前论文结构下，单TA更合适：

- 研究点1小论文本身就是单TA系统模型，前后章节口径一致
- 单TA更容易把 `LHH + 数字签名 + 掩码恢复 + 验证闭环` 讲圆
- 双TA真正带来的核心收益是“降低单点知情信任”，但它同时带来初始化职责拆分困难和 `2-of-2` 可用性下降，不适合在本章继续扩张
- 研究点2的真正创新不在“去单点信任”，而在“迟到训练结果的安全复用”

因此，本章采用：

- **协议层：单逻辑 TA**
- **系统层：如需可用性增强，可采用主备部署**

但不在本章引入 threshold TA 或 share recovery。

## 4. 核心设计思想

### 4.1 保留什么

研究点1中的下面这些内容应完整保留：

- 单TA初始化共享参数与掩码相关材料
- 客户端上传掩码参数
- 边缘侧按组收集在线集合并产生签名证据
- TA 基于在线集合计算窗口掩码和
- 云端恢复组级明文结果
- 边缘侧执行组级验证

也就是说，研究点2不是推翻研究点1，而是在其外层增加“迟到结果的合法复用机制”。

### 4.2 改什么

真正新增的是下面两点：

- 对于错过当前组窗口的客户端，不再简单丢弃其本地训练结果
- 为这些客户端单独设计一个新的 `late window`，使其以新的参与集合重新进入安全聚合流程

本研究点的核心原创点是：

> 不是直接复用旧的 masked upload，而是复用旧的本地训练结果，并在新的窗口中重新定义安全聚合关系。

### 4.3 验证边界怎么收

研究点2必须明确区分两类验证：

- **组级验证**：验证某个固定窗口 `omega` 中恢复出的组级明文结果，是否确实对应这个窗口里的 masked 聚合结果与 TA 掩码和
- **全局在线验证**：验证某次异步融合后云端立即下发的全局模型，是否严格等于所有已接纳组更新按照声明顺序和权重融合后的结果

前者在本章中是可以保住的，因为每个组窗口在封组后都拥有固定参与集合；TA 可以基于同一个集合计算窗口掩码和，边缘也可以围绕这个固定窗口执行哈希一致性检查。

后者则不能直接沿用研究点1的语义。原因在于组间半异步下，任意一个全局模型版本都同时依赖于组更新接纳顺序、陈旧度权重以及是否漏接纳或重复接纳某个组。因此，本章不把“异步全局模型逐次在线验证”作为主 claim，而是把验证主语收缩为“组窗口是否合法”。

## 5. 完整流程

### 5.1 初始化与提前分组

系统根据客户端的训练时延与通信时延构建画像：

`d_i = t_i^train + t_i^comm`

然后按 `d_i` 对客户端排序并划分为多个组：

`U_1, U_2, ..., U_G`

每个组 `U_g` 绑定一个边缘服务器 `e_g`。  
分组目标不是改变安全协议，而是让组内客户端完成时间更接近，从而减少组内同步等待。

### 5.2 组内正常训练与准时窗口聚合

在第 `r` 轮开始时，云端广播当前全局模型 `W^r`。  
组 `g` 内的客户端都基于同一个 `W^r` 开始本地训练，得到本地模型：

`w_i^{g,r}`

客户端随后上传掩码模型：

`wbar_i^{g,r} = w_i^{g,r} + m_i(g,r)`

边缘服务器在截止时间 `tau_g^r` 到达后封组，得到本轮准时参与集合：

`S_g^r subseteq U_g`

随后继续沿用研究点1流程：

- 边缘计算组级掩码和 `Wbar_g^r = sum_{i in S_g^r} wbar_i^{g,r}`
- 边缘计算 `H_masked(g,r) = LHH(Wbar_g^r)` 并签名绑定窗口元数据
- TA 基于同一个 `S_g^r` 计算组级掩码和 `M_g^r = sum_{i in S_g^r} m_i(g,r)`
- 云端恢复组级明文和 `W_g^r = Wbar_g^r - M_g^r`
- 边缘或验证方执行组级验证

通过验证后得到准时组更新：

`Wtilde_g^r = W_g^r / |S_g^r|`

### 5.2.1 组级验证关系

对任意一个已经封存的固定窗口 `omega`，它可以是准时窗口 `(g, r)`，也可以是晚到窗口 `(g, r, h, late)`。记该窗口的实际参与集合为：

`S_omega`

边缘服务器先计算该窗口内的 masked 参数和：

`Wbar_omega = sum_{i in S_omega} wbar_i`

TA 基于同一个 `S_omega` 计算窗口掩码和：

`M_omega = sum_{i in S_omega} m_i`

云端或验证方据此恢复窗口明文和：

`W_omega = Wbar_omega - M_omega`

随后构造三个哈希证据：

- `H_masked(omega) = LHH(Wbar_omega)`
- `H_mask(omega) = LHH(M_omega)`
- `H_plain(omega) = LHH(W_omega)`

组级验证关系定义为：

`H_plain(omega) + H_mask(omega) = H_masked(omega)`

其中 `+` 表示 `LHH` 的同态加法。  
与此同时，边缘服务器还应签名绑定最小窗口元数据，例如：

`(window_id, digest(S_omega), |S_omega|, H_masked(omega))`

只有当签名和哈希关系同时成立时，该窗口对应的组更新才被视为合法并可提交给云端。

需要明确的是，这一步保证的是“**该组窗口解掩码正确**”，而不是“**整个异步全局融合轨迹正确**”。换句话说，组级验证能证明某个窗口结果没有在解掩码阶段被篡改，但并不单独保证云端后续没有改 `lambda`、改顺序、漏组或重复接纳。

### 5.2.2 三个 ES、每个 ES 三个客户端的直观例子

为了避免整章只剩符号，可用一个最小系统说明研究点1到研究点2的递进关系。设系统中有三个边缘服务器：

- `ES1` 管理 `u11, u12, u13`
- `ES2` 管理 `u21, u22, u23`
- `ES3` 管理 `u31, u32, u33`

在第 `r` 轮训练开始时，云端向三个边缘统一广播全局模型 `W^r`。九个客户端都基于同一个 `W^r` 本地训练，并上传掩码参数而不是单步梯度。

假设本轮截止时间统一设为 `12s`，实际完成时间分别为：

- `ES1`：`u11 = 8s`，`u12 = 9s`，`u13 = 8s`
- `ES2`：`u21 = 10s`，`u22 = 11s`，`u23 = 10s`
- `ES3`：`u31 = 9s`，`u32 = 10s`，`u33 = 22s`

则三个准时窗口的实际参与集合分别为：

- `S_1^r = {u11, u12, u13}`
- `S_2^r = {u21, u22, u23}`
- `S_3^r = {u31, u32}`

此时，`ES3` 的准时窗口只会围绕 `{u31, u32}` 形成自己的组级证据。记：

`Wbar_(3,r) = wbar_31^(3,r) + wbar_32^(3,r)`

TA 基于同一个集合 `{u31, u32}` 计算：

`M_(3,r) = m_31(3,r) + m_32(3,r)`

恢复出的组明文和满足：

`W_(3,r) = Wbar_(3,r) - M_(3,r)`

对应的组级验证关系为：

`H_plain(3,r) + H_mask(3,r) = H_masked(3,r)`

如果这一步通过，则 `ES3` 的准时组更新 `Wtilde_3^r` 可以作为一个合法组结果进入云端；与此同时，`u33` 因为错过截止时间而不会被硬塞回当前窗口。

当 `u33` 在 `22s` 恢复在线后，研究点2进一步允许它进入新的晚到窗口：

`omega_(3,r,1)^late`

此时该窗口的参与集合可写为：

`S_(3,r,1)^late = {u33}`

`u33` 不是重放旧的 masked upload，而是基于新的窗口标识重新掩码上传。TA 也基于这个新的固定集合重新计算窗口掩码和，再执行一次完全相同的组级验证关系：

`H_plain((3,r,1)^late) + H_mask((3,r,1)^late) = H_masked((3,r,1)^late)`

如果该式通过，就得到一个合法的晚到组更新 `Wtilde_(3,r,1)^late`。随后，云端可以按组到达顺序半异步接纳 `Wtilde_1^r`、`Wtilde_2^r`、`Wtilde_3^r` 以及 `Wtilde_(3,r,1)^late`。这个例子也说明，本章能稳定保住的是“每个组窗口自身的验证正确性”，而不是“每一次异步产生的全局模型都在下发时立即可验证”。

### 5.3 迟到客户端的处理

若客户端 `u_i` 未能在 `tau_g^r` 前进入 `S_g^r`，则它不参与本轮准时窗口聚合。  
但与传统做法不同，它的本地训练结果不直接作废，而是进入“待复用状态”，本地缓存以下信息：

- 快照标识 `snapshot_id = (u_i, g, r)`
- 所属组 `g`
- 基模型轮次 `r`
- 本地训练结果 `w_i^{g,r}`
- 生命周期状态 `status in {fresh, locked, consumed, expired}`

注意，这里缓存的是本地训练结果，而不是旧的掩码上传结果。旧窗口已经关闭，原有参与集合和掩码关系也已经结束，因此不能把旧的 masked update 直接塞回去。

其中各状态的含义如下：

- `fresh`：尚未被复用，可进入后续 late window
- `locked`：已被某个待执行的 late window 选中，暂时不能再次分配
- `consumed`：已成功参与一次合法晚到窗口并被云端接纳，不能再次复用
- `expired`：超过允许陈旧度上界，或已被同一客户端在更新基模型下的新训练结果覆盖

这样定义的目的是保证每一份迟到训练结果最多只被系统接纳一次，避免重复计入。

### 5.4 Late Window 重入机制

当迟到客户端恢复在线后，边缘服务器为基于同一全局基模型 `W^r` 完成训练但错过原窗口、且当前状态仍为 `fresh` 的客户端建立一个迟到池：

`B_g^r`

当 `B_g^r` 达到阈值，或等待时间达到上限时，边缘服务器创建新的晚到窗口：

`omega_{g,r,h}^{late}`

其中 `h` 表示同一组、同一基轮次下第 `h` 个晚到窗口。  
该窗口对应新的固定参与集合：

`S_{g,r,h}^{late}`

进入该集合前，边缘服务器需检查两项条件：

- 对应快照的 `status = fresh`
- 当前陈旧度未超过上界 `s <= s_max`

满足条件的快照先被标记为 `locked`，避免并发地被多个 late window 重复选中。

这时，这些客户端不是提交旧上传，而是基于新的窗口标识重新掩码：

`wbar_i^{g,r,h,late} = w_i^{g,r} + m_i(g,r,h,late)`

随后再次完整执行研究点1的组级安全闭环：

- 边缘汇总晚到窗口掩码和
- TA 按 `S_{g,r,h}^{late}` 计算新的窗口掩码和
- 云端恢复晚到窗口组级明文
- 边缘执行组级验证

验证通过后，得到一个合法的晚到组更新：

`Wtilde_{g,r,h}^{late} = W_{g,r,h}^{late} / |S_{g,r,h}^{late}|`

随后，窗口内所有对应快照被标记为 `consumed`。  
若窗口执行失败，则这些快照恢复为 `fresh`；若在等待期间已经超过 `s_max`，则直接标记为 `expired`。

### 5.4.1 最小 anti-replay 元数据

为了防止同一份迟到训练结果被重复提交或多次接纳，系统至少维护一份轻量复用账本，而不要求本章实现完整 checkpoint audit。

建议账本条目包含：

- `snapshot_id`
- `window_id = (g, r, h, late)`
- `base_round = r`
- `status`
- `accepted_flag`
- `accepted_version`

其中核心约束为：

- 一个 `snapshot_id` 最多只能对应一个 `accepted_flag = 1` 的晚到窗口
- 已经 `consumed` 的快照不能重新进入任何窗口
- 已经 `expired` 的快照不能再参与系统

这份轻量账本的作用是防止直接重放与重复计入，而不是承担完整全局模型审计职责。

### 5.4.2 自适应陈旧度上限 `s_max`

上文默认将 `s_max` 视为一个固定超参数，用于控制迟到训练结果最多允许陈旧到什么程度。这种写法最简单，也最容易实现；但在真实异构环境下，固定 `s_max` 往往会出现两种问题：

- 当系统整体较快、掉线较少时，固定 `s_max` 可能过于宽松，接纳了没有必要参与的高陈旧更新
- 当系统整体较慢、掉线较多时，固定 `s_max` 可能过于严格，导致大量本可复用的迟到结果被直接标记为 `expired`

因此，可进一步引入一个受动态边界控制思想启发的轻量增强机制：  
不直接照搬 DVAFL 中 DWP 的 ETA 窗口公式，而是将其“根据上一轮观测反馈动态调整接纳边界”的思想迁移为 `s_max` 的自适应更新。

记第 `r` 轮或第 `r` 个检查点统计到的迟到结果总数为：

`N_late^r`

其中，仅将那些**因陈旧度超过当前上限 `s_max^r` 而被判为 `expired`** 的结果计为：

`N_expired^r`

据此定义复用压力：

`P_r = N_expired^r / max(1, N_late^r)`

其中：

- `P_r` 越大，说明当前 `s_max^r` 过紧，系统丢弃了太多迟到结果
- `P_r` 越小，说明当前 `s_max^r` 偏松，系统可以在保证复用率的同时适当收紧接纳范围

设目标丢弃比例为 `P_target`，调整步长为 `lambda_s`，则可将下一轮上限更新为：

`s_max^{r+1} = clip(round(s_max^r + lambda_s * (P_r - P_target)), s_min, s_cap)`

其中：

- `clip(·)` 用于把 `s_max` 限制在可接受区间 `[s_min, s_cap]`
- `round(·)` 用于保持 `s_max` 为整数轮数
- `s_min` 和 `s_cap` 分别是最小、最大允许陈旧度

为避免样本过少时的剧烈抖动，还可加入一个最小统计门槛 `n_min`：

- 若 `N_late^r < n_min`，则令 `s_max^{r+1} = s_max^r`

需要强调的是，自适应 `s_max` 不应被包装为本章新的独立创新点，而更适合作为 `late-window stale reuse` 主协议之上的**轻量增强机制**。它既不改变单TA掩码流程，也不改变组级验证关系，只是动态调整“什么程度的迟到结果仍值得接纳”这一边界。

### 5.5 云端组级半异步融合

云端统一接收两类已经通过组级验证的组更新：

- 准时组平均模型 `Wtilde_g^r`
- 晚到组平均模型 `Wtilde_{g,r,h}^{late}`

晚到组更新虽然合法，但其基模型较旧，因此需要定义陈旧度：

`s_{g,r,h}^{late} = v_cur - r`

其中 `v_cur` 为云端当前全局版本号。  
随后云端对该更新施加轻量陈旧度权重：

`phi(s) = 1 / (1 + beta s)`

最终以组级半异步方式更新全局模型。  
对准时组更新，可直接采用普通参数平均式融合：

`W^{v+1} = (1 - lambda_0) W^v + lambda_0 Wtilde_g^r`

对晚到组更新，则采用带陈旧度降权的参数平均式融合：

`lambda_{late} = lambda_0 * phi(s)`

`W^{v+1} = (1 - lambda_{late}) W^v + lambda_{late} Wtilde_{g,r,h}^{late}`

其中：

- 若 `s > s_max`，则该晚到组更新不再被接纳
- 对准时组更新可视为 `s = 0`

若系统启用自适应 `s_max`，则可在每轮融合结束后或每个检查点结束后，根据上一阶段的 `P_r` 更新下一阶段所使用的陈旧度上限。这样，晚到结果的接纳边界就不再是固定常数，而能够随系统状态自动收紧或放宽。

需要再次说明的是，这里的“通过验证”仅表示相应组窗口已经完成固定参与集合下的合法性验证，并不等价于“每一次异步融合后生成的新全局模型都已完成在线验证”。

## 6. 本研究点的真正创新点

研究点2的创新点不应表述为“做了异步”，而应固定成下面三条。

### 创新点1：单TA约束下的组级半异步扩展

在不破坏研究点1单TA掩码流程和组级验证闭环的前提下，将原本整轮同步的训练组织方式扩展为“组内定时同步、组间半异步”的分层结构。

### 创新点2：支持迟到训练结果安全复用的 Late-Window 机制

针对掉线或迟到客户端训练结果在传统方案中被直接丢弃的问题，提出 late-window 重入机制，使迟到客户端可在新的固定参与集合下重新掩码、重新验证，并以合法组平均模型形式参与后续全局融合。

### 创新点3：面向晚到组更新的组级陈旧度融合

不直接回灌旧窗口，也不在客户端级复用历史 masked update，而是在组级安全闭环完成后再对晚到组更新执行陈旧度感知融合，从而兼顾安全性、兼容性和训练有效性。

若篇幅允许，可进一步加入自适应 `s_max` 作为轻量增强，但不建议将其上升为第四条核心创新点。

## 7. 与相关方案的区别

### 7.1 与研究点1的区别

研究点1关注的是“如何安全聚合并验证当前在线客户端更新”。  
研究点2进一步关注“那些没能赶上当前窗口但已经完成训练的客户端，是否还能被安全地重新纳入系统”。

更细一点说，研究点1隐含的是整轮同步下较强的全局结果验证语义；而研究点2为了保留掩码并引入组间半异步，主动把验证粒度收缩到固定窗口上的组级验证，不再把“异步全局模型逐次在线验证”作为本章必须满足的条件。

### 7.2 与 FedFleet 的关系

`FedFleet` 给本研究点最直接的启发是：

- 提前分组
- 组内同步
- 组间半异步

本研究点沿用了这种训练组织骨架，但进一步回答了一个 `FedFleet` 没有处理的问题：

- 如果客户端已经完成训练，却因为迟到或掉线错过了当前窗口，它的结果能否被合法复用？

因此，本方案与 `FedFleet` 的关系应表述为：

- 借其分组半异步骨架
- 不停留在“只减少等待”
- 向前推进到“迟到结果的安全重入与组级复用”

### 7.3 与 MAFSA 的区别

`MAFSA` 的启发在于“不要直接丢掉掉线客户端的训练结果”。  
但本研究点不直接采用客户端级旧更新回灌，而是改造成更适配 HVSA 单TA验证框架的组级 `late-window stale reuse`。

因此，本方案与 `MAFSA` 的关系应表述为：

- 借用了“迟到结果值得复用”的问题意识
- 没有照搬其客户端级 stale update 复用方式
- 在单TA、组级验证、参数平均语义下重构为新的组级复用机制

### 7.4 与 GPVO-FL 的区别

`GPVO-FL` 与本文在体系结构上确实相似，都是云-边-端协同环境中的分组安全聚合与验证方案。  
但两者主问题不同：

- `GPVO-FL` 的重点是 grouped privacy-preserving 与 verification outsourcing
- 本研究点2的重点是 **迟到训练结果如何在已有安全聚合框架下被安全复用**

因此，本文不能把“分组隐私保护 + 边缘验证”再当作研究点2主贡献，否则容易与 `GPVO-FL` 重叠；研究点2真正需要强调的是 `late-window stale reuse` 与组级半异步复用机制。

### 7.5 与 SAEV-FL 的区别

`SAEV-FL` 的一个重要特点是用户本地持有 PRG 种子，TA 不掌握完整掩码，因此 TA 的隐私知情性更低。  
但它的全局解掩码与验证逻辑建立在用户侧解密和用户侧验证基础上。

本研究点不采用 SAEV-FL 的掩码设计，原因不是其思路不好，而是它与本文的主链路不兼容：

- 本文需要 TA 基于在线集合生成窗口掩码和，供云端恢复组级明文
- 本文需要 edge-centric verification，而不是把解密和验证职责重新移回用户侧

因此，SAEV-FL 更适合作为“TA 知识分离设计”的对照参考，而不是本章直接采用的协议骨架。

## 8. 建议题目

下面几个题目都可以，其中优先推荐前两个。

- 基于单TA可验证安全聚合的组级迟到更新复用半异步联邦学习方法
- 单TA约束下支持迟到训练结果安全复用的组级半异步联邦学习方法
- 面向掉线客户端训练结果复用的单TA可验证组级半异步聚合机制

## 9. 论文里最值得强调的一句话

研究点2最该强调的不是“我们也做了异步”，而是下面这句话：

> 本文不是简单放宽同步约束，而是在单TA可验证安全聚合框架下，提出一种不破坏原有掩码流程与组级验证闭环的 late-window stale reuse 机制，实现掉线或迟到客户端训练结果的安全复用。

## 10. 实验建议

围绕这个聚焦方向，实验最好保持克制，重点验证“复用是否真的有价值”。

建议至少做以下四组：

1. `Sync-HVSA`
2. `Grouped-HVSA + Discard-Late`
3. `Grouped-HVSA + Late-Window Reuse`
4. `Grouped-HVSA + Late-Window Reuse + Staleness Weighting`

若实验资源允许，建议额外补一个轻量消融：

5. `Grouped-HVSA + Late-Window Reuse + Adaptive s_max`

建议重点报告这些指标：

- 最终测试精度
- 达到目标精度所需墙钟时间
- 被丢弃训练结果比例
- 迟到结果复用率
- 掉线率升高时的鲁棒性
- 验证开销与通信开销

若加入自适应 `s_max` 消融，建议再补充：

- `expired` 比例随轮次变化
- `s_max` 随轮次变化曲线

## 11. 范围控制

为保证研究点2边界清楚，建议明确不把下面内容作为本章主线：

- 完整的全局 checkpoint audit
- 完整 HGA-FL bias correction
- 子模型分配
- 客户端级直接 stale update 回灌
- 异步全局模型的逐次在线验证
- 过重的资源感知训练预算设计
- threshold TA 或 share recovery

最稳妥的容量是：

- 1 个核心问题：迟到结果怎么合法复用
- 1 条主机制：late-window stale reuse
- 1 个主辅助机制：组级陈旧度加权
- 1 个可选轻量增强：自适应 `s_max`

这已经足够支撑研究点2成为一章完整内容。

## 12. 参考文献

以下文献可作为当前研究点2主线的核心参考文献。

1. Teng Lei, Wenjia Niu, Wen Zhang, Li Duan, Dedong Zhang, and Lin Li. *HVSA: A Lightweight and Verifiable Secure Aggregation Service for Hierarchical Federated Learning*. Manuscript, 2026.
2. Jiaming Xu and Shen Wang. "FedFleet: A Hierarchical Federated Learning Framework for Faster Convergence in Heterogeneous IoT Systems." *IEEE Internet of Things Journal*, vol. 12, no. 17, 2025, doi:10.1109/JIOT.2025.3579917.
3. Shiwen Zhang, Shuang Chen, Yujia Zhang, Wei Liang, Kuanching Li, Ling-Huey Li, and Keqin Li. "MAFSA: A Multi-Layer Asynchronous Federated Learning with Staleness-Awareness in Edge Computing." *Engineering Applications of Artificial Intelligence*, vol. 172, 114369, 2026, doi:10.1016/j.engappai.2026.114369.
4. Wei Xie, Runqun Xiong, and Junzhou Luo. "Hierarchical Global Asynchronous Federated Learning Across Multi-Center." In *Proceedings of Machine Learning Research*, vol. 260, *Asian Conference on Machine Learning (ACML)*, 2024.
5. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." arXiv:1602.05629, 2016.
6. Shiwen Zhang, Feixiang Ren, Wei Liang, Kuanching Li, and Nam Ling. "GPVO-FL: Grouped Privacy-Preserving and Verification-Outsourced Federated Learning in Cloud-Edge Collaborative Environment." *IEEE Transactions on Network and Service Management*, vol. 22, no. 5, 2025, doi:10.1109/TNSM.2025.3592122.
7. Shiwen Zhang, Feixiang Ren, Wei Liang, Kuanching Li, and Nam Ling. "SAEV-FL: Lightweight Secure Aggregation and Efficient Verification Scheme for Federated Learning in Cloud-Edge Collaborative Environment." *IEEE Transactions on Intelligent Vehicles*, vol. 11, no. 1, 2026, doi:10.1109/TIV.2025.3599909.

若论文篇幅允许，可再补充以下异步背景文献：

8. Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. "Asynchronous Federated Optimization." arXiv:1903.03934, 2019.
9. Lam M. Nguyen, Jianyu Wang, Qing Ling, Sabato Marco Siniscalchi, Parikshit Goswami, and Vladimir Matyáš. "Federated Learning with Buffered Asynchronous Aggregation." In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2022.
10. Ruirui Zhang, Xingze Wu, Yifei Zou, Zhenzhen Xie, Peng Li, Xiuzhen Cheng, Falko Dressler, and Dongxiao Yu. "Fed-RAA: Resource-Adaptive Asynchronous Federated Edge Learning With Theoretical Guarantee." *IEEE Transactions on Mobile Computing*, vol. 25, no. 6, 2026, doi:10.1109/TMC.2025.3649551.
