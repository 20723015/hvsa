# 研究点一对比实验：协议映射与计量口径（T0 初稿，复核中）

2026-10-20更新：用户授权的GPVO/VLMRS明确实例化已完成四点重复成本比较，156/156正确且完整验证通过（36预热、120正式），三张最终PNG目视核查通过。见[成本报告](../../docs/POINT1_DECIDED_MINIMUM_COST_COMPARISON_20261020.md)。旧兼容变体和条件诊断映射保持历史边界；新成本范围见§13，不宣称原文安全等价。

2026-10-19授权更新：GPVO/VLMRS的本项目实例化由复现者选定，见`configs/baseline_decisions_20261019.json`及`docs/POINT1_BASELINE_DECISIONS_20261019.md`。该范围不再等待唯一作者解释；新运行仍保留原文安全等价未证明的标记。新增GPVO完整计量路径和同输入试跑见§12，历史映射和归档不追改。

初稿日期：2026-10-17；复核更新：2026-10-18。对应计划：[POINT1_COMPARATIVE_EXPERIMENT_PLAN_20261017.md](../../docs/POINT1_COMPARATIVE_EXPERIMENT_PLAN_20261017.md)。
本文件记录当前实现，而非已经通过安全等价审查的正式基线。GPVO使用Z*_{N²}哈希的兼容变体，协调修改结果/承诺仍被接受；每轮初始化、哈希/承诺接口和安全性需复核后才能冻结。VLMRS及SAEV已有显式实例化条件下的真实回合，尚未通过原文等价与认证门槛。HVSA已补完整序列化封装、互斥计时、同步证据名单及输出域检查。
实现位于 `experiments/point1_compare/`，保留 `baselines/` 与既有实验记录；根目录`edge.py`补研究点一同步证据集合完整/去重及公开维度一维signed-int64输出检查。最新执行结果见[补全记录](../../docs/POINT1_IMPLEMENTATION_SUPPLEMENT_20261018.md)，历史范围见[执行报告](../../docs/POINT1_COMPARISON_E1_E5_20261018.md)。

## 0. 依据与范围

- GPVO-FL 原文：`E:/Essay/DFL/GPVO-FL_...pdf`（TNSM 2025，DOI 10.1109/TNSM.2025.3592122），本次核对了 PDF 第 3–6 页（预备知识、系统/威胁模型）与第 9–14 页（Fig.3 详尽流程、算法 2–5、验证正确性定理、安全分析、性能评估）。
- HVSA 本地协议：`CODEX.md`、`user.py`（单 TA 掩码上传）、`ta.py`（在线集合总掩码 + H_mask）、`edge.py`（组聚合、`meta_k` 签名、LHH 同态验证）、`cloud.py`（去掩码，不产出规范模型）、`main.py`（回合编排与确定性重构）。
- 计量口径：计划 §4.3（分角色、分事件、真实序列化字节、实测计算与建模网络分开）。

## 1. HVSA（研究点一真实单 TA 路径）

适配器 `adapters/hvsa_point1.py` 直接实例化根目录真实实体类（`TrustedAuthority`/`EdgeServer`/`CloudServer`/`User`），`rule_engine=None`，走FULL证据路径；不调用研究点二补偿、权重或checkpoint。纯协议实验输入是已编码的合成整数向量或真实训练快照，不在协议计时内重复量化。E7真实训练按 `user.py` 的float32乘10^6再截断int64规则生成快照。所有方案成本入口接收同一编码；掩码复用 `User._generate_ta_mask_int`，即 `PRG.mask_int(s_i,u_i,t,d)`。

| 论文/协议步骤 | 实现函数 | 测试用例 |
|---|---|---|
| 注册：用户与 TA 协商唯一种子 s_i（X25519+HKDF） | `User.register_with_ta`（user.py:57）+ `TrustedAuthority.register_user`（ta.py:315） | `test_hvsa_round_matches_reference`（种子一致性由后续去掩码正确性间接验证） |
| 客户端：量化 `×1e6→int64`，掩码 `PRG(s_i,u_i,t)`，上传单个掩码向量 | 适配器 `user_mask_upload` 阶段（`User._generate_ta_mask_int` + pickle 消息 → `EdgeServer.receive_masked_gradient`，edge.py:227） | 同上；消息字节数 = 实际 pickle 长度 |
| 边缘：预聚合 `W̄_k=Σx̃_i`；`H_k=LHH.Hash(W̄_k)`；签名 `meta_k=(edge_id,round,base_hash,digest(U_online),n_k,S_k,H_k)`；上报在线名单给 TA、转发掩码和与证据给云 | `EdgeServer.trigger_aggregation_and_forward`（edge.py:245–315，`S_k=n_k` 等权对齐 edge.py:265） | E1 正确回合验证通过；E5 篡改/重放被拒 |
| TA：`M_global=Σ_{u∈U_online}PRG(s_u,u,t)`；`H_mask=LHH.Hash(M_global)`；发云 | `TrustedAuthority.trigger_global_mask_computation`（ta.py:381–405） | 同上 |
| 云：`W_sum^raw=Σ_k W̄_k − M_global`；不产出规范模型 | `CloudServer.trigger_final_unmasking_and_averaging`（cloud.py:127–161） | E1 原始整数和逐维等于明文参考 |
| 边缘验证：先拒绝越域/错误形状结果，检查证据恰好覆盖同步注册边缘名单；重建并验签 `meta_k`；验证LHH等式；通过后确定性重构模型 | `EdgeServer.verify_global_result`；适配器中的模型重构 | `test_hvsa_coordinated_omission_and_duplication_rejected`、`test_hvsa_rejects_noncanonical_output_domain`；当前18类E5共1980次拒绝 |

## 2. GPVO-FL算法1–5的当前兼容变体

适配器 `adapters/gpvo_full.py` 实现了各阶段的代数回放，但其哈希实例化、生成元分布及跨轮密钥生命周期尚未证明等同原论文。它不基于 `baselines/gpvo_crypto.py`（历史掩码层基准，见§2.3）。

### 2.1 逐算法映射

| 原文步骤（PDF 页码） | 实现函数 | 测试用例 |
|---|---|---|
| Alg.1/Round1 初始化：TA 生成 DH/Com/LHF 公共参数、每边缘承诺密钥 τ_k、**每用户 DH 密钥对 (sk_i,pk_i)** 并下发 | `com_param_gen`（§III.C：`{N,g,p,q,e}`）+ `GPVOSession.__init__` | `test_gpvo_round_matches_reference`；`test_gpvo_commitment_homomorphism` |
| Alg.2/Round2 掩码：`s_{i,j}=DH.Agree(sk_i,pk_j)`（j∈{i−1,i+1,TA}，环内同组，i=1 时 i−1=m_k，i=m_k 时 i+1=1）；`x̃_i=x_i+ξ_{i,TA}−ξ_{i,i−1}+ξ_{i,i+1}`，ξ=PRG(s) | `gpvo_user_mask`（成对掩码两端用同一对称 pair 标签派生同一流，先经 `test_gpvo_pairwise_mask_cancellation` 验证抵消） | 同上 |
| Alg.3/Round3 预聚合：掉线补偿（在线邻居披露与掉线用户的成对掩码，边缘抵消残留）；`x_agg^k=Σ_online x̃_i+comp`；`h_k=LHF.Hash(x_agg^k)`；`c_k=Com.Commit(h_k,τ_k)`；上报 U_online 给 TA、发 {x_agg^k,c_k} 给云 | `run_gpvo_round` 的 edge 阶段（补偿符号见 §2.2） | `test_gpvo_dropout_compensation`（含相邻掉线、m_k=2 退化情形） |
| Round3 TA：`ξ=Σ_{u∈∪U_online}ξ_{u,TA}`；`h_TA=LHF.Hash(ξ)`；`r=Σ_k τ_k`；发 ξ 给云、{h_TA,r} 给边缘 | `run_gpvo_round` 的 TA 阶段 | 同上 |
| Alg.4/Round4 云聚合：`x_agg=Σ_k x_agg^k−ξ`；`c_agg=Com.Aggregate({c_k})=∏_k c_k mod N²`；广播 {x_agg,c_agg} | `run_gpvo_round` 的 cloud 阶段 | 同上 |
| Alg.5/Round5 边缘验证：`h_agg=LHF.Hash(x_agg)`；`h=h_TA·h_agg`；`Com.Decommit(c_agg,h,r)=1` 则通过并广播 | `run_gpvo_round` 的 verify 阶段（当前整数域兼容实现） | E5小维度及协调修改诊断；不代表完整性验收通过 |

### 2.2 解释决策与复核修正

1. **Round 2 无秘密分享**：原文 Fig.3 Round 2 与算法 2 只含"收钥→3×DH.Agree→3×PRG→密文→发送"，无任何 Shamir/秘密分享步骤（Shamir 仅出现在相关工作 §II 引用 [20]）。`baselines/gpvo_crypto.py:50–51` 的 `os.urandom(32)+share(...)` 不计入 GPVO 成本，也不进入本适配器。
2. **掉线补偿符号**：原文 Fig.3 Round 3 写 `x_agg^k=Σ_online x̃_i+Σ_{u∈dropout}(ξ_{i,i−1}−ξ_{i,i+1})`，而算法 2 中 `ξ_{i,i−1}` 在密文里是**减**项、`ξ_{i,i+1}` 是**加**项，且 DH 共享密钥对称（s_{i,j}=s_{j,i}）。以 3 用户环推演：用户 2 掉线后残留为 `+PRG(s12)−PRG(s23)`，需补偿 `−PRG(s12)+PRG(s23)`，即原文公式的相反符号（原文该式按字面执行会把残留翻倍，与定理 1 矛盾）。实现取抵消正确形式：`comp_i=−PRG(s_{i−1,i})·[i−1 在线] + PRG(s_{i,i+1})·[i+1 在线]`，由 E1 明文参考逐维一致与 E4 定向掉线案例裁决。披露方向：边缘向掉线用户的**在线邻居**请求其持有的成对掩码值。
3. **承诺及生成元**：当前采用 `Com.Commit(m,τ)=m^e·g^τ mod N²`，原文§III.C确有`e>N²`的表述。但“同态求和/乘积成立”不等于承诺作为聚合认证安全。当前`g`从Z*_{N²}抽取，原文描述为Z*_N中的最大阶元素；分布和安全条件未确认等价，不标作无偏的正式复现。
4. **跨轮密钥生命周期**：原文省略PRG轮次参数，不能推出密钥长期不变。PDF第14页§VIII明示TA在每个训练轮次分发公参和密钥对。当前固定令牌和`GPVOSession`跨轮复用只是实现选择，其重放接受不能归因于原论文。新增新初始化回归测试记录：每轮换DH密钥与承诺随机量后，旧结果/承诺被新轮辅助信息拒绝。
5. **LHF实例化**：当前为 `h(m)=∏g_i^{m_i} mod N²`，与原文实验设置中的P-256不同。普通EC点序列化通常不保留整数乘法同态，说明原文接口需要澄清；不能由此宣布任意EC实现必然失败，也不能将替换后的整数哈希成本等同原论文成本。本变体未给出碰撞抗性或协议安全等价归约，RSA模数3072位本身不能补上这一证明。
6. **撤销旧编码说明**：当前实现根本不生成GPVO的EC哈希点，原初稿“直接按(x‖y)编码即可”的第6条与第5条冲突，已撤销。P-256哈希与RSA承诺的接口属于待确认项。
7. **群内环与 m_k**：环在每组的注册用户上（u_1..u_{m_k}）；m_k=1 退化（i−1=i+1=i）不支持，输入生成保证每组 ≥2 用户；m_k=2 时 i−1=i+1 为同一邻居，两掩码项相消，退化为单 TA 掩码，协议仍正确（E1 覆盖）。

8. **协调修改用例**：对当前变体，公开计算`x'=x+δ`、`c'=c·LHF(δ)^e mod N²`，诚实验证入口仍接受。构造不需要TA辅助信息或私钥；原因是当前验证等式没有将云提供的聚合承诺锚定到真实输入证据。这只是兼容变体的已复现限制，未经原文完整接口和身份/会话语义核对，不作为论文漏洞声明。

### 2.3 与历史基准的关系

`baselines/gpvo_crypto.py`、`baselines/hvsa_crypto.py`、`baselines/vlmrs_crypto.py` 保留为历史掩码层基准（`exp_overhead_crypto.py` 引用），本适配器不复用、不修改。已知历史问题（计划 §3）在本适配器中的对应处理：无额外 Shamir（§2.2-1）；无按邻居数累加 RTT（本目录只记实测计算与真实序列化字节，网络等待若引入则单列"模型估算"并保留依赖图）；`KA.gen()` 返回 `(pub, priv)`，适配器统一 `pub, priv = KA.gen()` 并在测试中断言 `KA.agree(priv_a, pub_b) == KA.agree(priv_b, pub_a)`。

## 3. 算术域与安全参数（两方案对齐项）

| 项 | 取值 | 依据 |
|---|---|---|
| 客户端量化 | `× QUANTIZE_SCALE=10^6 → int64`，等权平均 | user.py:84、计划 §4.2-2 |
| 掩码域 | int64，PRG 掩码 ∈ [−10^9, 10^9)（拒绝采样） | utils.PRG |
| 聚合算术 | 原始整数和逐维一致（np.int64；参考值用 Python int 累加防溢出核对）；平均/反量化误差相对同编码明文参考单列 | 计划 §E1 |
| 密钥协商 | X25519 + HKDF-SHA256（~128-bit EC） | utils.KA；原文 §VIII.B 用 EC-DH |
| 线性同态哈希 | HVSA：P-256；GPVO当前变体：Z*_{N²}乘法LHF，与原文P-256实例化未对齐 | 不据此声称同安全等级/性能等价 |
| 签名（HVSA） | Ed25519 | utils.SIG |
| 承诺（GPVO） | RSA型模数3072位、`c=m^e·g^τ mod N²`、`e>N²`；参数分布及认证锚定待复核 | 3072位只记录模数，不等于已证明整个兼容变体达到128位安全 |
| LHH 生成元缓存 | `lru_cache(maxsize=4096)`：d>4096 时每次哈希部分重算，不假设全缓存 | utils.LHH:70；计划 §E2 |

## 4. 信任与角色差异（表1 依据）

| 维度 | GPVO-FL（原文 §IV.B） | HVSA（CODEX.md 威胁模型） |
|---|---|---|
| 云 | 恶意/懒惰（可篡改、可跳步） | 恶意（可篡改/丢弃/重放），其产出必须被验证 |
| 边缘 | 诚实但好奇；原文声称云可与至多 M−1 边缘、N−1 用户合谋 | 可与云共享视图；隐私按实际发布聚合中 `N−c≥2` 判断；不假设全部边缘共同作恶时仍有完整性 |
| 用户 | 诚实但好奇 | 同左 |
| TA | 完全可信、不合谋 | 完全可信、不合谋、可用性为工程假设 |
| 验证证据 | 边缘组哈希 + 承诺（云端可聚合为单一 c_agg）；TA 发 {h_TA, r}，无逐边缘签名 | 边缘逐组 `meta_k` Ed25519 签名（绑定轮次/基模型/在线摘要/计数/H_k）+ LHH 同态等式 |
| 掉线 | 未上传：在线邻居披露成对掩码、边缘补偿残留（§2.2-2）；已上传后离线：贡献照常计入（成对掩码已对消） | 在线名单驱动：TA 只为实际在线者出总掩码；已上传者照常计入 |
| 掩码结构 | 1×TA + 2×环邻居；固定密钥是当前变体设置，原文每轮初始化需另评估 | 1×TA种子派生，带轮次令牌 |

E5 只针对各协议实际承诺/签名的字段下结论：GPVO 无 `meta_k` 类签名字段，"篡改聚合上下文"用例对其标注"不适用（原文无该字段）"，不因测试器强加字段而宣布其不安全（计划 §E5）。

## 5. 初始化与复用规则（E3 依据）

| 项 | HVSA | GPVO-FL |
|---|---|---|
| 当前初始化 | 用户/TA密钥、种子、LHH公参、边缘签名密钥 | 当前会话建立Com/LHF参数、用户DH和边缘随机量；不代表原文允许其跨训练轮长期复用 |
| 每轮工作 | PRG带新轮次，边缘哈希+签名，TA总掩码+哈希 | 当前3×DH+3×PRG及证据/验证；正式每轮密钥更新和分发成本待加入 |
| 复用规则 | LHH参数、签名密钥、种子 | 每个参数逐项核对原文，不能统称“全部跨轮可复用” |
| 多轮计时拆分 | 首次初始化 / 首轮 / 稳定轮 / T 轮累计 / 每轮摊销，同左 | 同左 |

## 6. 计量与输出契约（T1）

- 适配器统一入口：`run_round(session, vectors, online, round_t, ...)`，返回 `RoundResult{completed, raw_sum, verified, phases[{phase,role,elapsed_ms}], messages[{link,direction,phase,payload_kind,bytes}], failure_reason}`。
- 原始观测行字段（raw.jsonl）：`scheme, run_id, seed, group, repeat, N_selected, N_uploaded, M, d, round, phase, role, elapsed_ms, sent_bytes, success, verified, error_reason`；掉线扫描另记 `N_registered`（E4 时启用）。
- 通信字节：新HVSA适配器补齐云→边缘完整证据、TA→边缘辅助、逐客户端广播副本和初始化封装。历史运行只有部分消息，不能回填或视为全网真实载荷。新GPVO尚未与HVSA计量边界对齐。
- 计时口径：单进程串行回放的墙钟时间统称"本地协议回放耗时"；不含任何 RTT/带宽建模。E2 起如加并行调度另称"模型估算的轮次时延"。
- `manifest.json`：运行命令、配置、种子、平台/Python/库版本、论文版本（GPVO TNSM 2025 DOI；HVSA chapter03/CODEX.md）、git 提交号、量化与域参数、RSA 参数与实例化决策指针（§2.2）、并发方式、计时口径、验证开关。
- 2026-10-18复核后的新运行另存完整配置、各源码SHA-256、变体标识和未满足项；不向旧manifest回填运行时并未采集的源码摘要。旧契约检查仅核对落盘内部一致性，不验证历史源码身份、协议安全或完整通信。
- Schema 2在运行开始时归档源码；完成时校验五项输出摘要、逐次攻击/诚实上下文、覆盖数量、原始计时及发送者/接收者。汇总计数一致不等于协议安全通过；包含错误接受的诊断也可通过“记录完整性”检查。
- `run_performance.py`仅执行HVSA及其控制组，支持E2/E3/E4/E6；三路掩码控制组不等同GPVO，关闭验证记录`verified=null`。目前入口已实现与小规模试跑通过，不代表正式矩阵全部完成。

## 7. VLMRS：实际组件与原文算法映射

| 原文步骤 | 当前实际执行位置 | 已有证据与未解决项 |
|---|---|---|
| 算法1：两组DH密钥、公布公钥 | `vlmrs_fullgraph.VLMRSFullGraphProtocol.__init__` | 真实X25519；全图ε=1、全局共享验证密钥是显式选择，非已冻结作者参数 |
| 算法1：分享β_j及NSK_j，逐接收者AE加密并经服务器转发 | `sharing.share_secret`；`vlmrs_fullgraph.run_round`的shares阶段 | 真正t-out-of-N多项式，AES-GCM封装包含发送者/接收者/轮次；不是历史hash模拟；N≥门限解释与正文一致，算法印刷不等式不照用 |
| 式3：个人与成对PRG掩码、向服务器上传 | `vlmrs_fullgraph.run_round.make_upload` | DH两端派生同一轮次流并抵消；个人β每轮重新采样，长期DH私钥复用选择单列 |
| §5.2：恢复在线者β或缺席者NSK并消除残留 | `sharing.reconstruct_secret`；`vlmrs_fullgraph.run_round.unmask` | 正常、前/后掉线逐维正确；门限不足明确失败；不返回无验证的假成功 |
| 式4—10、算法2：HF、(A,B,L,Q)、证明聚合、逐客户端四项检查 | `vlmrs_verification.VLMRSVerification`；全回合剩余客户端verify阶段 | BLS12-381真实群和配对；HF采用秘密系数线性向量实例；公开可逆d条件下协调缩放被接受，尚未通过认证门槛 |
| 多轮复用 | `recover_past_upload_with_later_key` | 后轮重构长期NSK后可恢复旧输入的条件诊断已落盘；不能直接旋转所有密钥后冒充原文复用收益 |

诊断入口 `probe_vlmrs_fullgraph.py`；归档运行 `20261018-210622-574404_vlmrs_fullgraph_diagnostic`，N6/d257/t4。对应测试 `test_point1_sharing.py`、`test_point1_vlmrs.py`。它不是VLMRS正式E1/E5/E3或性能排名，完整条件与原文位置见[实施审查](../../docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md)。

## 8. SAEV：实际完整条件回合映射

| 原文步骤 | 当前实际执行位置 | 条件及证据 |
|---|---|---|
| §VI初始化：CRT的a_i/A_i/B_i/C_i，保密分发；HH公参和r_i | `saev_full.SAEVProtocol.__init__` | 全宽Python整数；a_i两两互素且在指定区间；有限域使用ffdhe3072，r_i按非零字段元素解释；不是作者代码 |
| 算法1/3：x_i+C_iξ_i，盲化同态证明 | `saev_full.run_round.mask_and_prove` | 实际PRG、乘法HH；从未将CRT密文强行转成8字节 |
| 边缘预聚合、在线名单与TA的K/R | `saev_full.run_round.preaggregate/auxiliary` | 各链路真实传输；K/R仅交客户端，不在云侧提前解密 |
| 云汇总密文和证明 | `saev_full.run_round.aggregate` | 云只处理密文；完整大整数群运算和真实封装长度 |
| 算法2/4：逐客户端解密和验证 | `saev_full.run_round.decrypt/verify` | 中心化有符号解码要求a_i>2Nβ；正常/掉线正确；仅改密文拒绝，协调改密文及证明接受 |

入口 `probe_saev_full.py`，归档运行 `20261018-211729-065159_saev_full_diagnostic`，N20/M5/d257。测试 `test_point1_saev.py`。此条件变体未通过完整认证门槛，没有冒充E8三预热五观测或7850维性能结果。

## 9. 版本与成本验收

当前正式HVSA性能批次使用每轮1800秒硬限时及E6逐轮交替控制组。新运行源码在开始时归档；旧N扫描早于输出域检查，保留版本边界。图表只能取完成且严格复核的实际数据；timed_out、运行中或受限诊断不并入完整协议排名。当前计划所需的GPVO/VLMRS基线仍未完成，不能以这些映射表代替验收。

## 10. OVP：逐坐标完整条件链路映射（N20/d257诊断已完成）

| 原文步骤 | 代码位置 | 条件与当前证据 |
|---|---|---|
| Fig.3 Round 0：TA离线密钥初始化、公钥交换 | `ovp_full.OVPProtocol.__init__` | BLS12-381 DH为实例化选择；真实配对基及公钥交换小链路测试通过；与直接配对的一致性通过 |
| Round 1：α分割，两路PRNG；两份AE上传 | `OVPProtocol.run_round`的split/upload阶段 | 完整F_q向量、AES-GCM；原文实验CBC/1024位差异明确；负数及大数的字段抵消测试已通过 |
| Round 1/2：双方交换在线名单并取交集 | online阶段 | 小链路和N20/d257独立案例均将只交一份的用户排除，返回正确交集参考 |
| Round 2：各服务器总流的加法分享、修正x_vn | `additive_shares`及correct_input_part | N−1个真实随机字段向量与最后一个约束份额；不模拟成空函数；含交集排除的纯字段测试通过 |
| Round 2/3：每个坐标配对标签及云聚合 | coordinate_tags/aggregate_field_vectors_and_tags | 原文§VII.C的O(M)验证解释；真实GT幂和全宽序列化，不使用一个标量标签代替向量 |
| Round 3/4：相反份额反馈、秘密β、逐客户端验证 | coordinate_feedback/verify_every_coordinate | 两VS秘密分离；N4/d2组件与N20/d257完整诊断均执行；单改结果拒绝，协调改结果/标签/反馈被接受 |

入口 `probe_ovp_full.py --after-batch <batch.json>`。两项真实组件测试通过后，5个独立诊断案例已在HVSA性能批次正常结束后串行完成：`20261019-170131-784975_ovp_full_diagnostic`，严格核验通过、execution_errors=0。每个完整上传者实际检查全部257坐标；协调攻击不读取VS秘密β或明文参考。这是条件链路诊断，不替代正式E8、原文安全判决或跨论文性能排名；完整条件见实施审查§9。

## 11. HVSA定向可用性边界

`probe_hvsa_boundary.py`使用真实HVSA实体，在N100/M5/d7850下执行3份独立输入。运行`20261019-175022-317305_e4_hvsa_boundary`中，同组20人在上传前全部离线时只收到4/5组，未完成且verified=null；上传后该20人离线时贡献仍计入，完整验证通过并向80人下发；所有人上传后离线时同样正确、验证通过，模型接收者为空。未改根协议或增设空组证据。

manifest明确预期边界；输入NPZ、每组上传人数、实际接收者与全部事件归档。核验器重算明文参考及消息接收者，不把预期同步屏障失败算作密码拒绝或成功聚合；3例不与E4的80条随机轨迹合并成功率。8项边界证据测试及实际记录严格核验通过。

## 12. 用户授权的明确实例化与同输入计量

`gpvo_decided.py`采用Z*_(N²)乘法LHF、3072位RSA模数、素数e>N²且互素、均匀单位元g（不声称已证最大阶）。公共参数只初始化一次，每轮重新生成TA/用户DH密钥与组承诺随机量，生成/分发计入每轮；不加原消息路径以外的承诺签名或账本。使用gmpy2整数模幂，结果与旧后端一致；参数/后端差异公开。上传前掉线的掩码揭示在用户侧真正计算并逐接收者发送，不重复将该工作计到边缘。

VLMRS固定全邻居、ceil(2N/3)恢复阈值、公开根指数7、共享秘密线性HF和两组长期DH密钥复用。原有缩放及历史恢复条件现象保留；不凭空加秘密根指数或刷新策略消除反例。它是平面方案，M标为不适用。

`run_decided_comparison.py`为三方案使用同一份实际NPZ及上传/接收集合，轮间旋转执行顺序；HVSA使用当前源码重新采集必要点。初始化、角色互斥计时、真实逐接收者消息、参考和及实际返回值归档。`--pilot`有三类可用性案例、每例一次，无预热；非pilot默认诚实回合3预热+10观测，两种范围不混用。

试跑`20261019-182131-848006_decided_baseline_pilot`共9/9正确且验证通过，严格记录核验通过；相关9项回归测试通过。原始单次成本与初始化见`docs/POINT1_DECIDED_BASELINE_PILOT_20261019.md`。这是明确实例化的完整链路试跑，不是作者代码、安全等价证明、完整E5或正式重复成本分布。

## 13. 明确实例化的四点重复成本矩阵（2026-10-20完成）

N=20/50、维度1000/7850，分层M=5、VLMRS无边缘角色。每方案每点3预热+10正式，四点156个完整回合全部逐维正确且完整验证通过。续跑批次`20261020-212351-556444_decided_minimum_continuation`保留两个N20完整点并新跑两个N50点，78个保留+78个新增；旧中断2个预热回合不进入统计。四点关键协议源码与授权配置摘要相同，N20/N50跨日采集限制单列。

成本报告保留全部角色互斥工作、串行回放、初始化与真实逐接收者字节；图仅用120次正式观测，报告中位数与IQR，不填估算RTT。最终图及分析在`figures/point1_compare/20261020_decided_minimum_matrix_v2/`，归档与分析核对凭据为`docs/POINT1_DECIDED_COST_EVIDENCE_AUDIT_20261020.json`。HVSA相对GPVO的客户端及系统工作较低；N20/d7850通信比VLMRS高15.04%。缓存容量和密码后端不同，不能把全部成本差归因于协议结构，也不能据此写同等安全的SOTA排名。

生产参数完整E5、跨方案同人数多轮与分阶段掉线仍待执行。本节的成本验收不改变此前条件诊断中错误接受/历史恢复现象及其适用条件。

## 14. 六类生产参数异常补测（2026-10-23完成）

运行`20261023-204145-042317_decided_integrity`与20日成本矩阵的协议关键源码一致。两方案各100份32/257维和10份7850维当前输入，232个完整诚实上下文均正确且全部指定验证者通过；六类共1320次单代表验证者调用，公开构造及逐次记录独立核验通过。GPVO零扩维属于本项目接口检查，不能当成原论文密码学缺陷；协调修改、重放和VLMRS后来私钥恢复的条件行为如实保留。详见`docs/POINT1_DECLARED_INTEGRITY_RESULTS_20261023.md`。同人数多轮及三方案掉线不因本矩阵完成而自动变成已完成。

## 15. HVSA同人数多轮（2026-10-23完成）

复用冻结`iter_setting`，固定40人/变化40人各50轮，N50/M5/d7850，每轮同一归档输入、交替执行次序。100/100原始和正确且全部边缘验证通过，真实返回和及40个模型接收者核验一致；初始化各一次。旧50/40轨迹及跨日成本不混入。见`docs/POINT1_HVSA_PAIRED40_RESULTS_20261023_v3.md`；跨方案多轮和完整掉线仍待补。

## 16. 用户暂停后的分段补充（2026-10-24完成）

旧run保持paused_by_user，252项带固定SHA独立导出；新进程只执行原索引8/9的48项，严格来源与完整账本核验通过。E3为原连续60回合；E4为8+2两个密钥/初始化状态，重启成本单列、图按时段分行并显示全部原始点，不能声称连续10轮同一长期密钥或同条件性能。报告与bundle见`docs/POINT1_SEGMENTED_TRAJECTORY_RESULTS_20261024.md`及`figures/point1_compare/20261024_segmented_trajectories/bundle.json`。原大规模及统一条件连续性能仍待扩展。


## 17. 跨方案定向掉线边界（2026-10-24完成）

`run_directed_dropout.py`按预声明12种情形、N20/M5/d257、3共同输入池执行108次真实调用，87次正确且完整验证、21次未完成，全部符合预期。每方案仅初始化一次；全部指定边缘/客户端验证及实际接收者记录经独立核验。环连续/交替两人缺席的GPVO揭示消息分别2/4条；HVSA整组上传前缺席停于屏障，VLMRS剩13人低于固定阈值14；HVSA/GPVO全体上传后离线由边缘验完但无客户端接收。可用性未完成不计作篡改检出或成功聚合。

报告`docs/POINT1_DIRECTED_DROPOUT_RESULTS_20261024.md`、新论文稿v4及独立证据凭据已保存，9项归档/拒伪测试通过，最终PNG实际目视核查通过。核验器修订保留原档与副本，冻结协议及实跑驱动不改。该范围为功能诊断，不是正式性能分布；既有安全条件和大规模/连续性能缺口保留。


## 18. N50/d7850三方案同人数50轮

`run_large_multirun.py`的6个状态各初始化一次，固定40人/变化40人各50轮、三方案共300次真实调用；300次正确且完整验证、0次未完成。分片journal保存全部返回记录与原始阶段/通信，公开长期状态指纹及GPVO逐轮新会话指纹已核验。报告`docs/POINT1_LARGE_MULTIRUN_RESULTS_20261024.md`及论文新稿v5保存实际初始化、首轮、后续轮和真实前缀累计/摊销。10项归档/拒伪测试通过，3张PNG实际核查。只读环境记录不是硬件锁定，条件安全行为和后端差异不变；N100掉线及E4统一环境观测仍待执行。


## 19. N100/d7850分阶段掉线

`run_large_dropout.py`执行前/后×四档×10条×三方案，共240次真实调用，240次正确且完整验证、0次未完成；24个状态各连续10条、各初始化一次。上传后退出者的已上传贡献保留；全部指定验证者调用7600次，实际和/接收者/邻居补偿与分片摘要独立核验。报告`docs/POINT1_LARGE_DROPOUT_RESULTS_20261025.md`、论文v6、12项检查及3张PNG核查凭据保存。连续单状态问题已补，温度/频率未锁定；明确实例化及既有条件诊断保持，不声称原文安全等价或SOTA。


## 20. 独立明文训练对照（2026-10-26）

`run_training_controls.py`新增同编码明文与独立浮点FedAvg，3种子×20轮。同编码分支60/60重现旧HVSA基模型、量化本地模型、整数和及测试正确数；浮点分支各自训练，float64均值后转float32。旧60次安全协议/300次边缘验证不重复计数。全部聚合与测试预测复算，另按(seed+round)%20每轮每分支重跑一个客户端SGD，共120次。正式10项检查与2张PNG核查通过；结果见`docs/POINT1_TRAINING_CONTROL_RESULTS_20261026.md`，论文稿v7。仅IID线性模型，不宣称精度等价、领先或三方案训练排名。
