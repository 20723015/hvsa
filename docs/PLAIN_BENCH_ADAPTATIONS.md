# PLAIN_BENCH_ADAPTATIONS — 明文基线适配决策记录

用途：盲审/答辩应答材料。逐基线记录"论文/原文语义 → 本实验框架实现"的
每一条适配决策及其依据，支撑竞争基线表的公平性声明（论文 §4.6）。
配套代码：`plain_bench.py`（实现）、`run_plain_bench.py`（批跑）、
`tests/test_plain_bench.py`（手算对拍单测）。
创建：2026-10-01（盲审补全执行计划 阶段 1.5）。

---

## 0. 共同框架（三臂共享的决策）

| 决策 | 内容 | 依据 |
| --- | --- | --- |
| 拓扑映射 | 冻结链路的 5 ES × 4 用户分层拓扑在逐客户端粒度下压平：20 个客户端直接对服务端异步上传。组包（GroupPackage）语义不保留——它是本文方法的结构，不是基线的。 | FedBuff/MAFSA/DVAFL 原文均为客户端-服务器（或客户端-ES-云，聚合单元仍是客户端更新）；压平不改变"更新在到达前于客户端本地累积陈旧度"的核心动态 |
| 数据划分 | `datasets.distribute_data`，调用序列与 `main_point2.py:243-252` 逐行一致（`np.random.seed(seed)` + `torch.manual_seed(seed)` 之后）。同 (dataset, seed, iid, α) 的划分与主矩阵 run 完全相同。 | 基线比较的"同数据"锚点 |
| 本地训练 | `models.train_local`（SGD momentum=0.9、NLLLoss），与冻结链路同一实现。DATASET_CONFIGS 直接 `from main_point2 import`。 | "同训练语义"锚点；消除了"基线用了不同优化器"的质疑面 |
| 延迟生成 | 复刻 `main_point2.py:483`（`default_rng(seed)`）与 `:531-548`：采用其 group-by-speed 分支（`:535`）的**逐用户抽取**方式 `int(rng.integers(min_delay, max_delay+1))`，得每客户端延迟。同种子同分布。 | 冻结链路非分组分支只抽 5 个 edge 级延迟（逐包粒度），逐客户端粒度必须逐用户抽；抽取方式取原文 group_by-speed 分支的既有代码，避免新造分布 |
| 掉线模型 | 逐轮独立 Bernoulli（`dropout_rate=0.1` 默认，与主矩阵一致），掉线判定发生在训练完成时（`:643-646` 同口径）；burst 状态机同 `:640-646`。 | 与冻结链路掉线口径一致 |
| 聚合触发 | K 到达触发（FIFO 取前 K，`buffer_k=12`）。**三臂同触发**——触发器是实验控制变量，不是被比较的方法差异。K=12 ≈ 主方法 G3v 每 checkpoint 聚合质量（N_Q 均值 11.2，results_thesis_core G3v 实测）。 | 隔离各方法在"同节奏聚合"下的机制差异；等计算量比较（阶段 6.1）以 checkpoint_Ns 累积曲线为 x 轴 |
| 缓冲时效 | `s_max=3`（超期更新丢弃并计数）作为**共享系统约束**对三臂一致生效。 | FedBuff 原文即采用有界陈旧度；不给基线任何一项本方法独享的机制，也不让基线吞下本方法没有的惩罚 |
| 到达后重排 | 客户端上传（或掉线）后立即以下一到达时刻的最新全局版本开始下一轮训练。 | FedBuff/DVAFL 的连续训练语义；与冻结链路"edge 封包后回到 idle 下一 tick 重新开始"的节奏同构 |
| 产物契约 | npz/meta 31 字段复刻 `main_point2.py:1662-1808`；verify/attack 相关字段填 0/None（明文无验证语义）。新增 `arm`/`buffer_k`/`peak_rss_mb`。 | run_thesis_core 口径的 manifest 字段（`_collect_products`）可直接消费；`peak_rss_mb` 为阶段 1.3 要求的内存字段（per-role tracemalloc 于阶段 4.1 升级） |
| 通信记账 | `upload = 4B × d × 更新数`（明文 float32），`broadcast = 4B × d × 训练启动次数`（每轮下发）。**协议侧 int64 8B/元素口径不可与之直接混比**，论文引用时分开列。 | 口径声明是公平性的前提：掩码域 int64 是本文协议的设计代价，明文基线按其实际部署形态计 float32 |

## 1. FedBuff（`--arm fedbuff`）

**原文**：Nguyen et al., "FedBuff: Closed-Loop Federated Learning as A Service"（2022）。服务端 buffer 收集客户端更新，K 条到达即聚合；聚合含陈旧度函数 a(τ) 与有界陈旧度。

| 适配点 | 决策 | 依据/出处 |
| --- | --- | --- |
| 聚合权重 | α_i = 1/(wait_i+1)^σ，wait_i = 聚合 tick − 到达 tick（缓冲内等待），σ=0.5（`--wait-sigma`）。σ=0 退化为标准样本量加权平均（单测锚定）。 | 计划书指定"复用 wait 口径"（`main_point2.py:793-796` 同式）；σ=0.5 与主矩阵 σ 消融值一致，避免引入第五个 σ 值 |
| 聚合公式 | W_{v+1} = Σ α_i·S_i·w_i / Σ α_i·S_i（完整模型加权平均）。 | FedBuff 聚合即缓冲更新的加权平均；用完整模型而非增量等价（w_i = W(r_i)+Δ_i，权重作用于 w_i 与作用于 Δ_i 在 W_v 基准下一致） |
| 无补偿 | 不做任何版本补偿（FedBuff 无此机制）。 | 原文语义 |
| 代码 | `fedbuff_aggregate`；机制计数：stale_reuse_counts = 每次聚合中 base_version < v_now 的更新数（版本陈旧参与度）。 | — |

## 2. MAFSA（`--arm mafsa`）

**原文**：Zhang et al., MAFSA（Algorithm 1 "Staleness-Aware Algorithm"；`reference/009-2026-MAFSA.pdf` p.5）。客户端 Bernoulli 在线；掉线轮**在本地旧模型上继续训练并缓存**（offline-train-and-cache）；上线轮同时上报新鲜更新 w_t 与过期更新 w_{t_i}；ES 聚合 θ = Σ|D_i|(α·w_t + (1−α)·w_{t_i})/Σ|D_i|，α(E)=e^{−βE}（Eq.(10)(11)）。

| 适配点 | 决策 | 依据/出处 |
| --- | --- | --- |
| 掉线语义 | 掉线轮：不下载全局模型，在本地模型（有缓存则在缓存之上累积）继续 `train_local`，缓存 flat 与 stale_version。**与冻结链路 `user.py:165-194`（p2_train_offline_and_cache）同语义明文复刻**。 | MAFSA Algorithm-1 掉线分支 (else 分支, "Perform local update by w^{t_i−1}_i")；冻结链路已实现同语义，两边一致避免"两套 MAFSA" |
| 复用混合 | 上线轮：fresh 训练后 w = α·w_new + (1−α)·w_stale，α = exp(−β·E_i)，E_i = 当前训练基版本 − 缓存版本，β=0.3（`--staleness-beta`）。**与冻结链路 `user.py:306-310` 同式同方向**（α 衰减 fresh——这是与本文 ρ 衰减 stale 的方向对照，论文 §4.2 的复用方向差异正来自于此）。 | MAFSA Eq.(10)(11)；冻结链路 reuse_rule='mafsa' 已按此实现并被既有消融使用 |
| 服务端聚合 | 客户端混合后，服务端做纯样本量加权平均（Eq.(10) 在客户端侧混合完成后的退化形式）。 | 原文 ES 聚合的混合发生在"上报对"上；本框架混合在客户端内完成后单条上传，数学上等价（同 α、同 |D_i| 权重） |
| 触发器 | 与三臂一致的 K 到达触发（原文为 T/T_c 分层周期触发）。 | 共同框架决策：触发器是控制变量。原文分层周期触发在压平拓扑下无对应物，K 到达是等价的"周期性回收"语义 |
| 复用计数 | stale_reuse_counts = 两次聚合之间实际发生的混合事件数（成功上传才计，`pending_reuse`），与冻结链路 `pending_stale_reuse_count` 的"确认消费"口径一致。 | 计数口径对齐冻结链路 |
| 上传失败 | 缓存**未混合**的新模型为 stale 候选（`user.py:319` + `:137-148` 同语义）。 | 与冻结链路一致（默认 upload_dropout_rate=0，主矩阵不触发） |

## 3. DVAFL（`--arm dvafl`）

**原文**：Chen et al., DVAFL（TMC 2026；`reference/016-2026-DVAFL.pdf` p.2/p.6）。核心：DWP 动态聚合窗口 + VCM 版本控制补偿 Δ̃g = (m̂_i/M̂)·ΔG（Eq.(13)），直接版本偏差 ΔG 即 Direct 模式（Eq.(14)）；矩校准 m_i（Eq.(15)/(17)）、M（Eq.(16)/(18)）+ Adam 式偏差校正（Eq.(19)）；聚合 w^{t+1}=Σ p_i w̃_i（Eq.(3)，样本量加权）。原文明确**不做**陈旧度降权（与 FedAsync 分界，p.6）。

| 适配点 | 决策 | 依据/出处 |
| --- | --- | --- |
| "版本陈旧度权重" | 实现为 VCM 补偿项的版本偏差 ΔG_i = W(v) − W(r_i)（补偿强度随版本差增长），**不引入降权**。 | DVAFL 原文以补偿替代降权（"fundamentally diverges from down-weighting schemes"）；给 DVAFL 加降权反而是失实实现 |
| VCM 补偿 | 对 base_version < v 的更新：w̃_i = w_i + ratio_i·γ·(W(v) − W(r_i))，γ=1.0（`--gamma-vcm`，与主方法 γ_vcm 同值）。fresh 更新（r_i = v）不补偿。 | Eq.(13)；fresh 更新 ΔG=0，与原文一致 |
| 矩校准 | 默认 moment1：ratio = clip(m̂_i/M̂, 0, 2)，m̂_i/M̂ 各自按 Eq.(15)(16) 做 λ=0.9 EMA + Eq.(19) 偏差校正（逐客户端参与计数 t_i）。moment2 用 ratio²（Eq.(17)(18)）。direct 模式 ratio≡1（Eq.(14)）。 | Eq.(15)-(19) 逐式实现；偏差校正在逐客户端粒度下可行（冻结链路的组级 moment1 适配无客户端计数，用组内平均范数近似——plain 版更贴原文） |
| M̂ 冷启动 | 首次聚合（无 W(v−1)）或 m̂/M̂ 缺失时 ratio=1（退化 Direct）。 | 与冻结链路 cloud.py moment 模式的退化口径一致 |
| ratio 限幅 | clip(ratio, 0, 2)，与冻结链路 `cloud.py:631` 一致。 | 两边同口径，避免"plain 版更激进"的混淆 |
| 聚合 | Eq.(3) 纯样本量加权平均（无降权）。 | 原文 Eq.(3) |
| DWP 窗口 | 不实现（用三臂共享的 K 到达触发替代）。 | DWP 依据"预计完成时间"动态调窗，需要原文的客户端进度上报信道；压平拓扑下用共享触发器（共同框架决策）。**此项是 DVAFL 适配中语义损失最大的一条，论文引用时注明"K 到达触发适配"** |

## 4. 已知局限（论文表述时的限定语）

1. **粒度差异**：ours_plain 为冻结仿真器的组级轨迹（组包语义 + LHH 掩码
   域量化），三基线为逐客户端明文轨迹。等时间（tick）轴上两者的"到达-
   聚合"节奏由各自调度器生成——这是被比较的对象本身，不是实现缺陷；
   等计算量轴（checkpoint_Ns 累积）由阶段 6.1 对齐。
2. **量化差异**：冻结链路在掩码域做 int64 量化（QUANTIZE_SCALE=1e6），
   明文基线为 float64 无量化。阶段 0.2 的对拍（verify-on/off 逐值一致）
   证明掩码与验证机制对学习轨迹中性，量化误差由协议自身的数值边界覆盖
   （见 THESIS_FULL_DESIGN.md 数值实现层说明）。
3. **MAFSA 分层触发**：原文 T/T_c 分层周期触发未复现（压平拓扑无对应物）；
   DVAFL DWP 动态窗口未复现（需原文的进度上报信道）。两者都用共享的
   K 到达触发替代——对基线是中性化处理（触发节奏相同），不是削弱。
4. **超参来源**：三臂全部默认超参（σ=0.5 / β=0.3 / γ=1.0 / λ=0.9 / K=12 /
   s_max=3）逐一登记于本文档，均有主矩阵或原文出处，无自由调参；如盲审
   质疑超参选择，K 与 σ 的敏感性可作附录补充批次（不计入主线）。

## 5. 变更记录

| 日期 | 变更 | 触发 |
| --- | --- | --- |
| 2026-10-01 | 初版：三臂语义、共同框架、局限声明 | 阶段 1 实现（plain_bench.py 12 项单测全过） |
