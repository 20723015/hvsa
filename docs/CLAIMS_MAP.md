# CLAIMS_MAP — 贡献—定理—实验映射表（盲审应答底稿）

用途：论文每一处可被盲审质询的声明，对应到唯一的定理/实验证据与数据来源。
任何正文数字修改须先过本表的三方对账（论文 ↔ stats JSON ↔ manifest）。
创建：2026-10-01（盲审补全执行计划阶段 0.1）；对拍结果随阶段 0.2 更新。

---

## 1. 研究点一（第 3 章，HVSA 同步安全聚合）

| 论文声明 | 支撑定理/命题 | 实验证据 | 数据来源 | 状态 |
| --- | --- | --- | --- | --- |
| 客户端参数隐私（掩码 + 单 TA） | Theorem 3.1（聚合泄露下的客户端参数隐私；合谋阈值 c ≤ N−2） | DLG 梯度反演防御（掩码下 SSIM/PSNR）+ 掩码/明文量级分析（std 比 87.5×） | `exp_n2_dlg_inversion.py`、`exp_n7_masked_distribution.py` 产物 | 已有 |
| 云端聚合结果可验证 | Theorem 3.2（验证可靠性，归约到 LHH 抗碰撞 + Ed25519 EUF-CMA） | 攻击证据批跑：tamper_global / tamper_manifest / drop_replace_pkg / replay 全检出、clean 零误拒 | `results_thesis_core_attack/`、`run_attack_evidence.py`、AttackOutcome JSONL | 已有（3040 单元） |
| 客户端掉线鲁棒 | Theorem 3.3（在线集合驱动掩码重构） | 掉线率扫描实验 | 研究点一既有批次（不重跑，只核查） | 已有 |
| 验证通信与模型维度 d 无关（定长 3,670 B/checkpoint） | —（机制论证） | `exp_security_overhead.py`：Tiny/MLP/CNN 维度扫描 + 两个真实 run 分类型占比交叉验证 | `docs/OVERHEAD_EXPERIMENT.md` | 已有；阶段 4.2 扩展 d∈{1e4,1e5,5e5,1e6} + JSON 输出 |
| 密码学计算开销 per-role | — | `exp_overhead_crypto.py`、`exp_n4_phase_breakdown.py`（3 预热+10 实测口径） | `docs/OVERHEAD_EXPERIMENT.md` | 已有；阶段 4.1-4.2 补内存与 5e5/1e6 实测 |

## 2. 研究点二（第 4 章，检查点式组级半异步）

| 论文声明 | 支撑定理/命题 | 实验证据 | 数据来源 | 状态 |
| --- | --- | --- | --- | --- |
| 创新点 1（核心）：checkpoint 组级半异步降低等待、提高吞吐 | Theorem 4.1（检查点完整性检测） | 主表等时间/等计算量：G2(组间同步 K=5) vs G3(K=3)；等待 1.59→0.88 tick、吞吐 3.22→3.97 | `results_thesis_core/`（thesis_stats） | 已有 |
| 创新点 2（机制 A）：策略绑定可验证加权聚合（恶意云端不能自选权重） | Theorem 4.1 + 策略绑定归约（4.4 节） | 策略绑定攻击族：weight_forge / weight_relabel / source_downgrade / alpha_tamper / history_missing 全拒绝 | `results_thesis_core_attack/` | 已有；阶段 4.4 新增 drop_replace / history_missing 两类补全（history_missing 原为结构性不可测，见 §5 修复项） |
| 创新点 3（机制 B）：Direct-VCM 后置补偿改善 mixed-version 聚合 | 条件性单 checkpoint 偏差界（IID、一步更新假设）+ e_{t+1}≤L_T·e_t+ε_t 条件传播 | G3 vs G3v 配对差（6 单元 × 5 seeds）；γ 扫描消融（IID 单调递增、NI 反转）；vcm_mode 敏感性 | `results_thesis_core/`、`results_abl_*`、`results_vcm_*` | 已有；阶段 3 补 α=0.1 边界 |
| Theorem 4.2（客户端参数隐私保持） | Theorem 4.2（N_Q−c≥2 判定、掩码前混合不破坏链路） | 同 Theorem 3.1 实验链路 + 协议结构论证 | — | 已有 |
| 半异步不破坏隐私/验证链路 | Theorem 4.2 | 掩码/验证路径复用研究点一原语；攻击证据批跑同时覆盖 | — | 已有 |

## 3. Arm 语义速查（`run_thesis_core.py:59-94`，勿在论文外另造口径）

| Arm | 实现 | 语义 | 用途 |
| --- | --- | --- | --- |
| G1 | `fedavg_point2.py --rounds 20` | FedAvg 同步参考，20 轮 × 20 客户端 | 等更新量口径的同步参照（20×20=400 更新） |
| G2 | `main_point2.py -K 5 --s-max 3` | 组间同步（5 组全到齐才聚合），无补偿/无权重 | 同步 barrier 基线 |
| G3 | `-K 3 --s-max 3` | 检查点式组级半异步，无补偿 | 主方法骨架 |
| G3v | `-K 3 --s-max 3 --vcm --gamma-vcm 1.0` | ①+Direct-VCM，**论文最终主方法** | 主表/消融基座 |
| G5 | G3v + `--staleness-sigma 0.5` | ①+VCM+σ（消融） | σ 组件消融 |
| G3s | G3 + `--staleness-sigma 0.5` | 仅 σ（`--with-sigma-ablation` 才跑） | σ 消融 |
| GFB | `-K 1 --s-max 3` | 组包级 K=1 异步消融（**非 FedBuff 复现**） | checkpoint 阈值观察 |

等更新量口径：G1 = 20 轮 × 20 客户端 = 400 用户更新；thesis_core 各臂以 npz
`checkpoint_Ns` 累积曲线对齐（阶段 6.1 的等更新量比较模式）。

## 4. 阶段 0 核查结论（2026-10-01）

### 4.1 既有 thesis_core 矩阵的 verify 状态（阶段 0.3，已定论）

**结论：主矩阵全部 main_point2 臂（G2/G3/G3v/G5/GFB）为 verify-off。**

依据（三条独立证据）：
1. `run_thesis_core.py` ARMS 表内无 `--verify`；`main_point2.py:1979` 的
   `--verify` 为 `store_true` 默认 False。
2. 全部 run meta `verify_total=0, verify_pass=0`：`verification_attempts`
   仅在 `if args.verify:` 分支内累加（`main_point2.py:1102`）。
3. 论文 4.6 已有对应限定表述（"主矩阵各臂为验证关闭运行，其验证耗时字段
   不构成验证开销证据，验证开销以 4.4 节 verify-on 批次为准"）。

`verify_seconds_total` 非零之谜（≠验证开销）：`_t_verify0` 计时器在
`main_point2.py:1100` 无条件启动，verify-off 时它度量的是 commit 事务
（stage→finalize→publish→notify）耗时 ≈ 60–75 ms/checkpoint（G3v CIFAR10
20 个 checkpoint 合计 1.2–1.5 s）。**该字段在 verify-off run 中不得被引用为
验证开销**——论文已正确回避，后续表格生成（阶段 6.2）须保持该纪律。
LHH 验证计算 O(d) 的实测证据只来自 verify-on 批次（阶段 4.2）。

### 4.2 Ours-Plain 对拍（阶段 0.2）

单元：G3v × CIFAR10 × IID × s42（`-K 3 --s-max 3 --reuse-rule none --vcm
--gamma-vcm 1.0 --max-checkpoints 20 --max-ticks 300 --deterministic`）。

- verify-off 参照：`results_thesis_core/G3v_CIFAR10_IID_s42`，
  `arr_digest=649ba34b26ea2cff`。
- verify-on 复跑：`results_p0_verify_pair/G3v_CIFAR10_IID_s42_von/`。
- 判定结果：**【待对拍 run 完成后填写】**（一致 → Ours-Plain 复用 30 个
  G3v run；不一致 → 补跑 30 run）。
- ES delay profiles 两侧一致（`{'edge_0':1,'edge_1':4,'edge_2':4,
  'edge_3':3,'edge_4':3}`），RNG 消耗顺序未被 verify 开关改变。

## 5. 论文待删点与主线痕迹清单（阶段 7.1 执行，此处登记）

| 位置 | 内容 | 处置 |
| --- | --- | --- |
| `chapter04.tex:1527-1528` | "σ 度量（版本差/相对位移两种度量）的跨数据集复测与自适应档位（Aσ）机制未执行，须另行补跑后方可报告。" | 删除（σ 主线降为消融口径后该缺口声明随之失效） |
| `chapter04.tex:1640-1646` | `\paragraph{（7）未执行项}` 整段（ρ_max/β_c 未扫、复用×缓冲交互（I2）、D_fair/Δ_timeout、自适应 s_max 无对应开关） | 删除整段（含 `\paragraph` 行；计划写 1641-1645，实际段首在 1640） |
| `chapter04.tex:130-131` | 威胁模型列举"自适应衰减档位（Aσ）" | 降为消融/展望口径或删除 |
| `chapter04.tex:338` | "自适应 s_max、公平保底 D_fair、选择策略" 并列入未来开关 | 移出主线表述 |
| `chapter04.tex:370` | "启用 σ/Aσ 衰减策略时…" | 保留但标注为消融组件 |
| `chapter04.tex:734、766` | Aσ 状态/策略在实现中的描述 | 降为消融/附录口径 |
| `chapter04.tex:981-997` | σ/Aσ 策略条件正确性分析 | 保留为条件性分析（不作为主线声明），检查与"降为消融"口径一致 |
| `chapter04.tex:1259-1261` 附近 | "两处如实声明缺口…"（达到目标精度墙钟未截取 + verify-off 限定） | verify-off 限定保留（永远正确）；目标精度墙钟缺口由阶段 6 等时间口径补齐后改写 |
| `chapter04.tex:1283-1300`（`tab:ch4-baselines`） | 竞争基线表当前声明"四个基线均未在本实验框架内完成同环境复现" | 阶段 2 完成后整表换为 5 方法实测 booktabs 表（阶段 6.2 生成），"未复现"声明删除 |

Aσ / I2（reuse×buffer 交互）/ 自适应 s_max 的定位纪律：**不进入主线**；
Aσ 与 σ 只作为消融存在，I2 与自适应 s_max 降为未来工作。措辞中不出现
"首次/开创"类绝对表述；新颖性声明一律以检索日志（阶段 5）为前提。

## 6. 验收映射（执行计划 → 证据）

| 验收项 | 来源阶段 | 数据落点 |
| --- | --- | --- |
| 竞争表 5 方法 × 6 单元 × 5 seeds（g1_ref / fedbuff / mafsa / dvafl / ours_plain） | 阶段 2（150 run：30 复用 G1 + 30 ours_plain 视对拍 + 90 新跑） | `results_plain_bench/` + `results_thesis_core/` |
| 等计算量与等时间并列比较 | 阶段 6.1（`plain_bench_stats.py`，checkpoint_Ns 累积曲线为 x 轴） | stats JSON |
| α=0.1 Non-IID 边界（Direct-VCM 适用边界） | 阶段 3（`results_n1_alpha01/`，不并入主表） | 该 root manifest |
| 维度/内存/通信/攻击开销闭环 | 阶段 4.1-4.3 | `exp_security_overhead` JSON + per-role 表 |
| 3040 既有单元 + 新增攻击全拒绝、clean 零误拒 | 阶段 4.4-4.5（history_missing v≥1 修复 + 回归对拍） | AttackOutcome JSONL + 存档对拍记录 |
| 数值三方对账零差异 | 阶段 6.3 | 论文 ↔ stats JSON ↔ manifest |
| 正文无 Aσ/I2/自适应 s_max 未执行主线 | 阶段 7.1（§5 清单） | 论文 diff |
| 限定性表述 + 限定检索覆盖 | 阶段 5（9 列比较矩阵、证据级别标注） | `docs/search_log/COMPARISON_MATRIX.md` |

## 7. 口径纪律（写给所有后续改写者）

1. G3v 是唯一主方法；G3 是骨架消融、G5/G3s 是 σ 消融、GFB 是组包级消融
   （不得称 FedBuff 复现）。
2. 等更新量：G1=400 用户更新为锚；其余臂以 `checkpoint_Ns` 累积对齐。
3. verify-off 主矩阵的 `verify_seconds_total` 是 commit 事务耗时，不是验证
   开销；验证开销只引 verify-on 批次。
4. Direct-VCM 结论按数据写：IID 稳定、Non-IID 未观察到显著增益 → 只陈述
   适用边界；"收紧准入而非扩大补偿"仅作假设。
5. 基线公平性表述（阶段 1.6 文档定稿）：ours_plain 为冻结仿真器组级轨迹；
   三个外部基线为逐客户端轨迹（同种子、同延迟分布、各自调度器生成），
   差异如实声明，不声称"同框架逐行复现"。
6. **递进关系一句话（门槛 8 锚定句，摘要/结论口径与此逐字一致）**：
   研究点一建立"单 TA 掩码 + LHH 验证"的同步安全聚合与验证链路
   （Theorem 3.1–3.3），研究点二把异步边界收缩到 checkpoint 级，在不破坏
   该链路的前提下补齐 mixed-version 补偿（Direct-VCM）与策略绑定可验证
   加权（Theorem 4.1–4.2）。

## 8. 阶段 4 执行记录（2026-10-01）

### 8.1 冻结文件唯一修复（已批准，计划 4.5）

- 位置：`main_point2.py` history_missing 注入分支（原 :1086-1089）加
  `checkpoint_version < 1 → 不适用` 守卫，约 7 行。
- 依据：`edge.py:720-726` 的 history 检查仅 v>0 生效 → v=0 时该攻击
  结构性不可测，注入会产生"未注入却记为接受"的伪阳性单元。
- 回归证明（三件套，执行记录见 `results_regress_attack/`）：
  1. `regress_attack_fix.py` CPU 重跑 raw_sum s42 整类 ↔ 存档 JSONL 逐条
     逐字段对拍 → `REGRESSION_VERDICT.txt`；
  2. 修复后全量 pytest（修复前 92 passed 基线，后置 log
     `overnight_pytest_postfix.log`）；
  3. 新攻击批跑以其自身验收（拒绝率 100%）兜底。
- 对性能轨迹零影响的论证：守卫只存在于 `if args.attack in (...)` 分支内，
  `--attack none`（全部性能批次）不可达。

### 8.2 产物与脚本

| 项 | 脚本/产物 | 状态 |
| --- | --- | --- |
| per-role 内存（4.1） | `exp_memory_overhead.py`（运行时包装，冻结文件零改动；tracemalloc + RSS + 分析式对象图口径） | 冒烟通过（Tiny：es_verify 1.19MB / ta_authorize 0.01MB / commit 0.87MB / cloud 0.36MB / ta_mask 0.95MB）；正式探针入第二轮队列 |
| 维度扫描（4.2） | `exp_security_overhead.py --scan`（合成 checkpoint harness，d∈{1e4,1e5,5e5,1e6} × {G2,G3v}，JSON 输出） | 运行中；**口径妥协**：d≤1e5 为 3+10，d>1e5 为 3+5（单次真实 LHH 在争用 CPU 上达 245-280µs/dim，全 3+10 两臂约 30h+，超出整夜预算；每格 reps 字段如实记录） |
| per-role 通信（4.3） | `exp_comm_per_role.py`（静态 type→role 映射，消费既有 comm_bytes_by_type） | ✔ `comm_per_role.json`（142 runs：TA 恒定 15,360 B/run 与 d 无关；verify-off 无 receipt 字段自洽） |
| 新攻击类（4.4） | `run_attack_evidence.py` CLASSES + `run_attack_v2.py`（独立 root `results_thesis_core_attack_v2/`）+ `run_seed_attack_chunked.py` 同步 | 入第二轮队列 |
| 验收口径修正 | `summarize_class` 以**适用（已注入）单元**计拒绝率（计划措辞"各 100 适用单元"）；不适用单元单列，避免误计为"攻击被采纳" | 已实现并修正两个既有测试口径；旧 4 类 applicable==attempted，数值不变 |

### 8.3 阶段 5/6/7 执行记录（2026-10-01 晚）

| 项 | 产物 | 状态 |
| --- | --- | --- |
| OpenAlex 检索（5.1） | `lit_search_openalex.py`（礼貌池 + 深退避，QUERIES 与 Crossref 共用） | ✔ 316 条去重命中 → `docs/search_log/openalex_hits.json` |
| 引文追踪（5.1） | `lit_citation_chase.py`（12 种子，doi→arxiv_doi→title 解析，前/后向 cap=50，失效 ID 容错） | 后台运行中 → `citation_chase_summary.json` + `agent_raw/` |
| 种子清单（5.1） | `docs/search_log/seed_papers.json`（12 篇题录） | ✔ |
| 覆盖度检验数据驱动（5.2） | `build_search_log.py` §3 重写（三库 × 种子 DOI/归一标题匹配）+ §6 局限性声明重写 | ✔（当前实测：精确 4 / 模糊 1 / 全未命中 7，OpenAlex+引文数据落地后重跑更新） |
| 9 列比较矩阵（5.4） | `docs/search_log/COMPARISON_MATRIX.md`（15 行 × 9 列，证据级别标注，递进关系一句话） | ✔ |
| 统计（6.1） | `plain_bench_stats.py`（等更新量模式：checkpoint_Ns 累积为 x 轴；跨 root 汇入 G3v→ours_plain、G1→g1_ref，G1 更新轴按每轮 20 用户合成） | 就绪，等批次 |
| 表格（6.2） | `thesis_tables.py`（9 种 booktabs 渲染器，`chapters/gen/*.tex`） | 就绪；另修 thesis_stats.py 潜在 bug（main_table 未写入 JSON） |
| 对账（6.3） | `reconcile_thesis_numbers.py`（A 完整性 / B JSON↔manifest 重算 / C 攻击单元 / D 开销产物 / E 论文数字） | ✔ 管线可用：B 段 thesis_core 34 格零差异；C 段实测既有单元 **3040** 与计划口径一致；其余等批次 |
| 论文改写（7.1） | chapter04.tex 两处未执行清单删除（1526-1528 改写为展望口径、1640-1646 段删除）+ Aσ/自适应 s_max 主线痕迹清零（威胁模型清单、调度增强列表 σ/Aσ 措辞、981 段标题标注"可选组件"） | ✔ |
| 口径统一（7.2） | 摘要/第 1 章/第 4 章小结/第 5 章结论扫描：无"首次/开创"类表述；"IID 显著、Non-IID 未观察到增益"限定已就位；3040→4040 单元数更新待 v2 数据落地 | ✔（余一项【待数据】） |
| 编译（7.4） | main.pdf / main_blind.pdf 均 rc=0，0 错误 0 overfull 无未定义引用 | ✔ |
| 盲审 QA（7.4） | `E:\bjtu-thesis\docs\blind_review_qa.md`（创新性/基线公平性/理论边界 3 类 11 问） | ✔ |

### 8.4 中断与恢复（2026-10-01 23:3x，恢复记录）

**中断事实**：首轮隔夜任务在 21:19 前后全部终止（verify-on 对拍停在
checkpoint 2、回归重跑停在 97/127 条、两个队列脚本未及创建主日志），
根因判断：首批任务挂在本会话的后台进程树下 + 该机器已知有睡眠/更新
打断史（run_seed_attack_chunked.py 文档字符串即为此而写）。教训已固化：
**对拍 run 不再作全队列门闩**。

**恢复模型（三路独立计划任务，Task Scheduler 进程树，WakeToRun +
StartWhenAvailable + 40h 限额，会话退出不影响）**：
- `hvsa_gpu_queue` → `overnight_queue_v2.sh`：plain 90 → α01 61 → α10 31
  → 修复后 pytest → 新攻击 10 → 内存探针 → **对拍（最长，放最后）→ 对拍
  判定**。各阶段独立日志、可重入（manifest 指纹跳过）。
- `hvsa_cpu_work` → 回归重跑（REGRESSION_VERDICT.txt）→ 维度扫描
  （已改**逐格落盘**，中断保留已完成格）。
- `hvsa_lit_work` → 引文追踪剩余 10 种子（summary 按 key 合并不丢已
  完成）→ 重生成 `检索日志.md`。
- 代码加固：scan 增量落盘；LIT_MAILTO 环境变量可覆盖占位邮箱。

**晨间流水线**：`morning_pipeline.sh`（三 root stats → thesis_tables →
reconcile 全量 → 门槛 1/4/5 判定打印）。人工收尾仅剩：把
`chapters/gen/*.tex` 换入 chapter04（含 3040→4040 与攻击缺口段更新）→
latexmk 重编译两版 PDF。

**次日待办**：核验 PARITY_VERDICT → 填入本文件 §4.2 → 跑
morning_pipeline.sh → 论文换表 → 重编译。

### 8.5 阶段 4 的论文表述锚点

- 通信与 d 无关的证据 = `comm_per_role.json` 的 TA 角色（receipt/notify
  定长）+ `exp_security_overhead` 维度扫描的 manifest 定长项；
- LHH 验证计算 O(d) = 维度扫描 `es_verify` 列实测（1e5/1e6 真实 LHH），
  外推值只作交叉校验并列表述；
- 内存 per-role 表 = `memory_overhead_report.json`（tracemalloc 口径；
  GPU 显存不在追踪范围，RSS 差值含 CUDA 上下文噪声——报告中已注明）。
