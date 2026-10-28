# thesis-core 实验流水线 (第 2–3 周工程部分)

本文件记录"稳过盲审"四周计划中**工程侧**的落地方式: 主表批跑入口、统计与图表、
安全证据批次, 以及新增指标的计量口径。论文写作与数值回填 (第 1、4 周) 不在
本流水线范围内; 本流水线**不新增任何协议功能**。

## 1. 主表批跑: `run_thesis_core.py`

```bash
python run_thesis_core.py plan                 # 打印计划与规模
python run_thesis_core.py run                  # 全量串行执行 (可中断续跑)
python run_thesis_core.py run --arms G3v --datasets CIFAR10 --seeds 42
python run_thesis_core.py verify-repeat        # 复现轨迹逐值校验
```

计划规模 (默认 172 runs):

| 组 | 臂 | 规模 | 说明 |
|---|---|---|---|
| 主矩阵 | G2/G3/G3v/G5 | 4 × 3 数据集 × 2 分布 × 5 seeds = 120 | G3v 为主方法 |
| 同步参考 | G1 (FedAvg) | 3 × 2 × 5 = 30 | `fedavg_point2.py` |
| 阈值消融 | GFB | 2 数据集 × 2 × 5 = 20 | 组包级 K=1；非 FedBuff 复现 |
| 复现校验 | G3v (CIFAR10, seed 42) | 2 × 2 = 2 | 主表 run + `_rep` run |
| 可选消融 | G3s (仅 σ) | 3 × 2 × 5 = 30 | `--with-sigma-ablation` |

强约束 (fail-fast):

- 每个任务由 runner 统一注入 `--deterministic` (严格确定性; 臂定义里不允许自带);
- 默认串行 (`workers=1`); `--workers>1` 必须显式 `--allow-parallel`, 且会告警
  (并行污染本机计时与 GPU 轨迹);
- 每次运行独立目录 `results_thesis_core/<tag>/`, 内含 `point2_data_*.npz`、
  `point2_meta_*.json`、`run_<tag>.log`;
- `results_thesis_core/thesis_core_manifest.jsonl` 记录配置摘要 (`config_hash`)、
  曲线摘要 (`arr_digest`)、最终 tick、执行/采纳更新数、验证耗时、通信量与本机
  运行时间；续跑只在 tag、任务配置和源码内容摘要组成的 `job_fingerprint`
  全部一致时跳过，源码或参数变化会强制覆盖同 tag 旧产物。
- `--extra-args` 仅供冒烟，禁止写入正式 root；须另用
  `--root results_thesis_smoke`，避免 Tiny 结果进入论文统计。

`verify-repeat` 要求同 seed 同配置两次运行的 `acc_history` 与 `checkpoint_ticks`
**逐值一致** (严格确定性模式); 失配即退出码 1。

### GFB（组包级 K=1 阈值消融）的口径边界

本模拟器的最小聚合单元是**组包**。GFB 取 buffer=1 (`-K 1`, 陈旧度仍受 `s_max`
约束), 即“任一组包到达即聚合”的最激进异步档。它只检验 checkpoint 阈值，
**不等价于逐客户端粒度的 FedBuff 复现，不能作为 FedBuff 竞争基线**。

### FedAvg 参考口径

`fedavg_point2.py` 已对齐主表口径: `--deterministic` (严格模式)、
`--output-dir`/`--run-id` 独立产物、`config_hash` 只哈希科学配置、
meta 含 `arr_digest`/git/环境摘要。产物与 `main_point2.py` 同名规则
(`point2_data_{run_id}.npz`), 便于统一解析; 其 `checkpoint_ticks` 为轮次索引。

## 2. 统计与图表: `thesis_stats.py`

```bash
python thesis_stats.py                              # 默认 root=results_thesis_core
python thesis_stats.py --root results_x --out stats_x
```

输出到 `<root>/stats/`: `thesis_stats_report.md`、`thesis_stats.json`、
`thesis_main_curves.png` (六格 accuracy–ticks, mean±sample std 带)、
`thesis_eff_acc.png` (精度—效率)。

统计口径:

- 全部**同 seed 配对**; 报告 mean ± sample std (ddof=1)、配对差均值、
  95% CI (t 分布, n=5 → t=2.776)、Cohen's dz、正种子数;
- 三个比较是算法机制阶梯: `G3−G2` (半异步)、`G3v−G3` (后置补偿)、
  `G5−G3v` (可选 σ 组件)；第三创新“策略绑定”由安全攻击表支撑；
- **等时间比较**: 预算 = 同 seed 的 G3 末次 tick; 各臂取该 tick 之前的最后精度，
  直接报告 `G3/G3v/G5−G2@T` 的配对 CI 与效应量；
- 正式模式缺少任一主矩阵 seed 即退出；部分结果必须显式使用 `--allow-partial`，
  且不会产生“稳定提升”判定；完整五种子时，配对均值 > 0、≥4/5 为正且
  95% CI 不跨 0 才写“稳定提升”；
- ticks 一律解释为**模拟调度时间**; 本机墙钟只作计算成本, 不作分布式加速声明。

## 3. 安全证据: `run_attack_evidence.py`

```bash
python run_attack_evidence.py --dry-run
python run_attack_evidence.py --units 100 --seed 42
python run_attack_evidence.py --units 5 --model Tiny     # 冒烟
```

四类攻击 × 100 个适用攻击单元 + 同规模 clean control; 一个**单元** = 一次
checkpoint 尝试 (`main_point2.py` 每个候选写一条 AttackOutcome JSONL):

| 类 | 成员攻击 | 配置要求 | 检出层 |
|---|---|---|---|
| raw_sum | `tamper_global` | `--vcm` | LHH 等式 |
| manifest_replay | `tamper_manifest`, `replay` | `--vcm` | manifest/签名, exactly-once |
| weight_source | `weight_forge`, `weight_relabel`, `source_downgrade` | `--participation-debias ipw` | ES 独立复算权重/来源绑定 |
| comp_param | `alpha_tamper` | `--staleness-sigma 0.5` | ES 复算 α |

输出 `<root>/attack_evidence.json` 与 `attack_evidence.md`: 每类的
单元数/已尝试/检出/被采纳(攻击成功)/拒绝率/不适用数/检出阶段直方图, 以及
每个 run 的验证耗时与通信量; clean control 报误拒率。验收判定:
实际注入的适用攻击单元达到目标、每个成员攻击均有覆盖、子进程成功、
**攻击拒绝率 100% 且 clean 数量达标且误拒率 0%**。

## 4. 指标计量口径 (main_point2 新增)

- **执行/采纳更新数**: `group_packages_sealed` (ES 封装组包数 = 组级训练执行数),
  `user_updates_executed` (成功上传的用户本地更新数), `adopted_packages` /
  `adopted_user_updates` (进入已提交 checkpoint 的部分); 执行与采纳之差即
  "算了但未被采纳"的浪费量。
- **验证耗时**: `verify_seconds_total` 与逐候选 `verify_seconds_per_checkpoint`,
  覆盖 ES 验证 + vote + TA 授权 + 提交事务 (不含本地训练与评测)。
- **通信量**: `comm_total_bytes` 与 `comm_bytes_by_type`, 按协议消息字段计算
  (**不含 TLS/序列化/帧头开销**), 口径见 `main_point2._CommAccount`:
  `upload = 8B×dim`; `package = 8B×dim + 33 + 64 + 48×n_users`;
  `manifest = 16B×dim + (33+64+48)×|Q|`; `vote = 256B×|ES|`; `receipt = 320B`;
  `notify = 128B×(1+|ES|)`。

## 5. 边界与已知限制

- 本流水线不新增协议功能; 事务恢复只覆盖进程内 (跨进程恢复需 ES/TA 共识状态
  持久化, 当前 ES 状态为内存态, 遗留 journal 一律 fail-closed 丢弃)。
- 攻击实验强制 `--verify` (fail-fast); 旧 runner 中 `--attack` 且无 `--verify`
  的任务已移除。
- 主表所有单元必须同代码版本、同配置、5 seeds、`--deterministic`; 旧批次
  (`results_*`、`ASIG_*`) 只作探索记录, 不进入论文统计。
- 精度—效率图中的本机时间仅用于说明计算成本, 不作为分布式墙钟加速证据。
