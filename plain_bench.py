"""明文逐客户端基线仿真器 (盲审补全执行计划 阶段 1).

目的: 为论文竞争基线表提供同环境、同数据、同延迟分布的 FedBuff / MAFSA /
DVAFL 明文实测, 替换正文中"四个基线均未在本实验框架内完成同环境复现"的
缺口声明 (thesis 补全计划阶段 2, 90 runs)。

与冻结链路 (main_point2.py) 的关系 —— 本文件是全新明文支路, 不改动任何
冻结文件:
  - 复用 datasets.distribute_data (同调用序列: np.random.seed + torch
   .manual_seed 之后), 保证同 (dataset, seed, iid, alpha) 数据划分与
    main_point2 / fedavg_point2 完全一致;
  - 复用 models.get_model / train_local / evaluate / flatten_parameters
    (SGD momentum=0.9, NLLLoss), 本地训练语义与冻结链路一致;
  - 延迟/掉线随机流复刻 main_point2.py:483 (default_rng(seed)) 与
    :531-548 的生成方式 (同种子同分布), 在其上生成逐客户端到达序列;
  - npz/meta 输出契约复刻 main_point2.py:1662-1808, 保证 run_thesis_core
    口径的 manifest 字段 (arr_digest/final_acc/wall_clock/checkpoint_Ns
    等) 可被 thesis_stats / run_plain_bench 直接消费。

三臂语义 (忠实适配, 逐条依据见 docs/PLAIN_BENCH_ADAPTATIONS.md):
  - fedbuff: 逐客户端 buffer, K 到达触发; 聚合权重 α_i=1/(wait_i+1)^σ
    (main_point2.py:793-796 wait 口径复用); 无补偿。
  - mafsa: Algorithm-1 掉线语义 = offline-train-and-cache (user.py:165-194
    同语义明文复刻); 上线时 w = α·w_new + (1−α)·w_stale, α=exp(−β·E_i)
    (user.py:306-310 同式; MAFSA Eq.(10)(11)); 服务端样本量加权平均。
  - dvafl: VCM 补偿 Δ̃_i = (m̂_i/M̂)·γ·(W(v)−W(r_i)) (DVAFL Eq.(13)),
    m̂_i/M̂ 为一阶矩 EMA + Adam 式偏差校正 (Eq.(15)(16)(19); moment2 用
    Eq.(17)(18)); 聚合 Eq.(3) 样本量加权 (DVAFL 明确不做陈旧度降权)。

公平性口径 (阶段 1.6, 写入论文): ours_plain 走冻结仿真器组级轨迹
(verify-off G3v); 本文件三个基线为逐客户端轨迹 —— 同种子、同延迟分布、
同数据划分, 但调度器与聚合粒度是各方法自己的 (这正是被比较的对象)。
共享的系统级约束: 缓冲时效 s_max (超期更新丢弃并计数) 对三臂一视同仁。

用法:
  python plain_bench.py --arm fedbuff --dataset MNIST --seed 42
  python plain_bench.py --arm mafsa --dataset CIFAR10 --noniid --alpha 0.5
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, copy, hashlib, json, logging, math, time
import platform, subprocess
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

import models
from datasets import load_data, distribute_data
from main_point2 import DATASET_CONFIGS

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EDGE_SERVERS = 5
USERS_PER_EDGE = 4

_ARTIFACT_FIELDS = {'run_id', 'output_dir', 'tag', 'force'}

# 臂默认超参 (来源见 PLAIN_BENCH_ADAPTATIONS.md; CLI 可覆盖)
ARM_DEFAULTS = {
    'fedbuff': {'wait_sigma': 0.5},
    'mafsa': {'staleness_beta': 0.3},
    'dvafl': {'gamma_vcm': 1.0, 'moment_mode': 'moment1',
              'moment_lambda': 0.9},
}


# ============================================================
#  纯聚合函数 (单测锚点: tests/test_plain_bench.py 手算对拍)
# ============================================================

@dataclass
class PlainUpdate:
    """一条到达服务端的客户端更新 (明文)."""
    user_id: str
    user_idx: int
    base_version: int          # 训练开始时的全局版本 r_i
    flat: np.ndarray           # 训练后完整模型 w_i (float64)
    delta_norm: float          # ‖w_i − W(r_i)‖
    S_i: int                   # 本地样本量 |D_i|
    tick_arrived: int


def fedbuff_aggregate(updates: list, alphas: list) -> np.ndarray:
    """FedBuff 聚合: 等待时间公平加权平均 W = Σ α_i·S_i·w_i / Σ α_i·S_i.

    α_i = 1/(wait_i+1)^σ (main_point2.py:793-796 wait 口径复用);
    σ=0 时退化为标准样本量加权平均。
    """
    num = np.zeros_like(updates[0].flat, dtype=np.float64)
    denom = 0.0
    for u, a in zip(updates, alphas):
        num += a * u.S_i * u.flat
        denom += a * u.S_i
    return num / max(denom, 1e-12)


def mafsa_blend(new_flat: np.ndarray, stale_flat: np.ndarray,
                E_i: int, beta: float) -> np.ndarray:
    """MAFSA 复用混合: w = α·w_new + (1−α)·w_stale, α = exp(−β·E_i).

    与 user.py:306-310 (reuse_rule='mafsa') 同式同方向: α 衰减 fresh,
    E_i = 当前训练基版本 − 缓存版本 (user.py:286 同口径)。
    """
    alpha_i = math.exp(-beta * E_i)
    return alpha_i * new_flat + (1.0 - alpha_i) * stale_flat


def dvafl_moment_update(m_prev, count_prev, delta_norm, lam):
    """DVAFL Eq.(15): m_i = λ·m_i + (1−λ)·‖g_i‖; 返回 (m_i, 参与计数+1)."""
    m = ((1.0 - lam) * delta_norm if m_prev is None
         else lam * m_prev + (1.0 - lam) * delta_norm)
    return m, count_prev + 1


def dvafl_bias_correct(m, count, lam):
    """DVAFL Eq.(19): m̂ = m / (1 − λ^t); count<=0 或 m 为空时原样返回."""
    if m is None or count <= 0:
        return m
    denom = 1.0 - (lam ** count)
    return m / denom if denom > 1e-12 else m


def dvafl_global_moment_update(M_prev, count_prev, drift_norm, lam):
    """DVAFL Eq.(16): M = λ·M + (1−λ)·‖G‖; ‖G‖=‖W(v)−W(v−1)‖."""
    M = ((1.0 - lam) * drift_norm if M_prev is None
         else lam * M_prev + (1.0 - lam) * drift_norm)
    return M, count_prev + 1


def dvafl_compensation(flat: np.ndarray, base_version: int, version: int,
                       history: dict, m_hat, M_hat, gamma: float,
                       moment_mode: str) -> tuple:
    """DVAFL Eq.(13) VCM 补偿: Δ̃_i = (m̂_i/M̂)·γ·(W(v)−W(r_i)).

    moment_mode='moment1' 用一阶矩比值 (Eq.(15)(16));
    'moment2' 用 (m̂/M̂)² (Eq.(17)(18), 与冻结链路 cloud.py
    vcm_mode='moment2' 的 ratio² 口径一致);
    'direct' 比值恒 1 (Eq.(14), 即 Direct-VCM)。
    M̂/m̂ 不可用 (冷启动) 时退化为 Direct (与冻结链路口径一致)。
    返回 (校正后模型 w̃_i, ‖C_i‖)。
    """
    if base_version >= version or base_version not in history:
        return flat.copy(), 0.0
    G = history[version] - history[base_version]
    if (moment_mode == 'direct' or M_hat is None or M_hat <= 0.0
            or m_hat is None or m_hat <= 0.0):
        ratio = 1.0
    else:
        ratio = min(2.0, max(0.0, m_hat / M_hat))
        if moment_mode == 'moment2':
            ratio *= ratio
    C = ratio * gamma * G
    return flat + C, float(np.linalg.norm(C))


def dvafl_aggregate(corrected_flats: list, sizes: list) -> np.ndarray:
    """DVAFL Eq.(3) 聚合: w^{t+1} = Σ p_i·w̃_i, p_i = |D_i|/Σ|D_i| (无降权)."""
    total = sum(sizes)
    num = np.zeros_like(corrected_flats[0], dtype=np.float64)
    for w, s in zip(corrected_flats, sizes):
        num += s * w
    return num / max(total, 1)


# ============================================================
#  逐客户端事件驱动仿真
# ============================================================

@dataclass
class _Client:
    idx: int
    uid: str
    loader: object
    S_i: int
    delay: int
    model: object                       # 本地模型 (训练后状态保留, MAFSA 离线续训用)
    state: str = 'idle'                 # idle | running
    base_version: int = 0               # 当前训练轮的基版本 r_i
    ready_tick: int = 0
    # MAFSA offline-train-and-cache 状态 (user.py 同语义)
    stale_flat: np.ndarray | None = None
    stale_version: int = -1
    # DVAFL 一阶矩状态 (Eq.(15))
    m_i: float | None = None
    m_i_count: int = 0


def run_plain_simulation(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    if getattr(args, 'model', ''):
        cfg = dict(cfg, model=args.model)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- 产物管理与 config_hash (契约同 main_point2.py:140-177) ---
    _raw_payload = {k: (v if isinstance(v, (int, float, str, bool,
                                            type(None))) else str(v))
                    for k, v in vars(args).items()}
    scientific_config = {k: v for k, v in _raw_payload.items()
                         if k not in _ARTIFACT_FIELDS}
    artifact_config = {k: _raw_payload[k] for k in _ARTIFACT_FIELDS
                       if k in _raw_payload}
    config_hash = hashlib.sha256(json.dumps(
        scientific_config, sort_keys=True, ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:16]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = args.run_id or (f"{args.arm}_{args.dataset}_{timestamp}")
    out_npz = os.path.join(out_dir, f"point2_data_{run_id}.npz")
    out_meta = os.path.join(out_dir, f"point2_meta_{run_id}.json")
    if os.path.exists(out_npz) and not args.force:
        old_hash = None
        if os.path.exists(out_meta):
            try:
                with open(out_meta, 'r', encoding='utf-8') as f:
                    old_hash = json.load(f).get('config_hash')
            except Exception:
                old_hash = None
        if old_hash == config_hash:
            print(f"[skip] 同 config_hash ({config_hash}) 结果已存在, "
                  f"跳过: {out_npz} (--force 覆盖)")
            return
        print(f"[abort] 目标已存在且配置不同 (old={old_hash}, new={config_hash}):"
              f" {out_npz} — 换 --run-id 或加 --force")
        sys.exit(2)

    # --- 确定性开关 (与 main_point2 --deterministic 同口径) ---
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            raise SystemExit(f"--deterministic 严格模式初始化失败: {e}")

    # --- 日志 ---
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    log_name = os.path.join(out_dir, f"point2_log_{run_id}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_name, 'w', 'utf-8'),
                  logging.StreamHandler()])
    logger.info(f"=== 明文基线仿真 arm={args.arm} ===")
    logger.info(f"Config: {args}")

    # --- 数据准备 (调用序列与 main_point2.py:179-253 一致) ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds = load_data(args.dataset)
    total_users = NUM_EDGE_SERVERS * USERS_PER_EDGE
    train_loaders = distribute_data(
        train_ds,
        total_users,
        cfg['batch_size'],
        iid=not args.noniid,
        max_samples_per_user=cfg['samples_per_user'],
        dirichlet_alpha=args.alpha,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    init_model = models.get_model(args.dataset, cfg['model'])
    init_flat, model_meta = models.flatten_parameters(init_model)
    dim = init_flat.shape[0]

    # --- 内存埋点 (阶段 1.3: 进程级峰值 RSS; per-role tracemalloc 于阶段 4.1) ---
    import psutil
    _proc = psutil.Process()
    peak_rss_mb = float(_proc.memory_info().rss) / (1024 * 1024)

    # --- 延迟生成 (复刻 main_point2.py:483 + 531-548, 同种子同分布) ---
    # frozen 链路非 group-by-speed 分支对每 edge 抽 1 个延迟; 逐客户端粒度
    # 需要逐客户端延迟, 故采用 group_by_speed 分支 (:535) 的逐用户抽取方式:
    # 同一 default_rng(seed)、同一 [min_delay, max_delay] 均匀分布。
    rng = np.random.default_rng(args.seed)
    user_delays = [int(rng.integers(args.min_delay, args.max_delay + 1))
                   for _ in range(total_users)]
    # group-by-speed: 复刻 main_point2.py:537-544 的排序语义 (延迟多重集
    # 不变, 只有"快客户端先到"的次序效应; 逐客户端粒度无组包, 不重分组)
    if args.group_by_speed:
        user_delays = sorted(user_delays)
    logger.info(f"逐客户端延迟 (ticks): {user_delays}")

    # --- 客户端初始化 ---
    clients: list[_Client] = []
    for i in range(total_users):
        clients.append(_Client(
            idx=i, uid=f"U-{i}", loader=train_loaders[i],
            S_i=len(train_loaders[i].dataset), delay=user_delays[i],
            model=copy.deepcopy(init_model)))

    # --- 服务端状态 ---
    version: int = 0
    history: dict[int, np.ndarray] = {0: init_flat.astype(np.float64).copy()}
    buffer: list[PlainUpdate] = []
    user_offline_left: dict[str, int] = {c.uid: 0 for c in clients}
    adoption_counts = [0] * total_users   # 各客户端更新被聚合采纳次数

    # --- 指标 (npz 契约字段, 契约同 main_point2.py:505-523) ---
    p2_metrics = {
        'checkpoint_ticks': [],
        'wait_times': [],
        'stale_reuse_counts': [],
        'comp_gain': 0.0,
        'expired_buffer_count': 0,
        'duplicate_contributions': 0,
        'throughput': [],
        'verification_attempts': 0,
        'verification_rejections': 0,
        'replay_attempts': 0,
        'replay_rejections': 0,
        'verify_seconds_list': [],
        'user_updates_executed': 0,
        'group_packages_sealed': 0,
        'checkpoint_attempts': 0,
        'attack_outcomes': [],
    }
    ckpt_Ns: list[int] = []               # 每次聚合采纳的更新数 (N_Q 对应)
    ckpt_Ss: list[int] = []               # 每次聚合采纳的样本量 (S_Q 对应)
    # 明文通信记账 (口径见 PLAIN_BENCH_ADAPTATIONS.md: 上传/下发按
    # float32 4B/元素计; 协议侧 int64 8B/元素, 不可直接混比)
    comm_bytes = {'upload': 0, 'broadcast': 0}

    # DVAFL 全局矩状态 (Eq.(16))
    M_hat: float | None = None
    M_count: int = 0
    # mafsa 复用人次: 自上次聚合以来发生的混合事件数 (stale_reuse_counts 口径)
    pending_reuse = 0

    current_tick = 0
    checkpoint_count = 0
    acc_history: list[tuple[int, float]] = []
    wc_start = time.time()

    logger.info(f"系统初始化完成: {total_users} 用户 (arm={args.arm}, "
                f"buffer_K={args.buffer_k}, s_max={args.s_max})")

    while (checkpoint_count < args.max_checkpoints
           and current_tick < args.max_ticks):
        current_tick += 1

        # 0. 启动空闲客户端 (含首 tick 的全体 kickoff): 基=当前版本,
        # ready=tick+delay (与冻结链路 4a "对 idle edge 启动新 package"
        # 的启动语义同构, 粒度为客户端)
        for c in clients:
            if c.state == 'idle':
                c.state = 'running'
                c.base_version = version
                c.ready_tick = current_tick + c.delay

        # 1. 到达处理: 本 tick 就绪的客户端 (按客户端序号, 确定序)
        for c in clients:
            if c.state != 'running' or current_tick < c.ready_tick:
                continue
            # 掉线判定 (main_point2.py:640-646 同口径: 逐轮独立掷硬币 +
            # burst 状态机)
            if user_offline_left[c.uid] > 0:
                user_offline_left[c.uid] -= 1
                dropped = True
            else:
                dropped = rng.random() < args.dropout_rate
                if dropped and args.dropout_burst > 0:
                    user_offline_left[c.uid] = args.dropout_burst

            if dropped:
                if args.arm == 'mafsa':
                    # MAFSA Algorithm-1 掉线分支: 本地旧模型上继续训练并缓存
                    # (user.py:165-194 p2_train_offline_and_cache 同语义;
                    # 掉线多轮在缓存之上累积续训; 不下载全局模型)
                    if c.stale_flat is not None:
                        models.unflatten_parameters(c.model, c.stale_flat,
                                                    model_meta)
                    models.train_local(c.model, c.loader,
                                       cfg['local_epochs'], cfg['lr'])
                    stale, _ = models.flatten_parameters(c.model)
                    c.stale_flat = stale.astype(np.float64).copy()
                    c.stale_version = c.base_version
                    logger.debug(f"Tick {current_tick}: {c.uid} 掉线, "
                                 f"离线续训缓存 (base v{c.base_version})")
                # 非 mafsa 臂: 掉线即跳过 (main_point2 --dropout-mode skip
                # 基线语义), 本轮无产出
            else:
                # 在线: 下载当前全局模型为锚 (通信: broadcast), 本地训练
                models.unflatten_parameters(c.model,
                                            history[c.base_version],
                                            model_meta)
                comm_bytes['broadcast'] += 4 * dim
                models.train_local(c.model, c.loader,
                                   cfg['local_epochs'], cfg['lr'])
                new_flat, _ = models.flatten_parameters(c.model)
                new_flat = new_flat.astype(np.float64)
                blended = new_flat
                did_blend = False
                if (args.arm == 'mafsa' and c.stale_flat is not None
                        and c.stale_version >= 0
                        and c.base_version - c.stale_version > 0):
                    # MAFSA 复用混合 (user.py:306-310 同式); 成功上传才
                    # 计入复用人次 (与冻结链路 pending_stale_reuse_count
                    # 的"确认消费"口径一致)
                    blended = mafsa_blend(new_flat, c.stale_flat,
                                          c.base_version - c.stale_version,
                                          args.staleness_beta)
                    did_blend = True
                delta = blended - history[c.base_version]
                if rng.random() < args.upload_dropout_rate:
                    # 上传失败 (main_point2.py:737-739 同口径): 缓存**未混合**
                    # 的新模型为 stale 候选 (user.py:319 + 137-148 同语义)
                    if args.arm == 'mafsa':
                        c.stale_flat = new_flat.copy()
                        c.stale_version = c.base_version
                    logger.debug(f"Tick {current_tick}: {c.uid} 上传失败")
                else:
                    buffer.append(PlainUpdate(
                        user_id=c.uid, user_idx=c.idx,
                        base_version=c.base_version, flat=blended,
                        delta_norm=float(np.linalg.norm(delta)),
                        S_i=c.S_i, tick_arrived=current_tick))
                    p2_metrics['user_updates_executed'] += 1
                    comm_bytes['upload'] += 4 * dim
                    if args.arm == 'dvafl':
                        c.m_i, c.m_i_count = dvafl_moment_update(
                            c.m_i, c.m_i_count,
                            float(np.linalg.norm(delta)),
                            args.moment_lambda)
                    if args.arm == 'mafsa':
                        if did_blend:
                            pending_reuse += 1
                        c.stale_flat = None
                        c.stale_version = -1
                    logger.debug(f"Tick {current_tick}: {c.uid} 上传 "
                                 f"(base v{c.base_version})")

            # 无论掉线与否, 立即开始下一轮 (基=当前全局版本; 掉线客户端
            # 也重掷下一轮机会, 与逐包独立掉线口径一致)
            c.state = 'running'
            c.base_version = version
            c.ready_tick = current_tick + c.delay

        # 2. 缓冲时效检查 (共享系统约束, 三臂一致): v − r_i > s_max 丢弃
        if buffer:
            alive = []
            for u in buffer:
                if version - u.base_version > args.s_max:
                    p2_metrics['expired_buffer_count'] += 1
                    logger.debug(f"Tick {current_tick}: {u.user_id} 更新过期 "
                                 f"(s={version - u.base_version} > {args.s_max})")
                else:
                    alive.append(u)
            buffer = alive

        # 3. K 到达触发聚合 (FIFO 取前 K; 与冻结链路 fifo 策略一致)
        if len(buffer) >= args.buffer_k:
            selected = buffer[:args.buffer_k]
            buffer = buffer[args.buffer_k:]
            v_now = version
            stale_n = sum(1 for u in selected if u.base_version < v_now)
            if args.arm == 'fedbuff':
                alphas = []
                for u in selected:
                    wait = current_tick - u.tick_arrived
                    alphas.append(1.0 / (wait + 1.0) ** args.wait_sigma)
                final_model = fedbuff_aggregate(selected, alphas)
                comp_norms = {u.user_id: 0.0 for u in selected}
                reuse_n = stale_n
            elif args.arm == 'dvafl':
                # 全局矩先推进 (Eq.(16): 每次聚合对应一次全局迭代; 首次
                # 聚合无上一版本 → M̂ 冷启动, 补偿退化为 Direct)
                if v_now >= 1:
                    drift = float(np.linalg.norm(history[v_now]
                                                 - history[v_now - 1]))
                    M_hat, M_count = dvafl_global_moment_update(
                        M_hat, M_count, drift, args.moment_lambda)
                M_bc = dvafl_bias_correct(M_hat, M_count, args.moment_lambda)
                corrected, comp_norms = [], {}
                for u in selected:
                    c = clients[u.user_idx]
                    m_bc = dvafl_bias_correct(c.m_i, c.m_i_count,
                                              args.moment_lambda)
                    w_tilde, c_norm = dvafl_compensation(
                        u.flat, u.base_version, v_now, history,
                        m_bc, M_bc, args.gamma_vcm, args.moment_mode)
                    corrected.append(w_tilde)
                    comp_norms[u.user_id] = c_norm
                final_model = dvafl_aggregate(corrected,
                                              [u.S_i for u in selected])
                reuse_n = stale_n
            else:  # mafsa: 客户端已混合, 服务端样本量加权平均 (Eq.(10) 退化)
                num = np.zeros(dim, dtype=np.float64)
                den = 0
                for u in selected:
                    num += u.S_i * u.flat
                    den += u.S_i
                final_model = num / max(den, 1)
                comp_norms = {u.user_id: 0.0 for u in selected}
                reuse_n = pending_reuse   # 混合事件数即复用人次

            version += 1
            history[version] = final_model.astype(np.float64).copy()
            checkpoint_count += 1
            if args.arm == 'mafsa':
                pending_reuse = 0

            # 指标记录 (契约同 main_point2.py:1421-1434)
            p2_metrics['checkpoint_ticks'].append(current_tick)
            p2_metrics['stale_reuse_counts'].append(reuse_n)
            for u in selected:
                p2_metrics['wait_times'].append(
                    current_tick - u.tick_arrived)
                adoption_counts[u.user_idx] += 1
            p2_metrics['comp_gain'] += sum(comp_norms.values())
            prev_tick = (p2_metrics['checkpoint_ticks'][-2]
                         if len(p2_metrics['checkpoint_ticks']) > 1 else 0)
            S_Q = sum(u.S_i for u in selected)
            p2_metrics['throughput'].append(
                S_Q / max(1, p2_metrics['checkpoint_ticks'][-1] - prev_tick))
            p2_metrics['checkpoint_attempts'] += 1
            ckpt_Ns.append(len(selected))
            ckpt_Ss.append(S_Q)

            # 评测 (契约同 main_point2.py:1439-1442)
            eval_model = copy.deepcopy(init_model)
            models.unflatten_parameters(eval_model, final_model, model_meta)
            acc, _ = models.evaluate(eval_model, test_loader)
            acc_history.append((checkpoint_count, acc))

            logger.info(
                f"✓ Agg {checkpoint_count} (tick {current_tick}): "
                f"v{version - 1}→v{version} N_Q={len(selected)} "
                f"S_Q={S_Q} stale={stale_n} "
                f"comp={sum(comp_norms.values()):.4f} Acc={acc:.2f}%")

            peak_rss_mb = max(peak_rss_mb,
                              float(_proc.memory_info().rss) / (1024 * 1024))

    # --- 汇总 (契约同 main_point2.py:1481-1590) ---
    wc_elapsed = time.time() - wc_start
    avg_wait = (float(np.mean(p2_metrics['wait_times']))
                if p2_metrics['wait_times'] else 0.0)
    max_wait = (float(np.max(p2_metrics['wait_times']))
                if p2_metrics['wait_times'] else 0.0)
    # 冻结链路 stale_in_checkpoint 口径: 选中且 base 落后的更新数;
    # 明文臂以聚合时 selected 的 stale 数计 (与 stale_reuse_counts 同值,
    # mafsa 臂的复用混合同样计入 —— 其本质就是把陈旧缓存带进聚合)
    stale_in_checkpoint = int(sum(p2_metrics['stale_reuse_counts']))
    total_stale_reuses = int(sum(p2_metrics['stale_reuse_counts']))
    final_acc = acc_history[-1][1] if acc_history else 0.0
    fairness_var = float(np.var(adoption_counts))

    logger.info(f"\n=== 仿真完成: {checkpoint_count} aggregations, "
                f"{current_tick} ticks, 墙钟 {wc_elapsed:.1f}s ===")
    logger.info(f"Accuracy history: {[(c, round(a, 2)) for c, a in acc_history]}")
    logger.info(f"  最终准确率: {final_acc:.2f}%")
    logger.info(f"  平均等待: {avg_wait:.1f} ticks, 最大: {max_wait:.1f}")
    logger.info(f"  执行用户更新: {p2_metrics['user_updates_executed']}, "
                f"过期更新: {p2_metrics['expired_buffer_count']}, "
                f"采纳公平性方差: {fairness_var:.2f}")
    logger.info(f"  明文通信 (4B/元素计): upload="
                f"{comm_bytes['upload'] / 1024:.1f} KiB, broadcast="
                f"{comm_bytes['broadcast'] / 1024:.1f} KiB")

    # --- npz 落盘 (契约同 main_point2.py:1662-1739, 31 字段兼容) ---
    tmp_npz = out_npz[:-4] + '.tmp.npz'
    np.savez(
        tmp_npz,
        acc_history=np.array(acc_history),
        checkpoint_Ns=np.array(ckpt_Ns),
        checkpoint_Ss=np.array(ckpt_Ss),
        wait_times=np.array(p2_metrics['wait_times']),
        stale_reuse_counts=np.array(p2_metrics['stale_reuse_counts']),
        checkpoint_ticks=np.array(p2_metrics['checkpoint_ticks']),
        wall_clock=wc_elapsed,
        final_acc=final_acc,
        avg_wait=avg_wait,
        fairness_var=fairness_var,
        comp_gain=p2_metrics['comp_gain'],
        expired_buffer_count=p2_metrics['expired_buffer_count'],
        stale_in_checkpoint=stale_in_checkpoint,
        total_stale_reuses=total_stale_reuses,
        final_acc_ema=np.nan,
        ema_acc_history=np.array([]),
        final_acc_cma=np.nan,
        cma_acc_history=np.array([]),
        cma_beta_history=np.array([]),
        cma_risk_history=np.array([]),
        cma_stale_risk_history=np.array([]),
        cma_conflict_history=np.array([]),
        cma_mode='none',
        cma_update_seconds=0.0,
        cma_eval_seconds=0.0,
        participation_counts=np.array([]),
        debias_weights_float=np.array([]),
        debias_weights_int=np.array([]),
        debias_weight_ratio=np.array([]),
        participation_cv=np.array([]),
        label_tv=np.array([]),
        label_tv_eq=np.array([]),
        label_tv_cum=np.array([]),
        label_tv_cum_eq=np.array([]),
        cum_sel_weighted=np.array([]),
        cum_sel_equal=np.array([]),
        cum_weight_quality=np.array([]),
        cum_weight_cv=np.array([]),
        selected_groups=np.array([]),
        weight_source='none',
        vcm_train_norm=np.array([]),
        vcm_comp_norm=np.array([]),
        vcm_trust_eta=np.array([]),
        vcm_trust_ratio=1.0,
        vcm_alignment=np.array([]),
        alignment_sigma=np.array([]),
        g5_shadow_acc_history=np.array([]),
        final_acc_g5_shadow=np.nan,
        verify_pass_count=0,
        verify_total_count=0,
        verification_rejection_count=0,
        duplicate_contributions=0,
        replay_attempts=0,
        replay_rejections=0,
        attack_detected=0.0,
        user_updates_executed=p2_metrics['user_updates_executed'],
        group_packages_sealed=0,
        checkpoint_attempts=p2_metrics['checkpoint_attempts'],
        adopted_packages=checkpoint_count,
        adopted_user_updates=int(p2_metrics['user_updates_executed']),
        verify_seconds_total=0.0,
        verify_seconds_per_checkpoint=np.array([]),
        comm_total_bytes=sum(comm_bytes.values()),
        comm_bytes_by_type=json.dumps(comm_bytes),
        comm_counts_by_type=json.dumps({}),
        attack_outcome_count=0,
        arm=args.arm,
        buffer_k=args.buffer_k,
        peak_rss_mb=peak_rss_mb,
        config=str(args),
    )
    os.replace(tmp_npz, out_npz)
    logger.info(f"数据已保存: {out_npz}")

    # --- meta 落盘 (契约同 main_point2.py:1743-1808) ---
    def _git_info():
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL).decode().strip()
            dirty = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL).strip())
        except Exception:
            return None, None
        return commit, dirty

    git_commit, git_dirty = _git_info()
    meta = {
        'run_id': run_id,
        'config_hash': config_hash,
        'args': _raw_payload,
        'git_commit': git_commit,
        'git_dirty': git_dirty,
        'dataset': args.dataset,
        'seed': args.seed,
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'deterministic': bool(args.deterministic),
        'timestamp': timestamp,
        'final_acc': float(final_acc),
        'wall_clock': float(wc_elapsed),
        'verify_pass': 0,
        'verify_total': 0,
        'attack': 'none',
        'attack_detected': 0.0,
        'attack_log': None,
        'attack_outcomes': 0,
        'user_updates_executed': int(p2_metrics['user_updates_executed']),
        'group_packages_sealed': 0,
        'checkpoint_attempts': int(p2_metrics['checkpoint_attempts']),
        'adopted_packages': int(checkpoint_count),
        'adopted_user_updates': int(p2_metrics['user_updates_executed']),
        'verify_seconds_total': 0.0,
        'comm_total_bytes': int(sum(comm_bytes.values())),
        'comm_bytes_by_type': dict(comm_bytes),
        'npz': os.path.basename(out_npz),
        'config_mode_version': 'repro-v1',
        'scientific_config': scientific_config,
        'artifact_config': artifact_config,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_version': (torch.version.cuda if torch.version.cuda else None),
        'cudnn_version': (str(torch.backends.cudnn.version())
                          if torch.backends.cudnn.is_available() else None),
        'torch_num_threads': torch.get_num_threads(),
        'commandline': ' '.join(sys.argv),
        'arr_digest': hashlib.sha256(
            np.asarray(acc_history, dtype=np.float64).tobytes()
        ).hexdigest()[:16],
        'arm': args.arm,
        'buffer_k': int(args.buffer_k),
        'peak_rss_mb': float(peak_rss_mb),
    }
    tmp_meta = out_meta + '.tmp'
    with open(tmp_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, out_meta)
    logger.info(f"运行元数据已保存: {out_meta}")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="明文逐客户端基线仿真 (fedbuff/mafsa/dvafl)")
    parser.add_argument('--arm', required=True, choices=list(ARM_DEFAULTS),
                        help='基线臂 (默认超参见 ARM_DEFAULTS 与 '
                             'PLAIN_BENCH_ADAPTATIONS.md)')
    parser.add_argument('--dataset', default='MNIST',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10'])
    parser.add_argument('--model', default='',
                        help='覆盖默认模型 (默认取 DATASET_CONFIGS; '
                             'Tiny 仅用于冒烟/单测)')
    parser.add_argument('--tag', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noniid', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--max-checkpoints', type=int, default=20)
    parser.add_argument('--max-ticks', type=int, default=300)
    parser.add_argument('--min-delay', type=int, default=1)
    parser.add_argument('--max-delay', type=int, default=5)
    parser.add_argument('--group-by-speed', action='store_true',
                        help='复刻 main_point2 同名开关的延迟序列语义 '
                             '(逐客户端粒度下仅影响延迟排序)')
    parser.add_argument('--buffer-k', type=int, default=12,
                        help='K 到达触发 (默认 12 ≈ 主方法每 checkpoint '
                             '聚合质量 N_Q 均值 11.2)')
    parser.add_argument('--s-max', type=int, default=3,
                        help='缓冲时效上限 (共享系统约束, 三臂一致)')
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--dropout-burst', type=int, default=0)
    parser.add_argument('--upload-dropout-rate', type=float, default=0.0)
    parser.add_argument('--staleness-beta', type=float,
                        default=ARM_DEFAULTS['mafsa']['staleness_beta'],
                        help='mafsa: β_c (α=exp(−β·E))')
    parser.add_argument('--wait-sigma', type=float,
                        default=ARM_DEFAULTS['fedbuff']['wait_sigma'],
                        help='fedbuff: 等待时间公平加权 σ (0=等权)')
    parser.add_argument('--gamma-vcm', type=float,
                        default=ARM_DEFAULTS['dvafl']['gamma_vcm'],
                        help='dvafl: VCM 补偿强度 γ')
    parser.add_argument('--moment-mode', default='moment1',
                        choices=['direct', 'moment1', 'moment2'],
                        help='dvafl: Eq.(14) direct / Eq.(15)(16) moment1 / '
                             'Eq.(17)(18) moment2')
    parser.add_argument('--moment-lambda', type=float,
                        default=ARM_DEFAULTS['dvafl']['moment_lambda'],
                        help='dvafl: 矩 EMA 系数 λ')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()
    if not (0.0 <= args.dropout_rate < 1.0):
        parser.error("--dropout-rate must be in [0, 1)")
    if not (0.0 <= args.upload_dropout_rate < 1.0):
        parser.error("--upload-dropout-rate must be in [0, 1)")
    if args.buffer_k < 1:
        parser.error("--buffer-k must be >= 1")
    if args.s_max < 0:
        parser.error("--s-max must be >= 0")

    run_plain_simulation(args)
