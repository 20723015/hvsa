"""
研究点2: 单TA 检查点式组级半异步安全聚合

对应文档 THESIS_POINT2_STALENESS_AWARE_REUSE.md

运行方式:
    python main_point2.py                     # 默认 MNIST (无 stale reuse, 无 VCM)
    python main_point2.py --stale-reuse       # 开启 stale reuse
    python main_point2.py --stale-reuse --vcm # 开启 stale reuse + Direct-VCM
    python main_point2.py --cma-mode risk     # RA-CMA 影子平均 (可选后处理组件)
    python main_point2.py --dataset CIFAR10   # CIFAR-10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, copy, hashlib, logging, time, math
import json, platform, subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
from utils import (QUANTIZE_SCALE, LHH, vpda_weights, AlignmentSigmaPolicy,
                   staleness_int_weights)
from point2_types import AttackOutcome, ProtocolConfig
from commit_tx import run_commit_transaction, recover_staged_commit
import models
from datasets import load_data, distribute_data

# ============================================================
#  配置
# ============================================================

DATASET_CONFIGS = {
    'MNIST': {
        'model': 'CNN', 'rounds': 30, 'local_epochs': 1,
        'lr': 0.01, 'samples_per_user': 350, 'batch_size': 64,
    },
    'FashionMNIST': {
        'model': 'CNN', 'rounds': 30, 'local_epochs': 1,
        'lr': 0.01, 'samples_per_user': 350, 'batch_size': 64,
    },
    'CIFAR10': {
        'model': 'CNN', 'rounds': 30, 'local_epochs': 5,
        'lr': 0.01, 'samples_per_user': 1500, 'batch_size': 64,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader_label_hist(loader, n_classes: int, normalize: bool = True):
    """从 DataLoader 提取标签分布 (V-PDA 标签 TV 诊断用).

    支持 torchvision Dataset (targets) 与 Subset; 提取失败返回 None。
    normalize=False 时返回原始计数 (累计 TV 用)。
    """
    try:
        ds = loader.dataset
        if hasattr(ds, 'indices'):            # Subset
            labels = np.asarray(ds.dataset.targets)[np.asarray(ds.indices)]
        elif hasattr(ds, 'targets'):
            labels = np.asarray(ds.targets)
        else:
            labels = np.asarray([y for _, y in ds])
        if labels.ndim > 1:                   # one-hot
            labels = labels.argmax(1)
        h = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(np.float64)
        return (h / max(1.0, h.sum())) if normalize else h
    except Exception:
        return None


# ============================================================
#  协议通信量记账 (论文需报告通信开销; 口径显式可复算)
# ============================================================

class _CommAccount:
    """按协议消息字段累计通信量 (bytes), 不含 TLS/序列化/帧头开销.

    口径 (dim = 模型维度, n_u = 组内在线用户数, n_q = |Q(v)|):
    - upload:   用户上传掩码更新 = dim × 8B (int64 掩码向量)
    - package:  组包上报 Cloud = dim × 8B (掩码和) + 33B (LHH 点)
                + 64B (Ed25519 签名) + 48B (token/base_hash 元数据) × n_u 计数项
    - manifest: Cloud 广播候选 = 2 × dim × 8B (raw sum + M_Q)
                + n_q × (33 + 64 + 48)B (H_k + 签名 + token/base 元数据)
    - vote:     ES → TA 签名投票 = 256B × |ES|
    - receipt:  TA → Cloud 提交凭证 = 320B
    - notify:   commit 通知 = 128B × (1 TA + |ES|)
    """

    KEYS = ('upload', 'package', 'manifest', 'vote', 'receipt', 'notify')

    def __init__(self):
        self.counts = {k: 0 for k in self.KEYS}
        self.bytes = {k: 0 for k in self.KEYS}

    def add(self, key: str, nbytes: int, count: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(count)
        self.bytes[key] = self.bytes.get(key, 0) + int(nbytes)

    def upload(self, dim: int) -> None:
        self.add('upload', 8 * dim)

    def package(self, dim: int, n_users: int) -> None:
        self.add('package', 8 * dim + 33 + 64 + 48 * max(1, n_users))

    def manifest(self, dim: int, n_q: int) -> None:
        self.add('manifest', 16 * dim + (33 + 64 + 48) * max(1, n_q))

    def vote(self, n_es: int) -> None:
        self.add('vote', 256 * max(1, n_es), count=max(1, n_es))

    def receipt(self) -> None:
        self.add('receipt', 320)

    def notify(self, n_es: int) -> None:
        self.add('notify', 128 * (1 + max(0, n_es)), count=1 + max(0, n_es))

    def total_bytes(self) -> int:
        return int(sum(self.bytes.values()))

    def to_dict(self) -> dict:
        return {'total_bytes': self.total_bytes(),
                'bytes': dict(self.bytes), 'counts': dict(self.counts)}


# ============================================================
#  离散事件模拟
# ============================================================

def run_point2_simulation(args: argparse.Namespace):
    cfg = DATASET_CONFIGS[args.dataset]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- 统一实验接口: 产物管理与 config_hash ---
    # 配置分离 (可复现性): config_hash 只针对**科学配置** (数据/种子/模型/
    # 训练/协议/攻击参数) —— 换 run_id/output_dir/tag/force/ta_state_path
    # 不应改变"同一实验"的判定; 产物参数单列 ArtifactConfig 存入元数据。
    _ARTIFACT_FIELDS = {'run_id', 'output_dir', 'tag', 'force',
                        'ta_state_path'}
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
    args_payload = _raw_payload
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = args.run_id or (f"{args.tag or args.dataset}_{timestamp}")
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
        # hash 不同 → 禁止静默覆盖 (异配置结果必须显式处理)
        print(f"[abort] 目标已存在且配置不同 (old={old_hash}, new={config_hash}):"
              f" {out_npz} — 换 --run-id 或加 --force")
        sys.exit(2)

    # 可复现性: 播种全局 RNG (数据划分) 与 torch (本地训练/DataLoader)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 日志 ---
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    # --tag: 并行批量实验时区分产物文件 (series_seed 等), 避免同秒覆盖
    log_name = os.path.join(
        out_dir,
        (f'point2_log_{run_id}.txt'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_name, 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # 复现性: CPU 路径由 OMP/MKL=1 保证; GPU 路径需显式开启确定性算法
    # (cudnn 卷积反向默认含非确定性原子累加, CIFAR 级实验会逐次漂移 ~0.5pp)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 严格确定性: 全局确定性算法开关 (不 warn_only, 不用第二次调用覆盖),
        # 不支持的算子直接失败
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            raise SystemExit(f"--deterministic 严格模式初始化失败: {e}")
        logger.info("确定性模式已开启 (严格模式: "
                    "cudnn.deterministic=True, benchmark=False, "
                    "use_deterministic_algorithms(True))")
    elif DEVICE.type == "cuda":
        logger.warning("未开启 --deterministic 且使用 GPU: CIFAR 级训练可能逐次漂移 "
                       "(同 seed 两次运行的准确率差可达 ~0.5pp), "
                       "跨批次同 seed 配对将不可靠")

    logger.info(f"=== HVSA 研究点2: 半异步安全聚合 ===")
    logger.info(f"Config: {args}")

    # --- 影子平均模式解析 (RA-CMA, 可选后处理组件, 非创新点) ---
    # --cma-mode 显式优先; --ema-decay 保留为历史兼容别名, 映射到 fixed 模式
    # (同时修正旧实现错误: 影子初始化为 W_1 后不应再做 Adam 式偏差校正除 (1−a^t))
    cma_mode = args.cma_mode
    cma_beta_fixed = args.cma_beta
    if cma_mode == 'none' and args.ema_decay > 0:
        cma_mode = 'fixed'
        cma_beta_fixed = args.ema_decay
        logger.info(f"--ema-decay {args.ema_decay} → 修正后固定影子平均 "
                    f"(mode=fixed, β={cma_beta_fixed}, 无偏差校正)")
    elif cma_mode != 'none' and args.ema_decay > 0:
        logger.warning(f"--cma-mode={cma_mode} 与 --ema-decay={args.ema_decay} "
                       f"同时给定, 忽略 ema-decay")
    if cma_mode != 'none':
        logger.info(f"影子平均启用: mode={cma_mode} "
                    + (f"β={cma_beta_fixed}" if cma_mode == 'fixed' else
                       f"β∈[{args.cma_beta_min},{args.cma_beta_max}] "
                       f"λs={args.cma_lambda_stale}"))

    # --- 1. 数据准备 ---
    train_ds, test_ds = load_data(args.dataset)
    total_users = args.num_edge_servers * args.users_per_edge
    train_loaders = distribute_data(
        train_ds,
        total_users,
        cfg['batch_size'],
        iid=not args.noniid,
        max_samples_per_user=cfg['samples_per_user'],
        dirichlet_alpha=args.alpha,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # V-PDA/UR-VCM 诊断: 每组 (edge) 标签分布 + 每用户原始标签计数
    # (累计 TV 用"实际参与用户"的原始计数, 而非组归属的归一化分布)
    user_label_counts: dict[str, np.ndarray] = {}
    global_label_counts = None
    if args.participation_debias == 'ipw' or args.vcm_mode == 'trust':
        try:
            n_classes = (len(test_ds.classes) if hasattr(test_ds, 'classes')
                         else int(np.asarray(test_ds.targets).max()) + 1)
        except Exception:
            n_classes = 10
        edge_label_hists = {}
        acc_hist = np.zeros(n_classes, dtype=np.float64)
        counts_acc = np.zeros(n_classes, dtype=np.float64)
        n_ok = 0
        for i in range(total_users):
            h = _loader_label_hist(train_loaders[i], n_classes)
            c = _loader_label_hist(train_loaders[i], n_classes, normalize=False)
            if h is None or c is None:
                continue
            eid = f"edge_{i // args.users_per_edge}"
            edge_label_hists[eid] = edge_label_hists.get(eid, 0.0) + h
            acc_hist += h
            counts_acc += c
            user_label_counts[f"U-{i}"] = c
            n_ok += 1
        if n_ok == total_users:
            global_label_hist = acc_hist / acc_hist.sum()
            for eid in edge_label_hists:
                edge_label_hists[eid] = edge_label_hists[eid] / edge_label_hists[eid].sum()
            global_label_counts = counts_acc
        else:
            edge_label_hists = None
            global_label_hist = None
            user_label_counts = {}
            logger.warning("标签分布提取失败, TV 偏移诊断关闭")

    init_model = models.get_model(args.dataset, args.model or cfg['model'])
    init_flat, model_meta = models.flatten_parameters(init_model)
    dim = init_flat.shape[0]

    # --- 2. 系统初始化 ---
    # TA 权威 ledger: 默认内存态 (每次运行独立); --ta-state-path 给定则持久化
    # 到 SQLite —— prepared_checkpoints / finalize 记录跨进程可核对审计
    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01",
                          enable_verification=args.verify,
                          ledger_path=(args.ta_state_path or None))
    ta.point2_reset()
    cloud = CloudServer()
    # staged checkpoint journal (P0-2): "提交成功要应用的变更"先落盘, 提交
    # 事务中途异常可凭它恢复完成 (finalize/publish/notify 均幂等)。正常
    # 提交完成后 journal 被清理。
    _stage_journal = os.path.join(out_dir, f"point2_stage_{run_id}.journal")
    cloud.point2_reset(init_flat, initial_version=0,
                       stage_path=_stage_journal)
    cloud._vcm_curve_lambda = args.vcm_curve_lambda  # 曲率感知 VCM 系数
    cloud._vcm_trust_ratio = args.vcm_trust_ratio    # UR-VCM 可信域比率 ρ

    # ES 端策略绑定配置: 与云端同一份协议参数 (γ/mode/momentum/σ/度量/策略),
    # 使 ES 能独立复算权重与补偿系数并与云端声明值比对。
    # 生效 γ 与聚合路径有关: σ-agg (非 V-PDA) 路径 γ 恒为 0 (无补偿)。
    eff_gamma = args.gamma_vcm if args.vcm else 0.0
    if args.participation_debias != 'ipw' and args.sigma_target == 'agg':
        eff_gamma = 0.0
    # 版本化协议配置: TA 与全部 ES 预置同一实例, digest 进入 checkpoint
    # manifest (bundle), 云端声明值只作审计。
    protocol_config = ProtocolConfig(
        K=args.K, s_max=args.s_max,
        participation_debias=args.participation_debias,
        sigma_target=args.sigma_target,
        staleness_sigma=args.staleness_sigma,
        staleness_metric=args.staleness_metric,
        sigma_policy=args.sigma_policy,
        sigma_alignment_threshold=args.sigma_alignment_threshold,
        sigma_alignment_ema=args.sigma_alignment_ema,
        sigma_alignment_warmup=args.sigma_alignment_warmup,
        sigma_alignment_boost_threshold=args.sigma_alignment_boost_threshold,
        sigma_alignment_high=args.sigma_alignment_high,
        gamma_vcm=eff_gamma, vcm_mode=args.vcm_mode,
        momentum=args.momentum, vcm_trust_ratio=args.vcm_trust_ratio,
        debias_kappa=args.debias_kappa, debias_min=args.debias_min,
        debias_max=args.debias_max, debias_scale=args.debias_scale,
    )
    policy_digest = protocol_config.digest()
    ta.p2_set_policy_config(protocol_config)
    # verify-on: Cloud 提交必须持 TA receipt (权威授权); verify-off 性能
    # 模式不注入公钥 (receipt 为 None, 兼容历史行为)
    if args.verify:
        cloud.p2_set_receipt_pub_key(ta.get_receipt_public_key())
    edges = []
    for i in range(args.num_edge_servers):
        es = EdgeServer(
            id=f"edge_{i}", cloud_server=cloud, ta=ta,
            enable_verification=args.verify,
            policy_config=protocol_config,
            debias_kappa=args.debias_kappa, debias_min=args.debias_min,
            debias_max=args.debias_max, debias_scale=args.debias_scale,
            vcm_trust_ratio=args.vcm_trust_ratio,
            gamma_vcm=eff_gamma, vcm_mode=args.vcm_mode,
            momentum=args.momentum,
            staleness_metric=args.staleness_metric,
            staleness_sigma=args.staleness_sigma,
            sigma_policy=args.sigma_policy,
            sigma_alignment_threshold=args.sigma_alignment_threshold,
            sigma_alignment_ema=args.sigma_alignment_ema,
            sigma_alignment_warmup=args.sigma_alignment_warmup,
            sigma_alignment_boost_threshold=args.sigma_alignment_boost_threshold,
            sigma_alignment_high=args.sigma_alignment_high,
            participation_debias=args.participation_debias,
            sigma_target=args.sigma_target,
        )
        edges.append(es)

    users = []
    user_edge_map: dict[str, EdgeServer] = {}
    user_loaders_map: dict[str, DataLoader] = {}
    ta_pub = ta.get_public_key_bytes()

    for i in range(total_users):
        u = User(f"U-{i}", copy.deepcopy(init_model))
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        u.point2_init_cache()
        users.append(u)
        edge = edges[i // args.users_per_edge]
        user_edge_map[u.id] = edge
        user_loaders_map[u.id] = train_loaders[i]

    # 上次运行崩溃遗留的 staged journal: 本进程的 ES/TA 共识状态是全新
    # 内存态, 凭旧 journal 跨进程完成提交会把 Cloud/TA 推到 ES 追不上的
    # 版本 (ES 无本地已验证候选, 拒绝推进) → fail-closed 丢弃并显式告警。
    # 进程内的提交崩溃由事务重试 + recover_staged_commit 收敛。
    if os.path.exists(_stage_journal):
        logger.warning(f"发现上次运行遗留的 staged journal {_stage_journal} "
                       f"(提交事务未完成); 跨进程恢复不安全 → 丢弃 (fail-closed)")
        cloud.p2_discard_staged(
            reason='startup: 遗留 journal, 跨进程恢复不安全')

    # --- 3. 模拟状态 ---
    current_tick: int = 0
    checkpoint_count: int = 0
    # 突发离线状态机: user_id → 剩余离线组包数 (dropout_burst>0 时启用)
    user_offline_left: dict[str, int] = {u.id: 0 for u in users}
    # 固定永久离线子集 (--permanent-dropout-frac f): 与 --dropout-burst 的
    # "每轮重掷 + 永久化" 不同, 这里在建模仿真开始时就确定哪 f 比例的设备
    # 永久离线 (真实部署语义: 那批设备就是连不上), 之后不再变化。
    if args.permanent_dropout_frac > 0:
        # 独立 RNG: 不消耗主随机流 (dropout/tick 序列), 保证与既有实验可比
        _perm_rng = np.random.default_rng(args.seed * 1000 + 7)
        _perm_offline = [u.id for u in users
                         if _perm_rng.random() < args.permanent_dropout_frac]
        for uid in _perm_offline:
            user_offline_left[uid] = 10 ** 9
        logger.info(f"固定永久离线: {len(_perm_offline)}/{total_users} 用户 "
                    f"(frac={args.permanent_dropout_frac}) {sorted(_perm_offline)}")
    wc_start_time = time.time()
    acc_history: list[tuple[int, float]] = []
    # 服务器端影子模型 (纯后处理, 不参与训练/下发, 零协议消息):
    #   W̃_t = β_t·W̃_{t-1} + (1−β_t)·W_t,  W̃_1 = W_1 (不做偏差校正)
    #   β_t: fixed=常数; stale/conflict/risk = βmin + (βmax−βmin)·clip(r_t,0,1)
    cma_acc_history: list[tuple[int, float]] = []
    cma_beta_history: list[tuple[int, float]] = []
    cma_risk_history: list[tuple[int, float]] = []
    cma_stale_risk_history: list[tuple[int, float]] = []
    cma_conflict_history: list[tuple[int, float]] = []
    cma_flat: np.ndarray | None = None          # W̃_t
    cma_prev_delta: np.ndarray | None = None    # Δ_{t-1} = W_{t-1} − W_{t-2} (方向冲突用)
    cma_update_seconds = 0.0                    # 影子向量更新累计耗时 (O(d))
    cma_eval_seconds = 0.0                      # 影子模型额外评测累计耗时
    W_v = init_flat.astype(np.float64).copy()

    # V-PDA 诊断记录 (每轮 checkpoint 快照; 仅 --participation-debias ipw 启用)
    vpda_active = (args.participation_debias == 'ipw')
    participation_counts_history: list[list] = []   # 每轮 A_k(t) (commit 前快照)
    debias_weights_float_history: list[list] = []   # 每轮浮点权重 q_k
    debias_weights_int_history: list[list] = []     # 每轮整数权重 w_k
    weight_ratio_history: list[float] = []          # 每轮 max(w)/min(w)
    participation_cv_history: list[float] = []      # 每轮入选频率变异系数
    label_tv_history: list[float] = []              # 每轮加权选中样本标签分布 TV
    label_tv_eq_history: list[float] = []           # 每轮等权选中 TV (无去偏基线)
    # 累计 TV (实际参与用户的原始标签计数, 跨 checkpoint 累计)
    cum_w_counts = (np.zeros_like(global_label_counts)
                    if global_label_counts is not None else None)
    cum_eq_counts = (np.zeros_like(global_label_counts)
                     if global_label_counts is not None else None)
    label_tv_cum_history: list[float] = []          # 累计加权 TV
    label_tv_cum_eq_history: list[float] = []       # 累计等权 TV
    # 累计权重质量: 累计加权/等权选中次数 与 min/max 比、CV
    cum_sel_weighted: dict[str, float] = {e.id: 0.0 for e in edges}
    cum_sel_equal: dict[str, int] = {e.id: 0 for e in edges}
    cum_sel_weighted_history: list[list] = []
    cum_sel_equal_history: list[list] = []
    cum_weight_quality_history: list[float] = []    # min/max 比 (1=完全均匀)
    cum_weight_cv_history: list[float] = []
    selected_groups_history: list[list] = []        # 每轮实际参与组 (0/1)
    # UR-VCM 诊断
    vcm_train_norm_history: list[float] = []
    vcm_comp_norm_history: list[float] = []
    vcm_trust_eta_history: list[float] = []
    # DA-VCM 纯诊断（不改变训练轨迹）: 比较安全聚合后的整体训练增量
    # Δtrain=R-B 与原始 VCM 补偿均值 C 的方向。只有聚合模型、公开基版本
    # 和公开样本量参与计算，不读取单用户/单组明文更新。
    vcm_alignment_history: list[list[float]] = []
    # Aσ-VCM: 用聚合级方向一致性决定是否启用陈旧度衰减。EMA 只消费
    # 已恢复的聚合模型与公开版本历史；策略状态为两阶段 (tentative 决策
    # 用于本轮 α, commit 成功才落地), 与 ES 验证侧共享同一公开规则实现。
    sigma_alignment = AlignmentSigmaPolicy(
        sigma_fallback=args.staleness_sigma,
        sigma_high=args.sigma_alignment_high,
        ema_beta=args.sigma_alignment_ema,
        warmup=args.sigma_alignment_warmup,
        boost_threshold=args.sigma_alignment_boost_threshold,
        mid_threshold=args.sigma_alignment_threshold,
    ) if args.sigma_policy == 'alignment' else None
    alignment_sigma_history: list[list[float]] = []
    weight_source: str | None = None       # none | vpda | staleness
    g5_acc_history: list[tuple[int, float]] = []    # 无去偏 (等权 G5 等价) 影子评测
    g5_history: dict[int, np.ndarray] = {0: init_flat.copy()}  # 影子自洽版本历史
    # edge_label_hists / global_label_hist 在数据准备段 (line ~136) 初始化

    # 每个 edge 的状态: 'idle' | 'training'
    edge_state: dict[str, str] = {e.id: 'idle' for e in edges}
    # 每个 edge 的训练完成 tick
    edge_ready_tick: dict[str, int] = {e.id: 0 for e in edges}
    # 每个 edge 的 package_seq
    edge_pkg_seq: dict[str, int] = {e.id: 0 for e in edges}
    # 每个 edge 当前的 base_version
    edge_base_v: dict[str, int] = {e.id: 0 for e in edges}

    rng = np.random.default_rng(args.seed)

    # 边级临时离线区间 (--edge-offline-spans "0:20-60,2:50-90", tick 口径):
    # 模拟"某条 edge 链路成段不可用"的真实暂时断连 (整组掉线, 会恢复)。
    # 与逐用户 i.i.d. 掉线不同, 它会让连续若干 checkpoint 缺失一整个数据区域。
    edge_offline_spans: dict[str, list] = {}
    for _span in (args.edge_offline_spans or "").split(","):
        _span = _span.strip()
        if not _span:
            continue
        try:
            _eidx, _rng = _span.split(":")
            _lo, _hi = _rng.split("-")
            edge_offline_spans.setdefault(f"edge_{int(_eidx)}", []).append(
                (int(_lo), int(_hi)))
        except Exception:
            raise SystemExit(f"--edge-offline-spans 格式错误: {_span!r} "
                             f"(应为 'edge索引:起tick-止tick', 逗号分隔)")
    if edge_offline_spans:
        logger.info(f"边级临时离线区间 (tick): {edge_offline_spans}")

    # 指标追踪
    p2_metrics = {
        'checkpoint_ticks': [],       # 每个 checkpoint 完成时的 tick
        'wait_times': [],             # 每个 selected 组包的等待时间
        'stale_reuse_counts': [],     # 每个 checkpoint 中复用了 stale 的客户端数
        'comp_gain': 0.0,             # 累计 compensation 贡献
        'expired_buffer_count': 0,    # 从 buffer 因过期移除的组包总数
        'duplicate_contributions': 0, # 重复贡献数 (exactly-once 违反计数)
        'throughput': [],             # 每个 checkpoint 的样本量吞吐
        'verification_attempts': 0,   # 已进入 ES 验证的 checkpoint 候选数
        'verification_rejections': 0,# 验证失败而被丢弃的 checkpoint 数
        'replay_attempts': 0,         # 重放攻击注入次数
        'replay_rejections': 0,       # 重放被 Aggregated_IDs 拒绝次数
        # 论文主表/安全证据所需: 验证耗时、通信量、执行/采纳更新数
        'verify_seconds_list': [],    # 每个候选的 ES 验证+vote+授权耗时 (s)
        'user_updates_executed': 0,   # 实际执行的用户本地更新 (成功上传数)
        'group_packages_sealed': 0,   # ES 封装的组包数 (= 组级训练执行数)
        'checkpoint_attempts': 0,     # 进入验证的 checkpoint 尝试数
        'attack_outcomes': [],        # AttackOutcome 记录 (JSONL 落盘)
    }
    comm = _CommAccount()             # 协议通信量记账

    # 使用每个用户数据集的真实样本量；IID 下通常相同，Non-IID 下可能不同。
    for i, u in enumerate(users):
        u.p2_set_sample_size(len(train_loaders[i].dataset))

    # 提前分组: 给每个 edge 分配固定的 delay profile
    edge_delay: dict[str, int] = {}
    if args.group_by_speed:
        # 同质性能分组 (FedFleet/EAFL 思路): 用户按能力(速度)排序后均分到 edge,
        # 组内延迟相近 → 减少组内等待、压低组包陈旧度方差
        user_delays = [int(rng.integers(args.min_delay, args.max_delay + 1))
                       for _ in range(total_users)]
        order = sorted(range(total_users), key=lambda i: user_delays[i])
        for pos, i in enumerate(order):
            u = users[i]
            user_edge_map[u.id] = edges[pos // args.users_per_edge]
        for e_idx, e in enumerate(edges):
            grp = [user_delays[i] for i in order
                   if i // args.users_per_edge == e_idx]
            edge_delay[e.id] = int(np.median(grp)) if grp else 1
        logger.info(f"同质性能分组启用: 用户延迟序列={user_delays}")
    else:
        for e in edges:
            edge_delay[e.id] = int(rng.integers(args.min_delay, args.max_delay + 1))

    logger.info(f"系统初始化完成: {total_users} 用户, {args.num_edge_servers} ES")
    logger.info(f"ES delay profiles (ticks): {edge_delay}")

    # --- AttackOutcome 记录 (安全证据 JSONL) ---
    # --attack 实验自动落盘到 out_dir; --attack-log 显式指定时即使 --attack
    # none (clean control) 也逐 checkpoint 记录, 供拒绝率/误拒率统计
    if args.attack_log:
        attack_log_path = args.attack_log
    elif args.attack != 'none':
        attack_log_path = os.path.join(out_dir,
                                       f"attack_outcomes_{run_id}.jsonl")
    else:
        attack_log_path = None
    _attack_fh: list = [None]

    def _state_snapshot() -> dict:
        """三方状态摘要 (断言'失败不推进任何状态'与分叉检测)."""
        return {
            'cloud_version': int(cloud.p2_current_version),
            'ledger_version': int(ta.ledger.next_checkpoint_version()),
            'es_committed': [int(e._p2_committed_version) for e in edges],
            'adopted_packages': int(sum(len(r.selected_packages)
                                        for r in cloud.p2_checkpoint_log)),
        }

    def _emit_outcome(*, applicable: bool, attempted: bool, detected: bool,
                      accepted: bool, stage: str, reason: str,
                      before: dict) -> None:
        rec = AttackOutcome(
            attack=args.attack, requested=(args.attack != 'none'),
            applicable=bool(applicable), attempted=bool(attempted),
            detected=bool(detected), accepted=bool(accepted),
            detection_stage=stage, reason_code=reason,
            state_before=before, state_after=_state_snapshot()).to_dict()
        p2_metrics['attack_outcomes'].append(rec)
        if attack_log_path:
            if _attack_fh[0] is None:
                _attack_fh[0] = open(attack_log_path, 'w', encoding='utf-8')
            _attack_fh[0].write(json.dumps(rec, ensure_ascii=False) + '\n')
            _attack_fh[0].flush()

    # --- 4. 主循环 ---
    pending_stale_reuse_count = 0  # checkpoint 间累计，checkpoint 触发后清零
    while checkpoint_count < args.max_checkpoints and current_tick < args.max_ticks:
        current_tick += 1
        cloud._gradient_dim = dim  # ensure dim is set for raw sum recovery

        # 4a. 对 idle edge 启动新 package
        for e in edges:
            if edge_state[e.id] == 'idle':
                # 组包只有在成功提交或过期后才能被下一包替代。
                # 验证失败时保留原包，避免新包覆盖待重试候选。
                if cloud.p2_has_pending_package(e.id):
                    continue
                base_v = cloud.p2_current_version
                base_hash = cloud.p2_hash_current_model()
                pkg_seq = edge_pkg_seq[e.id]
                pkg_token = f"{e.id}:v{base_v}:p{pkg_seq}"
                edge_pkg_seq[e.id] += 1

                e.p2_start_package(
                    base_version=base_v,
                    base_hash=base_hash,
                    package_token=pkg_token,
                )
                edge_state[e.id] = 'training'
                edge_base_v[e.id] = base_v
                edge_ready_tick[e.id] = current_tick + edge_delay[e.id]

                logger.debug(f"Tick {current_tick}: {e.id} start {pkg_token}")

        # 4b. 处理完成的 package
        for e in edges:
            if edge_state[e.id] == 'training' and current_tick >= edge_ready_tick[e.id]:
                pkg_token = e._p2_package_token
                base_v = edge_base_v[e.id]
                base_hash = e._p2_base_hash

                # 用户训练 + stale reuse 混合 + 掩码上传
                online_count = 0
                stale_reuse_count = 0
                for u in users:
                    if user_edge_map[u.id] != e:
                        continue
                    # 掉线判定: dropout_burst>0 时为突发离线状态机 (掉线后持续
                    # B 个组包, 模拟真实设备离线航班; 0=逐包独立掷硬币)
                    if any(_s <= current_tick < _t
                           for _s, _t in edge_offline_spans.get(e.id, [])):
                        # 该 edge 当前处于临时离线区间: 整组掉线 (会恢复)
                        dropped = True
                    elif user_offline_left[u.id] > 0:
                        user_offline_left[u.id] -= 1
                        dropped = True
                    else:
                        dropped = rng.random() < args.dropout_rate
                        if dropped and args.dropout_burst > 0:
                            user_offline_left[u.id] = args.dropout_burst
                    if dropped:
                        # 掉线语义选择:
                        #  offline_train: MAFSA 忠实语义——本地旧模型上继续训练并缓存,
                        #                 恢复上线后由 reuse 混合规则回收成果 (算力回收)
                        #  skip: 掉线即跳过 (基线)
                        #  (--dropout-reuse rebase: 旧语义, 掉线=缓存顶上, 与 offline_train 互斥)
                        if args.dropout_mode == 'offline_train':
                            u.p2_train_offline_and_cache(
                                train_loader=user_loaders_map[u.id],
                                local_epochs=cfg['local_epochs'],
                                lr=cfg['lr'],
                                global_version=base_v,
                                model_meta=model_meta)
                            # 断网不断算 (offline_train + rebase 组合语义):
                            # 离线轮也用当前 package token 的新掩码旁路上传缓存成果
                            # (1 版陈旧, 用户自填坑 — SecAgg 下唯一干净的填坑方式),
                            # 模拟"网络中断但设备继续训练"的持续贡献场景
                            if (args.dropout_reuse == 'rebase'
                                    and u.p2_get_stale_info()[0]):
                                import pickle as _pkl
                                dropped_int = u.p2_make_dropout_reuse_masked(
                                    global_flat=cloud.p2_global_model_history[base_v],
                                    global_version=base_v,
                                    package_token=pkg_token,
                                )
                                if dropped_int is not None:
                                    e.p2_receive_masked_gradient(
                                        _pkl.dumps({"user_id": u.id,
                                                    "masked_gradient": dropped_int}),
                                        pkg_token, base_v,
                                        sample_size=len(user_loaders_map[u.id].dataset))
                                    online_count += 1
                                    stale_reuse_count += 1
                                    pending_stale_reuse_count += 1
                                    u.p2_confirm_consumed()
                        elif (args.dropout_reuse == 'rebase'
                                and args.reuse_rule != 'none'
                                and u.p2_get_stale_info()[0]):
                            import pickle as _pkl
                            dropped_int = u.p2_make_dropout_reuse_masked(
                                global_flat=cloud.p2_global_model_history[base_v],
                                global_version=base_v,
                                package_token=pkg_token,
                            )
                            if dropped_int is not None:
                                e.p2_receive_masked_gradient(
                                    _pkl.dumps({"user_id": u.id,
                                                "masked_gradient": dropped_int}),
                                    pkg_token, base_v,
                                    sample_size=len(user_loaders_map[u.id].dataset))
                                online_count += 1
                                stale_reuse_count += 1
                                pending_stale_reuse_count += 1
                                u.p2_confirm_consumed()
                        continue

                    loader = user_loaders_map[u.id]
                    reuse_rule = args.reuse_rule

                    # 陈旧度自适应学习率 (Async-SGD 风格, 可选): lr_i = lr/(1+λ·E_i)
                    # 组包基版本落后的用户本地步幅更小, 与服务器侧 σ 正交
                    lr_eff = cfg['lr']
                    if args.staleness_lr_lambda > 0:
                        has_stale, stale_ver = u.p2_get_stale_info()
                        if has_stale:
                            E_i = base_v - stale_ver
                            if E_i > 0:
                                lr_eff = cfg['lr'] / (1.0 + args.staleness_lr_lambda * E_i)

                    masked_int = u.p2_train_mix_mask_and_upload(
                        global_flat=cloud.p2_global_model_history[base_v],
                        global_version=base_v,
                        package_token=pkg_token,
                        model_meta=model_meta,
                        train_loader=loader,
                        local_epochs=cfg['local_epochs'],
                        lr=lr_eff,
                        staleness_beta=args.staleness_beta,
                        reuse_rule=reuse_rule,
                        rho_max=args.rho_max,
                        fedprox_mu=args.fedprox_mu,
                        curriculum=args.curriculum,
                        dpr_q=args.dpr_q,
                        clip_upd_k=args.clip_upd_k,
                    )

                    if masked_int is None:
                        continue

                    # 训练后上传丢包 (产生 stale 候选的来源)
                    if rng.random() < args.upload_dropout_rate:
                        u.p2_confirm_upload_failed()
                        continue

                    import pickle as _pkl
                    msg = {"user_id": u.id, "masked_gradient": masked_int}
                    e.p2_receive_masked_gradient(
                        _pkl.dumps(msg), pkg_token, base_v,
                        sample_size=len(user_loaders_map[u.id].dataset),
                        upd_norm=getattr(u, '_p2_last_upd_norm', 0.0))
                    online_count += 1
                    p2_metrics['user_updates_executed'] += 1
                    comm.upload(dim)
                    # ES 封组前确认接收 → 立即清除 stale 候选 (文档 L305-307)
                    has_stale, _ = u.p2_get_stale_info()
                    if has_stale:
                        pending_stale_reuse_count += 1
                    u.p2_confirm_consumed()

                # 封组 (n_min 隐私前提不足则不封组)
                if e._p2_online_users:
                    pkg = e.p2_seal_group_package(tick=current_tick)
                    if pkg is not None:
                        cloud.p2_receive_group_package(pkg, arrival_tick=current_tick)
                        p2_metrics['group_packages_sealed'] += 1
                        comm.package(dim, pkg.n_k)

                edge_state[e.id] = 'idle'

        # 4c. Checkpoint 触发
        selected = cloud.p2_select_checkpoint(
            K=args.K, s_max=args.s_max,
            current_tick=current_tick,
            delta_timeout=args.delta_timeout,
            policy=args.policy,
            D_fair=args.D_fair,
        )

        if selected:
            # 陈旧度权重 (T-FedHA 风格幂衰减, σ=0 关闭): α_k = 1/(s_k+1)^σ
            # 度量可选 (arXiv:2603.08211 陈旧度度量问题):
            #   version=版本差 s_k (默认, 原行为)
            #   rel2=参数相对漂移 ||W(v)−W(r_k)||/||W(v)−W(0)||, α=exp(−σ·d) (云端可算, 零消息)
            #   wait=缓冲等待 ticks (FedBuff-fair-weighting 思路), α=1/(wait+1)^σ
            alphas = None
            if args.staleness_sigma > 0:
                v_now = cloud.p2_current_version
                alphas = []
                for pkg in selected:
                    if args.staleness_metric == 'rel2':
                        Wv = cloud.p2_global_model_history[v_now]
                        Wr = cloud.p2_global_model_history[pkg.base_version]
                        W0 = cloud.p2_global_model_history[0]
                        d = (np.linalg.norm(Wv - Wr)
                             / (np.linalg.norm(Wv - W0) + 1e-12))
                        alphas.append(math.exp(-args.staleness_sigma * d))
                    elif args.staleness_metric == 'wait':
                        s_k = current_tick - cloud.p2_arrival_ticks.get(
                            pkg.package_token, current_tick)
                        alphas.append(1.0 / (s_k + 1.0) ** args.staleness_sigma)
                    else:  # version (默认)
                        s_k = v_now - pkg.base_version
                        alphas.append(1.0 / (s_k + 1.0) ** args.staleness_sigma)

            # RA-CMA stale risk: q_s = Σ S_k·s_k / (s_max·Σ S_k) —— 样本量加权
            # 平均组包陈旧度, 归一化 [0,1]; 与 σ 权重同口径 (选中时刻版本 v_now)。
            # 纯公开标量元数据 (S_k, base_version), 零协议消息。
            stale_risk_t = 0.0
            if cma_mode != 'none':
                S_Q_sel = sum(pkg.S_k for pkg in selected)
                if S_Q_sel > 0 and args.s_max > 0:
                    stale_risk_t = (sum(pkg.S_k * (cloud.p2_current_version - pkg.base_version)
                                        for pkg in selected)
                                    / (args.s_max * S_Q_sel))

            # TA 构建 checkpoint 掩码 (加权掩码和, 不释放单组掩码)
            # sigma-target=agg: σ 权重整数化 (×2^16) 后真实进入掩码域与聚合平均
            # (整数标量乘保持 LHH 同态可验证); vcm 模式: 掩码域恒等权,
            # σ 只缩放 VCM 补偿项 (历史行为)
            # V-PDA (--participation-debias ipw): 逆参与权重同样整数化进掩码域,
            # 且 VCM 补偿项加权 (权重只改聚合, 不改变调度/通信)。
            vpda_active = (args.participation_debias == 'ipw')
            use_agg = ((args.sigma_target == 'agg' and alphas is not None)
                       or vpda_active)
            weights = None
            weight_source = None
            vpda_qs = None
            vpda_wmin = 1.0
            vpda_wmax = 1.0
            if vpda_active:
                if args.sigma_target == 'agg':
                    logger.warning("V-PDA 与 --sigma-target agg 同时启用: "
                                   "agg 语义(加权无 VCM)被 V-PDA 加权 VCM 覆盖")
                # A_k(t): 此前成功提交的 checkpoint 中各组入选次数 (当前轮不计入);
                # 权重均值在**全部组**上取 (公平基准=全体组, 非选中子集)
                A_dict = {}
                for r in cloud.p2_checkpoint_log:
                    for p in r.selected_packages:
                        A_dict[p.edge_id] = A_dict.get(p.edge_id, 0) + 1
                A_all = [A_dict.get(e.id, 0) for e in edges]
                counts = [A_dict.get(p.edge_id, 0) for p in selected]
                vpda_qs, weights = vpda_weights(
                    counts, kappa=args.debias_kappa, wmin=args.debias_min,
                    wmax=args.debias_max, scale=args.debias_scale,
                    all_counts=A_all)
                weight_source = 'vpda'
                vpda_wmin = min(vpda_qs)
                vpda_wmax = max(vpda_qs)
            elif use_agg:
                # 整数化用共享 helper: 与 ES 复算侧 (staleness_int_weights)
                # 完全同一实现, 避免舍入差异造成验证误拒
                weights = staleness_int_weights(alphas)
                weight_source = 'staleness'
            M_Q, H_mask_raw = ta.p2_build_checkpoint_mask(selected, alphas,
                                                          weights=weights)

            # Cloud 恢复 (加权) raw sum
            raw_sum_int = cloud.p2_recover_raw_sum(selected, M_Q,
                                                   weights=weights)
            gamma = args.gamma_vcm if args.vcm else 0.0

            # Aσ-VCM（方向自适应 σ）: 固定 σ 在 Non-IID 能稳定方差，
            # 但在 IID 会损失有效补偿。先用未衰减补偿 C0 与安全聚合训练
            # 增量 Δtrain 的一致性做试探性档位决策（σ_eff 立即生效于本轮
            # α）；EMA/探测/锁定状态仅在 checkpoint 成功提交后落地，与
            # ES 验证侧的 commit 通知保持同一事件序列，验证失败的候选
            # 不推进任何一侧的策略状态。只改变明文域 VCM 补偿系数。
            pending_sigma_diag = None
            if args.sigma_policy == 'alignment':
                S_policy = max(1, sum(pkg.S_k for pkg in selected))
                raw_policy = (raw_sum_int.astype(np.float64)
                              / QUANTIZE_SCALE)
                R_policy = raw_policy / S_policy
                B_num_policy = np.zeros_like(R_policy)
                C0_num_policy = np.zeros_like(R_policy)
                v_policy = cloud.p2_current_version
                Wv_policy = cloud.p2_global_model_history[v_policy]
                for pkg in selected:
                    Wr_policy = cloud.p2_global_model_history[pkg.base_version]
                    B_num_policy += pkg.S_k * np.asarray(Wr_policy,
                                                         dtype=np.float64)
                    if pkg.base_version < v_policy:
                        C0_num_policy += (pkg.S_k * gamma
                                          * (Wv_policy - Wr_policy))
                D_policy = R_policy - B_num_policy / S_policy
                C0_policy = C0_num_policy / S_policy
                d_policy = float(np.linalg.norm(D_policy))
                c_policy = float(np.linalg.norm(C0_policy))
                cos_policy = float(np.dot(C0_policy, D_policy)
                                   / (c_policy * d_policy + 1e-12))
                cos_valid = (c_policy > 0.0 and d_policy > 0.0)
                sigma_eff = sigma_alignment.tentative_eff(
                    cos_policy if cos_valid else None)
                # vcm-target 的浮点 alpha 不参与掩码恢复，因此可在看到安全
                # 聚合量后再确定；这不会暴露任何单组更新。
                alphas = []
                for pkg in selected:
                    if args.staleness_metric == 'rel2':
                        Wr = cloud.p2_global_model_history[pkg.base_version]
                        W0 = cloud.p2_global_model_history[0]
                        dist = (np.linalg.norm(Wv_policy - Wr)
                                / (np.linalg.norm(Wv_policy - W0) + 1e-12))
                        alphas.append(math.exp(-sigma_eff * dist))
                    elif args.staleness_metric == 'wait':
                        wait = current_tick - cloud.p2_arrival_ticks.get(
                            pkg.package_token, current_tick)
                        alphas.append(1.0 / (wait + 1.0) ** sigma_eff)
                    else:
                        stale = v_policy - pkg.base_version
                        alphas.append(1.0 / (stale + 1.0) ** sigma_eff)
                # (cos 原值, 状态落地用 cos, sigma_eff): diag 延迟到 commit
                # 后构建, 使 EMA/锁定标志与本轮决策一致
                pending_sigma_diag = (cos_policy,
                                      cos_policy if cos_valid else None,
                                      sigma_eff)

            # Direct-VCM 补偿: agg 模式无补偿 (权重已作用于聚合平均);
            # V-PDA 模式加权原始模型 + 加权 VCM (γ 由 --vcm 控制);
            # 其余等权路径保持历史行为
            if vpda_active:
                corrected_sum, final_model, comp_map = cloud.p2_weighted_vcm_average(
                    selected, raw_sum_int, weights, gamma_vcm=gamma,
                    vcm_mode=args.vcm_mode, alphas=alphas)
            elif use_agg:
                gamma = 0.0
                corrected_sum, final_model, comp_map = cloud.p2_weighted_average(
                    selected, raw_sum_int, weights)
            else:
                corrected_sum, final_model, comp_map = cloud.p2_apply_vcm_and_average(
                    selected, raw_sum_int, gamma_vcm=gamma, vcm_mode=args.vcm_mode,
                    alphas=alphas)

            # DA-VCM 影子诊断：只观测原始聚合 R、基模型重心 B 与补偿 C 的
            # 夹角，绝不回写 final_model。当前候选若随后验证失败，则不落盘。
            # 诊断首轮也保留（C=0 时 cos 记 0），便于每条记录和 checkpoint
            # / acc_history 一一对齐。
            alignment_diag = None
            if not use_agg:
                S_diag = max(1, sum(pkg.S_k for pkg in selected))
                raw_float_diag = (raw_sum_int.astype(np.float64)
                                  / QUANTIZE_SCALE)
                R_diag = raw_float_diag / S_diag
                B_num_diag = np.zeros_like(R_diag)
                for pkg in selected:
                    W_rk_diag = cloud.p2_global_model_history.get(pkg.base_version)
                    if W_rk_diag is not None:
                        B_num_diag += pkg.S_k * np.asarray(W_rk_diag,
                                                           dtype=np.float64)
                B_diag = B_num_diag / S_diag
                C_diag = (np.asarray(corrected_sum, dtype=np.float64)
                          - raw_float_diag) / S_diag
                D_diag = R_diag - B_diag
                d_norm_diag = float(np.linalg.norm(D_diag))
                c_norm_diag = float(np.linalg.norm(C_diag))
                cos_diag = float(np.dot(C_diag, D_diag)
                                 / (c_norm_diag * d_norm_diag + 1e-12))
                ratio_diag = c_norm_diag / (d_norm_diag + 1e-12)
                alignment_diag = [cos_diag, d_norm_diag, c_norm_diag,
                                  ratio_diag, float(cos_diag < 0.0)]

            # FedLAW 式聚合收缩 (权重和<1 = 隐式 weight decay, 短视界泛化正则)
            if args.agg_shrink < 1.0:
                final_model = final_model * args.agg_shrink

            # Server momentum (平滑模型轨迹, 关闭时恒等): W' = β·W(v)+(1−β)·final
            if args.momentum > 0:
                final_model = cloud.p2_apply_momentum(final_model, args.momentum)

            # 服务器优化器 (作用于整个 checkpoint 更新, 在收缩/动量之后)
            if args.server_opt == 'gate':
                tau_max = max(cloud.p2_current_version - p.base_version
                              for p in selected)
                final_model = cloud.p2_server_lr_gate(final_model, tau_max)
            elif args.server_opt == 'adam':
                final_model = cloud.p2_adam_update(final_model)
            elif args.server_opt == 'trc':
                final_model = cloud.p2_server_trust_region(final_model, args.trc_k)

            # 形成待验证的 checkpoint 候选。
            # 重要语义：此处只能构造候选，不能推进 Cloud 状态；必须在 ES
            # 验证通过后才允许 p2_commit_checkpoint()。
            broadcast_selected = selected
            broadcast_raw = raw_sum_int
            broadcast_final = final_model
            broadcast_corrected = corrected_sum
            broadcast_comp_map = comp_map
            broadcast_alphas = alphas
            broadcast_stale_risk = stale_risk_t

            # 攻击适用性/注入标记 (AttackOutcome 记录用; 由各攻击分支更新)
            attack_applicable = True
            attack_injected = (args.attack != 'none')

            # 攻击注入 (独立于性能基线, 仅攻击实验时启用)。ES 验证的对象
            # 是 Cloud 广播值，因此攻击实验必须把篡改后的候选送入验证。
            if args.attack == 'tamper_global':
                # Cloud 篡改 W_sum^raw: 归约到 Theorem 4.1 (i), 应被 Step 2 LHH 等式检测
                tampered_raw = raw_sum_int.copy()
                tampered_raw[0] += 1
                if vpda_active:
                    tampered_corrected, tampered_final, tampered_comp_map = cloud.p2_weighted_vcm_average(
                        selected, tampered_raw, weights, gamma_vcm=gamma,
                        vcm_mode=args.vcm_mode, alphas=alphas)
                else:
                    tampered_corrected, tampered_final, tampered_comp_map = cloud.p2_apply_vcm_and_average(
                        selected, tampered_raw, gamma_vcm=gamma, vcm_mode=args.vcm_mode,
                        alphas=alphas)
                broadcast_raw = tampered_raw
                broadcast_final = tampered_final
                broadcast_corrected = tampered_corrected
                broadcast_comp_map = tampered_comp_map
                logger.warning(f"Tick {current_tick}: [ATTACK] tamper_global (W_sum^raw[0]+=1)")
            elif args.attack == 'tamper_manifest':
                # 篡改选中的组包集合 (注入一个不存在的组包)
                from point2_types import GroupPackage
                fake_pkg = GroupPackage(
                    edge_id="fake_edge", package_token="fake:pkg",
                    base_version=0, base_hash=b'\x00',
                    online_users=["fake_u"], n_k=1, S_k=100,
                    masked_sum_int=np.zeros(dim, dtype=np.int64),
                    H_k=LHH.Hash(ta.lhh_pp, np.zeros(dim, dtype=np.int64)),
                    signature=b'\x00', created_tick=current_tick,
                )
                selected_attacked = selected + [fake_pkg]
                broadcast_selected = selected_attacked
                logger.warning(f"Tick {current_tick}: [ATTACK] tamper_manifest")
            elif args.attack == 'drop_replace_pkg':
                # 漏聚合某个组包
                attack_applicable = len(selected) > 1
                if len(selected) > 1:
                    selected_attacked = selected[:-1]
                    broadcast_selected = selected_attacked
                    logger.warning(f"Tick {current_tick}: [ATTACK] drop_replace_pkg "
                                    f"(removed {selected[-1].package_token})")

            # ES 验证 (验证 Cloud 广播值)
            bundle = {
                # commit 前的当前版本就是本次 checkpoint 的 base/current version。
                'checkpoint_version': cloud.p2_current_version,
                # 协议配置摘要 (manifest 绑定): ES/TA 校验声明 == 本地预置
                'policy_config_digest': policy_digest,
                'selected_packages': broadcast_selected,
                'raw_sum_int': broadcast_raw,
                'M_Q': M_Q,
                'H_mask_raw': H_mask_raw,
                'final_model': broadcast_final,
                'gamma_vcm': gamma,
                'vcm_mode': args.vcm_mode,
                'alphas': broadcast_alphas,
                'weights': (weights if use_agg else None),
                'weight_source': (weight_source if use_agg else 'none'),
                'momentum': args.momentum,
                'global_model_history': cloud.p2_global_model_history,
            }

            # 策略绑定攻击族 (4.5.3; 与性能基线分离): 直接篡改 bundle 的
            # 声明/公开数据, ES 验证必须拒绝且不推进版本与策略状态。
            # 配置不适用的攻击 (等权路径伪造权重等) 记录后跳过注入。
            if args.attack in ('weight_forge', 'weight_relabel',
                               'source_downgrade', 'alpha_tamper',
                               'history_missing'):
                bundle = dict(bundle)
                _applicable = True
                if args.attack == 'weight_forge':
                    if bundle['weights'] is None:
                        _applicable = False
                    else:
                        bundle['weights'] = [int(w) + 65536
                                             for w in bundle['weights']]
                elif args.attack == 'weight_relabel':
                    _cur = bundle.get('weight_source') or 'none'
                    if _cur == 'none':
                        _applicable = False
                    else:
                        bundle['weight_source'] = ('staleness' if _cur == 'vpda'
                                                   else 'vpda')
                elif args.attack == 'source_downgrade':
                    if bundle['weights'] is None:
                        _applicable = False
                    else:
                        bundle['weights'] = None
                        bundle['weight_source'] = 'none'
                elif args.attack == 'alpha_tamper':
                    if not bundle.get('alphas'):
                        _applicable = False
                    else:
                        # 减半注入: 新鲜包 (s_k=0 → α=1) 下全 1 与诚实值无差别,
                        # 减半保证注入值必然偏离公开规则复算值
                        bundle['alphas'] = [a * 0.5 for a in bundle['alphas']]
                elif args.attack == 'history_missing':
                    _h = dict(bundle['global_model_history'])
                    _h.pop(bundle['checkpoint_version'], None)
                    bundle['global_model_history'] = _h
                attack_applicable = _applicable
                if _applicable:
                    logger.warning(f"Tick {current_tick}: [ATTACK] {args.attack}")
                else:
                    logger.warning(f"Tick {current_tick}: [ATTACK] {args.attack} "
                                   f"在本配置下不适用, 本轮未注入")

            all_passed = True
            verify_detail = None
            votes = None
            _t_verify0 = time.perf_counter()
            if args.verify:
                p2_metrics['verification_attempts'] += 1
                # 全部 ES 验证每个 checkpoint (不只是被选中的): V-PDA 下每个
                # ES 必须从全部历史 checkpoint 独立累计组参与历史 A_k, 才能
                # 在验证侧独立重算权重; 顺带使所有 ES 对共识模型达成一致。
                # (详细结果用于 AttackOutcome 的检出阶段归因)
                votes = []
                for e in edges:
                    vote, res = e.p2_vote_checkpoint_with_result(bundle)
                    votes.append(vote)
                    if not res.passed:
                        all_passed = False
                        if verify_detail is None:
                            verify_detail = res
                        logger.error(f"Tick {current_tick}: {e.id} 验证失败! "
                                     f"({res.stage}: {res.reason})")
                # 每个 ES 实际生成一个签名 vote（包括 passed=False 的拒绝票）。
                comm.vote(len(votes))
            p2_metrics['checkpoint_attempts'] += 1
            _state_before = _state_snapshot()
            comm.manifest(dim, len(broadcast_selected))

            if not all_passed:
                # 安全协议语义：检测到攻击/验证失败即丢弃本次 checkpoint
                # 更新，不推进全局版本、不删除 buffer 中的组包、不计入有效吞吐。
                p2_metrics['verification_rejections'] += 1
                p2_metrics['verify_seconds_list'].append(
                    time.perf_counter() - _t_verify0)
                logger.error(
                    f"Tick {current_tick}: checkpoint rejected; "
                    f"global version remains v{cloud.p2_current_version}"
                )
                _emit_outcome(
                    applicable=attack_applicable, attempted=attack_injected,
                    detected=True, accepted=False,
                    stage=(verify_detail.stage if verify_detail is not None
                           else 'verify'),
                    reason='verify_failed', before=_state_before)
                continue

            # 只有验证通过后才进行原子式状态提交。verify-on 路径: 收集全部
            # ES 签名 vote → TA 验票 + 权威 ledger 校验 → 签发 CheckpointReceipt
            # → Cloud 凭 receipt 提交 → ES 见 receipt 才推进版本与策略状态。
            receipt = None
            if args.verify:
                try:
                    receipt = ta.p2_authorize_checkpoint(
                        broadcast_selected, votes,
                        checkpoint_version=bundle['checkpoint_version'],
                        trigger_mode=cloud.p2_last_trigger_mode,
                        config=protocol_config)
                    comm.receipt()
                except ValueError as e:
                    p2_metrics['verification_rejections'] += 1
                    p2_metrics['verify_seconds_list'].append(
                        time.perf_counter() - _t_verify0)
                    logger.error(f"Tick {current_tick}: TA 授权拒绝 "
                                 f"(vote/调度/ledger 校验失败): {e}")
                    _emit_outcome(
                        applicable=attack_applicable, attempted=attack_injected,
                        detected=True, accepted=False, stage='ta_authorize',
                        reason='authorize_rejected', before=_state_before)
                    continue
            # 提交事务 (P0-2): stage → TA finalize → publish → notify。
            # finalize 才是权威落地点: 若被协议拒绝, Cloud 尚未应用任何变更
            # (旧实现在此产生 "Cloud 已提交但 TA 未落地" 的分叉); finalize
            # 成功后各阶段幂等, 异常可凭 stage journal 恢复完成。
            try:
                result = run_commit_transaction(
                    cloud, ta, edges,
                    selected_packages=broadcast_selected,
                    final_model=broadcast_final, raw_sum_int=broadcast_raw,
                    M_Q=M_Q, H_mask_raw=H_mask_raw,
                    corrected_sum=broadcast_corrected,
                    comp_map=broadcast_comp_map, receipt=receipt)
            except ValueError as e:
                # 协议性拒绝 (stage 校验 / TA finalize 核对失败): 丢弃本次
                # 候选, 不推进任何权威状态 (fail-closed)
                p2_metrics['verification_rejections'] += 1
                p2_metrics['verify_seconds_list'].append(
                    time.perf_counter() - _t_verify0)
                logger.error(f"Tick {current_tick}: checkpoint rejected by "
                             f"commit 事务: {e}")
                _emit_outcome(
                    applicable=attack_applicable, attempted=attack_injected,
                    detected=True, accepted=False, stage='commit',
                    reason='commit_rejected', before=_state_before)
                continue
            except Exception as e:
                # 瞬时故障已由事务内部重试; 仍失败则尝试恢复完成 (幂等),
                # 恢复不了才向上抛 —— 不静默留下 Cloud/TA/ES 分叉
                logger.error(f"Tick {current_tick}: 提交事务异常 ({e}), "
                             f"尝试恢复完成...")
                result = recover_staged_commit(cloud, ta, edges)
                if result is None:
                    raise
            p2_metrics['verify_seconds_list'].append(
                time.perf_counter() - _t_verify0)
            comm.notify(len(edges))
            result.verification_passed = all_passed
            # 攻击裁决记录: 非 replay 攻击走到这里说明攻击被采纳 (攻击
            # 实验的失败情形); clean control 记 requested=False
            if args.attack != 'replay':
                _emit_outcome(
                    applicable=attack_applicable, attempted=attack_injected,
                    detected=False, accepted=True, stage='commit',
                    reason='committed', before=_state_before)

            if all_passed:
                checkpoint_count += 1

                if alignment_diag is not None:
                    vcm_alignment_history.append(alignment_diag)
                # Aσ 状态落地 + diag 记录: 仅成功提交的 checkpoint 推进
                # 策略状态, 与事务内 ES commit 通知同一事件序列
                # (TA/ES 的 commit 通知已由事务幂等完成, 此处不重复)
                if pending_sigma_diag is not None:
                    cos_raw, cos_state, sigma_eff_c = pending_sigma_diag
                    sigma_alignment.commit_observation(cos_state)
                    alignment_sigma_history.append(
                        [cos_raw,
                         (np.nan if sigma_alignment.cos_ema is None
                          else sigma_alignment.cos_ema),
                         sigma_eff_c,
                         float(sigma_alignment.locked is not None)])

                # V-PDA 无去偏影子轨迹 (G5 等价: 同包序列等权聚合 + 等权 VCM,
                # VCM 参考主历史, alphas/γ/vcm_mode 与主路径一致; 只用于同 run
                # 配对诊断, 不回灌训练/下发, 零协议消息)
                g5_acc = None
                if vpda_active:
                    # 无去偏影子轨迹 (G5 等价): 同包序列 + 等权聚合 + 等权 VCM,
                    # VCM 参考影子自己的版本历史 (v=0 起逐轮演进, 与主版本同步),
                    # 使影子是自洽的 G5 轨迹 (与真 G5 的唯一差异=包内容)
                    M_Q_eq, _ = ta.p2_build_checkpoint_mask(
                        broadcast_selected, None, weights=None,
                        audit_only=True)
                    raw_eq = np.zeros(dim, dtype=np.float64)
                    S_eq = 0
                    for pkg in broadcast_selected:
                        raw_eq += pkg.masked_sum_int.astype(np.float64)
                        S_eq += pkg.S_k
                    raw_eq = (raw_eq - M_Q_eq.astype(np.float64)) / QUANTIZE_SCALE
                    v_now = cloud.p2_current_version          # 提交后版本 = checkpoint t
                    v_ref = v_now - 1                         # 影子 VCM 参考点 (commit 前)
                    comp_eq = np.zeros_like(raw_eq)
                    for j, pkg in enumerate(broadcast_selected):
                        r_k = pkg.base_version
                        if r_k < v_ref and r_k in g5_history:
                            G_k = (g5_history[v_ref] - g5_history[r_k])
                            alpha = 1.0 if alphas is None else alphas[j]
                            C_k = alpha * pkg.S_k * gamma * G_k
                            if args.vcm_mode == 'span' and (v_ref - r_k) > 0:
                                C_k = C_k / (v_ref - r_k)
                            comp_eq += C_k
                    g5_flat = (raw_eq + comp_eq) / max(1.0, S_eq)
                    g5_history[v_now] = g5_flat
                    # 评测 (RNG 快照/恢复, 零副作用)
                    _np_state = np.random.get_state()
                    _torch_state = torch.random.get_rng_state()
                    _cuda_states = ([torch.cuda.get_rng_state(i)
                                     for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_available() else [])
                    try:
                        g5_eval_model = copy.deepcopy(init_model)
                        models.unflatten_parameters(g5_eval_model, g5_flat, model_meta)
                        g5_acc, _ = models.evaluate(g5_eval_model, test_loader)
                    finally:
                        np.random.set_state(_np_state)
                        torch.random.set_rng_state(_torch_state)
                        for _i, _s in enumerate(_cuda_states):
                            torch.cuda.set_rng_state(_s, _i)
                    g5_acc_history.append((checkpoint_count, g5_acc))

                    # 每轮诊断快照: A_k(t) (commit 前口径), 权重, max/min 比,
                    # 入选频率变异系数, 选中样本标签分布 TV 偏移
                    a_snapshot = [A_dict.get(e.id, 0) for e in edges]
                    participation_counts_history.append(a_snapshot)
                    if vpda_qs is not None:
                        debias_weights_float_history.append(list(vpda_qs))
                        debias_weights_int_history.append(list(weights))
                        wmin = min(weights)
                        weight_ratio_history.append(
                            float(max(weights)) / max(1, wmin))
                    ap1 = [c + 1 for c in a_snapshot]
                    participation_cv_history.append(
                        float(np.std(ap1)) / max(1e-12, float(np.mean(ap1))))
                    # 标签 TV 偏移: 加权选中分布 (实际) vs 等权选中分布 (无去偏
                    # 基线), 均相对全局标签分布; 下降 = tv_eq − tv_w (自含基线)
                    tv_w, tv_eq = 0.0, 0.0
                    if edge_label_hists is not None and global_label_hist is not None:
                        S_sel = sum(pkg.S_k for pkg in broadcast_selected)
                        p_w = np.zeros_like(global_label_hist)
                        p_eq = np.zeros_like(global_label_hist)
                        wsum = 0.0
                        for j, pkg in enumerate(broadcast_selected):
                            hist_k = edge_label_hists[pkg.edge_id]
                            p_eq += pkg.S_k * hist_k
                            wq = vpda_qs[j] if vpda_qs is not None else 1.0
                            p_w += wq * pkg.S_k * hist_k
                            wsum += wq * pkg.S_k
                        p_eq = p_eq / max(1, S_sel)
                        p_w = p_w / max(1e-12, wsum)
                        tv_w = float(0.5 * np.abs(p_w - global_label_hist).sum())
                        tv_eq = float(0.5 * np.abs(p_eq - global_label_hist).sum())
                    label_tv_history.append(tv_w)
                    label_tv_eq_history.append(tv_eq)

                    # 累计 TV: 用"实际参与用户"的原始标签计数跨 checkpoint 累计
                    # (加权按 q_k、等权按 1), 相对全局原始计数; 另记录累计权重质量
                    # (累计加权/等权选中次数的 min/max 比与 CV) 与实际参与组。
                    edge_ids = [e.id for e in edges]
                    sel_row = [0] * len(edges)
                    for j, pkg in enumerate(broadcast_selected):
                        wq = vpda_qs[j] if vpda_qs is not None else 1.0
                        cum_sel_equal[pkg.edge_id] += 1
                        cum_sel_weighted[pkg.edge_id] += float(wq)
                        sel_row[edge_ids.index(pkg.edge_id)] = 1
                        if cum_eq_counts is not None:
                            for u in pkg.online_users:
                                c = user_label_counts.get(u)
                                if c is not None:
                                    cum_eq_counts += c
                                    cum_w_counts += float(wq) * c
                    selected_groups_history.append(sel_row)
                    cum_sel_equal_history.append([cum_sel_equal[e.id] for e in edges])
                    cum_sel_weighted_history.append([round(cum_sel_weighted[e.id], 6)
                                                     for e in edges])
                    wvals = np.asarray([cum_sel_weighted[e.id] for e in edges],
                                       dtype=np.float64)
                    cum_weight_quality_history.append(
                        float(wvals[wvals > 0].min() / wvals.max())
                        if wvals.max() > 0 else 1.0)
                    cum_weight_cv_history.append(
                        float(np.std(wvals)) / max(1e-12, float(np.mean(wvals))))
                    if cum_eq_counts is not None and global_label_counts is not None:
                        p_cw = cum_w_counts / max(1e-12, cum_w_counts.sum())
                        p_ce = cum_eq_counts / max(1e-12, cum_eq_counts.sum())
                        p_g = global_label_counts / max(1e-12, global_label_counts.sum())
                        label_tv_cum_history.append(
                            float(0.5 * np.abs(p_cw - p_g).sum()))
                        label_tv_cum_eq_history.append(
                            float(0.5 * np.abs(p_ce - p_g).sum()))

                # UR-VCM 诊断 (trust 模式, 与 V-PDA 正交): ‖Δtrain‖ / ‖C̄‖ / η
                if args.vcm_mode == 'trust' and cloud.p2_last_vcm_diag is not None:
                    _d = cloud.p2_last_vcm_diag
                    vcm_train_norm_history.append(float(_d['train_norm']))
                    vcm_comp_norm_history.append(float(_d['comp_norm']))
                    vcm_trust_eta_history.append(float(_d['eta']))

                # 影子模型更新 (RA-CMA / 修正后 fixed EMA):
                #   W̃_t = β_t·W̃_{t-1} + (1−β_t)·W_t,  W̃_1 = W_1 (无偏差校正)
                #   q_s = ΣS_k·s_k/(s_max·ΣS_k)  (选中时计算, 见 broadcast_stale_risk)
                #   q_d = max(0, −cos(Δ_t, Δ_{t−1})),  Δ_t = W_t − W_{t−1}
                #   r_t = λs·q_s + (1−λs)·q_d
                #   β_t = βmin + (βmax−βmin)·clip(r_t,0,1)  (fixed: β_t = 常数)
                # 纯服务器端 O(d): 不回灌训练、不改变下发模型、零协议消息。
                if cma_mode != 'none':
                    _t0 = time.perf_counter()
                    v_t = cloud.p2_current_version          # 提交后 v = checkpoint t
                    W_t = cloud.p2_global_model_history[v_t]
                    q_s = broadcast_stale_risk
                    if v_t == 1:
                        # 第一轮无历史方向可用: q_d=0, 直接 W̃₁ = W₁
                        cma_prev_delta = (W_t - cloud.p2_global_model_history[0])
                        q_d = 0.0
                    else:
                        delta_t = W_t - cloud.p2_global_model_history[v_t - 1]
                        if cma_prev_delta is not None:
                            q_d = max(0.0, -np.dot(delta_t, cma_prev_delta)
                                      / (np.linalg.norm(delta_t)
                                         * np.linalg.norm(cma_prev_delta) + 1e-12))
                        else:
                            q_d = 0.0
                        cma_prev_delta = delta_t
                    lam_s = {'stale': 1.0, 'conflict': 0.0}.get(cma_mode,
                                                                 args.cma_lambda_stale)
                    r_t = lam_s * q_s + (1.0 - lam_s) * q_d
                    if cma_mode == 'fixed':
                        beta_t = cma_beta_fixed
                    else:
                        beta_t = (args.cma_beta_min
                                  + (args.cma_beta_max - args.cma_beta_min)
                                  * min(1.0, max(0.0, r_t)))
                    if cma_flat is None:
                        cma_flat = W_t.astype(np.float64).copy()
                    else:
                        cma_flat = (beta_t * cma_flat
                                    + (1.0 - beta_t) * W_t.astype(np.float64))
                    cma_beta_history.append((v_t, float(beta_t)))
                    cma_stale_risk_history.append((v_t, float(q_s)))
                    cma_conflict_history.append((v_t, float(q_d)))
                    cma_risk_history.append((v_t, float(r_t)))
                    cma_update_seconds += time.perf_counter() - _t0

                # 重放攻击: 提交后重发一个已消费组包, 验证 Aggregated_IDs
                # exactly-once 拒绝 (发生在 buffer 接收路径, 与 ES 验证无关)。
                if args.attack == 'replay':
                    replay_pkg = selected[0]
                    rejected = not cloud.p2_receive_group_package(
                        replay_pkg, arrival_tick=current_tick)
                    p2_metrics['replay_attempts'] += 1
                    p2_metrics['replay_rejections'] += int(rejected)
                    logger.warning(
                        f"Tick {current_tick}: [ATTACK] replay "
                        f"{replay_pkg.package_token} -> "
                        f"{'rejected (exactly-once)' if rejected else 'ACCEPTED (BUG)'}"
                    )
                    # 重放攻击的裁决在提交后 (Aggregated_IDs exactly-once 拒绝);
                    # accepted 指本次 checkpoint 本身正常提交
                    _emit_outcome(
                        applicable=True, attempted=True, detected=bool(rejected),
                        accepted=True,
                        stage=('replay_exactly_once' if rejected
                               else 'replay_accepted'),
                        reason=('replay_rejected' if rejected
                                else 'replay_accepted'),
                        before=_state_before)

                # 指标记录
                p2_metrics['checkpoint_ticks'].append(current_tick)
                p2_metrics['stale_reuse_counts'].append(pending_stale_reuse_count)
                pending_stale_reuse_count = 0
                for pkg in selected:
                    wait_time = (current_tick -
                                 cloud.p2_arrival_ticks.get(pkg.package_token, 0))
                    p2_metrics['wait_times'].append(wait_time)
                p2_metrics['comp_gain'] += sum(broadcast_comp_map.values())
                # throughput: 每 tick 处理的有效样本数
                prev_tick = (p2_metrics['checkpoint_ticks'][-2]
                            if len(p2_metrics['checkpoint_ticks']) > 1 else 0)
                p2_metrics['throughput'].append(
                    result.S_Q / max(1, p2_metrics['checkpoint_ticks'][-1] - prev_tick))

                # 评估。注意: 攻击注入可能篡改 final_model (tamper_global),
                # 评测对象必须与提交/验证一致 (broadcast_final), 否则 verify-off
                # 攻击实验会评测错误模型。
                eval_model = copy.deepcopy(init_model)
                models.unflatten_parameters(eval_model, broadcast_final, model_meta)
                acc, loss = models.evaluate(eval_model, test_loader)
                acc_history.append((checkpoint_count, acc))

                # 影子模型评测。评测前后保存/恢复 RNG 状态: 开启影子模块后
                # 原始训练轨迹逐值不变, 影子评测是零副作用的纯观察。
                cma_acc = None
                if cma_mode != 'none' and cma_flat is not None:
                    _t0 = time.perf_counter()
                    _np_state = np.random.get_state()
                    _torch_state = torch.random.get_rng_state()
                    _cuda_states = ([torch.cuda.get_rng_state(i)
                                     for i in range(torch.cuda.device_count())]
                                    if torch.cuda.is_available() else [])
                    try:
                        cma_eval_model = copy.deepcopy(init_model)
                        models.unflatten_parameters(cma_eval_model, cma_flat, model_meta)
                        cma_acc, _ = models.evaluate(cma_eval_model, test_loader)
                    finally:
                        np.random.set_state(_np_state)
                        torch.random.set_rng_state(_torch_state)
                        for _i, _s in enumerate(_cuda_states):
                            torch.cuda.set_rng_state(_s, _i)
                    cma_acc_history.append((checkpoint_count, cma_acc))
                    cma_eval_seconds += time.perf_counter() - _t0

                logger.info(
                    f"✓ Ckp {checkpoint_count} (tick {current_tick}): "
                    f"v{result.version}→v{cloud.p2_current_version} "
                    f"N_Q={result.N_Q} S_Q={result.S_Q} comp={sum(comp_map.values()):.4f} "
                    f"stale_users={p2_metrics['stale_reuse_counts'][-1]} Acc={acc:.2f}%"
                    + (f" AccCMA={cma_acc:.2f}%" if cma_acc is not None else "")
                )

        # 每 10 tick 打印一次状态
        if current_tick % 10 == 0:
            logger.debug(
                f"Tick {current_tick}: buffer={len(cloud.p2_package_buffer)} "
                f"checkpoints={checkpoint_count} v={cloud.p2_current_version}"
            )

    # --- 5. 指标汇总 ---
    wc_elapsed = time.time() - wc_start_time
    avg_wait = np.mean(p2_metrics['wait_times']) if p2_metrics['wait_times'] else 0
    max_wait = np.max(p2_metrics['wait_times']) if p2_metrics['wait_times'] else 0
    # 组包过期率: 从 buffer 因 s_k > s_max 移除的组包数
    expired_from_buffer = cloud.p2_expired_total
    p2_metrics['duplicate_contributions'] = cloud.p2_duplicate_contributions
    # stale 组包: checkpoint 中被选中且 base_version < version 的包数
    stale_in_checkpoint = sum(
        1 for r in cloud.p2_checkpoint_log
        for p in r.selected_packages
        if p.base_version < r.version
    )
    # stale reuse: 所有 checkpoint 中复用了 stale 的客户端总人次
    total_stale_reuses = sum(p2_metrics['stale_reuse_counts'])
    # 公平性: 各 edge 被选中次数的方差
    edge_select_counts = {e.id: 0 for e in edges}
    for r in cloud.p2_checkpoint_log:
        for p in r.selected_packages:
            edge_select_counts[p.edge_id] = edge_select_counts.get(p.edge_id, 0) + 1
    fairness_var = float(np.var(list(edge_select_counts.values())))

    # 最终准确率
    final_acc = acc_history[-1][1] if acc_history else 0.0
    final_acc_cma = (cma_acc_history[-1][1] if cma_acc_history else None)
    # 兼容别名: 只有真正的固定 EMA (fixed 模式) 才写 final_acc_ema,
    # 避免 stale/conflict/risk 结果被误标为固定 EMA; 关闭时保持 None
    final_acc_ema = final_acc_cma if cma_mode == 'fixed' else None
    final_acc_g5_shadow = (g5_acc_history[-1][1] if g5_acc_history else None)

    # 验证统计。验证失败的 checkpoint 不进入 checkpoint_log，因此不能
    # 通过已提交日志反推攻击检测率。
    verify_total = p2_metrics['verification_attempts']
    verify_pass = verify_total - p2_metrics['verification_rejections']
    attack_detected = 0.0
    if args.attack == 'replay':
        # 重放检测发生在 buffer 接收路径 (Aggregated_IDs 拒绝), 与 ES 验证无关。
        ra = p2_metrics['replay_attempts']
        attack_detected = (p2_metrics['replay_rejections'] / ra) if ra > 0 else 0.0
    elif args.attack != 'none' and verify_total > 0:
        attack_detected = (verify_total - verify_pass) / verify_total

    logger.info(f"\n=== 模拟完成: {checkpoint_count} checkpoints, "
                 f"{current_tick} ticks, 墙钟 {wc_elapsed:.1f}s ===")
    logger.info(f"Accuracy history: {acc_history}")
    logger.info(f"--- 指标 ---")
    logger.info(f"  最终准确率: {final_acc:.2f}%")
    if final_acc_cma is not None:
        _mean = lambda h: (float(np.mean([v for _, v in h])) if h else float('nan'))
        logger.info(f"  最终准确率(影子 {cma_mode}): {final_acc_cma:.2f}%")
        logger.info(f"  影子风险: β均值={_mean(cma_beta_history):.3f} "
                    f"r均值={_mean(cma_risk_history):.3f} "
                    f"(q_s均值={_mean(cma_stale_risk_history):.3f}, "
                    f"q_d均值={_mean(cma_conflict_history):.3f})")
        logger.info(f"  影子开销: 更新 {cma_update_seconds:.3f}s, "
                    f"额外评测 {cma_eval_seconds:.1f}s, "
                    f"内存 {2 * dim * 8 / 1e6:.2f} MB (W~ 与 Δ 两向量), "
                    f"通信 0 (协议零改动)")
    if vpda_active and participation_counts_history:
        _wmin = min(min(w) for w in debias_weights_float_history)
        _wmax = max(max(w) for w in debias_weights_float_history)
        logger.info(f"  V-PDA: 权重区间 [{_wmin:.3f},{_wmax:.3f}] "
                    f"(κ={args.debias_kappa}, 范围 [{args.debias_min},{args.debias_max}]), "
                    f"max/min 比均值 {np.mean(weight_ratio_history):.2f}, "
                    f"入选频率 CV 均值 {np.mean(participation_cv_history):.3f}, "
                    f"标签 TV 偏移均值 {np.mean(label_tv_history):.4f}")
        if final_acc_g5_shadow is not None:
            logger.info(f"  最终准确率(无去偏影子 G5): {final_acc_g5_shadow:.2f}% "
                        f"(同 run 配对 Δ={final_acc - final_acc_g5_shadow:+.2f}pp)")
        if cum_weight_quality_history:
            _q = np.asarray(cum_weight_quality_history)
            _cv = np.asarray(cum_weight_cv_history)
            logger.info(f"  累计权重质量: 末轮 min/max={_q[-1]:.3f} "
                        f"(1=完全均匀), CV 均值={_cv.mean():.3f}")
        if label_tv_cum_history:
            logger.info(f"  累计标签 TV: 加权={label_tv_cum_history[-1]:.4f}, "
                        f"等权={label_tv_cum_eq_history[-1]:.4f}, "
                        f"下降={100*(1 - label_tv_cum_history[-1]/max(1e-12, label_tv_cum_eq_history[-1])):.1f}%")
    if args.vcm_mode == 'trust' and vcm_trust_eta_history:
        _eta = np.asarray(vcm_trust_eta_history)
        logger.info(f"  UR-VCM(trust, ρ={args.vcm_trust_ratio}): η 均值={_eta.mean():.3f}, "
                    f"门控触发率={float((_eta < 1.0).mean()):.1%}, "
                    f"‖Δtrain‖ 均值={np.mean(vcm_train_norm_history):.4f}, "
                    f"comp_norm 均值={np.mean(vcm_comp_norm_history):.4f}")
    logger.info(f"  墙钟时间: {wc_elapsed:.1f}s")
    logger.info(f"  平均等待: {avg_wait:.1f} ticks")
    logger.info(f"  最大等待: {max_wait:.1f} ticks")
    logger.info(f"  Compensation gain: {p2_metrics['comp_gain']:.4f}")
    logger.info(f"  Stale reuse 总人次: {total_stale_reuses}")
    logger.info(f"  组包过期移除数: {expired_from_buffer}")
    logger.info(f"  Checkpoint 中 stale 组包数: {stale_in_checkpoint}")
    logger.info(f"  组间公平性 (选中次数方差): {fairness_var:.4f}")
    logger.info(f"  有效吞吐量 (sample/tick): {np.mean(p2_metrics['throughput']):.1f}" if p2_metrics['throughput'] else "  有效吞吐量: N/A")
    logger.info(f"  验证通过: {verify_pass}/{verify_total}")
    logger.info(f"  执行/采纳更新: 组包 {p2_metrics['group_packages_sealed']} 个, "
                f"用户更新 {p2_metrics['user_updates_executed']} 次 → 采纳 "
                f"{sum(len(r.selected_packages) for r in cloud.p2_checkpoint_log)} 组包 / "
                f"{sum(r.N_Q for r in cloud.p2_checkpoint_log)} 用户更新")
    logger.info(f"  checkpoint 验证耗时 (ES 验证+vote+授权+提交): 合计 "
                f"{sum(p2_metrics['verify_seconds_list']):.2f}s, 每次均值 "
                f"{(np.mean(p2_metrics['verify_seconds_list']) if p2_metrics['verify_seconds_list'] else 0.0):.4f}s")
    logger.info(f"  协议通信量 (按消息字段计): 合计 {comm.total_bytes()/1024:.1f} KiB "
                f"{comm.bytes}")
    if args.attack != 'none':
        logger.info(f"  Attack detection rate: {attack_detected:.2%}")
        logger.info(f"  AttackOutcome 记录: {len(p2_metrics['attack_outcomes'])} 条 "
                    f"→ {attack_log_path or '(未落盘)'}")
    if args.attack == 'replay':
        logger.info(f"  Replay attempts: {p2_metrics['replay_attempts']}, "
                    f"rejected: {p2_metrics['replay_rejections']}")

    # --- 6. 画图 ---
    if acc_history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左: 准确率 vs checkpoint
        ckps, accs = zip(*acc_history)
        axes[0].plot(ckps, accs, 'b-o', lw=2, label='W(v)')
        if cma_mode != 'none' and cma_acc_history:
            cc, ca = zip(*cma_acc_history)
            if cma_mode == 'fixed':
                cma_label = f'Fixed-EMA(β={cma_beta_fixed})'
            elif cma_mode == 'risk':
                cma_label = f'RA-CMA(λs={args.cma_lambda_stale})'
            else:
                cma_label = f'RA-CMA({cma_mode})'
            axes[0].plot(cc, ca, 'r-s', lw=2, label=cma_label)
            axes[0].legend(loc='lower right')
            # 次轴: 自适应 β_t 与风险 r_t 动态
            if cma_beta_history:
                _ax2 = axes[0].twinx()
                _cb, _cbeta = zip(*cma_beta_history)
                _cr, _crisk = zip(*cma_risk_history)
                _ax2.plot(_cb, _cbeta, 'g--', lw=1.2, alpha=0.7, label='β_t')
                _ax2.plot(_cr, _crisk, 'm--', lw=1.2, alpha=0.7, label='r_t')
                _ax2.set_ylabel('β_t / r_t')
                _ax2.set_ylim(0, 1)
                _ax2.grid(False)
                _ax2.legend(loc='upper right', fontsize=8)
        axes[0].set_xlabel('Checkpoint')
        axes[0].set_ylabel('Test Accuracy (%)')
        title_parts = [f'{args.dataset}']
        if args.reuse_rule != 'none':
            title_parts.append(f'+StaleReuse({args.reuse_rule})')
        if args.vcm:
            title_parts.append('+VCM')
        axes[0].set_title(' '.join(title_parts))
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 右: Checkpoint 规模统计
        Ss = [r.S_Q for r in cloud.p2_checkpoint_log]
        stale_counts = [
            sum(1 for p in r.selected_packages if p.base_version < r.version)
            for r in cloud.p2_checkpoint_log
        ]
        x = range(1, len(Ss) + 1)
        axes[1].bar(x, Ss, alpha=0.6, label='Total samples S_Q')
        axes[1].bar(x, stale_counts, alpha=0.8, label='Stale packages')
        axes[1].set_xlabel('Checkpoint')
        axes[1].set_ylabel('Count / Samples')
        axes[1].set_title('Checkpoint Composition')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"point2_plot_{run_id}.png")
        plt.savefig(plot_path, dpi=150)
        logger.info(f"图表已保存: {plot_path}")

    # --- 7. 保存数据 (临时文件 + 原子替换) ---
    if _attack_fh[0] is not None:
        _attack_fh[0].close()
    elif attack_log_path:
        # 无任何 checkpoint 尝试时也落一个空文件, 便于批跑工具统一处理
        open(attack_log_path, 'w', encoding='utf-8').close()
    npz_path = out_npz
    tmp_npz = out_npz[:-4] + '.tmp.npz'   # np.savez 只接受 .npz 结尾
    np.savez(
        tmp_npz,
        acc_history=np.array(acc_history),
        checkpoint_Ns=np.array([r.N_Q for r in cloud.p2_checkpoint_log]),
        checkpoint_Ss=np.array([r.S_Q for r in cloud.p2_checkpoint_log]),
        wait_times=np.array(p2_metrics['wait_times']),
        stale_reuse_counts=np.array(p2_metrics['stale_reuse_counts']),
        checkpoint_ticks=np.array(p2_metrics['checkpoint_ticks']),
        wall_clock=wc_elapsed,
        final_acc=final_acc,
        avg_wait=avg_wait,
        fairness_var=fairness_var,
        comp_gain=p2_metrics['comp_gain'],
        expired_buffer_count=expired_from_buffer,
        stale_in_checkpoint=stale_in_checkpoint,
        total_stale_reuses=total_stale_reuses,
        final_acc_ema=(np.nan if final_acc_ema is None else final_acc_ema),
        ema_acc_history=np.array(cma_acc_history if cma_mode == 'fixed' else []),
        final_acc_cma=(np.nan if final_acc_cma is None else final_acc_cma),
        cma_acc_history=np.array(cma_acc_history),
        cma_beta_history=np.array(cma_beta_history),
        cma_risk_history=np.array(cma_risk_history),
        cma_stale_risk_history=np.array(cma_stale_risk_history),
        cma_conflict_history=np.array(cma_conflict_history),
        cma_mode=cma_mode,
        cma_update_seconds=cma_update_seconds,
        cma_eval_seconds=cma_eval_seconds,
        participation_counts=np.array(participation_counts_history),
        debias_weights_float=np.array(debias_weights_float_history),
        debias_weights_int=np.array(debias_weights_int_history),
        debias_weight_ratio=np.array(weight_ratio_history),
        participation_cv=np.array(participation_cv_history),
        label_tv=np.array(label_tv_history),
        label_tv_eq=np.array(label_tv_eq_history),
        label_tv_cum=np.array(label_tv_cum_history),
        label_tv_cum_eq=np.array(label_tv_cum_eq_history),
        cum_sel_weighted=np.array(cum_sel_weighted_history),
        cum_sel_equal=np.array(cum_sel_equal_history),
        cum_weight_quality=np.array(cum_weight_quality_history),
        cum_weight_cv=np.array(cum_weight_cv_history),
        selected_groups=np.array(selected_groups_history),
        weight_source=weight_source,
        vcm_train_norm=np.array(vcm_train_norm_history),
        vcm_comp_norm=np.array(vcm_comp_norm_history),
        vcm_trust_eta=np.array(vcm_trust_eta_history),
        vcm_trust_ratio=args.vcm_trust_ratio,
        # columns: cos(C,Δtrain), ||Δtrain||, ||C||,
        #          ||C||/(||Δtrain||+eps), 1[cos<0]
        vcm_alignment=np.array(vcm_alignment_history),
        # columns: raw cos(C0,Δtrain), EMA(cos), effective sigma_t,
        #          decision_locked
        alignment_sigma=np.array(alignment_sigma_history),
        g5_shadow_acc_history=np.array(g5_acc_history),
        final_acc_g5_shadow=(np.nan if final_acc_g5_shadow is None
                             else final_acc_g5_shadow),
        verify_pass_count=verify_pass,
        verify_total_count=verify_total,
        verification_rejection_count=p2_metrics['verification_rejections'],
        duplicate_contributions=p2_metrics['duplicate_contributions'],
        replay_attempts=p2_metrics['replay_attempts'],
        replay_rejections=p2_metrics['replay_rejections'],
        attack_detected=attack_detected,
        # 论文主表/安全证据: 执行与采纳更新数、验证耗时、通信量
        user_updates_executed=p2_metrics['user_updates_executed'],
        group_packages_sealed=p2_metrics['group_packages_sealed'],
        checkpoint_attempts=p2_metrics['checkpoint_attempts'],
        adopted_packages=sum(len(r.selected_packages)
                             for r in cloud.p2_checkpoint_log),
        adopted_user_updates=sum(r.N_Q for r in cloud.p2_checkpoint_log),
        verify_seconds_total=float(sum(p2_metrics['verify_seconds_list'])),
        verify_seconds_per_checkpoint=np.array(
            p2_metrics['verify_seconds_list']),
        comm_total_bytes=comm.total_bytes(),
        comm_bytes_by_type=json.dumps(comm.bytes),
        comm_counts_by_type=json.dumps(comm.counts),
        attack_outcome_count=len(p2_metrics['attack_outcomes']),
        config=str(args),
    )
    os.replace(tmp_npz, npz_path)
    logger.info(f"数据已保存: {npz_path}")

    # --- 8. 运行元数据 (可复现性审计: 环境/代码状态/配置摘要) ---
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
        'policy_config_digest': policy_digest,
        'args': args_payload,
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
        'verify_pass': verify_pass,
        'verify_total': verify_total,
        'attack': args.attack,
        'attack_detected': attack_detected,
        'attack_log': (os.path.basename(attack_log_path)
                       if attack_log_path else None),
        'attack_outcomes': len(p2_metrics['attack_outcomes']),
        'user_updates_executed': int(p2_metrics['user_updates_executed']),
        'group_packages_sealed': int(p2_metrics['group_packages_sealed']),
        'checkpoint_attempts': int(p2_metrics['checkpoint_attempts']),
        'adopted_packages': int(sum(len(r.selected_packages)
                                    for r in cloud.p2_checkpoint_log)),
        'adopted_user_updates': int(sum(r.N_Q
                                        for r in cloud.p2_checkpoint_log)),
        'verify_seconds_total': float(sum(p2_metrics['verify_seconds_list'])),
        'comm_total_bytes': comm.total_bytes(),
        'comm_bytes_by_type': dict(comm.bytes),
        'npz': os.path.basename(out_npz),
    }
    meta['config_mode_version'] = 'repro-v1'
    meta['scientific_config'] = scientific_config
    meta['artifact_config'] = artifact_config
    meta['cuda_available'] = bool(torch.cuda.is_available())
    meta['cuda_version'] = (torch.version.cuda
                            if torch.version.cuda else None)
    meta['cudnn_version'] = (str(torch.backends.cudnn.version())
                             if torch.backends.cudnn.is_available() else None)
    meta['torch_num_threads'] = torch.get_num_threads()
    meta['commandline'] = ' '.join(sys.argv)
    meta['arr_digest'] = hashlib.sha256(
        np.asarray(acc_history, dtype=np.float64).tobytes()).hexdigest()[:16]
    tmp_meta = out_meta + '.tmp'
    with open(tmp_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, out_meta)
    logger.info(f"运行元数据已保存: {out_meta}")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVSA Point2: Semi-Async Secure Aggregation")
    parser.add_argument('--dataset', default='MNIST',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10'])
    parser.add_argument('--tag', default='',
                        help='产物文件名 tag (批量实验用): point2_{log,data,plot}_{tag}_{ts}.ext')
    parser.add_argument('--model', default=None, choices=['CNN', 'MLP', 'Tiny'],
                        help='model override (default: dataset 默认配置; Tiny=7850 维线性模型, 用于全密码学 verify 端到端)')
    parser.add_argument('--deterministic', action='store_true',
                        help='开启 GPU 确定性算法 (cudnn.deterministic/benchmark=False + '
                             'CUBLAS workspace), 保证同 seed 逐值可复现; 代价是 CIFAR 级'
                             '训练变慢。CPU 路径本就确定, 无需开启')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for data partition and local training')
    parser.add_argument('--noniid', action='store_true',
                        help='use Dirichlet Non-IID partition; default is IID')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration for --noniid')
    parser.add_argument('--num-edge-servers', type=int, default=5)
    parser.add_argument('--users-per-edge', type=int, default=4)
    parser.add_argument('-K', type=int, default=3, help='checkpoint trigger threshold')
    parser.add_argument('--s-max', type=int, default=3, help='max staleness')
    parser.add_argument('--max-checkpoints', type=int, default=20)
    parser.add_argument('--max-ticks', type=int, default=300)
    # 组间异构时延 (ticks)
    parser.add_argument('--min-delay', type=int, default=1, help='fastest group delay (ticks)')
    parser.add_argument('--max-delay', type=int, default=5, help='slowest group delay (ticks)')
    parser.add_argument('--group-by-speed', action='store_true',
                        help='同质性能分组: 用户按速度排序分组, 组内延迟相近 (FedFleet 思路)')
    # stale reuse (E5: 三规则消融)
    parser.add_argument('--reuse-rule', default='ours',
                        choices=['none', 'mafsa', 'ours', 'ours_delta'],
                        help='stale reuse rule (ours_delta = 增量空间重锚定混合)')
    parser.add_argument('--rho-max', type=float, default=0.5, help='max stale weight (ours only)')
    parser.add_argument('--staleness-beta', type=float, default=0.3, help='beta_c decay')
    # VCM
    parser.add_argument('--vcm', action='store_true', help='enable Direct-VCM compensation')
    parser.add_argument('--vcm-mode', default='full',
                        choices=['full', 'span', 'clip', 'moment1', 'moment2',
                                 'moment1c', 'moment1n', 'curve', 'trust'],
                        help='VCM 补偿模式: full=原始 Direct-VCM, span=按版本跨度归一, '
                             'clip=限幅到 raw_sum 10%%, moment1/moment2=矩校准 (DVAFL 式(15)-(18), '
                             'm^_k/M^ 缩放补偿步长), moment1c=保守矩校准 (上限1), '
                             'moment1n=自归一化矩校准 (比值除以自身EMA, 数据集尺度无关), '
                             'curve=曲率感知 (补偿方向加轨迹加速度外推), '
                             'trust=UR-VCM 更新相对可信域: R=原始聚合均值, '
                             'B=所选组基模型样本量加权均值, Δtrain=R-B, '
                             'C_avg=原始补偿均值, η=min(1, ρ‖Δtrain‖/(‖C_avg‖+ε)), '
                             'W=R+η·C_avg (云端与 ES 同公式独立重算; C_avg=0 恒等, '
                             'Δtrain=0 且 C_avg≠0 时抑制补偿)')
    parser.add_argument('--vcm-trust-ratio', type=float, default=1.0,
                        help='UR-VCM 可信域比率 ρ (仅 --vcm-mode trust): '
                             '补偿范数上限 = ρ·‖聚合训练增量‖ (默认 1.0)')
    parser.add_argument('--staleness-sigma', type=float, default=0.0,
                        help='陈旧度幂衰减权重 α_k=1/(s_k+1)^σ (σ>0 启用, 0=关闭)')
    parser.add_argument('--sigma-policy', default='fixed',
                        choices=['fixed', 'alignment'],
                        help='σ 启停策略: fixed=恒定; alignment=Aσ-VCM, '
                             '聚合补偿/训练方向 EMA 高于阈值时关闭衰减')
    parser.add_argument('--sigma-alignment-threshold', type=float, default=0.72,
                        help='Aσ-VCM 关闭衰减的探测期平均方向一致性阈值')
    parser.add_argument('--sigma-alignment-ema', type=float, default=0.8,
                        help='Aσ-VCM 方向一致性诊断 EMA 衰减系数')
    parser.add_argument('--sigma-alignment-warmup', type=int, default=2,
                        help='Aσ-VCM 锁定 σ 前所需的非零补偿方向样本数')
    parser.add_argument('--sigma-alignment-boost-threshold', type=float,
                        default=0.9,
                        help='Aσ-VCM 进入高一致性轻放大档的探测期均值阈值')
    parser.add_argument('--sigma-alignment-high', type=float, default=-0.25,
                        help='Aσ-VCM 高一致性档锁定的 σ（默认 -0.25，'
                             '即对 stale 补偿作有界轻放大）')
    parser.add_argument('--staleness-metric', default='version',
                        choices=['version', 'rel2', 'wait'],
                        help='σ 权重的陈旧度度量: version=版本差(默认); rel2=参数相对漂距 exp(-σ·d); wait=缓冲等待 ticks')
    parser.add_argument('--sigma-target', default='vcm', choices=['vcm', 'agg'],
                        help='σ 权重作用位置: vcm=缩放 VCM 补偿项(历史行为); '
                             'agg=聚合端加权 (整数化 w_k 进入掩码域, W=Σw_k·S_k·w^_k/Σw_k·S_k, '
                             'LHH 整数同态可验证, 此时无 VCM 补偿)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='server momentum β: W\'=β·W(v)+(1-β)·final (0=关闭)')
    parser.add_argument('--agg-shrink', type=float, default=1.0,
                        help='FedLAW 式聚合收缩: W←c·W (c<1, 权重和<1 = 隐式 weight decay; 1=关闭)')
    parser.add_argument('--server-opt', default='none', choices=['none', 'gate', 'adam', 'trc'],
                        help='服务器优化器: gate=FADAS delay-adaptive lr 门控 η=min{1,1/τ_max}; '
                             'adam=FedAdam (β1=0.9,β2=0.99,τ=1e-3,η=1.0); '
                             'trc=聚合更新信赖域裁剪 (历史‖Δ‖中位数×k); 均只消费聚合量, SecAgg 兼容')
    parser.add_argument('--trc-k', type=float, default=2.0,
                        help='trc 信赖域倍数: 步长上限 = k × 历史‖Δ‖中位数')
    parser.add_argument('--vcm-curve-lambda', type=float, default=0.5,
                        help='曲率感知 VCM (--vcm-mode curve) 的加速度外推系数 λ')
    parser.add_argument('--clip-upd-k', type=float, default=0.0,
                        help='客户端自剪裁 (arXiv:2606.13287): 上传更新范数超过自身历史 '
                             'EMA 的 k 倍时截断 (0=关闭); SecAgg 兼容, 掩码前执行')
    parser.add_argument('--staleness-lr-lambda', type=float, default=0.0,
                        help='陈旧度自适应学习率系数 λ: lr_i=lr/(1+λ·E_i) (0=关闭)')
    parser.add_argument('--fedprox-mu', type=float, default=0.0,
                        help='FedProx 近端项系数 μ: 损失+μ/2·||w-W(r_k)||^2 (0=关闭)')
    parser.add_argument('--ema-decay', type=float, default=0.0,
                        help='历史兼容别名: 等价于 --cma-mode fixed --cma-beta a '
                             '(0=关闭; 纯后处理不回灌训练; 已修正旧实现"初始化 W_1 后又除偏差项"错误)')
    parser.add_argument('--cma-mode', default='none',
                        choices=['none', 'fixed', 'stale', 'conflict', 'risk'],
                        help='风险自适应检查点模型平均 (RA-CMA, 可选后处理组件, 非创新点): 服务器端影子模型 '
                             'W~_t=β_t·W~_{t-1}+(1-β_t)·W_t, W~_1=W_1 (无偏差校正)。'
                             'fixed=常数 β; stale=仅陈旧风险 q_s=ΣS_k·s_k/(s_max·ΣS_k) 自适应 β; '
                             'conflict=仅方向冲突 q_d=max(0,-cos(Δ_t,Δ_{t-1})) 自适应 β; '
                             'risk=完整 r_t=λs·q_s+(1-λs)·q_d → '
                             'β_t=βmin+(βmax-βmin)·clip(r_t,0,1)。'
                             '纯服务器端 O(d), 不回灌训练/下发, 零协议消息 (默认 0=关闭)')
    parser.add_argument('--cma-beta', type=float, default=0.5,
                        help='fixed 模式常数 β (默认 0.5)')
    parser.add_argument('--cma-beta-min', type=float, default=0.3,
                        help='RA-CMA β 下限 βmin (默认 0.3)')
    parser.add_argument('--cma-beta-max', type=float, default=0.65,
                        help='RA-CMA β 上限 βmax (默认 0.65)')
    parser.add_argument('--cma-lambda-stale', type=float, default=0.5,
                        help='RA-CMA 风险融合权重 λs: r_t=λs·q_s+(1-λs)·q_d (默认 0.5)')
    # V-PDA 组参与去偏 (可验证逆参与权重聚合)
    parser.add_argument('--participation-debias', default='none',
                        choices=['none', 'ipw'],
                        help='V-PDA 组参与去偏: ipw=逆参与权重 '
                             'q_k=clip((mean_j(A_j+1)/(A_k+1))^kappa, wmin, wmax), '
                             'w_k=round(scale*q_k); 整数权重进掩码域 (M_Q=Σw_k·M_k), '
                             '加权恢复 raw sum, 加权 VCM 补偿, ES 按公开参与历史'
                             '独立重算权重拒绝伪造 (默认 none)')
    parser.add_argument('--debias-kappa', type=float, default=0.5,
                        help='V-PDA 权重指数 κ (默认 0.5)')
    parser.add_argument('--debias-min', type=float, default=0.5,
                        help='V-PDA 权重下限 (默认 0.5)')
    parser.add_argument('--debias-max', type=float, default=2.0,
                        help='V-PDA 权重上限 (默认 2.0)')
    parser.add_argument('--debias-scale', type=int, default=65536,
                        help='V-PDA 权重整数化 scale (默认 65536=2^16)')
    parser.add_argument('--curriculum', default='none',
                        choices=['none', 'bias_asc', 'bias_desc'],
                        help='FedBSS 偏差感知样本课程 (AAAI 2025): bias_asc=小偏差样本先训, bias_desc=方向对照')
    parser.add_argument('--dpr-q', type=float, default=0.0,
                        help='DPR 动态参数重置 (NeurIPS 2025 适配): 每步把漂移超过 q 分位的维度重置回全局锚 (0=关闭)')
    parser.add_argument('--gamma-vcm', type=float, default=1.0, help='VCM compensation strength')
    # buffer/checkpoint
    parser.add_argument('--delta-timeout', type=int, default=0, help='timeout trigger limit (0=disabled)')
    parser.add_argument('--policy', default='fifo', choices=['fifo', 'fresh_first', 'fair_fresh_first'],
                        help='buffer selection policy')
    parser.add_argument('--D-fair', type=int, default=2, help='fairness defer threshold')
    # dropout
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='pre-training dropout rate')
    parser.add_argument('--dropout-mode', default='skip', choices=['skip', 'offline_train'],
                        help='掉线语义: skip=掉线即跳过(基线); offline_train=MAFSA忠实语义(掉线轮本地继续训练并缓存, 恢复后reuse回收)')
    parser.add_argument('--edge-offline-spans', default='',
                        help='边级临时离线区间 (tick 口径): "edge索引:起-止", 逗号分隔, '
                             '如 "0:20-60,2:50-90"; 整组掉线且会恢复, 用于建模'
                             '"某条 edge 链路成段不可用"的真实暂时断连 (默认空=关闭)')
    parser.add_argument('--permanent-dropout-frac', type=float, default=0.0,
                        help='建模开始即固定永久离线的设备比例 (真实部署语义: 固定的那批设备'
                             '连不上); 与 --dropout-burst 的每轮重掷语义不同 (默认 0=关闭)')
    parser.add_argument('--dropout-burst', type=int, default=0,
                        help='突发离线: 掉线后持续 B 个组包 (模拟真实设备离线航班, '
                             'NI 下造成分布代表性损失; 0=逐包独立掉线)')
    parser.add_argument('--dropout-reuse', default='none', choices=['none', 'rebase'],
                        help='训练前掉线时是否用 stale 缓存补交贡献: rebase=重锚定复用 (reuse 救掉线)')
    parser.add_argument('--upload-dropout-rate', type=float, default=0.0,
                        help='post-training upload failure rate (stale source)')
    # verification & attack
    parser.add_argument('--verify', action='store_true', help='enable LHH/Ed25519 verification')
    parser.add_argument('--attack', default='none',
                        choices=['none', 'tamper_global', 'tamper_manifest',
                                 'drop_replace_pkg', 'replay',
                                 'weight_forge', 'weight_relabel',
                                 'source_downgrade', 'alpha_tamper',
                                 'history_missing'],
                        help='attack type (separate from performance baseline); '
                             '策略绑定族: weight_forge=伪造权重数值, '
                             'weight_relabel=改标来源, source_downgrade=降级为等权, '
                             'alpha_tamper=篡改衰减系数, history_missing=省略共识历史')
    parser.add_argument('--attack-log', default='',
                        help='AttackOutcome JSONL 输出路径 (默认: --attack 非 none 时 '
                             '写 <output-dir>/attack_outcomes_<run-id>.jsonl; 显式指定时 '
                             '即使 --attack none 也逐 checkpoint 记录 clean control)')
    parser.add_argument('--output-dir', default='.',
                        help='产物输出目录 (npz/plot/log/meta)')
    parser.add_argument('--run-id', default='',
                        help='运行标识 (默认 <tag|dataset>_<timestamp>)')
    parser.add_argument('--force', action='store_true',
                        help='目标 npz 已存在时强制重跑 (默认仅同 config_hash 跳过)')
    parser.add_argument('--ta-state-path', default='',
                        help='TA 权威 ledger 的 SQLite 文件路径 (默认内存态, '
                             '每次运行独立; 持久化用于跨进程审计)')

    args = parser.parse_args()

    # RA-CMA/EMA 都是凸组合；越界参数会把平均变成外推，可能产生看似有效但
    # 不属于该方法的实验结果，因此在任何数据加载或产物写入前 fail-fast。
    if not 0.0 <= args.cma_beta <= 1.0:
        parser.error("--cma-beta must satisfy 0 <= beta <= 1")
    if not (0.0 <= args.cma_beta_min <= args.cma_beta_max <= 1.0):
        parser.error(
            "--cma-beta-min/--cma-beta-max must satisfy "
            "0 <= beta_min <= beta_max <= 1"
        )
    if not 0.0 <= args.cma_lambda_stale <= 1.0:
        parser.error("--cma-lambda-stale must satisfy 0 <= lambda <= 1")
    if not 0.0 <= args.ema_decay <= 1.0:
        parser.error("--ema-decay must satisfy 0 <= decay <= 1")
    # UR-VCM 可信域比率: 必须为有限正数 (非正/NaN/Inf 会让 η 失去门控语义)
    if not (math.isfinite(args.vcm_trust_ratio) and args.vcm_trust_ratio > 0.0):
        parser.error("--vcm-trust-ratio must be a finite positive number")
    if not -1.0 <= args.sigma_alignment_threshold <= 1.0:
        parser.error("--sigma-alignment-threshold must be in [-1, 1]")
    if not (args.sigma_alignment_threshold
            <= args.sigma_alignment_boost_threshold <= 1.0):
        parser.error("--sigma-alignment-boost-threshold must satisfy "
                     "threshold <= boost_threshold <= 1")
    if not math.isfinite(args.sigma_alignment_high):
        parser.error("--sigma-alignment-high must be finite")
    if not 0.0 <= args.sigma_alignment_ema < 1.0:
        parser.error("--sigma-alignment-ema must satisfy 0 <= ema < 1")
    if args.sigma_alignment_warmup < 1:
        parser.error("--sigma-alignment-warmup must be >= 1")
    if args.sigma_policy == 'alignment' and args.sigma_target != 'vcm':
        parser.error("--sigma-policy alignment currently requires --sigma-target vcm")
    # V-PDA 权重参数合法性 (wmin>0 且 wmin<=wmax; kappa 非负)
    if not (0.0 < args.debias_min <= args.debias_max):
        parser.error("--debias-min/--debias-max must satisfy 0 < min <= max")
    if not (math.isfinite(args.debias_kappa) and args.debias_kappa >= 0.0):
        parser.error("--debias-kappa must be a finite non-negative number")
    if args.debias_scale < 1:
        parser.error("--debias-scale must be >= 1")
    if not (0.0 <= args.permanent_dropout_frac < 1.0):
        parser.error("--permanent-dropout-frac must satisfy 0 <= f < 1")

    # verify-on 边界 (fail-fast, 数据加载前): 只允许 ES 能独立重构的
    # 策略组合; 实验性优化模式固定在 verify-off 研究模式下运行。
    if args.attack != 'none' and not args.verify:
        parser.error("--attack requires --verify (攻击实验必须开验证)")
    if args.verify:
        _verifiable_modes = ('full', 'span', 'clip', 'trust')
        if args.vcm_mode not in _verifiable_modes:
            parser.error(f"--vcm-mode {args.vcm_mode} is not verifiable "
                         f"(verify-on allows {_verifiable_modes}; "
                         f"moment*/curve 为 verify-off 研究模式)")
        if args.agg_shrink != 1.0:
            parser.error("--agg-shrink != 1 与 verify-on 暂不兼容 (fail-fast)")
        if args.server_opt != 'none':
            parser.error("--server-opt 与 verify-on 暂不兼容 (fail-fast)")
        if args.vcm_curve_lambda not in (0.0, 0.5):
            pass  # curve 模式已被上方白名单拒绝

    run_point2_simulation(args)
