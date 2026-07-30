"""
研究点2: 单TA 检查点式组级半异步安全聚合

对应文档 THESIS_POINT2_STALENESS_AWARE_REUSE.md

运行方式:
    python main_point2.py                     # 默认 MNIST (无 stale reuse, 无 VCM)
    python main_point2.py --stale-reuse       # 开启 stale reuse
    python main_point2.py --stale-reuse --vcm # 开启 stale reuse + Direct-VCM
    python main_point2.py --dataset CIFAR10   # CIFAR-10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, copy, hashlib, logging, time, math
import numpy as np
import torch
from torch.utils.data import DataLoader

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
from utils import LHH
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
    'CIFAR10': {
        'model': 'CNN', 'rounds': 30, 'local_epochs': 5,
        'lr': 0.01, 'samples_per_user': 1500, 'batch_size': 64,
    },
}

DEFAULT_ARGS = {
    'dataset': 'MNIST',
    'num_edge_servers': 5,
    'users_per_edge': 4,
    'K': 3,                     # checkpoint 触发阈值
    's_max': 3,                 # 最大陈旧度
    'max_checkpoints': 20,     # 最多 checkpoint 数
    'max_ticks': 200,           # 最大模拟 tick
    'gamma_vcm': 1.0,           # Direct-VCM 补偿强度
    'staleness_beta': 0.3,      # β_c: stale reuse 衰减
    'dropout_rate': 0.1,        # 每 tick 客户端训练前掉线率
    'upload_dropout_rate': 0.0,  # 训练后上传丢包率 (控制 stale 来源)
    'reuse_rule': 'ours',       # 复用规则: none / mafsa / ours
    'rho_max': 0.5,             # stale 结果最大权重 (仅 ours)
    'enable_vcm': False,
    'enable_verification': False,
    'min_delay': 1,             # 最快组几个 tick
    'max_delay': 5,             # 最慢组几个 tick
    'delta_timeout': 0,         # 超时触发时限 (0=禁用)
    'policy': 'fifo',           # 缓冲选择策略: fifo / fresh_first / fair_fresh_first
    'D_fair': 2,                # 公平保底阈值 (仅 fair 策略)
    'attack': 'none',           # 攻击类型: none / tamper_global / tamper_manifest / drop_replace_pkg
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  离散事件模拟
# ============================================================

def run_point2_simulation(args: argparse.Namespace):
    cfg = DATASET_CONFIGS[args.dataset]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # --- 日志 ---
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f'point2_log_{args.dataset}_{timestamp}.txt', 'w', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"=== HVSA 研究点2: 半异步安全聚合 ===")
    logger.info(f"Config: {args}")

    # --- 1. 数据准备 ---
    train_ds, test_ds = load_data(args.dataset)
    total_users = args.num_edge_servers * args.users_per_edge
    train_loaders = distribute_data(train_ds, total_users, cfg['batch_size'],
                                     iid=True, max_samples_per_user=cfg['samples_per_user'])
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    init_model = models.get_model(args.dataset, cfg['model'])
    init_flat, model_meta = models.flatten_parameters(init_model)
    dim = init_flat.shape[0]

    # --- 2. 系统初始化 ---
    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01")
    ta.point2_reset()
    cloud = CloudServer()
    cloud.point2_reset(init_flat, initial_version=0)

    edges = []
    for i in range(args.num_edge_servers):
        es = EdgeServer(
            id=f"edge_{i}", cloud_server=cloud, ta=ta,
            enable_verification=args.enable_verification,
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

    # --- 3. 模拟状态 ---
    current_tick: int = 0
    checkpoint_count: int = 0
    wc_start_time = time.time()
    acc_history: list[tuple[int, float]] = []
    W_v = init_flat.astype(np.float64).copy()

    # 每个 edge 的状态: 'idle' | 'training'
    edge_state: dict[str, str] = {e.id: 'idle' for e in edges}
    # 每个 edge 的训练完成 tick
    edge_ready_tick: dict[str, int] = {e.id: 0 for e in edges}
    # 每个 edge 的 package_seq
    edge_pkg_seq: dict[str, int] = {e.id: 0 for e in edges}
    # 每个 edge 当前的 base_version
    edge_base_v: dict[str, int] = {e.id: 0 for e in edges}

    rng = np.random.default_rng(42)

    # 指标追踪
    p2_metrics = {
        'checkpoint_ticks': [],       # 每个 checkpoint 完成时的 tick
        'wait_times': [],             # 每个 selected 组包的等待时间
        'stale_reuse_counts': [],     # 每个 checkpoint 中复用了 stale 的客户端数
        'comp_gain': 0.0,             # 累计 compensation 贡献
        'expired_buffer_count': 0,    # 从 buffer 因过期移除的组包总数
        'duplicate_contributions': 0, # 重复贡献数 (exactly-once 违反计数)
        'throughput': [],             # 每个 checkpoint 的样本量吞吐
    }

    # 为每个用户设置样本量 (每一轮均匀)
    for u in users:
        u.p2_set_sample_size(cfg['samples_per_user'])

    # 提前分组: 给每个 edge 分配固定的 delay profile
    edge_delay: dict[str, int] = {}
    for e in edges:
        edge_delay[e.id] = rng.integers(args.min_delay, args.max_delay + 1)

    logger.info(f"系统初始化完成: {total_users} 用户, {args.num_edge_servers} ES")
    logger.info(f"ES delay profiles (ticks): {edge_delay}")

    # --- 4. 主循环 ---
    while checkpoint_count < args.max_checkpoints and current_tick < args.max_ticks:
        current_tick += 1
        cloud._gradient_dim = dim  # ensure dim is set for raw sum recovery
        tick_stale_reuse_count = 0  # 本 tick 中所有组包的 stale reuse 总人次

        # 4a. 对 idle edge 启动新 package
        for e in edges:
            if edge_state[e.id] == 'idle':
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
                    if rng.random() < args.dropout_rate:
                        continue  # 训练前掉线

                    loader = user_loaders_map[u.id]
                    reuse_rule = args.reuse_rule

                    masked_int = u.p2_train_mix_mask_and_upload(
                        global_flat=cloud.p2_global_model_history[base_v],
                        global_version=base_v,
                        package_token=pkg_token,
                        model_meta=model_meta,
                        train_loader=loader,
                        local_epochs=cfg['local_epochs'],
                        lr=cfg['lr'],
                        staleness_beta=args.staleness_beta,
                        reuse_rule=reuse_rule,
                        rho_max=args.rho_max,
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
                        sample_size=cfg['samples_per_user'])
                    online_count += 1
                    # ES 封组前确认接收 → 立即清除 stale 候选 (文档 L305-307)
                    has_stale, _ = u.p2_get_stale_info()
                    if has_stale:
                        tick_stale_reuse_count += 1
                    u.p2_confirm_consumed()

                # 封组
                if e._p2_online_users:
                    pkg = e.p2_seal_group_package(tick=current_tick)
                    cloud.p2_receive_group_package(pkg, arrival_tick=current_tick)

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
            # TA 构建 checkpoint 掩码
            M_Q, H_mask_raw = ta.p2_build_checkpoint_mask(selected)

            # Cloud 恢复 raw sum
            raw_sum_int = cloud.p2_recover_raw_sum(selected, M_Q)

            # Direct-VCM 补偿
            gamma = args.gamma_vcm if args.enable_vcm else 0.0
            corrected_sum, final_model, comp_map = cloud.p2_apply_vcm_and_average(
                selected, raw_sum_int, gamma_vcm=gamma)

            # 攻击注入 (独立于性能基线, 仅攻击实验时启用)
            if args.attack == 'tamper_global':
                tampered = final_model + rng.normal(0, 0.01, final_model.shape)
                logger.warning(f"Tick {current_tick}: [ATTACK] tamper_global")
                result = cloud.p2_commit_checkpoint(
                    selected, tampered, raw_sum_int, M_Q, H_mask_raw,
                    corrected_sum, comp_map)
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
                logger.warning(f"Tick {current_tick}: [ATTACK] tamper_manifest")
                result = cloud.p2_commit_checkpoint(
                    selected_attacked, final_model, raw_sum_int, M_Q, H_mask_raw,
                    corrected_sum, comp_map)
            elif args.attack == 'drop_replace_pkg':
                # 漏聚合某个组包
                if len(selected) > 1:
                    selected_attacked = selected[:-1]
                    logger.warning(f"Tick {current_tick}: [ATTACK] drop_replace_pkg "
                                    f"(removed {selected[-1].package_token})")
                    result = cloud.p2_commit_checkpoint(
                        selected_attacked, final_model, raw_sum_int, M_Q, H_mask_raw,
                        corrected_sum, comp_map)
                else:
                    result = cloud.p2_commit_checkpoint(
                        selected, final_model, raw_sum_int, M_Q, H_mask_raw,
                        corrected_sum, comp_map)
            else:
                result = cloud.p2_commit_checkpoint(
                    selected, final_model, raw_sum_int, M_Q, H_mask_raw,
                    corrected_sum, comp_map)

            # ES 验证
            bundle = {
                'checkpoint_version': result.version,
                'selected_packages': selected,
                'raw_sum_int': raw_sum_int,
                'M_Q': M_Q,
                'H_mask_raw': H_mask_raw,
                'final_model': final_model,
                'gamma_vcm': gamma,
                'global_model_history': cloud.p2_global_model_history,
            }

            all_passed = True
            selected_edge_ids = {p.edge_id for p in selected}
            for e in edges:
                if e.id in selected_edge_ids:
                    if not e.p2_verify_checkpoint(bundle):
                        all_passed = False
                        logger.error(f"Tick {current_tick}: {e.id} 验证失败!")

            result.verification_passed = all_passed

            if all_passed:
                checkpoint_count += 1

                # 指标记录
                p2_metrics['checkpoint_ticks'].append(current_tick)
                p2_metrics['stale_reuse_counts'].append(tick_stale_reuse_count)
                for pkg in selected:
                    wait_time = (current_tick -
                                 cloud.p2_arrival_ticks.get(pkg.package_token, 0))
                    p2_metrics['wait_times'].append(wait_time)
                p2_metrics['comp_gain'] += sum(comp_map.values())
                # throughput: 每 tick 处理的有效样本数
                prev_tick = (p2_metrics['checkpoint_ticks'][-2]
                            if len(p2_metrics['checkpoint_ticks']) > 1 else 0)
                p2_metrics['throughput'].append(
                    result.S_Q / max(1, p2_metrics['checkpoint_ticks'][-1] - prev_tick))

                # 评估
                eval_model = copy.deepcopy(init_model)
                models.unflatten_parameters(eval_model, final_model, model_meta)
                acc, loss = models.evaluate(eval_model, test_loader)
                acc_history.append((checkpoint_count, acc))

                logger.info(
                    f"✓ Ckp {checkpoint_count} (tick {current_tick}): "
                    f"v{result.version}→v{cloud.p2_current_version} "
                    f"N_Q={result.N_Q} S_Q={result.S_Q} comp={sum(comp_map.values()):.4f} "
                    f"stale_users={tick_stale_reuse_count} Acc={acc:.2f}%"
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

    logger.info(f"\n=== 模拟完成: {checkpoint_count} checkpoints, "
                 f"{current_tick} ticks, 墙钟 {wc_elapsed:.1f}s ===")
    logger.info(f"Accuracy history: {acc_history}")
    logger.info(f"--- 指标 ---")
    logger.info(f"  最终准确率: {final_acc:.2f}%")
    logger.info(f"  墙钟时间: {wc_elapsed:.1f}s")
    logger.info(f"  平均等待: {avg_wait:.1f} ticks")
    logger.info(f"  最大等待: {max_wait:.1f} ticks")
    logger.info(f"  Compensation gain: {p2_metrics['comp_gain']:.4f}")
    logger.info(f"  Stale reuse 总人次: {total_stale_reuses}")
    logger.info(f"  组包过期移除数: {expired_from_buffer}")
    logger.info(f"  Checkpoint 中 stale 组包数: {stale_in_checkpoint}")
    logger.info(f"  组间公平性 (选中次数方差): {fairness_var:.4f}")
    logger.info(f"  有效吞吐量 (sample/tick): {np.mean(p2_metrics['throughput']):.1f}" if p2_metrics['throughput'] else "  有效吞吐量: N/A")
    logger.info(f"  Verification passed: {sum(1 for r in cloud.p2_checkpoint_log if r.verification_passed)}/{len(cloud.p2_checkpoint_log)}")

    # --- 6. 画图 ---
    if acc_history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左: 准确率 vs checkpoint
        ckps, accs = zip(*acc_history)
        axes[0].plot(ckps, accs, 'b-o', lw=2)
        axes[0].set_xlabel('Checkpoint')
        axes[0].set_ylabel('Test Accuracy (%)')
        title_parts = [f'{args.dataset}']
        if args.reuse_rule != 'none':
            title_parts.append(f'+StaleReuse({args.reuse_rule})')
        if args.enable_vcm:
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
        plot_path = f'point2_plot_{args.dataset}_{timestamp}.png'
        plt.savefig(plot_path, dpi=150)
        logger.info(f"图表已保存: {plot_path}")

    # --- 7. 保存数据 ---
    np.savez(
        f'point2_data_{args.dataset}_{timestamp}.npz',
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
        config=str(args),
    )
    logger.info("数据已保存.")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVSA Point2: Semi-Async Secure Aggregation")
    parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10'])
    parser.add_argument('--num-edge-servers', type=int, default=5)
    parser.add_argument('--users-per-edge', type=int, default=4)
    parser.add_argument('-K', type=int, default=3, help='checkpoint trigger threshold')
    parser.add_argument('--s-max', type=int, default=3, help='max staleness')
    parser.add_argument('--max-checkpoints', type=int, default=20)
    parser.add_argument('--max-ticks', type=int, default=300)
    # stale reuse (E5: 三规则消融)
    parser.add_argument('--reuse-rule', default='ours', choices=['none', 'mafsa', 'ours'],
                        help='stale reuse rule')
    parser.add_argument('--rho-max', type=float, default=0.5, help='max stale weight (ours only)')
    parser.add_argument('--staleness-beta', type=float, default=0.3, help='beta_c decay')
    # VCM
    parser.add_argument('--vcm', action='store_true', help='enable Direct-VCM compensation')
    parser.add_argument('--gamma-vcm', type=float, default=1.0, help='VCM compensation strength')
    # buffer/checkpoint
    parser.add_argument('--delta-timeout', type=int, default=0, help='timeout trigger limit (0=disabled)')
    parser.add_argument('--policy', default='fifo', choices=['fifo', 'fresh_first', 'fair_fresh_first'],
                        help='buffer selection policy')
    parser.add_argument('--D-fair', type=int, default=2, help='fairness defer threshold')
    # dropout
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='pre-training dropout rate')
    parser.add_argument('--upload-dropout-rate', type=float, default=0.0,
                        help='post-training upload failure rate (stale source)')
    # verification & attack
    parser.add_argument('--verify', action='store_true', help='enable LHH/Ed25519 verification')
    parser.add_argument('--attack', default='none',
                        choices=['none', 'tamper_global', 'tamper_manifest', 'drop_replace_pkg'],
                        help='attack type (separate from performance baseline)')

    args = parser.parse_args()
    run_point2_simulation(args)
