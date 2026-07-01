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
    'dropout_rate': 0.1,        # 每 tick 客户端掉线率
    'enable_stale_reuse': False,
    'enable_vcm': False,
    'enable_verification': False,
    'min_delay': 1,             # 最快组几个 tick
    'max_delay': 5,             # 最慢组几个 tick
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
                for u in users:
                    if user_edge_map[u.id] != e:
                        continue
                    if rng.random() < args.dropout_rate:
                        continue

                    loader = user_loaders_map[u.id]

                    if args.enable_stale_reuse:
                        masked_int = u.p2_train_mix_mask_and_upload(
                            global_flat=cloud.p2_global_model_history[base_v],
                            global_version=base_v,
                            package_token=pkg_token,
                            model_meta=model_meta,
                            train_loader=loader,
                            local_epochs=cfg['local_epochs'],
                            lr=cfg['lr'],
                            staleness_beta=args.staleness_beta,
                        )
                    else:
                        # 无 stale reuse: 训练后直接掩码
                        u.set_global_model(
                            cloud.p2_global_model_history[base_v], model_meta)
                        u.local_train(loader, cfg['local_epochs'], cfg['lr'])
                        new_flat, _ = models.flatten_parameters(u.model)
                        dim_f = new_flat.shape[0]
                        model_int = (new_flat * 1_000_000).astype(np.int64)
                        mask_int = u._generate_ta_mask_int(pkg_token, dim_f)
                        masked_int = model_int + mask_int

                    if masked_int is not None:
                        import pickle as _pkl
                        msg = {"user_id": u.id, "masked_gradient": masked_int}
                        e.p2_receive_masked_gradient(
                            _pkl.dumps(msg), pkg_token, base_v)
                        online_count += 1

                # 封组
                if e._p2_online_users:
                    pkg = e.p2_seal_group_package(tick=current_tick)
                    cloud.p2_receive_group_package(pkg)

                edge_state[e.id] = 'idle'

        # 4c. Checkpoint 触发
        selected = cloud.p2_select_checkpoint(K=args.K, s_max=args.s_max)

        if selected:
            # TA 构建 checkpoint 掩码
            M_Q, H_mask_raw = ta.p2_build_checkpoint_mask(selected)

            # Cloud 恢复 raw sum
            raw_sum_int = cloud.p2_recover_raw_sum(selected, M_Q)

            # Direct-VCM 补偿
            gamma = args.gamma_vcm if args.enable_vcm else 0.0
            corrected_sum, final_model, comp_map = cloud.p2_apply_vcm_and_average(
                selected, raw_sum_int, gamma_vcm=gamma)

            # 提交 checkpoint
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
                # 被选中组的用户更新缓存
                for pkg in selected:
                    for uid in pkg.online_users:
                        u = next((uu for uu in users if uu.id == uid), None)
                        if u:
                            u.p2_set_accepted_cache(
                                final_model, cloud.p2_current_version)

                checkpoint_count += 1

                # 评估
                eval_model = copy.deepcopy(init_model)
                models.unflatten_parameters(eval_model, final_model, model_meta)
                acc, loss = models.evaluate(eval_model, test_loader)
                acc_history.append((checkpoint_count, acc))

                logger.info(
                    f"✓ Ckp {checkpoint_count} (tick {current_tick}): "
                    f"v{result.version}→v{cloud.p2_current_version} "
                    f"N_Q={result.N_Q} comp={sum(comp_map.values()):.4f} "
                    f"Acc={acc:.2f}%"
                )

        # 每 10 tick 打印一次状态
        if current_tick % 10 == 0:
            logger.debug(
                f"Tick {current_tick}: buffer={len(cloud.p2_package_buffer)} "
                f"checkpoints={checkpoint_count} v={cloud.p2_current_version}"
            )

    # --- 5. 结果 ---
    logger.info(f"\n=== 模拟完成: {checkpoint_count} checkpoints, "
                 f"{current_tick} ticks ===")
    logger.info(f"Accuracy history: {acc_history}")

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
        if args.enable_stale_reuse:
            title_parts.append('+StaleReuse')
        if args.enable_vcm:
            title_parts.append('+VCM')
        axes[0].set_title(' '.join(title_parts))
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 右: Checkpoint 规模统计
        Ns = [r.N_Q for r in cloud.p2_checkpoint_log]
        stale_counts = [
            sum(1 for p in r.selected_packages if p.base_version < r.version)
            for r in cloud.p2_checkpoint_log
        ]
        x = range(1, len(Ns) + 1)
        axes[1].bar(x, Ns, alpha=0.6, label='Total users N_Q')
        axes[1].bar(x, stale_counts, alpha=0.8, label='Stale packages')
        axes[1].set_xlabel('Checkpoint')
        axes[1].set_ylabel('Count')
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
    parser.add_argument('--gamma-vcm', type=float, default=1.0)
    parser.add_argument('--staleness-beta', type=float, default=0.3)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--stale-reuse', action='store_true')
    parser.add_argument('--vcm', action='store_true')
    parser.add_argument('--verify', action='store_true', help='enable LHH/Ed25519 verification')

    args = parser.parse_args()
    run_point2_simulation(args)
