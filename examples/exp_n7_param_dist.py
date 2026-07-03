"""
N7: 聚合模型参数分布一致性展示

参考 VLMRS Fig.6，在 HVSA 验证通过的前提下，
展示本地模型参数分布与聚合模型参数分布一致，
为验证机制提供可视化辅助证据。

运行: python exp_n7_param_dist.py
     python exp_n7_param_dist.py --dataset CIFAR10
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy, logging, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
import models
from datasets import load_data, distribute_data

# ============================================================
DATASET = 'MNIST'
NUM_EDGE = 3
USERS_PER_EDGE = 10
TOTAL_USERS = NUM_EDGE * USERS_PER_EDGE
ROUNDS = 50
LOCAL_EPOCHS = 1
LR = 0.01
BATCH_SIZE = 64
SAMPLES_PER_USER = 300
MODEL_NAME = 'CNN'
VERIFY = False

SAMPLE_EPOCHS = [20, 50]           # 早/晚期采样点
NUM_SAMPLE_USERS = 4               # 随机选 4 个用户展示
LAYERS_TO_PLOT = [                 # 要画哪些层
    ('conv1.weight', 'Conv1 Weight'),
    ('conv1.bias',   'Conv1 Bias'),
    ('conv2.weight', 'Conv2 Weight'),
    ('fc1.weight',   'FC1 Weight'),
]

TIMESTAMP = time.strftime('%Y%m%d-%H%M%S')
# ============================================================


def run_and_capture():
    train_ds, test_ds = load_data(DATASET)
    loaders = distribute_data(train_ds, TOTAL_USERS, BATCH_SIZE,
                               iid=True, max_samples_per_user=SAMPLES_PER_USER)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    init_model = models.get_model(DATASET, MODEL_NAME)
    init_flat, model_meta = models.flatten_parameters(init_model)
    dim = init_flat.shape[0]

    # 系统初始化
    ta = TrustedAuthority(gradient_dim=dim, ta_id='TA-01')
    cloud = CloudServer()
    edges = [EdgeServer(f'ES-{i}', cloud, ta, enable_verification=VERIFY)
             for i in range(NUM_EDGE)]

    users = []
    user_edge = {}
    ta_pub = ta.get_public_key_bytes()
    for i in range(TOTAL_USERS):
        u = User(f'U-{i}', copy.deepcopy(init_model))
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        users.append(u)
        user_edge[u.id] = edges[i // USERS_PER_EDGE]

    global_vec = init_flat.astype(np.float64).copy()
    captured = {}  # epoch -> {user_params, global_params}

    for r in range(ROUNDS):
        round_t = r + 1
        cloud.start_new_round(round_t, dim, NUM_EDGE)
        ta.start_new_round(round_t)
        for es in edges:
            es.start_new_round(round_t)

        for i, u in enumerate(users):
            u.set_global_model(global_vec, model_meta)
            u.local_train(loaders[i], LOCAL_EPOCHS, LR)
            u.mask_and_upload_model(round_t, user_edge[u.id])

        for es in edges:
            es.trigger_aggregation_and_forward()
        ta.trigger_global_mask_computation(cloud)
        avg_model, _, _ = cloud.trigger_final_unmasking_and_averaging()

        if avg_model is not None:
            global_vec = avg_model

        # 在指定 epoch 采样
        if round_t in SAMPLE_EPOCHS:
            rng = np.random.default_rng(round_t)
            sampled_users = rng.choice(users, NUM_SAMPLE_USERS, replace=False)

            local_params = []
            for u in sampled_users:
                u.set_global_model(global_vec, model_meta)
                local_params.append({
                    name: param.detach().cpu().numpy().flatten()
                    for name, param in u.model.named_parameters()
                    if any(l[0] == name for l in LAYERS_TO_PLOT)
                })

            # 全局模型
            tmp = copy.deepcopy(init_model)
            models.unflatten_parameters(tmp, global_vec, model_meta)
            global_params = {
                name: param.detach().cpu().numpy().flatten()
                for name, param in tmp.named_parameters()
                if any(l[0] == name for l in LAYERS_TO_PLOT)
            }

            captured[round_t] = {
                'local': local_params,
                'global': global_params,
            }
            logging.info(f'Epoch {round_t}: 已采样 {len(sampled_users)} 用户参数')

        if round_t % 10 == 0:
            tmp = copy.deepcopy(init_model)
            models.unflatten_parameters(tmp, global_vec, model_meta)
            acc, _ = models.evaluate(tmp, test_loader)
            logging.info(f'Round {round_t}/{ROUNDS} | Acc: {acc:.2f}%')

    return captured


def plot_distributions(captured):
    n_rows = len(LAYERS_TO_PLOT)
    n_cols = len(SAMPLE_EPOCHS) * 2  # local + global per epoch
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # 4 users

    for row, (layer_name, layer_label) in enumerate(LAYERS_TO_PLOT):
        for col, epoch in enumerate(SAMPLE_EPOCHS):
            data = captured[epoch]

            # 本地参数分布（左列）
            ax_local = axes[row][col * 2]
            local_vals = [u_params[layer_name] for u_params in data['local']]
            local_all = np.concatenate(local_vals)
            ax_local.hist(local_vals, bins=40, color=colors[:len(local_vals)],
                          alpha=0.6, stacked=False, density=True, histtype='stepfilled')
            ax_local.set_title(f'{layer_label}\nEpoch {epoch} — Local (4 users)',
                               fontsize=9)
            ax_local.set_ylabel('Density')
            ax_local.grid(True, linestyle='--', alpha=0.4)

            # 聚合模型分布（右列）
            ax_global = axes[row][col * 2 + 1]
            global_vals = data['global'][layer_name]
            ax_global.hist(global_vals, bins=40, color='#ff7f00', alpha=0.8,
                           density=True)
            ax_global.set_title(f'{layer_label}\nEpoch {epoch} — Aggregated',
                                fontsize=9)
            ax_global.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = f'exp_n7_param_dist_{DATASET}_{TIMESTAMP}.png'
    plt.savefig(save_path, dpi=200)
    logging.info(f'图表已保存: {save_path}')
    plt.show()


def main():
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info(f'=== N7: 参数分布一致性展示 ({DATASET}) ===')
    captured = run_and_capture()
    plot_distributions(captured)


if __name__ == '__main__':
    main()
