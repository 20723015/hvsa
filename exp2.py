import logging
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import concurrent.futures

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
import models
from datasets import load_data, distribute_data

TARGET_DATASET = 'MNIST'

CONFIGS = {
    'MNIST': {
        'model': 'CNN',
        'rounds': 50,
        'local_epochs': 1,
        'lr': 0.001,
        'samples_per_user': 350,
        'batch_size': 64
    },
    'FashionMNIST': {
        'model': 'CNN',
        'rounds': 40,
        'local_epochs': 2,
        'lr': 0.01,
        'samples_per_user': 600,
        'batch_size': 32
    },
    'CIFAR10': {
        'model': 'CNN',
        'rounds': 50,
        'local_epochs': 5,
        'lr': 0.01,
        'samples_per_user': 2000,
        'batch_size': 64
    }
}

CFG = CONFIGS[TARGET_DATASET]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]
NUM_EDGE_SERVERS = 2
USERS_PER_EDGE = 5
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiment_with_dropout(train_loaders, test_loader, initial_model, dropout_rate):
    logging.info(f"\n>>> 开始测试: 掉线率 = {dropout_rate * 100}% <<<")

    flat_params, model_meta = models.flatten_parameters(initial_model)
    gradient_dim = flat_params.shape[0]

    ta = TrustedAuthority(gradient_dim=gradient_dim, ta_id="TA-01")
    cloud = CloudServer()
    edge_servers = [EdgeServer(f"ES-{i}", cloud, ta, enable_verification=False)
                    for i in range(NUM_EDGE_SERVERS)]

    users = []
    user_edge_map = {}
    ta_pub = ta.get_public_key_bytes()

    for i in range(TOTAL_USERS):
        u = User(f"U-{i}", copy.deepcopy(initial_model))
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        users.append(u)
        user_edge_map[u.id] = edge_servers[i // USERS_PER_EDGE]

    acc_history = []
    global_model_vector = flat_params

    total_rounds = CFG['rounds']
    local_epochs = CFG['local_epochs']
    lr = CFG['lr']

    for r in range(total_rounds):
        round_t = r + 1

        cloud.start_new_round(round_t, gradient_dim, NUM_EDGE_SERVERS)
        ta.start_new_round(round_t)
        for es in edge_servers:
            es.start_new_round(round_t)

        active_users_info = []
        for i, u in enumerate(users):
            if np.random.rand() >= dropout_rate:
                active_users_info.append((i, u))

        active_user_count = len(active_users_info)

        def train_single_client(client_info):
            idx, client = client_info
            client.set_global_model(global_model_vector, model_meta)
            client.local_train(train_loaders[idx], local_epochs, lr)
            client.mask_and_upload_model(round_t, user_edge_map[client.id])

        if active_user_count > 0:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(train_single_client, active_users_info)

        for es in edge_servers:
            es.trigger_aggregation_and_forward()

        ta.trigger_global_mask_computation(cloud)
        global_avg_model, _, _ = cloud.trigger_final_unmasking_and_averaging()

        if global_avg_model is not None:
            global_model_vector = global_avg_model
            users[0].set_global_model(global_model_vector, model_meta)
            acc, _ = models.evaluate(users[0].model, test_loader)
            acc_history.append(acc)

            logging.info(f"[Dropout {int(dropout_rate * 100)}%] Round {round_t}/{total_rounds} | "
                         f"Online: {active_user_count}/{TOTAL_USERS} | Acc: {acc:.2f}%")
        else:
            logging.warning(f"[Dropout {int(dropout_rate * 100)}%] Round {round_t} | No users online!")
            acc_history.append(acc_history[-1] if acc_history else 0)

    return acc_history


def main():
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    LOG_FILENAME = f'exp2_dropout_log_{TARGET_DATASET}_{TIMESTAMP}.txt'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logging.info(f"=== 实验 2: 掉线率鲁棒性测试 ({TARGET_DATASET}) ===")
    logging.info(f"本次实验 ID: {TIMESTAMP}")
    logging.info(f"配置: {CFG}")
    logging.info(f"测试掉线率: {DROPOUT_RATES}")
    logging.info("Verification Mode: accuracy-only fast mode")

    import sys as _sys2
    _use_iid = '--noniid' not in _sys2.argv
    _alpha = float(_sys2.argv[_sys2.argv.index('--alpha') + 1]) if '--alpha' in _sys2.argv else 0.5

    full_train_dataset, full_test_dataset = load_data(TARGET_DATASET)
    user_loaders = distribute_data(full_train_dataset, TOTAL_USERS, CFG['batch_size'],
                                   iid=_use_iid,
                                   dirichlet_alpha=_alpha,
                                   max_samples_per_user=CFG['samples_per_user'])
    test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)

    initial_model = models.get_model(TARGET_DATASET, CFG['model'])

    results = {}
    for rate in DROPOUT_RATES:
        acc_curve = run_experiment_with_dropout(user_loaders, test_loader, initial_model, rate)
        results[rate] = acc_curve

    data_file = f'exp2_data_{TARGET_DATASET}_{TIMESTAMP}.npz'
    np.savez(data_file, **{f'rate_{r}': acc for r, acc in results.items()})
    logging.info(f"实验数据已保存至: {data_file}")

    plt.figure(figsize=(10, 7))
    epochs = range(1, CFG['rounds'] + 1)
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, rate in enumerate(DROPOUT_RATES):
        plt.plot(epochs, results[rate],
                 label=f'Dropout Rate = {int(rate * 100)}%',
                 color=colors[i % len(colors)], linewidth=2)

    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_file = f'exp2_plot_{TARGET_DATASET}_{TIMESTAMP}.png'
    plt.savefig(plot_file, dpi=300)
    logging.info(f"对比图已保存至: {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()
