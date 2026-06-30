import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
import models
from datasets import load_data, distribute_data

TARGET_DATASET = 'CIFAR10'

CONFIGS = {
    'MNIST': {
        'model': 'CNN',
        'rounds': 50,
        'local_epochs': 1,
        'lr': 0.01,
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
NUM_EDGE_SERVERS = 2
USERS_PER_EDGE = 5
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_VERIFICATION_MODE = False


def run_standard_fedavg(train_loaders, test_loader, initial_model):
    logging.info(f"--- [Baseline] Standard FedAvg ({TARGET_DATASET}) ---")

    global_model = copy.deepcopy(initial_model).to(DEVICE)
    acc_history = []
    user_models = [copy.deepcopy(initial_model).to(DEVICE) for _ in range(TOTAL_USERS)]

    for r in range(CFG['rounds']):
        for i in range(TOTAL_USERS):
            u_model = user_models[i]
            u_model.load_state_dict(global_model.state_dict())
            models.train_local(u_model, train_loaders[i], CFG['local_epochs'], CFG['lr'])

        edge_aggregated_weights = []
        for edge_idx in range(NUM_EDGE_SERVERS):
            start = edge_idx * USERS_PER_EDGE
            end = start + USERS_PER_EDGE
            edge_user_weights = [user_models[i].state_dict() for i in range(start, end)]

            edge_template = copy.deepcopy(edge_user_weights[0])
            for key in edge_template.keys():
                user_params_tensor = torch.stack([w[key].float() for w in edge_user_weights], dim=0)
                edge_template[key] = torch.mean(user_params_tensor, dim=0)

            edge_aggregated_weights.append(edge_template)

        global_state = global_model.state_dict()
        for key in global_state.keys():
            edge_params_tensor = torch.stack([w[key] for w in edge_aggregated_weights], dim=0)
            global_state[key] = torch.mean(edge_params_tensor, dim=0)

        global_model.load_state_dict(global_state)

        acc, _ = models.evaluate(global_model, test_loader)
        acc_history.append(acc)
        logging.info(f"[FedAvg] Round {r + 1}/{CFG['rounds']} | Acc: {acc:.2f}%")

    return acc_history


def run_proposed_scheme(train_loaders, test_loader, initial_model):
    logging.info(f"--- [Proposed] HVSA ({TARGET_DATASET}) ---")

    flat_params, model_meta = models.flatten_parameters(initial_model)
    gradient_dim = flat_params.shape[0]

    ta = TrustedAuthority(gradient_dim=gradient_dim, ta_id="TA-01")
    cloud = CloudServer()
    edge_servers = [EdgeServer(f"ES-{i}", cloud, ta, enable_verification=ENABLE_VERIFICATION_MODE)
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

    for r in range(CFG['rounds']):
        round_t = r + 1
        cloud.start_new_round(round_t, gradient_dim, NUM_EDGE_SERVERS)
        ta.start_new_round(round_t)
        for es in edge_servers:
            es.start_new_round(round_t)

        for i, u in enumerate(users):
            u.set_global_model(global_model_vector, model_meta)
            u.local_train(train_loaders[i], CFG['local_epochs'], CFG['lr'])
            u.mask_and_upload_model(round_t, user_edge_map[u.id])

        for es in edge_servers:
            es.trigger_aggregation_and_forward()

        ta.trigger_global_mask_computation(cloud)
        global_avg_model, _, _ = cloud.trigger_final_unmasking_and_averaging()

        if global_avg_model is not None:
            global_model_vector = global_avg_model
            users[0].set_global_model(global_model_vector, model_meta)
            acc, _ = models.evaluate(users[0].model, test_loader)
            acc_history.append(acc)
            logging.info(f"[Proposed] Round {round_t}/{CFG['rounds']} | Acc: {acc:.2f}%")
        else:
            acc_history.append(0)

    return acc_history


def main():
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

    LOG_FILENAME = f'exp1_log_{TARGET_DATASET}_{TIMESTAMP}.txt'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        force=True,
        handlers=[
            logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info(f"=== 实验配置: {TARGET_DATASET} ===")
    logging.info(f"本次实验 ID (时间戳): {TIMESTAMP}")
    logging.info(f"日志将保存至: {LOG_FILENAME}")
    logging.info(f"Current Config: {TARGET_DATASET} | Samples/User: {CFG['samples_per_user']} | Model: {CFG['model']}")
    if not ENABLE_VERIFICATION_MODE:
        logging.info("Verification Mode: accuracy-only fast mode")

    full_train_dataset, full_test_dataset = load_data(TARGET_DATASET)
    import sys as _sys
    _use_iid = '--noniid' not in _sys.argv
    _alpha = float(_sys.argv[_sys.argv.index('--alpha') + 1]) if '--alpha' in _sys.argv else 0.5
    user_loaders = distribute_data(full_train_dataset, TOTAL_USERS, CFG['batch_size'],
                                   iid=_use_iid,
                                   dirichlet_alpha=_alpha,
                                   max_samples_per_user=CFG['samples_per_user'])

    test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)

    initial_model = models.get_model(TARGET_DATASET, CFG['model'])

    acc_fedavg = run_standard_fedavg(user_loaders, test_loader, initial_model)
    acc_ours = run_proposed_scheme(user_loaders, test_loader, initial_model)

    data_file = f'exp1_data_{TARGET_DATASET}_{TIMESTAMP}.npz'
    np.savez(data_file, fedavg=acc_fedavg, ours=acc_ours)
    logging.info(f"实验数据已保存至: {data_file}")

    plt.figure(figsize=(8, 6))
    epochs = range(1, CFG['rounds'] + 1)

    plt.plot(epochs, acc_fedavg, 'k--', label='FedAvg', linewidth=2)
    plt.plot(epochs, acc_ours, 'r-', label='Proposed', linewidth=2, alpha=0.8)

    plt.title(f'Model Accuracy Convergence ({TARGET_DATASET})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = f'exp1_convergence_{TARGET_DATASET}_{TIMESTAMP}.png'
    plt.savefig(save_path, dpi=300)
    logging.info(f"结果图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
