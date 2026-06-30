"""
实验: Non-IID + 梯度泄露防御综合评估

用法:
    python exp_noniid.py                      # IID + Non-IID(alpha=0.5) 收敛对比
    python exp_noniid.py --alpha 1.0          # 指定 Dirichlet alpha
    python exp_noniid.py --dlg                # 跑 DLG 攻击防御
    python exp_noniid.py --all                # 全部实验
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy, logging, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
import models
from datasets import load_data, distribute_data

TARGET_DATASET = 'MNIST'

CONFIGS = {
    'MNIST':    {'model': 'CNN', 'rounds': 100, 'local_epochs': 1, 'lr': 0.01, 'samples_per_user': 300, 'batch_size': 64},
    'CIFAR10':  {'model': 'CNN', 'rounds': 100, 'local_epochs': 5, 'lr': 0.01, 'samples_per_user': 1500, 'batch_size': 64},
}

CFG = CONFIGS[TARGET_DATASET]
NUM_EDGE_SERVERS = 4
USERS_PER_EDGE = 25
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_VERIFICATION_MODE = True


def run_proposed(train_loaders, test_loader, initial_model, desc="HVSA"):
    flat_params, model_meta = models.flatten_parameters(initial_model)
    dim = flat_params.shape[0]

    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01")
    cloud = CloudServer()
    edges = [EdgeServer(f"ES-{i}", cloud, ta,
                        enable_verification=ENABLE_VERIFICATION_MODE)
             for i in range(NUM_EDGE_SERVERS)]

    users, user_edge = [], {}
    ta_pub = ta.get_public_key_bytes()
    for i in range(TOTAL_USERS):
        u = User(f"U-{i}", copy.deepcopy(initial_model))
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        users.append(u)
        user_edge[u.id] = edges[i // USERS_PER_EDGE]

    acc_history = []
    vec = flat_params

    for r in range(CFG['rounds']):
        cloud.start_new_round(r + 1, dim, NUM_EDGE_SERVERS)
        ta.start_new_round(r + 1)
        for es in edges:
            es.start_new_round(r + 1)

        for i, u in enumerate(users):
            u.set_global_model(vec, model_meta)
            u.local_train(train_loaders[i], CFG['local_epochs'], CFG['lr'])
            u.mask_and_upload_model(r + 1, user_edge[u.id])

        for es in edges:
            es.trigger_aggregation_and_forward()
        ta.trigger_global_mask_computation(cloud)
        avg, _, _ = cloud.trigger_final_unmasking_and_averaging()

        if avg is not None:
            vec = avg
            users[0].set_global_model(vec, model_meta)
            acc, _ = models.evaluate(users[0].model, test_loader)
            acc_history.append(acc)
            if (r + 1) % 20 == 0:
                logging.info(f"[{desc}] Round {r+1}/{CFG['rounds']} | Acc: {acc:.2f}%")
        else:
            acc_history.append(acc_history[-1] if acc_history else 0)

    return acc_history


def run_fedavg(train_loaders, test_loader, initial_model):
    global_model = copy.deepcopy(initial_model).to(DEVICE)
    user_models = [copy.deepcopy(initial_model).to(DEVICE) for _ in range(TOTAL_USERS)]
    acc_history = []

    for r in range(CFG['rounds']):
        for i in range(TOTAL_USERS):
            user_models[i].load_state_dict(global_model.state_dict())
            models.train_local(user_models[i], train_loaders[i], CFG['local_epochs'], CFG['lr'])

        global_state = global_model.state_dict()
        for key in global_state:
            stacked = torch.stack([user_models[i].state_dict()[key].float() for i in range(TOTAL_USERS)], dim=0)
            global_state[key] = torch.mean(stacked, dim=0)
        global_model.load_state_dict(global_state)

        acc, _ = models.evaluate(global_model, test_loader)
        acc_history.append(acc)

    return acc_history


def run_dlg_defense_exp():
    """DLG 攻击防御"""
    PRECISION = 1000000
    MASK_RANGE = 2 ** 30
    TASKS = {
        'MNIST':    {'checkpoints': [0, 20, 40, 60], 'max_iter': 65, 'model': 'CNN', 'lr': 1.0, 'tv_reg': 1e-6},
        'CIFAR10':  {'checkpoints': [0, 60, 120, 180], 'max_iter': 185, 'model': 'CNN', 'lr': 1.0, 'tv_reg': 1e-3},
    }

    all_results = {}; ground_truths = {}
    for ds in ['MNIST', 'CIFAR10']:
        cfg_d = TASKS[ds]
        train_ds, _ = load_data(ds)
        idx = np.random.randint(len(train_ds))
        gt_data, gt_label = train_ds[idx]
        gt_data = gt_data.view(1, *gt_data.shape).to(DEVICE)
        gt_label = torch.tensor([gt_label]).view(1).to(DEVICE)
        ground_truths[ds] = gt_data

        model = models.get_model(ds, cfg_d['model']).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        model.zero_grad()
        pred = model(gt_data)
        loss = criterion(pred, gt_label)
        loss.backward()
        original_dy = [p.grad.clone() for p in model.parameters()]

        all_results[ds] = {}
        all_results[ds]['FedAvg'] = _dlg_attack(model, original_dy, gt_data.shape, gt_label, cfg_d, f"{ds}-FedAvg")

        logging.info(f"  [{ds}] HVSA 加密梯度...")
        enc_dy = []
        for g in original_dy:
            grad_int = (g * PRECISION).type(torch.long)
            mask = torch.randint(-MASK_RANGE, MASK_RANGE, g.shape, device=DEVICE)
            enc_dy.append((grad_int + mask).float())
        all_results[ds]['Ours'] = _dlg_attack(model, enc_dy, gt_data.shape, gt_label, cfg_d, f"{ds}-Ours")

    _plot_dlg(ground_truths, all_results, TASKS)


def _dlg_attack(model, target_grad, shape, label, cfg_d, label_text):
    dummy = torch.randn(shape).to(DEVICE).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy], lr=cfg_d['lr'])
    snapshots = {}; state = {'iter': 0}
    if 0 in cfg_d['checkpoints']:
        snapshots[0] = dummy.detach().clone().cpu()

    def closure():
        opt.zero_grad(); dpred = model(dummy)
        dl = F.cross_entropy(dpred, label)
        ddy = torch.autograd.grad(dl, model.parameters(), create_graph=True)
        gd = sum(((gx - gy)**2).sum() for gx, gy in zip(ddy, target_grad))
        tv = sum((dummy[:,:,1:,:] - dummy[:,:,:-1,:]).pow(2).sum() + (dummy[:,:,:,1:] - dummy[:,:,:,:-1]).pow(2).sum())
        total = gd + tv * cfg_d['tv_reg']
        total.backward(); return total

    for _ in range(cfg_d['max_iter']):
        try: opt.step(closure)
        except: pass
        if torch.isnan(dummy).any() or torch.isinf(dummy).any():
            with torch.no_grad(): dummy.data.normal_()
        state['iter'] += 1
        if state['iter'] in cfg_d['checkpoints']:
            snapshots[state['iter']] = dummy.detach().clone().cpu()
    return snapshots


def _tensor_to_img(t):
    t = t.squeeze(0)
    img = t.squeeze(0).numpy() if t.shape[0] == 1 else t.permute(1,2,0).numpy()
    if np.isnan(img).any() or np.isinf(img).any():
        return np.random.rand(*img.shape)
    img = img - img.min()
    if img.max() > 0: img = img / img.max()
    return img


def _plot_dlg(gts, res, tasks):
    fig, axes = plt.subplots(4, 5, figsize=(16, 14))
    configs = [('MNIST','FedAvg','MNIST\n(FedAvg)'),('MNIST','Ours','MNIST\n(Ours)'),
               ('CIFAR10','FedAvg','CIFAR10\n(FedAvg)'),('CIFAR10','Ours','CIFAR10\n(Ours)')]
    for ri, (ds, met, title) in enumerate(configs):
        cps = tasks[ds]['checkpoints']
        gt = _tensor_to_img(gts[ds].cpu())
        ax = axes[ri, 0]
        ax.imshow(gt, cmap='gray' if len(gt.shape)==2 else None, interpolation='nearest')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        if ri in (0,2): ax.set_title("Original", fontsize=11, fontweight='bold')
        ax.axis('off')
        snaps = res[ds][met]
        for ci, it in enumerate(cps):
            ax = axes[ri, ci+1]
            if it in snaps:
                img = _tensor_to_img(snaps[it])
                ax.imshow(img, cmap='gray' if len(img.shape)==2 else None, interpolation='nearest')
                if ri in (0,2): ax.set_title(f"Iter {it}", fontsize=11, fontweight='bold')
            ax.axis('off')
    plt.tight_layout(); plt.subplots_adjust(hspace=0.4)
    plt.savefig('exp_noniid_dlg_defense.png', dpi=300)
    plt.show()


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for h in logging.root.handlers[:]: logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(f'exp_noniid_{TARGET_DATASET}_{timestamp}.txt','w','utf-8'),
                                  logging.StreamHandler()])

    do_iid = '--noniid-only' not in sys.argv
    do_dlg = '--dlg' in sys.argv or '--all' in sys.argv
    _alpha = float(sys.argv[sys.argv.index('--alpha')+1]) if '--alpha' in sys.argv else 0.5

    logging.info(f"=== Non-IID + DLG 综合实验 ===")
    logging.info(f"Dataset: {TARGET_DATASET} | Users: {TOTAL_USERS} | ES: {NUM_EDGE_SERVERS} | Rounds: {CFG['rounds']}")
    logging.info(f"Verification: {ENABLE_VERIFICATION_MODE} | Dirichlet alpha: {_alpha}")

    train_ds, test_ds = load_data(TARGET_DATASET)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    init_model = models.get_model(TARGET_DATASET, CFG['model'])

    all_results = {}

    if do_iid:
        logging.info("\n>>> IID 实验 ...")
        iid_loaders = distribute_data(train_ds, TOTAL_USERS, CFG['batch_size'], iid=True,
                                       max_samples_per_user=CFG['samples_per_user'])
        all_results['FedAvg_IID'] = run_fedavg(iid_loaders, test_loader, init_model)
        all_results['HVSA_IID'] = run_proposed(iid_loaders, test_loader, init_model, "HVSA-IID")

    logging.info(f"\n>>> Non-IID 实验 (Dirichlet alpha={_alpha}) ...")
    noniid_loaders = distribute_data(train_ds, TOTAL_USERS, CFG['batch_size'], iid=False,
                                      dirichlet_alpha=_alpha, max_samples_per_user=CFG['samples_per_user'])
    all_results['FedAvg_NonIID'] = run_fedavg(noniid_loaders, test_loader, init_model)
    all_results['HVSA_NonIID'] = run_proposed(noniid_loaders, test_loader, init_model, "HVSA-NonIID")

    np.savez(f'exp_noniid_data_{TARGET_DATASET}_{timestamp}.npz',
             **{k: np.array(v) for k, v in all_results.items()})
    logging.info(f"数据已保存: exp_noniid_data_{TARGET_DATASET}_{timestamp}.npz")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, CFG['rounds'] + 1)

    if do_iid:
        ax1.plot(epochs, all_results['FedAvg_IID'], 'k--', label='FedAvg', lw=2)
        ax1.plot(epochs, all_results['HVSA_IID'], 'r-', label='HVSA', lw=2)
    ax1.set_title(f'IID ({TARGET_DATASET})'); ax1.set_xlabel('Rounds'); ax1.set_ylabel('Accuracy (%)')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, all_results['FedAvg_NonIID'], 'k--', label='FedAvg', lw=2)
    ax2.plot(epochs, all_results['HVSA_NonIID'], 'r-', label='HVSA', lw=2)
    ax2.set_title(f'Non-IID alpha={_alpha} ({TARGET_DATASET})'); ax2.set_xlabel('Rounds'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_file = f'exp_noniid_plot_{TARGET_DATASET}_{timestamp}.png'
    plt.savefig(plot_file, dpi=300)
    logging.info(f"图已保存: {plot_file}")
    plt.show()

    if do_dlg:
        logging.info("\n>>> DLG 攻击防御实验 ...")
        run_dlg_defense_exp()

    logging.info("\n=== 实验完成 ===")


if __name__ == "__main__":
    main()
