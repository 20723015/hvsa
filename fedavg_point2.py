"""研究点2 FedAvg 对照基线: 与 main_point2.py 完全同参的 Standard FedAvg.

对齐点 (与 main_point2.py DATASET_CONFIGS['MNIST'], IID, seed 42):
  - 20 用户 (5 edge x 4 user), 每用户 350 样本, batch 64
  - local_epochs=1, lr=0.01, CNN MNIST
  - 20 轮全局通信 (对应 G2 max_checkpoints=20)
  - 分组平均 (edge 组内先平均再全局平均) = 数学上等价于直接全局平均

论文主表口径 (thesis-core 批跑): --deterministic 严格确定性、独立
--output-dir、npz/meta 原子落盘, meta 含 config_hash/git/环境摘要,
与 main_point2.py 的产物命名一致 (`point2_data_{run_id}.npz`)。

用法:
  python fedavg_point2.py [--rounds 20] [--seed 42] [--dataset MNIST]
  python fedavg_point2.py --deterministic --output-dir results_x --run-id G1_MNIST_IID_s42
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import models
import datasets
from datasets import load_data, distribute_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DATASET_CONFIGS = {
    'MNIST': {'model': 'CNN', 'local_epochs': 1, 'lr': 0.01,
              'samples_per_user': 350, 'batch_size': 64},
    'FashionMNIST': {'model': 'CNN', 'local_epochs': 1, 'lr': 0.01,
                     'samples_per_user': 350, 'batch_size': 64},
    'CIFAR10': {'model': 'CNN', 'local_epochs': 5, 'lr': 0.01,
                'samples_per_user': 1500, 'batch_size': 64},
}

NUM_EDGE_SERVERS = 5
USERS_PER_EDGE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ARTIFACT_FIELDS = {'run_id', 'output_dir', 'tag', 'force'}


def _enable_strict_determinism() -> None:
    """与 main_point2.py --deterministic 同口径的严格确定性开关."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"--deterministic 严格模式初始化失败: {e}")


def _git_info():
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL).strip())
    except Exception:  # noqa: BLE001
        return None, None
    return commit, dirty


def fedavg_weighted_average(keys, user_states, user_sizes,
                            num_edges, users_per_edge):
    """标准 FedAvg 样本量加权 (edge 内与 edge 间均按样本数加权).

    Dirichlet Non-IID 下各客户端样本数不同, 任何等权平均 (逐用户或逐
    edge 先平均) 都不再等价于标准 FedAvg; 分层拓扑写法必须逐层保留
    样本权重。该函数与直接全局样本量加权在数学上完全一致 (回归测试
    锚定这一点)。
    """
    new_state = {}
    for key in keys:
        edge_means, edge_sizes = [], []
        for e in range(num_edges):
            s, en = e * users_per_edge, (e + 1) * users_per_edge
            sizes = user_sizes[s:en]
            edge_n = max(1, sum(sizes))
            edge_sum = sum(user_states[i][key].float() * sizes[i - s]
                           for i in range(s, en))
            edge_means.append(edge_sum / edge_n)
            edge_sizes.append(edge_n)
        total_n = max(1, sum(edge_sizes))
        new_state[key] = (sum(m * n for m, n in zip(edge_means, edge_sizes))
                          / total_n)
    return new_state


def main():
    parser = argparse.ArgumentParser(description="研究点2 FedAvg 对照基线")
    parser.add_argument("--dataset", default="MNIST",
                        choices=["MNIST", "FashionMNIST", "CIFAR10"])
    parser.add_argument("--tag", default="",
                        help="产物文件名 tag (批量实验用)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="",
                        help="覆盖默认模型 (默认取 DATASET_CONFIGS; Tiny 仅用于冒烟)")
    parser.add_argument("--noniid", action="store_true",
                        help="Dirichlet Non-IID 数据划分")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration (--noniid 时生效)")
    parser.add_argument("--deterministic", action="store_true",
                        help="严格确定性模式 (论文主表必须开启)")
    parser.add_argument("--output-dir", default=".",
                        help="产物输出目录 (npz/meta)")
    parser.add_argument("--run-id", default="",
                        help="运行标识 (默认 <tag>_<timestamp>)")
    parser.add_argument("--force", action="store_true",
                        help="目标 npz 已存在时覆盖 (默认同 config_hash 跳过)")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    if args.model:
        cfg = dict(cfg, model=args.model)
    total_users = NUM_EDGE_SERVERS * USERS_PER_EDGE

    # 配置分离 (与 main_point2 一致): config_hash 只哈希科学配置
    raw_payload = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                       else str(v)) for k, v in vars(args).items()}
    scientific_config = {k: v for k, v in raw_payload.items()
                         if k not in _ARTIFACT_FIELDS}
    config_hash = hashlib.sha256(json.dumps(
        scientific_config, sort_keys=True, ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:16]

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or (f"{args.tag or ('fedavg_' + args.dataset)}_{timestamp}")
    out_npz = os.path.join(args.output_dir, f"point2_data_{run_id}.npz")
    out_meta = os.path.join(args.output_dir, f"point2_meta_{run_id}.json")
    if os.path.exists(out_npz) and not args.force:
        old_hash = None
        if os.path.exists(out_meta):
            try:
                with open(out_meta, 'r', encoding='utf-8') as f:
                    old_hash = json.load(f).get('config_hash')
            except Exception:  # noqa: BLE001
                old_hash = None
        if old_hash == config_hash:
            print(f"[skip] 同 config_hash ({config_hash}) 结果已存在: {out_npz}")
            return
        print(f"[abort] 目标已存在且配置不同 (old={old_hash}, new={config_hash}): "
              f"{out_npz} — 换 --run-id 或加 --force")
        sys.exit(2)

    if args.deterministic:
        _enable_strict_determinism()
        logging.info("确定性模式已开启 (严格模式)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds = load_data(args.dataset)
    loaders = distribute_data(train_ds, total_users, cfg['batch_size'],
                              iid=not args.noniid,
                              max_samples_per_user=cfg['samples_per_user'],
                              dirichlet_alpha=args.alpha)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    global_model = models.get_model(args.dataset, cfg['model']).to(DEVICE)
    user_models = [copy.deepcopy(global_model) for _ in range(total_users)]

    acc_history, elapsed_list = [], []
    start = time.time()
    for r in range(args.rounds):
        t0 = time.time()
        for i in range(total_users):
            user_models[i].load_state_dict(global_model.state_dict())
            models.train_local(user_models[i], loaders[i], cfg['local_epochs'], cfg['lr'])

        # 标准 FedAvg 样本量加权 (分层拓扑, 数学等价于全局样本量加权;
        # 由 fedavg_weighted_average 的回归测试锚定)
        new_state = fedavg_weighted_average(
            global_model.state_dict().keys(),
            [m.state_dict() for m in user_models],
            [len(loaders[i].dataset) for i in range(total_users)],
            NUM_EDGE_SERVERS, USERS_PER_EDGE)
        global_model.load_state_dict(new_state)

        acc, _ = models.evaluate(global_model, test_loader)  # (accuracy%, loss)
        acc_history.append(float(acc))
        elapsed_list.append(time.time() - t0)
        logging.info(f"[FedAvg] Round {r + 1}/{args.rounds} | Acc: {acc:.2f}%")

    wall = time.time() - start
    tmp_npz = out_npz[:-4] + '.tmp.npz'
    np.savez(tmp_npz,
             acc_history=np.array(acc_history),
             final_acc=acc_history[-1],
             wall_clock=wall,
             round_seconds=np.array(elapsed_list),
             rounds=args.rounds, seed=args.seed,
             dataset=args.dataset,
             num_edge_servers=NUM_EDGE_SERVERS,
             users_per_edge=USERS_PER_EDGE,
             # 与 main_point2 命名对齐: 等时间比较按 round 索引代替 tick
             checkpoint_ticks=np.arange(1, len(acc_history) + 1),
             config=str(args))
    os.replace(tmp_npz, out_npz)

    git_commit, git_dirty = _git_info()
    meta = {
        'run_id': run_id, 'config_hash': config_hash,
        'args': raw_payload, 'scientific_config': scientific_config,
        'git_commit': git_commit, 'git_dirty': git_dirty,
        'dataset': args.dataset, 'seed': args.seed,
        'platform': platform.platform(),
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'deterministic': bool(args.deterministic),
        'timestamp': timestamp,
        'final_acc': float(acc_history[-1]),
        'wall_clock': float(wall),
        'arr_digest': hashlib.sha256(
            np.asarray(acc_history, dtype=np.float64).tobytes()).hexdigest()[:16],
        'npz': os.path.basename(out_npz),
    }
    tmp_meta = out_meta + '.tmp'
    with open(tmp_meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, out_meta)

    print(f"\n=== FedAvg 完成: {args.rounds} rounds, 墙钟 {wall:.1f}s ===")
    print(f"Accuracy history: {[round(a, 2) for a in acc_history]}")
    print(f"数据已保存: {out_npz}")
    print(f"元数据已保存: {out_meta}")


if __name__ == "__main__":
    main()
