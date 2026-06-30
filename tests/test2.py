import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import copy

# --- 导入组件 ---
from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
import models
from datasets import load_data, distribute_data

# ==========================================
#      配置区：请确认文件名和参数
# ==========================================

# 1. 指定您现有的那个文件名
EXISTING_NPZ_FILE = 'exp2_dropout_data_recovered_MNIST_20260119-161542.npz'

# 2. 实验设置
TARGET_DATASET = 'MNIST'
NEW_RATES_TO_RUN = [0.1, 0.2]  # 需要补跑: 10%, 20%
ALL_RATES_FOR_PLOT = [0.0, 0.1, 0.2, 0.3]  # 最终画图展示这四个

# 3. MNIST 参数配置 (与之前保持一致)
CFG = {
    'model': 'CNN',
    'rounds': 50,
    'local_epochs': 1,
    'lr': 0.01,
    'samples_per_user': 350,
    'batch_size': 64
}

NUM_EDGE_SERVERS = 2
USERS_PER_EDGE = 5
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#      核心逻辑
# ==========================================

def run_experiment_with_dropout(train_loaders, test_loader, initial_model, dropout_rate):
    """运行单次掉线实验"""
    logging.info(f"\n>>> [补跑任务] 开始测试: 掉线率 = {dropout_rate * 100}% <<<")

    flat_params, model_meta = models.flatten_parameters(initial_model)
    gradient_dim = flat_params.shape[0]

    ta = TrustedAuthority(gradient_dim=gradient_dim)
    cloud = CloudServer()
    # accuracy-only fast mode: 补跑掉线曲线时显式关闭真实 LHH/Ed25519 验证。
    edge_servers = [EdgeServer(f"ES-{i}", cloud, ta, enable_verification=False)
                    for i in range(NUM_EDGE_SERVERS)]

    users = []
    user_edge_map = {}
    for i in range(TOTAL_USERS):
        u = User(f"U-{i}", copy.deepcopy(initial_model))
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta.get_public_key_bytes())
        users.append(u)
        user_edge_map[u.id] = edge_servers[i // USERS_PER_EDGE]

    acc_history = []
    global_model_vector = flat_params

    # 获取参数
    total_rounds = CFG['rounds']
    local_epochs = CFG['local_epochs']
    lr = CFG['lr']

    for r in range(total_rounds):
        round_t = r + 1
        cloud.start_new_round(round_t, gradient_dim, NUM_EDGE_SERVERS)
        ta.start_new_round(round_t)
        for es in edge_servers: es.start_new_round(round_t)

        active_count = 0
        for i, u in enumerate(users):
            # 掉线模拟
            if np.random.rand() < dropout_rate:
                continue

            active_count += 1
            u.set_global_model(global_model_vector, model_meta)
            u.local_train(train_loaders[i], local_epochs, lr)
            u.mask_and_upload_model(round_t, user_edge_map[u.id])

        for es in edge_servers: es.trigger_aggregation_and_forward()
        ta.trigger_global_mask_computation(cloud)
        global_avg_model, _, _ = cloud.trigger_final_unmasking_and_averaging()

        if global_avg_model is not None:
            global_model_vector = global_avg_model
            users[0].set_global_model(global_model_vector, model_meta)
            acc, _ = models.evaluate(users[0].model, test_loader)
            acc_history.append(acc)
            logging.info(
                f"[Dropout {int(dropout_rate * 100)}%] Round {round_t}/{total_rounds} | Online: {active_count} | Acc: {acc:.2f}%")
        else:
            acc_history.append(acc_history[-1] if acc_history else 0)
            logging.warning(f"[Dropout {int(dropout_rate * 100)}%] Round {round_t} | No users online!")

    return acc_history


def main():
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    LOG_FILENAME = f'exp2_supplementary_MNIST_log_{TIMESTAMP}.txt'

    # 配置日志
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'), logging.StreamHandler()]
    )

    logging.info(f"=== MNIST 补全实验 ===")
    logging.info(f"目标文件: {EXISTING_NPZ_FILE}")

    # 1. 加载旧数据
    if not os.path.exists(EXISTING_NPZ_FILE):
        logging.error(f"❌ 找不到文件: {EXISTING_NPZ_FILE}")
        return

    loaded_data = np.load(EXISTING_NPZ_FILE)
    results = {k: v for k, v in loaded_data.items()}
    logging.info(f"✅ 已加载旧数据: {list(results.keys())}")

    # 2. 准备环境
    full_train_dataset, full_test_dataset = load_data(TARGET_DATASET)
    user_loaders = distribute_data(full_train_dataset, TOTAL_USERS, CFG['batch_size'],
                                   iid=True, max_samples_per_user=CFG['samples_per_user'])
    test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)
    initial_model = models.get_model(TARGET_DATASET, CFG['model'])

    # 3. 补跑实验 (10%, 20%)
    for rate in NEW_RATES_TO_RUN:
        key_name = f'rate_{rate}'
        if key_name in results:
            logging.info(f"掉线率 {rate} 的数据已存在，跳过。")
            continue

        # 跑实验
        acc_curve = run_experiment_with_dropout(user_loaders, test_loader, initial_model, rate)
        results[key_name] = acc_curve

    # 4. 保存完整数据
    merged_filename = f'exp2_MNIST_MERGED_{TIMESTAMP}.npz'
    np.savez(merged_filename, **results)
    logging.info(f"🎉 完整数据已保存: {merged_filename}")

    # 5. 画图
    plt.figure(figsize=(10, 7))
    epochs = range(1, CFG['rounds'] + 1)
    # 定义颜色: 0%(蓝), 10%(橙), 20%(绿), 30%(红)
    colors = ['b', 'orange', 'g', 'r', 'purple']

    # 排序确保图例整齐
    rates_to_plot = sorted(ALL_RATES_FOR_PLOT)

    for i, rate in enumerate(rates_to_plot):
        key = f'rate_{rate}'
        # 兼容可能的 key 格式 (例如 rate_0.1 或 rate_0.1000001)
        # 这里做一个模糊查找
        found_key = None
        for k in results.keys():
            if f"rate_{rate}" in k:  # 简单匹配
                found_key = k
                break

        if found_key:
            data = results[found_key]
            # 截取前 50 轮 (防止长度不一致报错)
            if len(data) > CFG['rounds']: data = data[:CFG['rounds']]

            plt.plot(epochs, data,
                     label=f'Dropout Rate = {int(rate * 100)}%',
                     color=colors[i % len(colors)], linewidth=2)
        else:
            logging.warning(f"⚠️  画图时缺少掉线率 {rate} 的数据")

    plt.title(f'Impact of Client Dropout ({TARGET_DATASET})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_file = f'exp2_MNIST_plot_combined_{TIMESTAMP}.png'
    plt.savefig(plot_file, dpi=300)
    logging.info(f"📊 对比图已保存: {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()
