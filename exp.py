import logging
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

# --- 导入组件 (仅用于复用 User 类和模型逻辑，不使用安全模块) ---
from user import User
from cloud import CloudServer
import models
from datasets import load_data, distribute_data

# ==========================================
#      参数配置 (严格复刻之前的设置)
# ==========================================
TARGET_DATASET = 'CIFAR10'

# 这是您之前的“标准/完整”配置
CFG = {
    'model': 'CNN',
    'rounds': 50,  # 保持 50 轮
    'local_epochs': 5,  # 保持 5 个本地 epoch
    'lr': 0.01,
    'samples_per_user': 2000,  # 保持较大数据量
    'batch_size': 64
}

# 目标掉线率
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]

# 客户端设置
NUM_EDGE_SERVERS = 2
USERS_PER_EDGE = 5
TOTAL_USERS = NUM_EDGE_SERVERS * USERS_PER_EDGE  # 10个用户

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
#      核心逻辑 (纯净 FedAvg + 掉线模拟)
# ==========================================

def run_fedavg_with_dropout(train_loaders, test_loader, initial_model, dropout_rate):
    """
    使用 FedAvg 模拟掉线实验
    """
    logging.info(f"\n>>> [FedAvg Proxy] 开始测试: 掉线率 = {int(dropout_rate * 100)}% ...")

    # 1. 初始化用户
    # 注意：这里我们不需要 TA 和 EdgeServer 的复杂逻辑，只需要用户和云端聚合
    users = []
    for i in range(TOTAL_USERS):
        # 初始化用户，不需要注册密钥，因为跑的是明文
        u = User(f"U-{i}", copy.deepcopy(initial_model))
        users.append(u)

    acc_history = []

    # 全局模型初始化
    flat_params, model_meta = models.flatten_parameters(initial_model)
    global_model_vector = flat_params

    start_time = time.time()

    # 2. 训练循环
    for r in range(CFG['rounds']):
        round_t = r + 1

        # --- 步骤 A: 确定在线用户 ---
        online_users = []
        for i, u in enumerate(users):
            # 随机决定是否掉线
            if np.random.rand() >= dropout_rate:
                online_users.append(u)

        if len(online_users) == 0:
            logging.warning(f"Round {round_t}: 所有用户都掉线了！跳过本轮。")
            acc_history.append(acc_history[-1] if acc_history else 0)
            continue

        # --- 步骤 B: 在线用户本地训练 ---
        uploaded_models = []

        for u in online_users:
            # 1. 下载全局模型 (直接赋值，无解密开销)
            u.set_global_model(global_model_vector, model_meta)

            # 2. 本地训练 (这是最耗时的部分，但比加密快)
            # 获取对应的 DataLoader
            loader_idx = int(u.id.split('-')[1])
            u.local_train(train_loaders[loader_idx], CFG['local_epochs'], CFG['lr'])

            # 3. 获取模型参数 (不加密，直接取参数)
            flat, _ = models.flatten_parameters(u.model)
            uploaded_models.append(flat)

        # --- 步骤 C: 云端聚合 (FedAvg) ---
        # 直接求平均 (Mean)
        global_model_vector = np.mean(uploaded_models, axis=0)

        # --- 步骤 D: 评估 ---
        # 将新模型赋给第一个用户用于测试
        users[0].set_global_model(global_model_vector, model_meta)
        acc, _ = models.evaluate(users[0].model, test_loader)
        acc_history.append(acc)

        print(f"\r   Round {round_t}/{CFG['rounds']} | Online: {len(online_users)}/{TOTAL_USERS} | Acc: {acc:.2f}%",
              end="")

    duration = time.time() - start_time
    logging.info(f"\n   -> 完成! 耗时: {duration:.1f}s | 最终精度: {acc_history[-1]:.2f}%")
    return acc_history


def main():
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    LOG_FILENAME = f'exp2_CIFAR10_FedAvg_Proxy_{TIMESTAMP}.txt'

    # 配置日志
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'), logging.StreamHandler()]
    )

    logging.info(f"=== 实验 2: CIFAR-10 掉线率鲁棒性 (FedAvg 代跑版) ===")
    logging.info("说明: 使用 FedAvg 模拟无损加密方案的收敛曲线")
    logging.info(f"配置: {CFG}")
    logging.info(f"掉线率: {DROPOUT_RATES}")

    # 1. 准备数据
    logging.info("正在加载 CIFAR-10 数据...")
    full_train_dataset, full_test_dataset = load_data(TARGET_DATASET)
    user_loaders = distribute_data(full_train_dataset, TOTAL_USERS, CFG['batch_size'],
                                   iid=True, max_samples_per_user=CFG['samples_per_user'])
    test_loader = DataLoader(full_test_dataset, batch_size=256, shuffle=False)

    initial_model = models.get_model(TARGET_DATASET, CFG['model'])

    # 2. 依次运行实验
    results = {}
    for rate in DROPOUT_RATES:
        # 深拷贝模型，确保独立
        model_copy = copy.deepcopy(initial_model)
        acc_curve = run_fedavg_with_dropout(user_loaders, test_loader, model_copy, rate)
        results[rate] = acc_curve

    # 3. 保存数据 (格式与之前完全一致，方便后续画图)
    data_file = f'exp2_data_CIFAR10_{TIMESTAMP}.npz'
    np.savez(data_file, **{f'rate_{r}': acc for r, acc in results.items()})
    logging.info(f"数据已保存至: {data_file}")

    # 4. 绘图
    plt.figure(figsize=(10, 7))
    epochs = range(1, CFG['rounds'] + 1)

    # 颜色: 蓝, 橙, 绿, 红
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, rate in enumerate(DROPOUT_RATES):
        plt.plot(epochs, results[rate],
                 label=f'Dropout Rate = {int(rate * 100)}%',
                 color=colors[i], linewidth=2)

    plt.title(f'Impact of Client Dropout (CIFAR-10)')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_file = f'exp2_plot_CIFAR10_{TIMESTAMP}.png'
    plt.savefig(plot_file, dpi=300)
    logging.info(f"结果图已保存至: {plot_file}")
    plt.show()


if __name__ == "__main__":
    main()