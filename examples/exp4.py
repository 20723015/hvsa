import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# --- 导入组件 ---
try:
    from user import User
    from ta import TrustedAuthority
except ImportError:
    print("错误: 找不到核心模块，请确保文件在项目根目录下。")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ================= 实验配置 =================

# 实验 A: 固定维度，变化用户数量
EXP_A_FIXED_DIM = 500000  # 固定 50万参数
EXP_A_USERS = [10, 50, 100, 200, 300]  # 变量: 用户数 n

# 实验 B: 固定用户数量，变化模型维度
EXP_B_FIXED_USERS = 50  # 固定 50 个用户
EXP_B_DIMS = [1000000, 5000000, 10000000, 20000000, 50000000]  # 变量: 维度 dim


# ================= 测量函数 =================

def setup_user(dim):
    """辅助函数：初始化并注册一个用户"""
    ta = TrustedAuthority(gradient_dim=dim)
    user = User("U-Test", None)
    # 必须注册才能获得种子
    ta.register_user(user.id, user.get_public_key_bytes())
    user.register_with_ta(ta.get_public_key_bytes())
    return user


def measure_ours(dim):
    """
    测量 PP-HVFL (Ours)
    复杂度: O(1) regarding users, O(dim) regarding model
    """
    user = setup_user(dim)

    t0 = time.perf_counter()
    # 核心: 只生成 1 个掩码
    mask = user._generate_ta_mask_int(1, dim)
    # 模拟加法
    _ = mask + 1
    t_cost = time.perf_counter() - t0

    return t_cost


def measure_secagg(dim, num_users):
    """
    测量 Google SecAgg (Baseline)
    复杂度: O(N) regarding users, O(dim) regarding model
    """
    user = setup_user(dim)

    t0 = time.perf_counter()

    # 核心: 必须循环 N-1 次 (Pairwise masking)
    # 模拟 N-1 个邻居
    combined_mask = np.zeros(dim, dtype=np.int64)
    for _ in range(num_users - 1):
        m = user._generate_ta_mask_int(1, dim)
        combined_mask += m

    t_cost = time.perf_counter() - t0

    return t_cost


# ================= 主运行逻辑 =================

def run_dual_experiment():
    logging.info("========== 开始双变量对比实验 ==========")

    # --- 实验 A: 变化用户数量 (Impact of N) ---
    logging.info(f"\n[实验 A] 变化用户数量 n (固定维度={EXP_A_FIXED_DIM})")
    res_a = {'x': EXP_A_USERS, 'ours': [], 'secagg': []}

    for n in EXP_A_USERS:
        # Ours: 不在乎 n，只传维度
        t_o = measure_ours(EXP_A_FIXED_DIM)
        # SecAgg: 在乎 n
        t_s = measure_secagg(EXP_A_FIXED_DIM, n)

        res_a['ours'].append(t_o)
        res_a['secagg'].append(t_s)
        logging.info(f"  n={n} | Ours: {t_o:.4f}s | SecAgg: {t_s:.4f}s")

    # --- 实验 B: 变化模型维度 (Impact of Dimension) ---
    logging.info(f"\n[实验 B] 变化模型维度 dim (固定用户={EXP_B_FIXED_USERS})")
    res_b = {'x': EXP_B_DIMS, 'ours': [], 'secagg': []}

    for dim in EXP_B_DIMS:
        # Ours
        t_o = measure_ours(dim)
        # SecAgg
        t_s = measure_secagg(dim, EXP_B_FIXED_USERS)

        res_b['ours'].append(t_o)
        res_b['secagg'].append(t_s)
        logging.info(f"  dim={dim} | Ours: {t_o:.4f}s | SecAgg: {t_s:.4f}s")

    return res_a, res_b


def plot_dual_results(res_a, res_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图: 随用户数量变化 (Impact of Users) ---
    ax1.plot(res_a['x'], res_a['secagg'], 'r-s', linewidth=2, label='Google SecAgg')
    ax1.plot(res_a['x'], res_a['ours'], 'b-o', linewidth=2, label='PP-HVFL (Ours)')

    ax1.set_xlabel('Number of Clients ($n$)')
    ax1.set_ylabel('Computation Time (s)')
    ax1.set_title(f'(a) Impact of Client Number\n(Fixed Dim={EXP_A_FIXED_DIM})')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 右图: 随模型维度变化 (Impact of Dimension) ---
    # 处理横坐标标签 (10k, 1M...)
    x_labels = [f"{d // 1000}k" if d < 1000000 else f"{d // 1000000}M" for d in res_b['x']]

    ax2.plot(res_b['x'], res_b['secagg'], 'r-s', linewidth=2, label=f'Google SecAgg (n={EXP_B_FIXED_USERS})')
    ax2.plot(res_b['x'], res_b['ours'], 'b-o', linewidth=2, label='PP-HVFL (Ours)')

    ax2.set_xlabel('Model Parameters ($d$)')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_title(f'(b) Impact of Model Dimension\n(Fixed Users={EXP_B_FIXED_USERS})')
    # ax2.set_xticklabels(x_labels) # 自动刻度通常更好，也可以手动设置
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = 'exp4_final_dual_plot.png'
    plt.savefig(save_path, dpi=300)
    logging.info(f"\n结果图已保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    r_a, r_b = run_dual_experiment()
    plot_dual_results(r_a, r_b)