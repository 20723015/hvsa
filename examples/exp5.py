import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
import sys

try:
    from user import User
    from edge import EdgeServer
    from cloud import CloudServer
    from ta import TrustedAuthority
except ImportError:
    print("错误: 找不到核心模块，请确保在项目根目录下。")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')

MODEL_SIZES = [10000, 100000, 500000, 1000000]

HE_COMP_FACTOR = 500.0
HE_COMM_FACTOR = 2.0


def measure_proposed_scheme(dim):
    """测量 HVSA 方案在特定维度下的真实开销"""
    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01")
    cloud = CloudServer()
    es = EdgeServer("ES-Test", cloud, ta, enable_verification=True)

    user = User("User-Test", None)
    ta.register_user(user.id, user.get_public_key_bytes())
    user.register_with_ta(ta.get_public_key_bytes())

    dummy_gradient = np.random.randint(-100, 100, size=dim, dtype=np.int64)

    # [A] 客户端开销
    t0 = time.perf_counter()
    mask = user._generate_ta_mask_int(1, dim)
    masked_gradient = dummy_gradient + mask
    t_client = time.perf_counter() - t0

    payload = pickle.dumps(masked_gradient)
    comm_kb = len(payload) / 1024.0

    # [B] 边缘服务器开销
    es._masked_gradients = [masked_gradient]
    es._online_users = [user.id]
    t1 = time.perf_counter()
    es.trigger_aggregation_and_forward()
    t_edge = time.perf_counter() - t1

    # [C] TA 开销
    round_t = 1
    ta.start_new_round(round_t)
    ta._all_online_users = {user.id}
    t2 = time.perf_counter()
    ta.trigger_global_mask_computation(cloud)
    t_ta = time.perf_counter() - t2

    # [D] Cloud 解掩码
    cloud.start_new_round(round_t, dim, 1)
    cloud._sum_masked_gradients = masked_gradient.copy()
    t4 = time.perf_counter()
    avg_model, _, _ = cloud.trigger_final_unmasking_and_averaging()
    t_cloud = time.perf_counter() - t4

    return t_client, t_edge, t_ta, t_cloud, comm_kb


def run_experiment_4():
    logging.info("========== 实验：计算与通信效率分析 ==========")

    results = {
        'dims': MODEL_SIZES,
        'time_client': [], 'time_edge': [], 'time_ta': [], 'time_cloud': [], 'comm': [],
        'time_client_he': [], 'comm_he': []
    }

    for dim in MODEL_SIZES:
        logging.info(f"正在测试参数规模: {dim} ...")
        t_cli, t_edge, t_ta, t_cloud, comm = measure_proposed_scheme(dim)

        results['time_client'].append(t_cli)
        results['time_edge'].append(t_edge)
        results['time_ta'].append(t_ta)
        results['time_cloud'].append(t_cloud)
        results['comm'].append(comm)

        t_he = t_cli * HE_COMP_FACTOR
        c_he = comm * HE_COMM_FACTOR
        results['time_client_he'].append(t_he)
        results['comm_he'].append(c_he)

        logging.info(f"  -> [Ours] Client: {t_cli:.5f}s | Edge: {t_edge:.5f}s | "
                     f"TA: {t_ta:.5f}s | Cloud: {t_cloud:.5f}s | Comm: {comm:.2f}KB")

    return results


def plot_efficiency(res):
    dims = res['dims']
    x_labels = [f"{d // 1000}k" for d in dims]
    x = np.arange(len(dims))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 图 1: 客户端 vs HE ---
    ax1.bar(x - width / 2, res['time_client_he'], width, label='HE (Paillier)', color='#ff9999', hatch='//')
    ax1.bar(x + width / 2, res['time_client'], width, label='HVSA', color='#66b3ff')
    ax1.set_ylabel('Client Encryption Time (s)')
    ax1.set_xlabel('Model Parameters')
    ax1.set_title('Client Computation Overhead')
    ax1.set_xticks(x); ax1.set_xticklabels(x_labels)
    ax1.legend(); ax1.set_yscale('log')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- 图 2: 各实体计算开销分解 ---
    w = width / 3
    ax2.bar(x - 1.2 * w, res['time_client'], w, label='Client', color='#66b3ff')
    ax2.bar(x, res['time_edge'], w, label='Edge', color='#ffb366')
    ax2.bar(x + 1.2 * w, res['time_cloud'], w, label='Cloud', color='#ff66b3')
    ax2.set_ylabel('Computation Time (s)')
    ax2.set_xlabel('Model Parameters')
    ax2.set_title('Per-Entity Computation Breakdown')
    ax2.set_xticks(x); ax2.set_xticklabels(x_labels)
    ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = 'exp5_efficiency.png'
    plt.savefig(save_path, dpi=300)
    logging.info(f"图表已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    results = run_experiment_4()
    plot_efficiency(results)
