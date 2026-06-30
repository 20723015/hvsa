import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 实验设定 (完全对齐引文描述)
# ==========================================
N_TOTAL = 300  # 总用户数
M_EDGE = 30  # 边缘服务器数
G_SIZE = 10  # 每个边缘服务器管理的用户数 (分组大小)
D_DIM = 10000  # 梯度维度 d = 10000
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]  # 掉线率 0%, 10%, 20%, 30%

# 基础常数大小 (Bytes)
FLOAT_SIZE = 4
INT_SIZE = 8
SHARE_SIZE = 256  # 基线方案中，为每个掉线用户额外上传的恢复信息大小

# 耗时系数 (ms)
T_PRG = 1e-5  # PRG 生成一个元素的耗时
T_ADD = 1e-6  # 基础加法耗时
T_RECOVER = 0.5  # 基线方案服务器恢复一个掉线者掩码的耗时


def simulate_dropout_impact():
    res_baseline = {'u_comm': [], 'u_comp': [], 'cs_comm': [], 'cs_comp': []}
    res_hvsa = {'u_comm': [], 'u_comp': [], 'cs_comm': [], 'cs_comp': []}

    base_payload = D_DIM * (FLOAT_SIZE + INT_SIZE) / 1024  # 基础模型载荷 (KB)
    base_prg_time = D_DIM * T_PRG  # 基础掩码生成耗时 (ms)

    for rate in DROPOUT_RATES:
        n_active = int(N_TOTAL * (1 - rate))
        n_drop = N_TOTAL - n_active
        # 平均每个分组内掉线的用户数
        drop_per_group = G_SIZE * rate

        # ---------------------------------------------------------
        # 1. Baseline (引文中的方案：需要存活用户额外上传信息)
        # ---------------------------------------------------------
        # 通信：随掉线率增加，存活用户必须上传额外碎片信息
        u_comm_base = base_payload + (drop_per_group * SHARE_SIZE) / 1024
        # 计算：基本不变 (引文提到 "little impact on the computation overhead of users")
        u_comp_base = base_prg_time + (drop_per_group * 0.05)

        # CS 通信：接收所有存活用户的载荷和恢复碎片 (几乎持平或略微变化)
        cs_comm_base = (n_active * u_comm_base) / 1024  # MB
        # CS 计算：聚合存活用户 + 恢复掉线掩码
        cs_comp_base = (n_active * D_DIM * T_ADD) + (n_drop * T_RECOVER)

        res_baseline['u_comm'].append(u_comm_base)
        res_baseline['u_comp'].append(u_comp_base)
        res_baseline['cs_comm'].append(cs_comm_base)
        res_baseline['cs_comp'].append(cs_comp_base)

        # ---------------------------------------------------------
        # 2. HVSA (Ours：TA 动态解除掩码，即抛即走)
        # ---------------------------------------------------------
        # 通信：绝对常数，不受掉线率影响
        u_comm_hvsa = base_payload
        # 计算：绝对常数
        u_comp_hvsa = base_prg_time

        # CS 通信：只接收存活用户数据 + 1个TA全局掩码 (实际上随掉线率增加，总通信量还会略微减少)
        cs_comm_hvsa = (n_active * base_payload + D_DIM * INT_SIZE / 1024) / 1024  # MB
        # CS 计算：只聚合存活用户，TA全局掩码直接抵消，无需复杂恢复
        cs_comp_hvsa = n_active * D_DIM * T_ADD

        res_hvsa['u_comm'].append(u_comm_hvsa)
        res_hvsa['u_comp'].append(u_comp_hvsa)
        res_hvsa['cs_comm'].append(cs_comm_hvsa)
        res_hvsa['cs_comp'].append(cs_comp_hvsa)

    return res_baseline, res_hvsa


# ==========================================
# 绘图代码
# ==========================================
def plot_results(res_base, res_ours):
    x_labels = ['0%', '10%', '20%', '30%']
    x = np.arange(len(x_labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Impact of Dropout Rates ($N=300$, $M=30$, $d=10000$)', fontsize=16, fontweight='bold')

    # 1. User Communication
    axes[0, 0].bar(x - width / 2, res_base['u_comm'], width, label='Baseline', color='#888888', edgecolor='black')
    axes[0, 0].bar(x + width / 2, res_ours['u_comm'], width, label='HVSA (Ours)', color='#E15759', edgecolor='black')
    axes[0, 0].set_title('User Communication Overhead', fontsize=12)
    axes[0, 0].set_ylabel('Traffic (KB)')
    axes[0, 0].set_xticks(x);
    axes[0, 0].set_xticklabels(x_labels)
    axes[0, 0].legend()

    # 2. User Computation
    axes[0, 1].bar(x - width / 2, res_base['u_comp'], width, label='Baseline', color='#888888', edgecolor='black')
    axes[0, 1].bar(x + width / 2, res_ours['u_comp'], width, label='HVSA (Ours)', color='#E15759', edgecolor='black')
    axes[0, 1].set_title('User Computation Overhead', fontsize=12)
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_xticks(x);
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].legend()

    # 3. Server Communication
    axes[1, 0].plot(x_labels, res_base['cs_comm'], marker='o', ls='--', color='#888888', label='Baseline CS Traffic',
                    lw=2)
    axes[1, 0].plot(x_labels, res_ours['cs_comm'], marker='s', ls='-', color='#E15759', label='HVSA CS Traffic', lw=2)
    axes[1, 0].set_title('Cloud Server Communication Overhead', fontsize=12)
    axes[1, 0].set_ylabel('Total Traffic (MB)')
    axes[1, 0].legend()

    # 4. Server Computation
    axes[1, 1].plot(x_labels, res_base['cs_comp'], marker='o', ls='--', color='#888888', label='Baseline CS Latency',
                    lw=2)
    axes[1, 1].plot(x_labels, res_ours['cs_comp'], marker='s', ls='-', color='#E15759', label='HVSA CS Latency', lw=2)
    axes[1, 1].set_title('Cloud Server Computation Overhead', fontsize=12)
    axes[1, 1].set_ylabel('Total Latency (ms)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('dropout_impact.png', dpi=300)
    print("✅ 掉线率影响对比图已生成: dropout_impact.png")
    plt.show()


if __name__ == "__main__":
    res_b, res_h = simulate_dropout_impact()
    plot_results(res_b, res_h)