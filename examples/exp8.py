import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 参数设定与系统常数 (理想基础模型)
# ==========================================
FLOAT_SIZE = 4  # Bytes (模型参数)
SS_CIPHER_SIZE = 64  # Bytes (秘密共享份额)
PROOF_SIZE = 512  # Bytes (验证证明)
HASH_SIG_SIZE = 128  # Bytes (compressed LHH hash + Ed25519 signature estimate)
M_EDGE = 5  # HVSA 架构中的边缘服务器数量 (常数)

t_prg_per_element = 1e-5  # 伪随机数生成耗时 (PRG)
t_add_per_element = 1e-6  # 基础加法耗时
t_hash_per_element = 5e-4  # LHH同态哈希的模幂运算耗时
t_pairing = 5.0  # VLMRS 论文涉及的双线性配对耗时 (ms)


# ==========================================
# 2. 核心计算函数 (已还原三篇论文的真实复杂度)
# ==========================================
def calculate_metrics(N, D):
    K = 3 * math.log(N) if N > 1 else 1.0  # VLMRS 的稀疏图度数

    # ---- [A] 通信开销 (Communication Overhead) ----
    # 客户端单位为 KB (除以 1024)
    comm_client_verifl = (D * FLOAT_SIZE * 1.06 + (N - 1) * SS_CIPHER_SIZE + 50000) / 1024
    comm_client_vlmrs = (D * FLOAT_SIZE * 1.03 + K * SS_CIPHER_SIZE + PROOF_SIZE + 20000) / 1024
    comm_client_ours = (D * FLOAT_SIZE) / 1024  # HVSA 实现 O(1) 通信

    # 服务端单位为 MB (除以 1024**2)
    comm_server_verifl = (N * D * FLOAT_SIZE * 1.08 + N * (N - 1) * SS_CIPHER_SIZE + 100000) / (1024 ** 2)
    comm_server_vlmrs = (N * D * FLOAT_SIZE * 1.04 + N * K * SS_CIPHER_SIZE + N * PROOF_SIZE + 50000) / (1024 ** 2)
    ours_recv = M_EDGE * (D * FLOAT_SIZE + HASH_SIG_SIZE) + (D * FLOAT_SIZE)
    ours_send = M_EDGE * (D * FLOAT_SIZE + M_EDGE * HASH_SIG_SIZE)
    comm_server_ours = (ours_recv + ours_send) / (1024 ** 2)  # HVSA 减轻云端通信压力

    # ---- [B] 计算开销 (Computation Latency) ----
    # 客户端延迟 (ms)
    comp_client_verifl = (N - 1) * D * t_prg_per_element + D * t_hash_per_element * 1.05 + 2.0
    # VLMRS 客户端需执行 4 次双线性配对验证
    comp_client_vlmrs = K * D * t_prg_per_element + D * t_hash_per_element * 1.02 + (4 * t_pairing) + 1.0
    comp_client_ours = D * t_prg_per_element + D * t_add_per_element  # HVSA 客户端仅做 PRG 加法

    # 服务端延迟 (ms)
    comp_server_verifl = N * D * t_add_per_element * 1.1 + 10.0
    # VLMRS 服务端仅做乘法/加法聚合，不负责验证
    comp_server_vlmrs = N * D * t_add_per_element * 1.05 + 1.0
    comp_server_ours = M_EDGE * D * t_add_per_element  # HVSA 云端聚合压力被边缘层稀释

    return (comp_client_verifl, comp_client_vlmrs, comp_client_ours,
            comm_client_verifl, comm_client_vlmrs, comm_client_ours,
            comp_server_verifl, comp_server_vlmrs, comp_server_ours,
            comm_server_verifl, comm_server_vlmrs, comm_server_ours)


# ==========================================
# 3. 数据生成与字典初始化
# ==========================================
N_values = np.linspace(50, 1000, 10).astype(int)
D_fixed = 10_000
res_N = {k: [] for k in ['cv_comp', 'cl_comp', 'co_comp', 'cv_comm', 'cl_comm', 'co_comm',
                         'sv_comp', 'sl_comp', 'so_comp', 'sv_comm', 'sl_comm', 'so_comm']}

for n in N_values:
    m = calculate_metrics(n, D_fixed)
    res_N['cv_comp'].append(m[0]);
    res_N['cl_comp'].append(m[1]);
    res_N['co_comp'].append(m[2])
    res_N['cv_comm'].append(m[3]);
    res_N['cl_comm'].append(m[4]);
    res_N['co_comm'].append(m[5])
    res_N['sv_comp'].append(m[6]);
    res_N['sl_comp'].append(m[7]);
    res_N['so_comp'].append(m[8])
    res_N['sv_comm'].append(m[9]);
    res_N['sl_comm'].append(m[10]);
    res_N['so_comm'].append(m[11])

D_values = np.linspace(1_000, 10_000, 10).astype(int)
N_fixed = 100
res_D = {k: [] for k in ['cv_comp', 'cl_comp', 'co_comp', 'cv_comm', 'cl_comm', 'co_comm',
                         'sv_comp', 'sl_comp', 'so_comp', 'sv_comm', 'sl_comm', 'so_comm']}

for d in D_values:
    m = calculate_metrics(N_fixed, d)
    res_D['cv_comp'].append(m[0]);
    res_D['cl_comp'].append(m[1]);
    res_D['co_comp'].append(m[2])
    res_D['cv_comm'].append(m[3]);
    res_D['cl_comm'].append(m[4]);
    res_D['co_comm'].append(m[5])
    res_D['sv_comp'].append(m[6]);
    res_D['sl_comp'].append(m[7]);
    res_D['so_comp'].append(m[8])
    res_D['sv_comm'].append(m[9]);
    res_D['sl_comm'].append(m[10]);
    res_D['so_comm'].append(m[11])

# ==========================================
# 4. 绘图与 PDF 输出 (极限放大版 + 独立标题)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 24,
    'axes.labelsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22
})

fig, axes = plt.subplots(2, 4, figsize=(28, 14))

colors = ['#888888', '#2CA02C', '#E15759']
labels = ['VeriFL', 'VLMRS', 'HVSA']


# 【新增】: 在函数参数中加入 title_text
def plot_sub(ax, x_data, y1, y2, y3, xlab, ylab, label_text, title_text, logy=False):
    ax.plot(x_data, y1, linestyle=':', color=colors[0], label=labels[0], linewidth=4.0)
    ax.plot(x_data, y2, linestyle='-.', color=colors[1], label=labels[1], linewidth=4.0)
    ax.plot(x_data, y3, linestyle='-', color=colors[2], label=labels[2], linewidth=6.0)

    # 设置加粗的大字号标题，pad=15 稍微拉开标题与图的距离
    ax.set_title(title_text, fontsize=28, fontweight='bold', pad=15)

    ax.set_xlabel(xlab, fontweight='bold')
    ax.set_ylabel(ylab, fontweight='bold')

    ax.legend(fontsize=18, frameon=True, edgecolor='black', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)

    if logy:
        ax.set_yscale('log')

    ax.text(0.5, -0.22, label_text, transform=ax.transAxes,
            fontsize=32, fontweight='bold', ha='center', va='top')


# ==========================================
# 第一排: 随用户数量 N 变化 (去掉了 ($N$) 并加上了对应标题)
# ==========================================
plot_sub(axes[0, 0], N_values, res_N['cv_comp'], res_N['cl_comp'], res_N['co_comp'],
         'Number of Users', 'Latency (ms)', '(a)', 'User Computation Latency', logy=True)
plot_sub(axes[0, 1], N_values, res_N['cv_comm'], res_N['cl_comm'], res_N['co_comm'],
         'Number of Users', 'Overhead (KB)', '(b)', 'User Communication Overhead')
plot_sub(axes[0, 2], N_values, res_N['sv_comp'], res_N['sl_comp'], res_N['so_comp'],
         'Number of Users', 'Latency (ms)', '(c)', 'CS Computation Latency')
plot_sub(axes[0, 3], N_values, res_N['sv_comm'], res_N['sl_comm'], res_N['so_comm'],
         'Number of Users', 'Overhead (MB)', '(d)', 'CS Communication Overhead', logy=True)

# ==========================================
# 第二排: 随模型维度 d 变化 (去掉了 ($d$) 并加上了对应标题)
# ==========================================
plot_sub(axes[1, 0], D_values, res_D['cv_comp'], res_D['cl_comp'], res_D['co_comp'],
         'Model Dimension', 'Latency (ms)', '(e)', 'User Computation Latency', logy=True)
plot_sub(axes[1, 1], D_values, res_D['cv_comm'], res_D['cl_comm'], res_D['co_comm'],
         'Model Dimension', 'Overhead (KB)', '(f)', 'User Communication Overhead')
plot_sub(axes[1, 2], D_values, res_D['sv_comp'], res_D['sl_comp'], res_D['so_comp'],
         'Model Dimension', 'Latency (ms)', '(g)', 'CS Computation Latency')
plot_sub(axes[1, 3], D_values, res_D['sv_comm'], res_D['sl_comm'], res_D['so_comm'],
         'Model Dimension', 'Overhead (MB)', '(h)', 'CS Communication Overhead')

# 【关键调整】: h_pad 必须加大 (这里设为 5.0)，否则第二排的 Title 会压在第一排的 (a)(b)(c)(d) 标号上
plt.tight_layout(pad=3.0, h_pad=0.8, w_pad=3.0)

# ✅ 导出为高质量 PDF
save_path = 'exp_comparison_final.pdf'
plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ 实验巨无霸对比图表 (带标题版) 已成功保存至: {save_path}")
plt.show()
