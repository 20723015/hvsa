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
# 4. 绘图与输出 (4 行 x 2 列: 每行一个指标, 左列随 N / 右列随 d)
#    设计要点: 单一共享图例 + 左侧竖排行标 + 顶部列头 + 角标 (a)-(h),
#    去掉每张子图重复的标题与图例, 让 HVSA 曲线成为视觉焦点。
# ==========================================
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 19,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.1,
})

BLUE_DARK = '#14314F'

# 两条 baseline 用冷/中性色, HVSA 用醒目红并加粗加大
STYLE = {
    'VeriFL': dict(color='#8C8C8C', linestyle=':',  linewidth=2.6, marker='s',
                   markersize=7, markerfacecolor='white', markeredgewidth=1.6),
    'VLMRS': dict(color='#3B6FB0', linestyle='--', linewidth=2.6, marker='^',
                  markersize=7, markerfacecolor='white', markeredgewidth=1.6),
    'HVSA':  dict(color='#E23B41', linestyle='-',  linewidth=4.2, marker='o',
                  markersize=8, markerfacecolor='#E23B41', markeredgecolor='white',
                  markeredgewidth=1.2, zorder=5),
}

# 每行一个指标: (三条曲线的数据键, y 轴单位, 是否 log, 行标题)
ROWS = [
    (('cv_comp', 'cl_comp', 'co_comp'), 'Latency (ms)',  True,  'User Computation\nLatency'),
    (('cv_comm', 'cl_comm', 'co_comm'), 'Overhead (KB)', False, 'User Communication\nOverhead'),
    (('sv_comp', 'sl_comp', 'so_comp'), 'Latency (ms)',  True,  'CS Computation\nLatency'),
    (('sv_comm', 'sl_comm', 'so_comm'), 'Overhead (MB)', True,  'CS Communication\nOverhead'),
]
# 两列: (数据字典, x 轴数据, x 轴标签)
COLS = [
    (res_N, N_values, 'Number of Users'),
    (res_D, D_values, 'Model Dimension'),
]

fig, axes = plt.subplots(4, 2, figsize=(13, 15.5))
fig.subplots_adjust(left=0.095, right=0.985, top=0.895,
                    bottom=0.060, hspace=0.50, wspace=0.18)

panel_tags = iter(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'])

for r, (keys, ylab, logy, _metric) in enumerate(ROWS):
    for c, (res, xvals, xlab) in enumerate(COLS):
        ax = axes[r, c]
        for name, key in zip(['VeriFL', 'VLMRS', 'HVSA'], keys):
            ax.plot(xvals, res[key], label=name, **STYLE[name])

        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(xlab, fontweight='bold', labelpad=4)
        ax.set_ylabel(ylab, fontweight='bold', labelpad=4)
        ax.grid(True, which='major', linestyle='--', alpha=0.30, linewidth=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.18, linewidth=0.5)
        ax.tick_params(direction='out', length=4, width=1.0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.02)

        # 角标 (a)-(h): 左上角小圆牌
        ax.text(0.035, 0.955, next(panel_tags), transform=ax.transAxes,
                fontsize=17, fontweight='bold', va='top', ha='left', color=BLUE_DARK,
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#C4D0E0', alpha=0.9))

# --- 每行指标名: 横排放在该行两图正上方居中 (每个指标只出现一次) ---
for r, (_keys, _ylab, _logy, metric) in enumerate(ROWS):
    pL = axes[r, 0].get_position()
    pR = axes[r, 1].get_position()
    xc = (pL.x0 + pR.x1) / 2
    yt = pL.y1 + 0.010
    fig.text(xc, yt, metric.replace('\n', ' '), ha='center', va='bottom',
             fontsize=20, fontweight='bold', color=BLUE_DARK)

# --- 全图共享的单一图例 (顶部居中) ---
legend_handles = [Line2D([0], [0], **{k: v for k, v in STYLE[name].items()
                                       if k not in ('zorder',)}, label=name)
                  for name in ['VeriFL', 'VLMRS', 'HVSA']]
fig.legend(legend_handles, ['VeriFL', 'VLMRS', 'HVSA'],
           loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3,
           fontsize=19, frameon=True, edgecolor='#C4D0E0', framealpha=0.95,
           handlelength=2.6, columnspacing=2.4, borderpad=0.7)

# 导出: PDF (供海报嵌入) + PNG (供快速预览)
pdf_path = 'exp_comparison_final.pdf'
png_path = 'exp_comparison_final.png'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=200, bbox_inches='tight')
print("[OK] saved:", pdf_path, "|", png_path)
