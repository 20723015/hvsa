import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'exp1_data_MNIST_20260118-224054.npz'
data = np.load(file_path)
acc_fedavg = data['fedavg']
acc_ours = data['ours']

# 2. 设置全局参数 (应用完全一致的极限放大版尺寸)
plt.rcParams.update({
    'font.size': 24,          # 基础字体大小提升至 24
    'axes.labelsize': 28,     # X/Y 轴标签提升至 28
    'xtick.labelsize': 22,    # X 轴刻度数字提升至 22
    'ytick.labelsize': 22     # Y 轴刻度数字提升至 22
})

# 画布大小统一调整为 (10, 6)
plt.figure(figsize=(10, 6))

# 3. 绘图 (线宽加粗至 4.0，同时放大标记点)
epochs = range(1, len(acc_fedavg) + 1)
# markersize 从 6 提升到 12，完美压住 4.0 的线宽
plt.plot(epochs, acc_fedavg, 'k--', label='FedAvg', linewidth=4.0, markevery=5, markersize=12)
# 【重要修复】：把 HSVA 再次纠正为 HVSA！
plt.plot(epochs, acc_ours, 'r-', label='HVSA', linewidth=4.0, alpha=0.8, markevery=5, markersize=12)

# 4. 坐标轴与网格 (标签加粗，网格线加粗)
plt.xlabel('Communication Rounds', fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)

# 5. 图例设置 (统一 22 号巨型字体)
plt.legend(
    loc='lower right',
    fontsize=22,             # 图例字体提升至 22
    frameon=True,
    edgecolor='black',       # 边框改黑更清晰
    framealpha=0.9,          # 降低透明度防遮挡
    shadow=True,             # 保留阴影效果
    borderpad=1.0,
    handlelength=2.5
)

# 6. 保存与显示 (增加 pad 防止边缘被裁)
plt.tight_layout(pad=2.0)
plt.savefig('mnist_exp1.pdf', dpi=300, bbox_inches='tight')
print("MNIST 精度对比图 (极限放大版) 已保存！")
plt.show()