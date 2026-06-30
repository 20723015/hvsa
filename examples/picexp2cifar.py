import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt


def plot_cifar10_from_npz(file_path):
    # 1. 读取 CIFAR-10 数据
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到 CIFAR-10 文件 {file_path}")
        return

    # 2. 设置全局绘图风格 (极限放大版，与 MNIST 保持一致)
    plt.rcParams.update({
        'font.size': 24,  # 基础字体大小提升至 24
        'axes.labelsize': 28,  # X/Y 轴标签提升至 28
        'xtick.labelsize': 22,  # X 轴刻度数字提升至 22
        'ytick.labelsize': 22  # Y 轴刻度数字提升至 22
    })
    plt.figure(figsize=(10, 6))

    # 3. 按掉线率排序键值 (假设命名规则仍包含 dropout_rate)
    keys = sorted(data.files, key=lambda x: float(x.split('_')[1]))

    # 4. 定义样式
    styles = [
        {'color': 'black', 'linestyle': '-'},
        {'color': 'blue', 'linestyle': '--'},
        {'color': 'red', 'linestyle': '-.'},
        {'color': 'green', 'linestyle': ':'}
    ]

    # 5. 绘图
    for i, key in enumerate(keys):
        acc_history = data[key]
        dropout_rate = float(key.split('_')[1])
        epochs = range(1, len(acc_history) + 1)

        plt.plot(epochs, acc_history,
                 label=f'Dropout Rate: {int(dropout_rate * 100)}%',
                 linewidth=4.0,  # 线宽加粗到 4.0
                 **styles[i % len(styles)])

    # 6. 图表修饰 (匹配放大的字体并加粗)
    plt.xlabel('Communication Rounds', fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontweight='bold')

    # 调整图例大小和边框防遮挡
    plt.legend(
        loc='lower right',
        fontsize=22,  # 图例字体提升至 22
        frameon=True,
        edgecolor='black',  # 图例边框改黑
        framealpha=0.9  # 降低透明度防遮挡
    )

    plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)  # 网格线加粗

    # 增加 pad 以防止巨大的字体被裁切
    plt.tight_layout(pad=2.0)

    # 7. 保存并显示
    output_filename = 'exp2-3cifar10.pdf'
    plt.savefig(output_filename, dpi=300,bbox_inches='tight')
    print(f"CIFAR-10 对比图已保存至: {output_filename}")
    plt.show()


if __name__ == "__main__":
    # 请更新为对应的 CIFAR-10 数据文件路径
    plot_cifar10_from_npz('exp2_data_CIFAR10_20260312-151849.npz')