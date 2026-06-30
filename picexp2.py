import numpy as np
import matplotlib.pyplot as plt


def plot_mnist_from_npz(file_path):
    # 1. 读取 MNIST 数据
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到 MNIST 文件 {file_path}")
        return

    # 2. 设置全局绘图风格 (极限放大版)
    plt.rcParams.update({
        'font.size': 24,  # 基础字体大小提升至 24
        'axes.labelsize': 28,  # X/Y 轴标签提升至 28
        'xtick.labelsize': 22,  # X 轴刻度数字提升至 22
        'ytick.labelsize': 22  # Y 轴刻度数字提升至 22
    })
    plt.figure(figsize=(10, 6))

    # 3. 按掉线率排序键值
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
                 linewidth=4.0,  # 线宽加粗到 4.0，压住巨大的字体
                 **styles[i % len(styles)])

    # 6. 图表修饰
    plt.xlabel('Communication Rounds', fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontweight='bold')

    # 调整图例大小和边框
    plt.legend(
        loc='lower right',
        fontsize=22,  # 图例字体提升至 22
        frameon=True,
        edgecolor='black',  # 图例边框改黑，更清晰
        framealpha=0.9  # 降低透明度，防止遮挡后面的线导致看不清
    )

    plt.grid(True, linestyle='--', alpha=0.6, linewidth=1.5)  # 网格线也稍微加粗

    # 增加 pad 以防止巨大的字体被裁切
    plt.tight_layout(pad=2.0)

    # 7. 保存并显示
    output_filename = 'exp2-mnist3.pdf'
    plt.savefig(output_filename, dpi=300,bbox_inches='tight')
    print(f"MNIST 对比图已保存至: {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_mnist_from_npz('exp2_data_MNIST_20260311-235642.npz')