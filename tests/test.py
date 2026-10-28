import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np
import logging

# ================= 配置区 =================
# 请将此处修改为您生成的那个 txt 日志文件名
LOG_FILENAME = 'exp2_dropout_log_MNIST_20260119-161542.txt'

# 您希望救援出的掉线率数据 (根据您的描述是 0 和 0.3)
TARGET_RATES = [0.0, 0.3]
TOTAL_ROUNDS = 50  # 您的实验总轮次


# ==========================================

def recover_data():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"正在从日志文件 {LOG_FILENAME} 中恢复数据...")

    data_store = {f'rate_{r}': [] for r in TARGET_RATES}

    # 正则表达式匹配日志行
    # 格式示例: [Dropout 30%] Round 1/50 | Online: 7/10 | Acc: 12.34%
    pattern = re.compile(r'\[Dropout (\d+)%\] Round (\d+)/\d+ .* Acc: ([\d\.]+)%')

    try:
        with open(LOG_FILENAME, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    rate_percent = int(match.group(1))
                    round_num = int(match.group(2))
                    acc = float(match.group(3))

                    # 将百分比转回小数 (30 -> 0.3)
                    rate_key = f'rate_{rate_percent / 100}'

                    # 如果 float 转换有微小误差 (如 0.3000004)，做个字符串截断匹配
                    # 这里简化处理：直接匹配我们预设的 key
                    if rate_key in data_store:
                        data_store[rate_key].append(acc)
    except FileNotFoundError:
        logging.error("错误：找不到日志文件，请检查文件名！")
        return

    # 检查并保存
    final_results = {}
    for rate in TARGET_RATES:
        key = f'rate_{rate}'
        recovered_acc = data_store[key]

        if len(recovered_acc) == 0:
            logging.warning(f"警告：未在日志中找到 rate={rate} 的数据。")
        else:
            logging.info(f"成功恢复 rate={rate} 的数据，共 {len(recovered_acc)} 轮。")
            # 截取或补全到 50 轮 (防止中间断掉)
            if len(recovered_acc) > TOTAL_ROUNDS:
                recovered_acc = recovered_acc[:TOTAL_ROUNDS]
            final_results[key] = recovered_acc

    if final_results:
        save_name = LOG_FILENAME.replace('.txt', '.npz').replace('log', 'data_recovered')
        np.savez(save_name, **final_results)
        logging.info(f"✅ 数据救援成功！已保存为: {save_name}")
        logging.info("您现在可以使用 exp2_supplementary.py 脚本来跑 10% 和 20% 的实验了。")
    else:
        logging.error("❌ 救援失败，未提取到有效数据。")


if __name__ == "__main__":
    recover_data()