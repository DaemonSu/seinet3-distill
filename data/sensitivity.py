import re
from collections import defaultdict
import matplotlib.pyplot as plt

# 记录所有实验结果：(T, M, B, threshold) -> 评估指标
results = {}

# 记录每个(T, M, B)组合对应的最佳 threshold
best_thresholds = {}

def parse_log_file(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = {}
    best_thresholds = {}

    current_params = None
    threshold_idx_map = [0.75, 0.80, 0.85, 0.90, 0.95]
    current_threshold_index = -1
    buffer = {}

    for line in lines:
        line = line.strip()

        # 匹配超参数
        if "[GridSearch] Params:" in line:
            match = re.search(r"Params: \{'temperature': ([\d\.]+), 'base_margin': ([\d\.]+), 'beta': ([\d\.]+)\}", line)
            if match:
                T = float(match.group(1))
                M = float(match.group(2))
                B = float(match.group(3))
                current_params = (T, M, B)
                current_threshold_index = 0
                buffer = {}
            continue

        # 指标提取（使用正则以避免 tqdm 前缀干扰）
        closed_match = re.search(r"closed_acc:\s*([\d\.]+)", line)
        open_match = re.search(r"open_recognition_rate:\s*([\d\.]+)", line)
        overall_match = re.search(r"overall_acc:\s*([\d\.]+)", line)
        f1_match = re.search(r"f1_open:\s*([\d\.]+)", line)

        if closed_match:
            buffer['closed_acc'] = float(closed_match.group(1))
        if open_match:
            buffer['open_rate'] = float(open_match.group(1))
        if overall_match:
            buffer['overall_acc'] = float(overall_match.group(1))
        if f1_match:
            buffer['f1_open'] = float(f1_match.group(1))

            # 四项指标都到齐了，写入
            if len(buffer) == 4 and current_params is not None:
                if current_threshold_index < len(threshold_idx_map):
                    threshold = threshold_idx_map[current_threshold_index]
                    key = current_params + (threshold,)
                    results[key] = buffer.copy()
                    current_threshold_index += 1
                buffer.clear()

        # 解析 Best open_threshold
        if "Best open_threshold" in line:
            match = re.search(r"Best open_threshold for this combination: ([\d\.]+)", line)
            if match and current_params:
                best_thresholds[current_params] = float(match.group(1))

    print(f"✅ 成功解析结果条目数: {len(results)}")
    return results, best_thresholds


def find_best_overall(results):
    best_key = None
    best_score = -1
    for key, val in results.items():
        if val['overall_acc'] > best_score:
            best_score = val['overall_acc']
            best_key = key
    print("\n🏆 Best Overall Accuracy Combination:")
    print(f"Params: T={best_key[0]}, M={best_key[1]}, B={best_key[2]}, Threshold={best_key[3]}")
    print(f"Scores: {results[best_key]}")
    return best_key

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_sensitivity(results, best_thresholds, fixed_params: dict, vary: str):
    """
    绘制敏感性分析图。

    :param results: 日志解析得到的完整结果字典，key 是 (temperature, base_margin, beta, threshold)
    :param best_thresholds: 每组 (temperature, base_margin, beta) 对应的最佳 threshold
    :param fixed_params: 指定固定的两个或三个参数（必须包含 temperature, base_margin, beta 三个中除 vary 外的两个）
    :param vary: 指定的变化参数，可以是 'temperature', 'base_margin', 'beta', 'threshold' 中的一个
    """
    valid_fields = ['temperature', 'base_margin', 'beta', 'threshold']
    assert vary in valid_fields, f"vary 参数必须是 {valid_fields} 中的一个"

    param_names = ['temperature', 'base_margin', 'beta', 'threshold']
    param_idx = {name: i for i, name in enumerate(param_names)}

    filtered = {}
    for key, metrics in results.items():
        temp, margin, beta, threshold = key
        param_values = {
            'temperature': temp,
            'base_margin': margin,
            'beta': beta,
            'threshold': threshold
        }

        # 如果 vary 不是 threshold，则过滤掉非最佳 threshold 组合
        if vary != 'threshold':
            key_no_threshold = (temp, margin, beta)
            if best_thresholds.get(key_no_threshold, None) != threshold:
                continue
        else:
            # 如果 vary 是 threshold，则需要固定前面三个参数
            required_keys = ['temperature', 'base_margin', 'beta']
            if not all(k in fixed_params for k in required_keys):
                raise ValueError(f"When varying 'threshold', fixed_params must include: {required_keys}")
            if any(abs(param_values[k] - fixed_params[k]) > 1e-6 for k in required_keys):
                continue

        # 检查是否满足固定参数条件
        match = True
        for k, v in fixed_params.items():
            if k != vary and abs(param_values[k] - v) > 1e-6:
                match = False
                break
        if match:
            filtered[param_values[vary]] = metrics

    if not filtered:
        print("❌ 没有找到满足指定固定参数条件的记录。请检查 fixed_params 设置。")
        return

    # 排序并提取数据
    x_vals = sorted(filtered.keys())
    closed_acc = [filtered[x]['closed_acc'] for x in x_vals]
    open_rate = [filtered[x]['open_rate'] for x in x_vals]
    overall_acc = [filtered[x]['overall_acc'] for x in x_vals]
    f1_open = [filtered[x]['f1_open'] for x in x_vals]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, closed_acc, marker='o', label='Closed Acc')
    plt.plot(x_vals, open_rate, marker='o', label='Open Recog. Rate')
    plt.plot(x_vals, overall_acc, marker='o', label='Overall Acc')
    plt.plot(x_vals, f1_open, marker='o', label='F1 Open')

    plt.xlabel(f'Varying Parameter: {vary}')
    plt.ylabel('Metric Value')
    plt.title(f'Sensitivity Analysis on {vary}\nFixed Parameters: {fixed_params}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    log_path = "log3.txt"  # 修改为你的日志文件路径
    results, best_thresholds = parse_log_file(log_path)
    best_key = find_best_overall(results)
    # plot_sensitivity(results, best_key)
    plot_sensitivity(results, best_thresholds=best_thresholds,fixed_params={'temperature': 0.13, 'base_margin': 0.5},vary='beta')
    plot_sensitivity(results,  best_thresholds=best_thresholds,fixed_params={'temperature': 0.13, 'beta': 0.6}, vary='base_margin')
    plot_sensitivity(results,  best_thresholds=best_thresholds,fixed_params={'beta': 0.6, 'base_margin': 0.5}, vary='temperature')

