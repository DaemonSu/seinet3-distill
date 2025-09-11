import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV数据
df = pd.read_csv('results2.csv')

# 转换 open_set_f1 单位为百分比
df['open_set_f1'] *= 100

# 定义指标名称和标签
metrics = ['closed_set_acc', 'open_set_recognition_rate', 'overall_acc', 'open_set_f1']
metric_labels = ['Closed-set Acc (%)', 'Open-set Recog (%)', 'Overall Acc (%)', 'Open-set F1 (%)']

# 找到 overall_acc 最大值对应的参数组合
best_row = df.loc[df['overall_acc'].idxmax()]
best_params = {
    'temperature': best_row['temperature'],
    'base_margin': best_row['base_margin'],
    'beta': best_row['beta']
}
print("Best Parameters (max overall_acc):", best_params)


# 绘图函数
def plot_sensitivity(fixed_keys, varied_key, filename):
    fixed_cond = (df[fixed_keys[0]] == best_params[fixed_keys[0]]) & \
                 (df[fixed_keys[1]] == best_params[fixed_keys[1]])
    subset = df[fixed_cond].sort_values(by=varied_key)

    if subset.empty:
        print(f"[Warning] No data for {varied_key} variation with {fixed_keys} fixed.")
        return

    x_labels = subset[varied_key].tolist()
    bar_data = [subset[m].tolist() for m in metrics]

    # 绘制柱状图
    plt.figure(figsize=(9, 5))
    bar_width = 0.2
    x = range(len(x_labels))

    for i, values in enumerate(bar_data):
        plt.bar([p + i * bar_width for p in x], values, width=bar_width, label=metric_labels[i])

    plt.xlabel(f"{varied_key} (varied)")
    plt.ylabel("Metric Value (%)")
    plt.xticks([p + 1.5 * bar_width for p in x], [f"{v:.2f}" for v in x_labels])
    plt.title(f"Param Sensitivity: {varied_key} (fixed {fixed_keys[0]}={best_params[fixed_keys[0]]}, "
              f"{fixed_keys[1]}={best_params[fixed_keys[1]]})")

    # ✅ 设置 Y 轴从 80% 开始，更容易看出细微变化
    plt.ylim(85, max([max(values) for values in bar_data]) + 1)

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


# 执行三组敏感性分析
plot_sensitivity(['base_margin', 'beta'], 'temperature', 'sensitivity_temperature.pdf')
plot_sensitivity(['temperature', 'beta'], 'base_margin', 'sensitivity_base_margin.pdf')
plot_sensitivity(['temperature', 'base_margin'], 'beta', 'sensitivity_beta.pdf')
