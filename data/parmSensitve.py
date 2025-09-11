import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 读取日志文件
# ------------------------------
log_file = "experiment_log.txt"
with open(log_file, "r") as f:
    lines = f.readlines()

# ------------------------------
# 正则匹配
# ------------------------------
param_pattern = re.compile(r"\[GridSearch\] Params: (.*)")
threshold_pattern = re.compile(
    r"\[Threshold=(\d+\.\d+)\] Closed Acc=(\d+\.\d+)  Open Recog=(\d+\.\d+)  Overall Acc=(\d+\.\d+)  F1_open=(\d+\.\d+)"
)

# ------------------------------
# 提取阈值0.99的数据
# ------------------------------
data = []
current_params = None

for line in lines:
    line = line.strip()
    param_match = param_pattern.match(line)
    if param_match:
        current_params = eval(param_match.group(1))
        continue

    thresh_match = threshold_pattern.match(line)
    if thresh_match and current_params is not None:
        threshold = float(thresh_match.group(1))
        if threshold == 0.99:
            row = {
                **current_params,
                "Threshold": threshold,
                "Closed_Acc": float(thresh_match.group(2)),
                "Open_Recog": float(thresh_match.group(3)),
                "Overall_Acc": float(thresh_match.group(4)),
                "F1_open": float(thresh_match.group(5)),
            }
            data.append(row)

df = pd.DataFrame(data)
df.to_csv("threshold_0.99_results.csv", index=False)
print("CSV 已保存: threshold_0.99_results.csv")

# ------------------------------
# 找到最佳参数（以 Overall_Acc 最大）
# ------------------------------
best_row = df.loc[df["Overall_Acc"].idxmax()]
print("最佳参数:", best_row.to_dict())

# ------------------------------
# 参数敏感性分析函数（分组柱状图，四个指标）
# ------------------------------
def plot_sensitivity_multi(df, fixed_params, vary_param, save_path="sensitivity.pdf"):
    # 筛选固定参数的数据
    mask = pd.Series(True, index=df.index)
    for k, v in fixed_params.items():
        mask &= (df[k] == v)
    plot_df = df[mask].sort_values(vary_param)

    metrics = ["Closed_Acc", "Open_Recog", "Overall_Acc", "F1_open"]
    n_metrics = len(metrics)
    x = np.arange(len(plot_df))
    width = 0.18

    plt.figure(figsize=(10,6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, plot_df[metric], width, label=metric)

    plt.xlabel(vary_param, fontsize=12, fontname='Times New Roman')
    plt.ylabel("Metric Value", fontsize=12, fontname='Times New Roman')
    plt.title(f"Parameter Sensitivity for {vary_param}", fontsize=14, fontname='Times New Roman')
    plt.xticks(x + width*(n_metrics-1)/2, plot_df[vary_param].astype(str), fontsize=10, fontname='Times New Roman')
    plt.yticks(fontsize=10, fontname='Times New Roman')
    plt.ylim(0,1.05)
    plt.legend(fontsize=10, frameon=False)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"参数敏感性柱状图已保存: {save_path}")

# ------------------------------
# 三次敏感性分析
# ------------------------------
# 1. beta 敏感性，固定 temperature, base_margin
plot_sensitivity_multi(df,
                 fixed_params={"temperature": best_row["temperature"], "base_margin": best_row["base_margin"]},
                 vary_param="beta",
                 save_path="sensitivity_beta.pdf")

# 2. base_margin 敏感性，固定 temperature, beta
plot_sensitivity_multi(df,
                 fixed_params={"temperature": best_row["temperature"], "beta": best_row["beta"]},
                 vary_param="base_margin",
                 save_path="sensitivity_base_margin.pdf")

# 3. temperature 敏感性，固定 base_margin, beta
plot_sensitivity_multi(df,
                 fixed_params={"base_margin": best_row["base_margin"], "beta": best_row["beta"]},
                 vary_param="temperature",
                 save_path="sensitivity_temperature.pdf")
