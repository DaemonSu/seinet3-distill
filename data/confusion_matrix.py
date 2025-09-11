import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 混淆矩阵数据
cm = np.array([
[540,0,0,0,0,0,0,0,0],
[0,476,0,0,0,0,0,0,3],
[0,0,489,0,0,0,0,0,4],
[0,0,0,513,0,0,0,0,5],
[0,0,0,0,493,0,0,0,1],
[0,0,0,0,0,492,0,0,16],
[0,0,0,0,0,0,467,0,4],
[0,0,0,0,0,0,0,495,2],
[0,15,34,7,33,30,2,2,3877]
])

# 类别标签
labels = [f"ID{i:02d}" for i in range(8)] + ["Open"]

# 将混淆矩阵转换为行归一化百分比
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 绘制热力图
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm_percent, annot=True, fmt=".2f", cmap="YlGnBu",
    xticklabels=labels, yticklabels=labels,
    cbar=True, linewidths=0.5, linecolor='gray'
)

# 设置 colorbar 标题为 "Percentage"
cbar = ax.collections[0].colorbar
cbar.set_label("Percentage (%)", fontsize=12)

# 图表标题和标签
plt.title("Confusion Matrix (Percentage per Row)", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)

# 调整布局
plt.tight_layout()

# 保存为 PDF
plt.savefig("confusion_matrix_percentage.pdf", format="pdf")
plt.close()

print("百分比 confusion matrix 已保存为 confusion_matrix_percentage.pdf")
