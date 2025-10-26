import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================================
# 读取实验记录
# =========================================================
with open("model/incremental_metrics.json", "r") as f:
    hist = json.load(f)

steps = sorted(int(k) for k in hist.keys())
classes = sorted(int(c) for c in hist[str(steps[-1])]["per_class_acc_now"].keys())

# =========================================================
# 构建每类每阶段准确率矩阵
# =========================================================
acc_matrix = np.full((len(classes), len(steps)), np.nan)

for si, s in enumerate(steps):
    per_class = hist[str(s)].get("per_class_acc_now", {})
    for ci, c in enumerate(classes):
        if str(c) in per_class:
            acc_matrix[ci, si] = per_class[str(c)]

# =========================================================
# 计算平均遗忘率 (AFR)
# =========================================================
afr_per_step = []
for si in range(len(steps)):
    F_c = []
    for ci in range(len(classes)):
        accs = acc_matrix[ci, :si+1]
        if not np.isnan(accs).all():
            max_prev = np.nanmax(accs[:-1]) if si > 0 else accs[0]
            last = accs[si]
            F_c.append(max(0, max_prev - last))
    afr_per_step.append(np.nanmean(F_c))

afr_per_step = np.nan_to_num(afr_per_step)

# =========================================================
# 计算总准确率 / 旧类准确率 / 新类准确率
# =========================================================
total_acc = [hist[str(s)]["total_acc"] for s in steps]
old_acc = [hist[str(s)]["old_acc"] for s in steps]
new_acc = [hist[str(s)]["new_acc"] for s in steps]

# =========================================================
# 绘图部分
# =========================================================
os.makedirs("figs", exist_ok=True)
plt.rcParams["font.family"] = "Times New Roman"

# ---------- 图1：Accuracy vs Step ----------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(steps, total_acc, "-o", label="Total Acc")
plt.plot(steps, old_acc, "-o", label="Old Acc")
plt.plot(steps, new_acc, "-o", label="New Acc")
plt.xlabel("Incremental Step")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Step")
plt.legend()
plt.grid(True)

# ---------- 图2：Average Forgetting vs Step ----------
plt.subplot(1, 2, 2)
plt.plot(steps, afr_per_step, "-o", color="red", label="Average Forgetting Rate")
plt.xlabel("Incremental Step")
plt.ylabel("Average Forgetting (%)")
plt.title("Average Forgetting vs Step")
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig("figs/accuracy_forgetting_curves.png", dpi=300)
plt.savefig("accuracy_forgetting_curves.pdf", bbox_inches="tight", format='pdf')
plt.show()

# =========================================================
# 绘制每类的准确率随阶段变化的热力图
# =========================================================
# plt.figure(figsize=(10, 6))
# sns.heatmap(
#     acc_matrix,
#     xticklabels=steps,
#     yticklabels=classes,
#     cmap="viridis",
#     cbar_kws={'label': 'Accuracy (%)'}
# )
# plt.xlabel("Incremental Step")
# plt.ylabel("Class ID")
# plt.title("Class-wise Accuracy Evolution")
# plt.tight_layout()
# # plt.savefig("figs/classwise_accuracy_heatmap.png", dpi=300)
# plt.savefig("classwise_accuracy_heatmap.pdf", bbox_inches="tight", format='pdf')
# plt.show()

plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    acc_matrix,
    xticklabels=steps,
    yticklabels=False,  # 初始不显示全部标签，先手动控制
    cmap="viridis",
    cbar_kws={'label': 'Accuracy (%)'}
)

# 控制 y 轴标签显示间隔
num_classes = acc_matrix.shape[0]
tick_interval = max(1, num_classes // 20)  # 每20个class显示一个label
yticks_to_show = np.arange(0, num_classes, tick_interval)
ax.set_yticks(yticks_to_show + 0.5)  # heatmap的格点中心偏移
ax.set_yticklabels([str(c) for c in yticks_to_show], rotation=45, ha='right', fontsize=8)

# x轴标签倾斜一点点更美观
plt.xticks(rotation=0, fontsize=9)
plt.yticks(fontsize=8)
plt.xlabel("Incremental Step")
plt.ylabel("Class ID")
plt.title("Class-wise Accuracy Evolution")
plt.tight_layout()
plt.savefig("classwise_accuracy_heatmap.pdf", bbox_inches="tight", format='pdf')


# =========================================================
# 输出统计结果
# =========================================================
mean_final_forgetting = np.nanmean(afr_per_step[1:])  # 第0步不算
print("==== Incremental Performance Summary ====")
print(f"Average Forgetting per Step: {afr_per_step}")
print(f"Mean Forgetting (overall): {mean_final_forgetting:.2f}%")
print(f"Final Total Acc: {total_acc[-1]:.2f}% | Old: {old_acc[-1]:.2f}% | New: {new_acc[-1]:.2f}%")
