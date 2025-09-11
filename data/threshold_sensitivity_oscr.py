import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# =======================
# 配置参数
# =======================
thresholds = np.arange(0.5, 1.0, 0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据加载 ============
# 假设你有 MixedDataset 类
config = parse_args()
mixed_testset = MixedDataset(config.test_mixed)
mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=False)

# ============ 加载模型 ============
encoder = FeatureExtractor(1024).to(device)
classifier = ClassifierHead(1024, config.num_classes).to(device)

ckpt1 = torch.load(os.path.join(config.save_dir, '26-q/', 'encoder.pth'), map_location=device)
ckpt2 = torch.load(os.path.join(config.save_dir, '26-q/', 'classifier.pth'), map_location=device)

encoder.load_state_dict(ckpt1['encoder'])
classifier.load_state_dict(ckpt2['classifier'])

encoder.eval()
classifier.eval()

# =======================
# Step 1: 提取整个测试集特征和 logits
# =======================
all_max_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(mixed_loader, desc="Extracting Features"):
        x, y = x.to(device), y.to(device)
        feat = encoder(x)
        logits = classifier(feat)
        prob = F.softmax(logits, dim=1)
        max_prob, pred = prob.max(dim=1)

        all_max_probs.append(max_prob.cpu())
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())

all_max_probs = torch.cat(all_max_probs)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
num_known = config.num_known  # 已知类数量
is_known = all_labels < num_known  # bool tensor

# =======================
# Step 2: 定义 OSCR 函数
# =======================
def compute_oscr(y_true, y_pred, is_known_mask):
    y_true = y_true.clone()
    y_pred = y_pred.clone()
    correct_known = (y_pred[is_known_mask] == y_true[is_known_mask]).sum().item()
    n_known = is_known_mask.sum().item()

    correct_unknown = (y_pred[~is_known_mask] == -1).sum().item()
    n_unknown = (~is_known_mask).sum().item()

    if n_known + n_unknown == 0:
        return 0.0
    return (correct_known + correct_unknown) / (n_known + n_unknown)

# =======================
# Step 3: 阈值循环计算指标
# =======================
closed_set_acc = []
open_set_detect_rate = []
f1_scores = []
auroc_list = []
oscr_list = []

all_labels_np = all_labels.numpy()
all_preds_np = all_preds.numpy()
all_max_probs_np = all_max_probs.numpy()
is_known_np = is_known.numpy()

for tau in thresholds:
    pred_known = all_max_probs >= tau
    final_preds = all_preds.clone()
    final_preds[~torch.tensor(pred_known)] = -1  # 标记未知

    final_preds_np = final_preds.numpy()

    closed_acc = accuracy_score(all_labels_np[is_known_np], final_preds_np[is_known_np])
    closed_set_acc.append(closed_acc)

    detect_rate = (final_preds_np[~is_known_np] == -1).sum() / max((~is_known_np).sum(), 1)
    open_set_detect_rate.append(detect_rate)

    f1 = f1_score(all_labels_np[is_known_np], final_preds_np[is_known_np], average='macro')
    f1_scores.append(f1)

    auroc = roc_auc_score(is_known_np.astype(int), all_max_probs_np)
    auroc_list.append(auroc)

    oscr = compute_oscr(all_labels, final_preds, is_known)
    oscr_list.append(oscr)

    print(f"Threshold: {tau:.2f} | Closed Acc: {closed_acc:.4f} | Open Detect: {detect_rate:.4f} "
          f"| F1: {f1:.4f} | AUROC: {auroc:.4f} | OSCR: {oscr:.4f}")

# =======================
# Step 4: 保存结果为 CSV
# =======================
df = pd.DataFrame({
    "Threshold": thresholds,
    "Closed_Acc": closed_set_acc,
    "Open_Detect_Rate": open_set_detect_rate,
    "F1_Score": f1_scores,
    "AUROC": auroc_list,
    "OSCR": oscr_list
})
df.to_csv("threshold_sensitivity_results.csv", index=False)
print("CSV 保存完成：threshold_sensitivity_results.csv")

# =======================
# Step 5: 绘制 TIFS 风格曲线 & 保存 PDF
# =======================
plt.figure(figsize=(8, 6))
plt.plot(thresholds, closed_set_acc, 'o-', label='Closed-set Acc')
plt.plot(thresholds, open_set_detect_rate, 's-', label='Open-set Detect Rate')
plt.plot(thresholds, f1_scores, '^-', label='F1 Score')
plt.plot(thresholds, auroc_list, 'd-', label='AUROC')
plt.plot(thresholds, oscr_list, 'x-', label='OSCR')

plt.xlabel('Threshold', fontsize=12, fontname='Times New Roman')
plt.ylabel('Metric Value', fontsize=12, fontname='Times New Roman')
plt.title('Threshold Sensitivity Analysis', fontsize=14, fontname='Times New Roman')
plt.xticks(fontsize=10, fontname='Times New Roman')
plt.yticks(fontsize=10, fontname='Times New Roman')
plt.legend(fontsize=10, frameon=False)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('threshold_sensitivity_oscr.pdf')
print("PDF 保存完成：threshold_sensitivity_oscr.pdf")
