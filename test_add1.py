import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import parse_args
from dataset import KnownDataset
from model_mix import FeatureExtractor, ClassifierHead
from util.utils import expected_calibration_error, fpr_at_95_tpr, load_object


from sklearn.metrics import confusion_matrix

def test_incremental(config, test_loader):
    # === 加载模型 ===
    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(config.device)
    classifier = torch.load(os.path.join(config.save_dir, 'classifier_incremental.pth')).to(config.device)
    contrastive_layer = torch.load(os.path.join(config.save_dir, 'mid_incremental.pth')).to(config.device)

    encoder.eval()
    classifier.eval()
    contrastive_layer.eval()

    all_preds, all_labels = [], []

    total_correct, total_num = 0, 0
    old_correct, old_num = 0, 0
    new_correct, new_num = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)
            y = y.to(config.device)  # 直接用设备编号作为标签

            feat = encoder(x)
            feat = contrastive_layer(feat)
            logits = classifier(feat)
            preds = logits.argmax(dim=1)

            total_correct += (preds == y).sum().item()
            total_num += x.size(0)

            mask_old = y < config.old_num_classes
            mask_new = y >= config.old_num_classes

            if mask_old.any():
                old_correct += (preds[mask_old] == y[mask_old]).sum().item()
                old_num += mask_old.sum().item()
            if mask_new.any():
                new_correct += (preds[mask_new] == y[mask_new]).sum().item()
                new_num += mask_new.sum().item()

            all_labels.append(y.cpu())
            all_preds.append(preds.cpu())

    acc_total = total_correct / total_num * 100
    acc_old = old_correct / old_num * 100 if old_num > 0 else 0
    acc_new = new_correct / new_num * 100 if new_num > 0 else 0
    print(f"Test Acc Total: {acc_total:.2f}% | Old: {acc_old:.2f}% | New: {acc_new:.2f}%")

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # ==== 混淆矩阵 ====
    np.set_printoptions(
        linewidth=2000,  # 每行最大字符数（设置足够大的值避免换行）
        threshold=np.inf # 强制打印所有元素（不省略）
        )
    labels_order = np.arange(all_preds.max() + 1)  # 直接按设备编号顺序
    cm = confusion_matrix(all_labels, all_preds, labels=labels_order)

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("Labels:", labels_order.tolist())
    print(cm)




if __name__ == "__main__":
    config = parse_args()
    # ============ 数据加载 ============

    mixed_testset = KnownDataset(config.test_add1)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=False)

    test_incremental( config,mixed_loader)
