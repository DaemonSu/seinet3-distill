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
    # 加载模型、映射
    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(config.device)
    classifier = torch.load(os.path.join(config.save_dir, 'classifier_incremental.pth')).to(config.device)
    mapping = load_object(os.path.join(config.save_dir, 'class_mapping.pkl'))
    label2idx = mapping['label2idx']
    idx2label = mapping['idx2label']

    encoder.eval()
    classifier.eval()

    all_preds_idx, all_labels_idx = [], []

    total_correct, total_num = 0, 0
    old_correct, old_num = 0, 0
    new_correct, new_num = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(config.device)
            # 映射真实标签 -> classifier index
            y_idx = torch.tensor([label2idx[int(l)] for l in y],
                                 dtype=torch.long, device=config.device)
            feat = encoder(x)
            logits = classifier(feat)
            preds_idx = logits.argmax(dim=1)

            total_correct += (preds_idx == y_idx).sum().item()
            total_num += x.size(0)

            mask_old = y_idx < config.old_num_classes
            mask_new = y_idx >= config.old_num_classes
            if mask_old.any():
                old_correct += (preds_idx[mask_old] == y_idx[mask_old]).sum().item()
                old_num += mask_old.sum().item()
            if mask_new.any():
                new_correct += (preds_idx[mask_new] == y_idx[mask_new]).sum().item()
                new_num += mask_new.sum().item()



            all_labels_idx.append(y_idx.cpu())
            all_preds_idx.append(preds_idx.cpu())

    acc_total = total_correct / total_num * 100
    acc_old = old_correct / old_num * 100 if old_num > 0 else 0
    acc_new = new_correct / new_num * 100 if new_num > 0 else 0
    print(f"Test Acc Total: {acc_total:.2f}% | Old: {acc_old:.2f}% | New: {acc_new:.2f}%")



    all_labels_idx = torch.cat(all_labels_idx).numpy()
    all_preds_idx = torch.cat(all_preds_idx).numpy()

    # 统计整体准确率
    total_acc = (all_labels_idx == all_preds_idx).mean() * 100

    print(f"Test Accuracy Total: {total_acc:.2f}%")

    # ==== 构建混淆矩阵 ====
    # labels_order = 所有类别 index（包含旧类、新类）
    labels_order = sorted(idx2label.keys())
    cm = confusion_matrix(all_labels_idx, all_preds_idx, labels=labels_order)

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    # 显示真实标签编号
    label_names = [idx2label[i] for i in labels_order]
    print("Labels:", label_names)
    print(cm)




if __name__ == "__main__":
    config = parse_args()
    # ============ 数据加载 ============

    mixed_testset = KnownDataset(config.test_add1)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=True)

    test_incremental( config,mixed_loader)
