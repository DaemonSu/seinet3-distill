import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import parse_args
from dataset import MixedDataset, KnownDataset
from model_mix import FeatureExtractor, ClassifierHead
from util.utils import expected_calibration_error, fpr_at_95_tpr
from util.visualize import visualize_features

import torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from collections import Counter

def test_mixed(encoder, classifier, test_loader, config):
    encoder.eval()
    classifier.eval()
    device = config.device

    all_preds = []
    all_labels = []
    all_probs = []     # list of batch tensors (cpu)
    known_scores = []
    unknown_scores = []
    all_feats = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            y_torch = y.to(device) if isinstance(y, torch.Tensor) else torch.tensor(y, device=device, dtype=torch.long)

            feat = encoder(x)
            logits = classifier(feat)   # [B, C]
            probs = F.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)

            # collect
            all_probs.append(probs.cpu())          # for ECE
            all_feats.append(feat.cpu().numpy())   # for t-SNE

            # known/unknown scores (if unknown labeled as -1)
            known_scores.extend(max_probs.cpu().numpy()[y_torch.cpu().numpy() != -1])
            unknown_scores.extend(max_probs.cpu().numpy()[y_torch.cpu().numpy() == -1])

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y_torch.cpu().numpy().tolist())

    # concat tensors / arrays
    all_probs = torch.cat(all_probs, dim=0) if len(all_probs) > 0 else torch.empty(0)
    all_feats = np.vstack(all_feats) if len(all_feats) > 0 else np.empty((0,))
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)



    # Debug prints: check label ranges and counts
    print("DEBUG: classifier output dim:", getattr(classifier, 'out_features', None) or logits.shape[1])
    print("DEBUG: test label unique:", np.unique(y_true))
    print("DEBUG: predicted unique:", np.unique(y_pred))
    print("DEBUG: pred distribution:", Counter(y_pred))

    # Closed-set mask: if you use -1 to denote unknown in test labels, mask them out
    closed_mask = (y_true != -1)
    open_mask = (y_true == -1)

    if closed_mask.sum() > 0:
        closed_acc = accuracy_score(y_true[closed_mask], y_pred[closed_mask])
    else:
        closed_acc = float('nan')

    if open_mask.sum() > 0:
        open_recognition_rate = np.mean(y_pred[open_mask] == -1)
    else:
        open_recognition_rate = float('nan')

    overall_acc = accuracy_score(y_true, y_pred)

    # if open labels exist compute F1 (binary unknown vs known)
    try:
        f1_open = f1_score(y_true == -1, y_pred == -1, zero_division=0)
    except Exception:
        f1_open = float('nan')

    # optional: ECE (需要你的 expected_calibration_error 函数)
    try:
        ece = expected_calibration_error(all_probs, torch.tensor(y_true, dtype=torch.long))
    except Exception as e:
        print("[Warn] ECE computation failed:", e)
        ece = float('nan')

    # optional: fpr95
    try:
        fpr95 = fpr_at_95_tpr(np.array(known_scores), np.array(unknown_scores))
    except Exception as e:
        print("[Warn] fpr95 computation failed:", e)
        fpr95 = float('nan')

    print("\n===== Mixed Test Results =====")
    print(f"Closed-set Accuracy      : {closed_acc * 100:.2f}%")
    print(f"Open-set Recognition Rate: {open_recognition_rate * 100 if not np.isnan(open_recognition_rate) else np.nan:.2f}%")
    print(f"Overall Accuracy         : {overall_acc * 100:.2f}%")
    print(f"Open-set F1 Score        : {f1_open:.4f}")
    print(f"ECE = {ece:.4f}, FPR@95TPR = {fpr95:.4f}")

    # confusion matrix
    try:
        np.set_printoptions(
            linewidth=2000,  # 每行最大字符数（设置足够大的值避免换行）
            threshold=np.inf  # 强制打印所有元素（不省略）
        )
        labels_order = sorted(np.unique(np.concatenate([y_true, y_pred])))
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print("Labels:", labels_order)
        print(cm)
    except Exception as e:
        print("[Warn] confusion_matrix failed:", e)

    # optional feature visualization
    try:
        if all_feats.shape[0] > 0:
            visualize_features(all_feats, y_true, known_class_count=config.old_num_classes, method='t-SNE')
    except Exception as e:
        print("[Warn] feature viz failed:", e)

    return closed_acc, open_recognition_rate, overall_acc, f1_open



if __name__ == "__main__":
    config = parse_args()
    # ============ 数据加载 ============

    mixed_testset = KnownDataset(config.test_closed)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=True)

    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(config.device)
    classifier = torch.load(os.path.join(config.save_dir,'classifier.pth')).to(config.device)


    # 假设已有：encoder, classifier, config, test_loader
    test_mixed(encoder, classifier, mixed_loader, config)
