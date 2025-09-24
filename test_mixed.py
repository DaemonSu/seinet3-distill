import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import parse_args
from dataset import MixedDataset
from model_mix import FeatureExtractor, ClassifierHead
from util.utils import expected_calibration_error, fpr_at_95_tpr
from util.visualize import visualize_features
import matplotlib.pyplot as plt


def test_mixed(encoder, classifier, test_loader, config):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    all_labelsdd = []
    all_probs = []
    all_logits = []

    all_feats = []
    known_scores, unknown_scores = [], []
    unknown_max_probs = []  # 记录未知样本 max_probs


    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(config.device), y.to(config.device)
            feat = encoder(x)

            logits = classifier(feat)
            prob = F.softmax(logits, dim=1)
            all_probs.append(prob.cpu())
            all_labelsdd.append(y.cpu())
            max_probs, preds = prob.max(dim=1)

            # Threshold-based prediction
            pred_labels = []
            for p, pred in zip(max_probs, preds):
                if p < config.open_threshold:
                    pred_labels.append(-1)
                    score = prob.max(1)[0]  # 同样用最大 softmax 概率
                    unknown_scores.extend(score.cpu().numpy())
                    # Open-set
                else:
                    pred_labels.append(pred.item())  # Closed-set prediction
                    score = prob.max(1)[0]  # 用最大 softmax 概率当 known_score
                    known_scores.extend(score.cpu().numpy())

            all_preds.extend(pred_labels)
            all_labels.extend(y.cpu().tolist())
            all_logits.append(max_probs.cpu().numpy())
            all_feats.extend(feat.cpu().numpy())

        # 记录真实未知样本的 max_probs（用于分布可视化）
    for prob_val, true_label in zip(max_probs, y):
        if true_label.item() == -1:
            unknown_max_probs.append(prob_val.item())


    visualize_features(np.array(all_feats),np.array(all_labels), known_class_count=20, method='t-SNE')
    all_logits = np.concatenate(all_logits)

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Masking
    closed_mask = y_true != -1
    open_mask = y_true == -1

    # Closed-set Accuracy
    if closed_mask.sum() > 0:
        closed_acc = accuracy_score(y_true[closed_mask], y_pred[closed_mask])
    else:
        closed_acc = float('nan')

    # Open-set Recognition Rate
    if open_mask.sum() > 0:
        open_recognition_rate = np.mean(y_pred[open_mask] == -1)
    else:
        open_recognition_rate = float('nan')

    # Overall Accuracy
    overall_acc = np.mean(y_pred == y_true)

    # Open-set F1 Score (binary classification: -1 vs not -1)
    f1_open = f1_score(y_true == -1, y_pred == -1)

    # 拼接
    all_probs = torch.cat(all_probs)
    all_labelsdd=torch.cat(all_labelsdd)

    # 计算指标
    ece = expected_calibration_error(all_probs, all_labelsdd)
    fpr95 = fpr_at_95_tpr(np.array(known_scores), np.array(unknown_scores))

    print(f"ECE = {ece:.4f}, FPR@95TPR = {fpr95:.4f}")


    print("\n===== Mixed Test Results =====")
    print(f"Closed-set Accuracy     : {closed_acc * 100:.2f}%")
    print(f"Open-set Recognition Rate: {open_recognition_rate * 100:.2f}%")
    print(f"Overall Accuracy        : {overall_acc * 100:.2f}%")
    print(f"Open-set F1 Score       : {f1_open:.4f}")

    # 混淆矩阵
    np.set_printoptions(
        linewidth=2000,  # 每行最大字符数（设置足够大的值避免换行）
        threshold=np.inf  # 强制打印所有元素（不省略）
    )
    try:
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds, labels=list(range(20)) + [-1]))
    except:
        print("Warning: Unable to compute confusion matrix for open-set labels.")

    return closed_acc, open_recognition_rate, overall_acc, f1_open


if __name__ == "__main__":
    config = parse_args()
    # ============ 数据加载 ============

    mixed_testset = MixedDataset(config.test_mixed)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=True)

    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(config.device)
    classifier = torch.load(os.path.join(config.save_dir,'classifier.pth')).to(config.device)


    # 假设已有：encoder, classifier, config, test_loader
    test_mixed(encoder, classifier, mixed_loader, config)
