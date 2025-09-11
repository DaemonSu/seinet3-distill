import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os

def compute_closed_acc(y_true, y_pred):
    mask = y_true != -1
    return accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() > 0 else np.nan

def compute_open_recognition_rate(y_true, y_pred):
    mask = y_true == -1
    return np.mean(y_pred[mask] == -1) if mask.sum() > 0 else np.nan

def compute_overall_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_f1_open(y_true, y_pred):
    return f1_score(y_true == -1, y_pred == -1)

def compute_auroc(known_scores, unknown_scores):
    y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_score)

def compute_oscr(known_scores, unknown_scores, correct_flags):
    """
    OSCR = 面积(正确率 vs 拒绝率)
    known_scores: 所有闭集样本置信度
    unknown_scores: 所有开集样本置信度
    correct_flags: 每个闭集样本是否预测正确 (1/0)，长度与 known_scores 一致
    """
    scores = np.concatenate([known_scores, unknown_scores])
    labels = np.concatenate([correct_flags, np.zeros(len(unknown_scores))])

    sorted_idx = np.argsort(-scores)  # 降序排序
    sorted_labels = labels[sorted_idx]

    cum_correct = np.cumsum(sorted_labels)
    total_correct = cum_correct[-1] if cum_correct[-1] > 0 else 1
    cum_samples = np.arange(1, len(sorted_labels)+1)

    tpr = cum_correct / total_correct
    fpr = cum_samples / len(sorted_labels)

    return np.trapz(tpr, fpr)

def compute_ece(probs, labels, n_bins=15):
    confidences, predictions = np.max(probs, axis=1), np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop_in_bin
    return ece

def compute_fpr95(known_scores, unknown_scores):
    y_true = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]

def evaluate_all_metrics(y_true, y_pred, probs, known_scores, unknown_scores, correct_flags):
    return {
        "CSRR": compute_closed_acc(y_true, y_pred),
        "OSRR": compute_open_recognition_rate(y_true, y_pred),
        "OA": compute_overall_acc(y_true, y_pred),
        "F1-open": compute_f1_open(y_true, y_pred),
        "AUROC": compute_auroc(known_scores, unknown_scores),
        "OSCR": compute_oscr(known_scores, unknown_scores, correct_flags),
        "ECE": compute_ece(probs, y_true),
        "FPR@95TPR": compute_fpr95(known_scores, unknown_scores)
    }

def save_metrics_to_csv(metrics_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("Metric,Value\n")
        for k, v in metrics_dict.items():
            f.write(f"{k},{v:.4f}\n")
