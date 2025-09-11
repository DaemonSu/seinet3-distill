import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt


def evaluate_openset_predictions(probs, labels, thresholds=None, plot_roc=True):
    """
    自动评估开集识别结果，并寻找最佳阈值。

    Args:
        probs: 模型输出的 sigmoid 概率值，形状 [N]
        labels: 真实标签（二分类：1表示known，0表示unknown），形状 [N]
        thresholds: 要测试的阈值列表，默认从 0.01 到 0.99 每隔 0.01 一个点
        plot_roc: 是否绘制 ROC 曲线

    Returns:
        best_threshold, metrics_dict
    """

    # 转为 numpy
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_f1 = 0
    best_metrics = {}
    best_threshold = 0

    print(f"{'Thresh':>7} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}")
    print("-" * 40)
    for t in thresholds:
        preds = (probs >= t).astype(np.float32)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        print(f"{t:7.2f} | {acc:6.3f} | {prec:6.3f} | {rec:6.3f} | {f1:6.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }

    # ROC AUC
    try:
        auc = roc_auc_score(labels, probs)
        best_metrics["auc"] = auc
    except:
        best_metrics["auc"] = None

    print(f"\n✅ Best Threshold: {best_threshold:.2f} | F1: {best_f1:.4f} | Acc: {best_metrics['accuracy']:.4f}")

    # ROC 曲线
    if plot_roc:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {best_metrics["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Open-set Recognition')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

    return best_threshold, best_metrics
