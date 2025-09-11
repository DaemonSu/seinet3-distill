import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import KnownDataset, UnknownDataset
from model_open import FeatureExtractor, ClassifierHead
from util.utils import set_seed
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import os
from config import parse_args

def evaluate_open_contrastive(config):
    set_seed(config.seed)

    # ============ 数据加载 ============
    known_testset = KnownDataset(config.test_closeData)
    known_loader = DataLoader(known_testset, batch_size=config.batch_size, shuffle=False)

    unknown_testset = UnknownDataset(config.test_openData)
    unknown_loader = DataLoader(unknown_testset, batch_size=config.batch_size, shuffle=False)

    # ============ 加载模型 ============
    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    ckpt = torch.load(os.path.join(config.save_dir, 'model_opencon2.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    encoder.eval()
    classifier.eval()

    all_preds, all_labels = [], []
    all_logits = []

    # ============ 已知类测试 ============
    for x, y in known_loader:
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            feat = encoder(x)
            logits = classifier(feat)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(F.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy())

    # ============ 未知类测试 ============
    for x, _ in unknown_loader:
        x = x.to(config.device)
        with torch.no_grad():
            feat = encoder(x)
            logits = classifier(feat)
            prob = F.softmax(logits, dim=1)
            max_probs, preds = prob.max(dim=1)
            unknown_labels = [-1] * x.size(0)
            all_labels.extend(unknown_labels)
            all_preds.extend([-1 if p < config.open_threshold else pred.item() for p, pred in zip(max_probs, preds)])
            all_logits.append(max_probs.cpu().numpy())

    # ============ 指标统计 ============
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    known_mask = all_labels != -1
    unknown_mask = all_labels == -1

    closed_acc = np.mean(all_preds[known_mask] == all_labels[known_mask]) * 100
    open_recog_rate = np.mean(all_preds[unknown_mask] == -1) * 100
    overall_acc = np.mean(all_preds == all_labels) * 100
    open_f1 = f1_score((all_labels == -1), (all_preds == -1))

    print(f"\n📊 Evaluation Results:")
    print(f"Closed-set Accuracy      : {closed_acc:.2f}%")
    print(f"Open-set Recognition Rate: {open_recog_rate:.2f}%")
    print(f"Overall Accuracy         : {overall_acc:.2f}%")
    print(f"Open-set F1 Score        : {open_f1:.4f}")

    # 混淆矩阵
    try:
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds, labels=list(range(10)) + [-1]))
    except:
        print("Warning: Unable to compute confusion matrix for open-set labels.")

if __name__ == "__main__":
    config = parse_args()
    evaluate_open_contrastive(config)
