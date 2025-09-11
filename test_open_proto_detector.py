import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import parse_args
from dataset import MixedDataset
from model_open import FeatureExtractor, ClassifierHead
from train_OpenProtoDetector import OpenProtoDetector
from util.utils import load_object


def test_with_open_proto_detector(config, encoder, classifier, open_detector, prototypes, test_loader):
    torch.set_printoptions(threshold=float('inf'))

    encoder.eval()
    classifier.eval()
    open_detector.eval()

    all_preds = []
    all_labels = []

    closed_correct = 0
    closed_total = 0
    open_total = 0
    open_detected = 0
    open_cls_total = 0
    open_cls_correct = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="🔍 Testing"):
            x = x.to(config.device)
            y = y.to(config.device)

            # ========== 1. 特征提取与预测 ==========
            feat = encoder(x)
            logits = classifier(feat)
            probs = F.softmax(logits, dim=1)
            max_probs, pred_classes = probs.max(dim=1)

            # ========== 2. 获取对应预测原型（用于 OpenDetector） ==========
            proto = prototypes.get(pred_classes)  # shape: [B, D]

            # ========== 3. OpenDetector 判断是否为 Known 类 ==========
            scores = open_detector(max_probs.unsqueeze(1), feat, proto)
            result=torch.sigmoid(scores)
            is_known = (result > 0.5).int()
            print(result)
            # ========== 4. 遍历每个样本进行评估 ==========
            for i in range(x.size(0)):
                true_label = y[i].item()
                pred_class = pred_classes[i].item()
                known_flag = is_known[i].item()
                pred_is_open = 1 - known_flag  # 1 表示 open 类

                if true_label != -1:
                    # 已知类（Closed-set）
                    closed_total += 1
                    if not pred_is_open:
                        if pred_class == true_label:
                            closed_correct += 1
                        all_preds.append(pred_class)
                    else:
                        all_preds.append(-1)
                    all_labels.append(true_label)
                else:
                    # 开集（Open-set）
                    open_total += 1
                    if pred_is_open:
                        open_detected += 1
                        all_preds.append(-1)
                        open_cls_total += 1
                        if pred_class == true_label:
                            open_cls_correct += 1
                    else:
                        all_preds.append(pred_class)
                    all_labels.append(-1)

    # ========== 5. 统计指标 ==========
    closed_acc = closed_correct / closed_total if closed_total > 0 else 0
    open_recognition_rate = open_detected / open_total if open_total > 0 else 0
    overall_acc = accuracy_score(all_labels, all_preds)
    open_cls_acc = open_cls_correct / open_cls_total if open_cls_total > 0 else 0

    print("\n📊 Evaluation Metrics:")
    print(f"Closed-set Accuracy        : {closed_acc:.4f}")
    print(f"Open-set Recognition Rate  : {open_recognition_rate:.4f}")
    print(f"Overall Accuracy           : {overall_acc:.4f}")
    print(f"Open-set Classification Acc: {open_cls_acc:.4f}")

    try:
        print("\n📉 Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds, labels=list(range(10)) + [-1]))
    except Exception as e:
        print("⚠️ Warning: Unable to compute confusion matrix.", str(e))

    return all_preds, all_labels


if __name__ == "__main__":
    config = parse_args()

    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    # 加载模型
    ckpt = torch.load(os.path.join(config.save_dir, 'model_opencon2.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    # 加载原型与 open detector
    prototype = load_object(os.path.join(config.save_dir, 'prototype2.pkl'))
    open_proto_detector = OpenProtoDetector(feature_dim=config.embedding_dim).to(config.device)
    open_proto_detector.load_state_dict(torch.load(os.path.join(config.save_dir, 'open_proto_detector.pth'), map_location=config.device))

    # 数据加载
    mixed_testset = MixedDataset(config.test_mixed)
    test_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=True)

    # 运行测试
    test_with_open_proto_detector(config, encoder, classifier, open_proto_detector, prototype, test_loader)

