import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KnownDataset, UnknownDataset, MixedDataset
from model_mix import FeatureExtractor, ClassifierHead
from loss import SupConLoss_DynamicMargin
from util.utils import set_seed, adjust_lr, topaccuracy, save_object
from config import parse_args
import itertools


def train_open_contrastive(config, criterion_params):
    set_seed(config.seed)

    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, config.batch_size, shuffle=True)

    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, config.num_classes).to(config.device)

    supcon_loss_fn = SupConLoss_DynamicMargin(
        temperature=criterion_params["temperature"],
        base_margin=criterion_params["base_margin"],
        beta=criterion_params["beta"]
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr)

    for epoch in range(config.epochs):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0, 0

        known_iter = iter(known_loader)
        unknown_iter = iter(unknown_loader)

        for i in range(len(known_loader)):
            try:
                x_known, y_known = next(known_iter)
            except StopIteration:
                known_iter = iter(known_loader)
                x_known, y_known = next(known_iter)

            try:
                x_unknown, _ = next(unknown_iter)
            except StopIteration:
                unknown_iter = iter(unknown_loader)
                x_unknown, _ = next(unknown_iter)

            x_known, y_known = x_known.to(config.device), y_known.to(config.device)
            x_unknown = x_unknown.to(config.device)

            # 提取特征
            feat_known = encoder(x_known)
            feat_unknown = encoder(x_unknown)

            # 闭集分类损失
            logits = classifier(feat_known)
            ce_loss = ce_loss_fn(logits, y_known)

            logits_unknown = classifier(feat_unknown)
            probs_unknown = F.softmax(logits_unknown, dim=1)
            max_probs, _ = probs_unknown.max(dim=1)

            # 惩罚：鼓励开集样本的最大概率越低越好
            # 比如超过阈值的部分才被惩罚（soft margin）
            penalty = torch.clamp(max_probs - config.open_threshold_train, min=0)
            penalty_loss = penalty.mean()
            uniform = torch.full_like(probs_unknown, 1.0 / probs_unknown.size(1))
            kl_loss = F.kl_div(probs_unknown.log(), uniform, reduction='batchmean')

            # 对比损失：已知类之间 + 已知类 vs 未知类
            feat_all = torch.cat([feat_known, feat_unknown], dim=0)
            labels_all = torch.cat([y_known, torch.full((x_unknown.size(0),), -1, device=config.device)], dim=0)
            con_loss = supcon_loss_fn(feat_all, labels_all)

            loss = con_loss + ce_loss + penalty_loss + 2 * kl_loss

            # print(
            #     f"[Epoch {epoch + 1}] con_loss: {con_loss:.4f} | ce_loss: {ce_loss:.2f}| penalty_loss: {penalty_loss:.2f}| kl_loss: {kl_loss:.2f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = topaccuracy(logits, y_known)
            total_loss += loss.item()
            total_acc += acc
        if epoch % 8== 0:
            print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(known_loader):.4f} | Acc: {total_acc / len(known_loader):.2f}")

        adjust_lr(optimizer, epoch, config)
    temperature = criterion_params["temperature"]
    base_margin = criterion_params["base_margin"]
    beta = criterion_params["beta"]

    # 拼接成字符串，例如：temp0.09_margin0.5_beta1.2
    folder_name = f"{config.save_dir}{temperature}_{base_margin}_{beta}"

    # 创建文件夹
    os.makedirs(folder_name,exist_ok=True)

    torch.save({'encoder': encoder.state_dict()},
               os.path.join(folder_name, 'encoder.pth'))

    torch.save({'classifier': classifier.state_dict()},
               os.path.join(folder_name, 'classifier.pth'))
    return encoder, classifier




import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

def collect_logits_and_labels(encoder, classifier, val_loader, config):
    """一次性计算 logits 和 labels"""
    encoder.eval()
    classifier.eval()

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Collecting logits"):
            x = x.to(config.device)
            feat = encoder(x)
            logits = classifier(feat)

            all_logits.append(logits.cpu().numpy())
            all_labels.extend(y.cpu().tolist())

    return np.vstack(all_logits), np.array(all_labels)


def evaluate_with_threshold(all_logits, all_labels, open_threshold):
    """使用已有 logits 和 labels 在不同阈值下做评估"""
    probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    pred_labels = np.where(max_probs < open_threshold, -1, preds)

    closed_mask = all_labels != -1
    open_mask = all_labels == -1

    closed_acc = accuracy_score(all_labels[closed_mask], pred_labels[closed_mask]) if closed_mask.sum() > 0 else float('nan')
    open_recognition_rate = np.mean(pred_labels[open_mask] == -1) if open_mask.sum() > 0 else float('nan')
    overall_acc = np.mean(pred_labels == all_labels)
    f1_open = f1_score(all_labels == -1, pred_labels == -1)

    print(f"[Threshold={open_threshold:.3f}] Closed Acc={closed_acc:.4f}  "
          f"Open Recog={open_recognition_rate:.4f}  Overall Acc={overall_acc:.4f}  F1_open={f1_open:.4f}")

    return overall_acc


import itertools

def run_grid_search(config):
    param_grid = {
        # "temperature": [ 0.09, 0.11, 0.13,0.15,0.17],
        # "base_margin": [ 0.1, 0.2, 0.3, 0.4, 0.5],
        # "beta": [  0.2,  0.4,  0.6, 0.8 , 1.0],

        "temperature": [0.13],
        "base_margin": [0.1, 0.2, 0.3, 0.4, 0.5],
        "beta": [0,0.1],
    }

    best_score = -1
    best_params = None
    best_threshold = None
    results = []

    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        print(f"\n[GridSearch] Params: {params}")

        # 训练模型
        encoder, classifier = train_open_contrastive(config, params)

        # 验证集
        val_dataset = MixedDataset(config.val_data)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # ===== 用法 =====
        logits, labels = collect_logits_and_labels(encoder, classifier, val_loader, config)

        for t in [0.60,0.70, 0.80, 0.90, 0.95, 0.97, 0.99]:
            score = evaluate_with_threshold(logits, labels, t)
            if score > best_score:
                best_score = score
                best_threshold = t
                best_params=params
                results.append({**params, "threshold": t, "overall_acc": score})

        print(f"Best threshold: {best_threshold}, score: {best_score:.4f}")



    print("\n========== Grid Search Complete ==========")
    print(f"Best Params: {best_params}")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Score: {best_score:.4f}")

    return best_params, best_threshold, best_score, results




if __name__ == "__main__":
    config = parse_args()
    run_grid_search(config)
