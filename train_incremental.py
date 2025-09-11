import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict

from config import parse_args
from util.utils import set_seed, topaccuracy, adjust_lr, save_object
from dataset import KnownDataset, UnknownDataset
from model_mix import FeatureExtractor, ClassifierHead

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_new, logits_old, mask_old):
        """
        logits_new: [B, num_new_classes] 新分类器输出
        logits_old: [B, num_old_classes] 旧分类器输出
        mask_old: [B] bool mask，标记哪些样本是旧类
        """
        if mask_old.sum() == 0:
            return torch.tensor(0.0, device=logits_new.device)
        logp = F.log_softmax(logits_new[:, :logits_old.size(1)] / self.temperature, dim=1)
        p = F.softmax(logits_old / self.temperature, dim=1)
        loss = F.kl_div(logp[mask_old], p[mask_old], reduction='batchmean') * (self.temperature ** 2)
        return loss

def train_incremental(config):
    set_seed(config.seed)

    # ============ 新数据加载 ============
    new_trainset = KnownDataset(config.train_data_new)
    new_loader = DataLoader(new_trainset, config.incr_batch_size, shuffle=True)

    # ============ 加载旧类缓存 ============
    with open(os.path.join(config.save_dir, 'feature_cache.pkl'), 'rb') as f:
        feature_cache = pickle.load(f)

    cache_data, cache_labels = [], []
    for cls, feats in feature_cache.items():
        cache_data.extend(feats)
        cache_labels.extend([cls] * len(feats))
    cache_data = torch.tensor(cache_data, dtype=torch.float32).to(config.device)
    cache_labels = torch.tensor(cache_labels, dtype=torch.long).to(config.device)

    # ============ 模型 ============
    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(config.device)
    encoder.eval()  # 冻结特征提取器

    # ====== 构建 label↔index 映射 ======
    old_classes = list(range(config.old_num_classes))
    new_classes = sorted({y for _, y in new_trainset})

    label2idx = {}
    idx2label = {}

    # 旧类直接映射
    for cls in old_classes:
        label2idx[cls] = cls
        idx2label[cls] = cls

    start_idx = config.old_num_classes
    for i, cls in enumerate(new_classes):
        if cls not in label2idx:  # 防止重复
            label2idx[cls] = start_idx + i
            idx2label[start_idx + i] = cls

    n_classes_total = max(label2idx.values()) + 1
    classifier = ClassifierHead(1024, n_classes_total).to(config.device)

    # 旧分类器
    classifier_old = torch.load(os.path.join(config.save_dir, 'classifier.pth')).to(config.device)
    classifier_old.eval()

    # ============ 损失 & 优化器 ============
    ce_loss_fn = nn.CrossEntropyLoss()
    distill_loss_fn = DistillationLoss(temperature=config.distill_temperature)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)

    # ============ 训练 ============
    for epoch in range(config.epochs):
        classifier.train()
        total_loss, total_acc = 0, 0

        for x_new, y_new in new_loader:
            x_new = x_new.to(config.device)
            # 标签映射到 classifier index
            y_new_idx = torch.tensor([label2idx[int(l)] for l in y_new],
                                     dtype=torch.long, device=config.device)

            # 随机采样旧类缓存
            if len(cache_data) > 0:
                idx = torch.randperm(cache_data.size(0))[:x_new.size(0)*3]
                x_cache = cache_data[idx]
                y_cache_idx = torch.tensor([label2idx[int(l)] for l in cache_labels[idx].cpu()],
                                           dtype=torch.long, device=config.device)
            else:
                x_cache = torch.empty(0, 1024).to(config.device)
                y_cache_idx = torch.empty(0, dtype=torch.long).to(config.device)

            # 特征
            with torch.no_grad():
                feat_new = encoder(x_new)
            feat_cache = x_cache  # 缓存已是特征

            feat_all = torch.cat([feat_new, feat_cache], dim=0)
            labels_all = torch.cat([y_new_idx, y_cache_idx], dim=0)

            logits_all = classifier(feat_all)

            # CE loss
            ce_loss = ce_loss_fn(logits_all, labels_all)

            # 知识蒸馏（旧类）
            mask_old = labels_all < config.old_num_classes
            if mask_old.any():
                with torch.no_grad():
                    logits_old_all = classifier_old(feat_all)
                kd_loss = distill_loss_fn(
                    logits_all[:, :config.old_num_classes],
                    logits_old_all[:, :config.old_num_classes],
                    mask_old
                )
            else:
                kd_loss = 0.0

            loss = ce_loss + config.alpha_kd * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits_all.argmax(dim=1)
            acc = (preds == labels_all).float().mean().item() * 100
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(new_loader):.4f} | Acc: {total_acc/len(new_loader):.2f}")
        adjust_lr(optimizer, epoch, config)

    # ============ 更新缓存 ============
    classifier.eval()
    all_data = DataLoader(new_trainset, batch_size=config.batch_size, shuffle=False)
    for x, y in all_data:
        x = x.to(config.device)
        # 注意映射真实标签→index
        y_idx = torch.tensor([label2idx[int(l)] for l in y], dtype=torch.long, device=config.device)
        with torch.no_grad():
            feat = encoder(x)
            logits = classifier(feat)
            preds = logits.argmax(dim=1)
            correct_mask = preds == y_idx

            for idx_class in y_idx.unique():
                idx_class = int(idx_class.item())
                real_class = idx2label[idx_class]  # 映射回真实标签
                if real_class in feature_cache:  # 旧类不更新
                    continue
                mask_cls = (y_idx == idx_class) & correct_mask
                if mask_cls.any():
                    if real_class not in feature_cache:
                        feature_cache[real_class] = []
                    feature_cache[real_class].extend(feat[mask_cls].cpu().tolist())
                    if len(feature_cache[real_class]) > config.max_feature_per_class:
                        feature_cache[real_class] = feature_cache[real_class][-config.max_feature_per_class:]

    save_object(feature_cache, os.path.join(config.save_dir, 'feature_cache.pkl'))
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save(classifier, os.path.join(config.save_dir, 'classifier_incremental.pth'))
    # 同时保存映射
    save_object({'label2idx': label2idx, 'idx2label': idx2label},
                os.path.join(config.save_dir, 'class_mapping.pkl'))
    print("增量训练完成，模型、映射和特征缓存已保存。")



if __name__ == "__main__":
    config = parse_args()
    train_incremental(config)
