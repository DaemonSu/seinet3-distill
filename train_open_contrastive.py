import torch
import torch.nn as nn
from dataset import KnownDataset, UnknownDataset
from model_mix import FeatureExtractor, ClassifierHead
from loss import SupConLoss_OpenSet, SupConLoss_DynamicMargin,SupCon_OpenSet_Mixed
import os
from config import parse_args

from torch.utils.data import DataLoader

from util.utils import adjust_lr
import torch.nn.functional as F


def train_open_contrastive(config):
    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, batch_size=config.batch_size, shuffle=True)
    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, batch_size=config.open_batch_size, shuffle=True)

    encoder = FeatureExtractor(in_channels=3, feature_dim=config.embedding_dim).to(config.device)
    classifier = ClassifierHead(config.embedding_dim, config.num_classes).to(config.device)

    ce_loss_fn = nn.CrossEntropyLoss()
    # supcon_loss_fn = SupConLoss_OpenSet()
    supcon_loss_fn =SupConLoss_DynamicMargin()
    # supcon_loss_fn =SupCon_OpenSet_Mixed()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr, weight_decay=1e-4)

    feature_cache = {cls: [] for cls in range(config.old_num_classes)}



    for epoch in range(config.epochs):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0, 0

        known_iter = iter(known_loader)
        unknown_iter = iter(unknown_loader)

        for i in range(len(known_loader)):
            x_known, y_known = next(known_iter, (None, None))
            if x_known is None:
                known_iter = iter(known_loader)
                x_known, y_known = next(known_iter)

            x_unknown, _ = next(unknown_iter, (None, None))
            if x_unknown is None:
                unknown_iter = iter(unknown_loader)
                x_unknown, _ = next(unknown_iter)

            x_known, y_known = x_known.to(config.device), y_known.to(config.device)
            x_unknown = x_unknown.to(config.device)

            feat_known = encoder(x_known)
            with torch.no_grad():  # 避免开集污染 BN
                feat_unknown = encoder(x_unknown)

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

            loss = 2.4 * con_loss + 2.0 * ce_loss + 0.05* kl_loss

            if epoch%10 == 0:
                print(f"[Epoch {epoch + 1}] ce: {ce_loss:.4f} | con: {con_loss:.4f} | kl: {kl_loss:.4f} | penalty: {penalty_loss:.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            acc = (preds == y_known).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(known_loader):.4f} | Acc: {total_acc/len(known_loader):.4f}")
        adjust_lr(optimizer, epoch, config)

        if epoch % 10 == 0:
            valset = KnownDataset(config.val_closed)
            val_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False)
            encoder.eval()
            classifier.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    logits = classifier(encoder(x))
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"[Epoch {epoch + 1}] Val Acc: {acc:.4f}")



    # ============ 模型保存 ============
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save(encoder, os.path.join(config.save_dir, 'encoder.pth'))
    torch.save(classifier, os.path.join(config.save_dir, 'classifier.pth'))
    # ============ 训练后构建特征缓存 ============
    build_feature_cache(encoder, classifier, config)

def build_feature_cache(encoder, classifier, config):
    """训练结束后重新提取各类别特征缓存"""
    print("\nBuilding feature cache...")

    encoder.eval()
    classifier.eval()

    dataset = KnownDataset(config.train_data_close)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    feature_cache = {cls: [] for cls in range(config.num_classes)}

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            feat = encoder(x)
            logits = classifier(feat)
            preds = logits.argmax(dim=1)
            correct_mask = preds.eq(y)

            for cls in y.unique():
                cls = int(cls.item())
                mask_cls = (y == cls) & correct_mask
                if mask_cls.any():
                    feats_cls = feat[mask_cls].cpu()
                    feature_cache[cls].extend(feats_cls.tolist())
                    # 限制缓存大小
                    if len(feature_cache[cls]) > config.max_feature_per_class:
                        feature_cache[cls] = feature_cache[cls][-config.max_feature_per_class:]

    # 保存缓存
    os.makedirs(config.save_dir, exist_ok=True)
    import pickle
    with open(os.path.join(config.save_dir, 'feature_cache.pkl'), 'wb') as f:
        pickle.dump(feature_cache, f)

    print("Feature cache built and saved successfully.")
if __name__ == "__main__":

    config = parse_args()
    train_open_contrastive(config)







