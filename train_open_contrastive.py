import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PrototypeMemory import PrototypeMemory
from dataset import KnownDataset, UnknownDataset
from model_mix import FeatureExtractor, ClassifierHead
from loss import SupConLoss_DynamicMargin
from util.utils import set_seed, adjust_lr, topaccuracy, save_object
import os
from config import parse_args
import torch.nn.functional as F

def train_open_contrastive(config):
    set_seed(config.seed)

    # ============ 数据加载 ============
    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, batch_size=config.batch_size, shuffle=True)

    # ============ 模型初始化 ============
    encoder = FeatureExtractor( in_channels=3, feature_dim=1024).to(config.device)
    classifier = ClassifierHead(1024, config.num_classes).to(config.device)

    # ============ 损失函数 ============
    supcon_loss_fn = SupConLoss_DynamicMargin()
    # supcon_loss_fn =    SupConLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # ============ 优化器 ============
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr)

    prototype = PrototypeMemory(config.num_classes, config.embedding_dim, config.device,config.prototype_momentum)

    feature_cache = {cls: [] for cls in range(config.old_num_classes)}
    max_feature_per_class = config.max_feature_per_class  # 每类最多保存多少特征
    # ============ 训练开始 ============
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


            loss = con_loss+ ce_loss +  penalty_loss + 2 * kl_loss
            print(
                f"[Epoch {epoch + 1}] con_loss: {con_loss:.4f} | ce_loss: {ce_loss:.2f}| penalty_loss: {penalty_loss:.2f}| kl_loss: {kl_loss:.2f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新原型
            prototype.update(feat_known.detach(), y_known.detach())
            # ================= 保存少量特征到缓存 =================
            # 预测并生成 mask
            preds = logits.argmax(dim=1)
            correct_mask = (preds == y_known)

            # 遍历每个类别
            for cls in y_known.unique():
                cls = int(cls.item())
                # 只取正确分类的样本
                mask_cls = (y_known == cls) & correct_mask
                if mask_cls.any():
                    feats_cls = feat_known[mask_cls].detach().cpu()
                    feature_cache[cls].extend(feats_cls.tolist())
                    # 保留最新 max_feature_per_class 个
                    if len(feature_cache[cls]) > max_feature_per_class:
                        feature_cache[cls] = feature_cache[cls][-max_feature_per_class:]

            acc = topaccuracy(logits, y_known)
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(known_loader):.4f} | Acc: {total_acc/len(known_loader):.2f}")

        adjust_lr(optimizer, epoch, config)

    # 模型保存
    os.makedirs(config.save_dir, exist_ok=True)

    torch.save(encoder, os.path.join(config.save_dir,  'encoder.pth'))

    # 保存整个分类器模型（结构+参数）
    torch.save(classifier, os.path.join(config.save_dir,'classifier.pth'))

    # 将原型以文件的形式保存到文件夹中
    save_object(prototype, 'model/prototype2.pkl')

    # ================= 保存特征缓存 =================
    import pickle
    with open(os.path.join(config.save_dir, 'feature_cache.pkl'), 'wb') as f:
        pickle.dump(feature_cache, f)

if __name__ == "__main__":

    config = parse_args()
    train_open_contrastive(config)







