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

    encoder = FeatureExtractor(in_channels=3, feature_dim=1024).to(config.device)
    classifier = ClassifierHead(1024, config.num_classes).to(config.device)

    ce_loss_fn = nn.CrossEntropyLoss()
    # supcon_loss_fn = SupConLoss_OpenSet()
    # supcon_loss_fn =SupConLoss_DynamicMargin()
    supcon_loss_fn =SupCon_OpenSet_Mixed()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr, weight_decay=1e-4)

    feature_cache = {cls: [] for cls in range(config.old_num_classes)}

    for epoch in range(config.epochs):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0, 0

        known_iter = iter(known_loader)

        for i in range(len(known_loader)):
            x_known, y_known = next(known_iter, (None, None))
            if x_known is None:
                known_iter = iter(known_loader)
                x_known, y_known = next(known_iter)


            x_known, y_known = x_known.to(config.device), y_known.to(config.device)

            feat_known = encoder(x_known)

            logits = classifier(feat_known)
            ce_loss = ce_loss_fn(logits, y_known)


            # 对比损失：已知类之间 + 已知类 vs 未知类
            con_loss,_ = supcon_loss_fn(feat_known, y_known)
            if epoch <= 150:
                loss = 0.01 * con_loss + ce_loss
            else:
                loss = 0.5*con_loss + ce_loss



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                    if len(feature_cache[cls]) > config.max_feature_per_class:
                        feature_cache[cls] = feature_cache[cls][-config.max_feature_per_class:]


            acc = (preds == y_known).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(known_loader):.4f} | Acc: {total_acc/len(known_loader):.4f}")
        adjust_lr(optimizer, epoch, config)

    # ============ 模型保存 ============
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save(encoder, os.path.join(config.save_dir, 'encoder.pth'))
    torch.save(classifier, os.path.join(config.save_dir, 'classifier.pth'))

    # ================= 保存特征缓存 =================
    import pickle
    with open(os.path.join(config.save_dir, 'feature_cache.pkl'), 'wb') as f:
        pickle.dump(feature_cache, f)
if __name__ == "__main__":

    config = parse_args()
    train_open_contrastive(config)







