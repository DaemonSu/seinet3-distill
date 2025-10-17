import math

from config import parse_args
from dataset import KnownDataset
from loss import SupConLoss_DynamicMargin, SupConLoss
from model_mid import IncrementalContrastiveModel
from model_mix import FeatureExtractor, ClassifierHead
from util.utils import set_seed, adjust_lr, adjust_lr_add
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam


def load_maybe_model(path, device):
    """
    尝试加载：若是 state_dict 则返回 None + state_dict，
    若是整个 model 对象则返回 model.
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'state_dict' in obj and 'arch' in obj:
        # 如果是自定义保存格式
        return obj  # caller handles
    if isinstance(obj, dict) and any(k.startswith('layer') or k.startswith('fc') for k in obj.keys()):
        # 很可能是 state_dict
        return obj
    # 其他情况是 model object
    return obj

def to_tensor_cache(feat_list, device):
    if len(feat_list) == 0:
        return torch.empty(0)
    t = torch.tensor(feat_list, dtype=torch.float32, device=device)
    return t

def sample_cache_balanced(feature_cache, sample_per_class, device):
    """
    从 feature_cache（dict: label -> list(features)）中采样，
    尽量做到每个 class sample_per_class 个，若不足则随机重复采样。
    返回: feats_tensor (N, D), labels_tensor (N,)
    """
    feats = []
    labels = []
    for cls, flist in feature_cache.items():
        if len(flist) == 0:
            continue
        farr = torch.tensor(flist, dtype=torch.float32, device=device)
        if farr.size(0) >= sample_per_class:
            idx = torch.randperm(farr.size(0), device=device)[:sample_per_class]
            sel = farr[idx]
        else:
            # repeat-random
            idx = torch.randint(0, farr.size(0), (sample_per_class,), device=device)
            sel = farr[idx]
        feats.append(sel)
        labels.extend([cls] * sel.size(0))
    if len(feats) == 0:
        return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    feats = torch.cat(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return feats, labels




def distillation_kl_loss(student_logits, teacher_logits, num_old_classes, T=2.0):
    """
    增量训练蒸馏损失（KL Divergence）：
    - 只对旧闭集类别进行蒸馏
    - student_logits: [B, total_classes] (含旧 + 新设备)
    - teacher_logits: [B, num_old_classes] (只含旧闭集)
    - num_old_classes: 旧闭集类别数量
    - T: 蒸馏温度
    """
    if num_old_classes == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # 取 student 对应旧类的列
    student_old = student_logits[:, :num_old_classes]  # [B, old_num]
    teacher_old = teacher_logits[:, :num_old_classes]  # [B, old_num]

    # KL divergence
    log_p_s = F.log_softmax(student_old / T, dim=1)
    p_t = F.softmax(teacher_old / T, dim=1)

    kd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
    return kd_loss


def train_incremental(config):
    """
    增量训练：
    - logits 列直接对应设备编号
    - 每次增量只扩展新类
    - feature_cache 按设备编号更新
    """
    device = config.device
    torch.manual_seed(config.seed)

    # ================= 新数据加载 =================
    new_trainset = KnownDataset(config.train_data_add1)
    new_loader = DataLoader(new_trainset, batch_size=config.incr_batch_size, shuffle=True)

    # ================= 读取缓存特征 =================
    cache_path = os.path.join(config.save_dir, 'feature_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            feature_cache = pickle.load(f)
    else:
        feature_cache = {}

    cache_feats, cache_labels = [], []
    for cls, feats in feature_cache.items():
        cache_feats.extend(feats)
        cache_labels.extend([cls] * len(feats))
    cache_feats = torch.tensor(cache_feats, dtype=torch.float32, device=device) if len(cache_feats) > 0 else torch.empty(0, config.embedding_dim, device=device)
    cache_labels = torch.tensor(cache_labels, dtype=torch.long, device=device) if len(cache_labels) > 0 else torch.empty(0, dtype=torch.long, device=device)

    # ================= 加载 encoder & 旧分类器 =================
    encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(device)
    encoder.eval()

    classifier_old_path = os.path.join(config.save_dir, 'classifier.pth')
    classifier_old = torch.load(classifier_old_path).to(device) if os.path.exists(classifier_old_path) else None
    if classifier_old: classifier_old.eval()

    # ================= 构建分类器 =================
    # 计算总类数 = 最大设备编号 +1
    all_device_ids = set(cache_labels.cpu().tolist())
    all_device_ids.update([int(y) for _, y in new_trainset])
    n_classes_total = max(all_device_ids) + 1

    classifier = ClassifierHead(config.embedding_dim, n_classes_total).to(device)
    model = IncrementalContrastiveModel(
        encoder, classifier,
        in_dim=config.embedding_dim,
        hidden_dim=512,
        feat_dim=config.embedding_dim
    ).to(device)

    model.encoder.eval()
    model.classifier.train()
    model.contrastive_layer.train()

    ce_loss_fn = nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.05)
    optimizer = torch.optim.Adam(
        list(model.contrastive_layer.parameters()) + list(model.classifier.parameters()),
        lr=config.incre_lr
    )

    # ================= 训练 =================
    for epoch in range(config.epochs2):
        total_loss, total_acc = 0, 0
        for x_new, y_new in new_loader:
            x_new = x_new.to(device)
            y_new_idx = y_new.to(device)  # logits 列直接对应设备编号

            # ------- 采样旧类缓存 --------
            ratio = 0.6 # e.g., 5 或 8，可放在 config
            num_old_classes = config.old_num_classes
            if num_old_classes > 0:
                N_old = x_new.size(0) * ratio
                per_class = max(1, math.ceil(N_old / num_old_classes))
                feat_cache_encoder, y_cache_idx = sample_cache_balanced(
                    feature_cache, sample_per_class=per_class, device=device
                )
            else:
                feat_cache_encoder = torch.empty(0, config.embedding_dim, device=device)
                y_cache_idx = torch.empty(0, dtype=torch.long, device=device)

            # ------- Teacher 流：encoder 输出 → old_classifier（不经过 contrastive）--------
            with torch.no_grad():
                feat_new_enc = encoder(x_new)  # [B_new, D]
                teacher_logits_new = (
                    classifier_old(feat_new_enc) if classifier_old is not None else None
                )

                if feat_cache_encoder.numel() > 0 and classifier_old is not None:
                    teacher_logits_old = classifier_old(feat_cache_encoder)  # 旧缓存
                    teacher_logits = torch.cat([teacher_logits_new, teacher_logits_old], dim=0)
                else:
                    teacher_logits = teacher_logits_new

            # ------- Student 流：encoder 输出 → contrastive → new_classifier --------
            feat_all_enc = torch.cat([feat_new_enc, feat_cache_encoder], dim=0)  # [B_all, D]
            feat_all_proj = model.contrastive_layer(feat_all_enc)  # projector
            # 拆分新旧特征的投影结果
            feat_new_proj = feat_all_proj[:x_new.size(0)]  # 新特征：需要整形，不做蒸馏
            feat_cache_proj = feat_all_proj[x_new.size(0):]  # 旧特征：需要保留原样，计算蒸馏损失

            # 特征蒸馏损失：仅旧缓存特征，且需detach原始Encoder特征（避免梯度回传影响冻结的Encoder）
            feat_distill_loss = 0.0
            if feat_cache_encoder.numel() > 0:
                # 旧缓存的原始Encoder特征（detach，冻结Encoder的保障）
                feat_cache_enc_detach = feat_cache_encoder.detach()
                # MSE损失：投影后的旧特征 ≈ 原始旧特征
                feat_distill_loss = F.mse_loss(feat_cache_proj, feat_cache_enc_detach)
                # 建议权重：0.5~2.0（根据实验调整，避免过大压制新类学习）
                # feat_distill_loss *= 1.0

            logits_all = classifier(feat_all_proj)  # 新分类器输出
            labels_all = torch.cat([y_new_idx, y_cache_idx], dim=0)

            # ------- 损失计算 --------
            # CE loss
            # ce_loss = ce_loss_fn(logits_all, labels_all)

            is_new = (labels_all >= num_old_classes)  # 根据编号判定新类
            ce_per_sample = F.cross_entropy(logits_all, labels_all, reduction='none')
            weight = torch.where(is_new, torch.tensor(1.8, device=device), torch.tensor(1.0, device=device))
            ce_loss = (ce_per_sample * weight).mean()




            # Contrastive loss on projected features
            con_loss = contrastive_loss_fn(feat_all_proj, labels_all)

            loss =  2.2*ce_loss +  1.0 * con_loss + 0.5*feat_distill_loss

            if epoch % 8 == 0:
                print(f"[Epoch {epoch + 1}] ce: {ce_loss:.4f} | con: {con_loss:.4f}| feat_distill_loss: {feat_distill_loss:.4f}")





            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits_all.argmax(dim=1)
            acc = (preds == labels_all).float().mean().item() * 100
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(new_loader):.4f} | Acc: {total_acc / len(new_loader):.2f}")
        adjust_lr_add(optimizer, epoch, config)

    # ================= 更新缓存 =================
    model.classifier.eval()
    model.contrastive_layer.eval()

    for x, y in DataLoader(new_trainset, batch_size=config.open_batch_size, shuffle=False):
        x = x.to(device)
        y_idx = y.to(device)
        with torch.no_grad():
            feat_encoder = encoder(x)
            feat_contrast = model.contrastive_layer(feat_encoder)
            logits = model.classifier(feat_contrast)
            preds = logits.argmax(dim=1)
            correct_mask = preds == y_idx

            for cls in y_idx.unique():
                mask_cls = (y_idx == cls) & correct_mask
                if mask_cls.any():
                    feats_to_save = feat_encoder[mask_cls].detach().cpu().tolist()
                    if int(cls.item()) not in feature_cache:
                        feature_cache[int(cls.item())] = []
                    feature_cache[int(cls.item())].extend(feats_to_save)
                    if len(feature_cache[int(cls.item())]) > config.max_feature_per_class:
                        feature_cache[int(cls.item())] = feature_cache[int(cls.item())][-config.max_feature_per_class:]

    # ================= 保存 =================
    os.makedirs(config.save_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(feature_cache, f)
    torch.save(model.classifier, os.path.join(config.save_dir, 'classifier_incremental.pth'))
    torch.save(model.contrastive_layer, os.path.join(config.save_dir, 'mid_incremental.pth'))

    print("✅ 增量训练完成，模型和特征缓存已保存。")


if __name__ == "__main__":
    config = parse_args()
    train_incremental(config)
