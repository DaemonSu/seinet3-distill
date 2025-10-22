import os
import torch
import torch.nn.functional as F
import pickle
import math
from torch.utils.data import DataLoader

from config import parse_args
from model_mid import IncrementalContrastiveModel
from model_mix import FeatureExtractor, ClassifierHead
from loss import SupConLoss
from dataset import KnownDataset
from test_mixed_enhanced import evaluate_incremental
from util.utils import set_seed, adjust_lr, adjust_lr_add

# ============== 基础工具 ==============

def find_latest_file(dir_path, prefix):
    files = [f for f in os.listdir(dir_path) if f.startswith(prefix)]
    if not files:
        return None

    def extract_step(filename):
        # 去掉前缀与后缀，兼容 .pth / .pkl
        base = filename.replace(prefix, "")
        base = base.replace(".pth", "").replace(".pkl", "")
        try:
            return int(base)
        except ValueError:
            return -1  # 无法解析时忽略
    files = [f for f in files if extract_step(f) >= 0]
    if not files:
        return None

    files.sort(key=extract_step)
    return os.path.join(dir_path, files[-1])


def sample_cache_balanced(feature_cache, sample_per_class, device):
    feats, labels = [], []
    for cls, flist in feature_cache.items():
        if len(flist) == 0:
            continue
        farr = torch.tensor(flist, dtype=torch.float32, device=device)
        if farr.size(0) >= sample_per_class:
            idx = torch.randperm(farr.size(0), device=device)[:sample_per_class]
            sel = farr[idx]
        else:
            idx = torch.randint(0, farr.size(0), (sample_per_class,), device=device)
            sel = farr[idx]
        feats.append(sel)
        labels.extend([cls] * sel.size(0))
    if len(feats) == 0:
        return torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)
    feats = torch.cat(feats, dim=0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return feats, labels


# ============== 连续增量训练核心 ==============

def train_incremental_multi(config):
    device = config.device
    os.makedirs(config.save_dir, exist_ok=True)
    torch.manual_seed(config.seed)

    # ---------- 检查上一阶段模型 ----------
    encoder_path = find_latest_file(config.save_dir, "encoder_step")
    classifier_path = find_latest_file(config.save_dir, "classifier_step")

    if encoder_path and classifier_path:
        print(f"加载上一阶段模型: {encoder_path}, {classifier_path}")
        encoder = torch.load(encoder_path, map_location=device).to(device)
        classifier_old = torch.load(classifier_path, map_location=device).to(device)
    else:
        print("未检测到旧模型，使用初始训练阶段结果。")
        encoder = torch.load(os.path.join(config.save_dir, 'encoder.pth')).to(device)
        classifier_old = torch.load(os.path.join(config.save_dir, 'classifier.pth')).to(device)
    encoder.eval()
    classifier_old.eval()

    try:
        old_num_classes = classifier_old.classifier[-1].out_features
    except AttributeError:
        old_num_classes = classifier_old.fc.out_features  # 向后兼容旧结构

    # ---------- 加载缓存 ----------
    # ================= 读取缓存特征 =================
    cache_path = find_latest_file(config.save_dir, "feature_cache_step")
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            feature_cache = pickle.load(f)
    else:
        # 没有旧缓存则初始化为空
        feature_cache = {}

    # 兼容老结构：如果是list或dict of list
    # if isinstance(feature_cache, list):
    #     # 新逻辑的flatten结构
    #     feature_cache_dict = {}
    #     for feat, label in feature_cache:
    #         label = int(label)
    #         if label not in feature_cache_dict:
    #             feature_cache_dict[label] = []
    #         feature_cache_dict[label].append(feat)
    #     feature_cache = feature_cache_dict
    # elif not isinstance(feature_cache, dict):
    #     raise ValueError("feature_cache format not recognized.")

    if isinstance(feature_cache, dict) and "features" in feature_cache:
        feature_cache = feature_cache["features"]
        cache_version = feature_cache.get("version", 0)
    else:
        feature_cache = feature_cache
        cache_version = 0

    # ---------- 加载新类数据 ----------
    new_trainset = KnownDataset(config.train_data_add)
    new_loader = DataLoader(new_trainset, batch_size=config.incr_batch_size, shuffle=True)

    # ---------- 计算类别扩展 ----------
    all_device_ids = {int(k) for k in feature_cache.keys()}
    for _, y in new_trainset:
        all_device_ids.add(int(y))
    n_classes_total = max(all_device_ids) + 1


    # all_device_ids = {int(k) for k in feature_cache.keys()}
    # for _, y in new_trainset:
    #     all_device_ids.add(int(y))
    # n_classes_total = max(all_device_ids) + 1


    # ---------- 构建新分类器与模型 ----------
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

    ce_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.05)
    optimizer = torch.optim.Adam(
        list(model.contrastive_layer.parameters()) + list(model.classifier.parameters()),
        lr=config.incre_lr
    )

    # ---------- 增量训练 ----------
    for epoch in range(config.epochs2):
        total_loss, total_acc = 0, 0

        for x_new, y_new in new_loader:
            x_new, y_new = x_new.to(device), y_new.to(device)

            # 采样旧类缓存
            ratio = config.ratio
            if old_num_classes > 0:
                N_old = x_new.size(0) * ratio
                per_class = max(1, math.ceil(N_old / old_num_classes))
                feat_cache_encoder, y_cache_idx = sample_cache_balanced(feature_cache, per_class, device)
            else:
                feat_cache_encoder = torch.empty(0, config.embedding_dim, device=device)
                y_cache_idx = torch.empty(0, dtype=torch.long, device=device)

            with torch.no_grad():
                feat_new_enc = encoder(x_new)

            feat_all_enc = torch.cat([feat_new_enc, feat_cache_encoder], dim=0)
            feat_all_proj = model.contrastive_layer(feat_all_enc)
            feat_new_proj = feat_all_proj[:x_new.size(0)]
            feat_cache_proj = feat_all_proj[x_new.size(0):]

            feat_distill_loss = 0.0
            if feat_cache_encoder.numel() > 0:
                feat_distill_loss = F.mse_loss(feat_cache_proj, feat_cache_encoder.detach())

            logits_all = classifier(feat_all_proj)
            labels_all = torch.cat([y_new, y_cache_idx], dim=0)

            is_new = (labels_all >= old_num_classes)
            ce_per_sample = F.cross_entropy(logits_all, labels_all, reduction='none')
            weight = torch.where(is_new, torch.tensor(1.8, device=device), torch.tensor(1.0, device=device))
            ce_loss = (ce_per_sample * weight).mean()

            con_loss = contrastive_loss_fn(feat_all_proj, labels_all)
            loss = 2.0 * ce_loss + 1.0 * con_loss + 0.5 * feat_distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits_all.argmax(dim=1)
            acc = (preds == labels_all).float().mean().item() * 100
            total_loss += loss.item()
            total_acc += acc

        print(f"[Step {step} | Epoch {epoch+1}] Loss={total_loss/len(new_loader):.4f} | Acc={total_acc/len(new_loader):.2f}")
        adjust_lr_add(optimizer, epoch, config)

    # ---------- 更新缓存 ----------
    model.classifier.eval()
    model.contrastive_layer.eval()

    for x, y in DataLoader(new_trainset, batch_size=config.open_batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            feat_enc = encoder(x)
            feat_contrast = model.contrastive_layer(feat_enc)
            logits = model.classifier(feat_contrast)
            preds = logits.argmax(dim=1)
            mask = preds == y
            for cls in y.unique():
                m = (y == cls) & mask
                if m.any():
                    feats = feat_enc[m].detach().cpu().tolist()
                    feature_cache.setdefault(int(cls.item()), []).extend(feats)
                    if len(feature_cache[int(cls.item())]) > config.max_feature_per_class:
                        feature_cache[int(cls.item())] = feature_cache[int(cls.item())][-config.max_feature_per_class:]

    # ---------- 保存 ----------
    cache_obj = {"version": step, "features": feature_cache}
    with open(os.path.join(config.save_dir, f"feature_cache_step{step}.pkl"), "wb") as f:
        pickle.dump(cache_obj, f)
    torch.save(model.classifier, os.path.join(config.save_dir, f"classifier_step{step}.pth"))
    torch.save(model.encoder, os.path.join(config.save_dir, f"encoder_step{step}.pth"))
    torch.save(model.contrastive_layer, os.path.join(config.save_dir, f"contrastive_step{step}.pth"))

    print(f"✅ 增量 Step {step} 训练完成，模型与缓存已保存。")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

    config = parse_args()
    num_increments = 10
    initial_classes = 50
    increment_size = 5

    for step in range(1, num_increments + 1):
        config.train_data_add = f"G:/seidataforCIL-init/train-add{step}"
        train_incremental_multi(config)

        test_data = f"G:/seidataforCIL-init/test-add{step}"
        num_old_classes = initial_classes + (step - 1) * increment_size
        evaluate_incremental(
            model_path=f"model/classifier_step{step}.pth",
            encoder_path=f"model/encoder_step{step}.pth",
            projector_path=f"model/contrastive_step{step}.pth",  # ✅ 新增
            test_data_path=test_data,
            num_old_classes=num_old_classes,
            device=config.device
        )

