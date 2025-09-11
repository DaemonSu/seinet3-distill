import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from dataset import KnownDataset, UnknownDataset
from model_open import FeatureExtractor, OpenDetector, ClassifierHead
from config import parse_args


def train_open_detector_from_pretrained(config):
    print("Loading pretrained encoder and classifier...")

    # ============ 加载模型 ============
    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    ckpt = torch.load(os.path.join(config.save_dir, 'model_opencon2.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    encoder.eval()
    classifier.eval()

    print("✅ Pretrained models loaded!")

    open_detector = OpenDetector(feature_dim=config.embedding_dim , num_classes=config.num_classes).to(config.device)
    optimizer = optim.Adam(open_detector.parameters(), lr=config.lr)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    known_trainset = KnownDataset(config.train_data)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.val2)
    unknown_loader = DataLoader(unknown_trainset, batch_size=config.batch_size, shuffle=True)

    known_iter = iter(known_loader)
    unknown_iter = iter(unknown_loader)

    for epoch in range(config.epochs):
        total_loss = 0
        for _ in tqdm(range(len(known_loader)), desc=f"Epoch {epoch + 1}"):
            try:
                x_known, _ = next(known_iter)
            except StopIteration:
                known_iter = iter(known_loader)
                x_known, _ = next(known_iter)

            try:
                x_unknown, _ = next(unknown_iter)
            except StopIteration:
                unknown_iter = iter(unknown_loader)
                x_unknown, _ = next(unknown_iter)

            x_known = x_known.to(config.device)
            x_unknown = x_unknown.to(config.device)

            with torch.no_grad():
                feat_known = encoder(x_known)
                logits_known = classifier(feat_known)
                feat_unknown = encoder(x_unknown)
                logits_unknown = classifier(feat_unknown)

            open_inputs_known  = torch.cat([feat_known, logits_known], dim=1)
            open_inputs_unknown  = torch.cat([feat_unknown, logits_unknown], dim=1)
            open_inputs = torch.cat([open_inputs_known, open_inputs_unknown], dim=0)  # [B_total, D+num_classes]

            # 标签仍然按顺序构建
            open_labels = torch.cat([
                torch.zeros(x_known.size(0), device=config.device),
                torch.ones(x_unknown.size(0), device=config.device)
            ], dim=0)

            # preds = open_detector(open_inputs_feat,open_inputs_logits).squeeze()
            preds = open_detector(open_inputs).squeeze()

            loss = bce_loss_fn(preds, open_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Phase 2] Epoch {epoch + 1}: Open Detector Loss={total_loss / len(known_loader):.4f}")

    torch.save(open_detector.state_dict(),os.path.join(config.save_dir, 'open_detector.pth'))
    print("✅ Open detector model saved!")


if __name__ == "__main__":
    config = parse_args()
    train_open_detector_from_pretrained(config)
