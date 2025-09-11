import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import parse_args
from dataset import KnownDataset, UnknownDataset
from model_open import FeatureExtractor, ClassifierHead
from util.utils import load_object


class OpenProtoDetector(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=128):
        super().__init__()
        self.feat_proj = nn.Linear(feature_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 修复关键点
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, max_prob, feat, proto):
        feat_proj = self.feat_proj(feat)
        proto_proj = self.feat_proj(proto)
        sim = F.cosine_similarity(feat_proj, proto_proj, dim=1).unsqueeze(1)
        x = torch.cat([max_prob, sim], dim=1)  # shape: [B, 2]
        return self.classifier(x).squeeze(1)



def train_open_proto_detector(config, encoder, classifier, prototypes):
    print("✅ Training OpenProtoDetector...")

    encoder.eval()
    classifier.eval()

    open_detector = OpenProtoDetector(
        feature_dim=config.embedding_dim
    ).to(config.device)

    optimizer = torch.optim.Adam(open_detector.parameters(), lr=config.lr)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    known_set = KnownDataset(config.train_data)
    unknown_set = UnknownDataset(config.val2)

    known_loader = DataLoader(known_set, batch_size=config.batch_size, shuffle=True)
    unknown_loader = DataLoader(unknown_set, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs2):
        total_loss = 0
        known_iter = iter(known_loader)
        unknown_iter = iter(unknown_loader)

        for _ in tqdm(range(len(known_loader)), desc=f"Epoch {epoch+1}"):
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

            x_known = x_known.to(config.device)
            y_known = y_known.to(config.device)
            x_unknown = x_unknown.to(config.device)

            with torch.no_grad():
                feat_known = encoder(x_known)
                logits_known = classifier(feat_known)
                probs_known = F.softmax(logits_known, dim=1)
                max_prob_known, _ = probs_known.max(dim=1, keepdim=True)
                proto_known = prototypes.get(y_known)

                feat_unknown = encoder(x_unknown)
                logits_unknown = classifier(feat_unknown)
                probs_unknown = F.softmax(logits_unknown, dim=1)
                max_prob_unknown, _ = probs_unknown.max(dim=1, keepdim=True)
                proto_unknown = torch.zeros_like(proto_known)

            score_known = open_detector(max_prob_known, feat_known, proto_known)
            score_unknown = open_detector(max_prob_unknown, feat_unknown, proto_unknown)

            labels = torch.cat([
                torch.ones(score_known.size(0), device=config.device),
                torch.zeros(score_unknown.size(0), device=config.device)
            ])
            scores = torch.cat([score_known, score_unknown])

            loss = bce_loss_fn(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(known_loader):.4f}")

    torch.save(open_detector.state_dict(), os.path.join(config.save_dir, 'open_proto_detector.pth'))
    print("✅ OpenProtoDetector trained and saved.")


if __name__ == "__main__":
    config = parse_args()

    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    ckpt = torch.load(os.path.join(config.save_dir, 'model_opencon2.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    protoPath = os.path.join(config.save_dir, 'prototype2.pkl')
    prototype = load_object(protoPath)

    encoder.eval()
    classifier.eval()
    train_open_proto_detector(config, encoder, classifier, prototype)
