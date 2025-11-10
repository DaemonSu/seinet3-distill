import torch
import torch.nn as nn
from dataset import KnownDataset
from model_simple import SimpleSEINet
from config import parse_args

from torch.utils.data import DataLoader


def train_open_contrastive(config):
    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, batch_size=config.batch_size, shuffle=True)

    valset = KnownDataset(config.val_closed)
    val_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False)

    model=SimpleSEINet().to(config.device)

    criterion  = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for x, y in known_loader:
            x = x.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(known_loader.dataset)
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")
        if epoch % 10 == 0:

            model.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    # print(x.mean().item(), x.std().item(), x.min().item(), x.max().item())

                    x = x.to(config.device)
                    y=y.to(config.device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            print(f"[Epoch {epoch + 1}] Val Acc: {acc:.4f}")


if __name__ == "__main__":

    config = parse_args()
    train_open_contrastive(config)
