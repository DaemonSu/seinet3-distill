import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= Dataset =================
class NpyDataset(Dataset):
    def __init__(self, file_list, label_map):
        self.data, self.labels = [], []
        for f in file_list:
            sample = np.load(f)  # [SAMPLE_SIZE, 3]
            self.data.append(sample)
            self.labels.append(label_map[f])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ================= 小型网络 =================
class SmallNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,L,3] -> [B,3,L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ================= 小规模训练 =================
def small_train_test(data_folder):
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]

    # 提取设备编号，例如 device_19_0000.npy -> 19
    devices = sorted(list(set(int(f.split('_')[1]) for f in files)))[:2]  # 取两个设备
    label_map = {}
    selected_files = []
    label_counter = 0
    for dev in devices:
        dev_str = f"{dev:02d}"  # 0 -> "00", 19 -> "19"
        dev_files = [f for f in files if f"device_{dev_str}_" in f][:4] # 每个设备取前4条样本
        for f in dev_files:
            selected_files.append(f)
            label_map[f] = label_counter
        label_counter += 1

    # 划分训练/测试集
    train_files = selected_files[:len(selected_files) // 2]
    test_files = selected_files[len(selected_files) // 2:]

    train_dataset = NpyDataset(train_files, label_map)
    test_dataset = NpyDataset(test_files, label_map)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model = SmallNet(num_classes=len(devices)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"[Epoch {epoch}] Test Acc: {correct / total:.4f}")


if __name__ == "__main__":
    small_train_test("G:/seidataforCIL/val-closed")
