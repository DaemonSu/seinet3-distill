import torch


class OpenSetDataset(torch.utils.data.Dataset):
    def __init__(self, dists, logits, open_flags):
        self.dists = torch.tensor(dists, dtype=torch.float32)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.open_flags = torch.tensor(open_flags, dtype=torch.float32)  # 1=open set, 0=closed set

    def __len__(self):
        return len(self.dists)

    def __getitem__(self, idx):
        return self.dists[idx], self.logits[idx], self.open_flags[idx]
