from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class SEIDataset(Dataset):
    def __init__(self, data_path):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npy')]
        self.labels = [int(f.split('_')[1]) for f in self.files]  # 获取 device_xx_xxxx.npy 的 xx 作为label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        label = self.labels[idx]
        # print(data.shape)
        # print(label)  # 打印每个批次的标签
        return torch.tensor(data, dtype=torch.float), label



# 文件：dataset.py

import torch
from torch.utils.data import Dataset
import os
import numpy as np

# 文件：dataset.py

import torch
from torch.utils.data import Dataset
import os
import numpy as np
import re

class KnownDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.samples = []

        # 文件名示例：device_09_5549.npy
        # pattern = re.compile(r'device_(\d+)_\d+\.npy')
        pattern = re.compile(r'device_(\d+)_\d+(?:_[\w\d]+)?\.npy')

        for file in os.listdir(root):
            if file.endswith('.npy'):
                match = pattern.match(file)
                if match:
                    label = int(match.group(1))  # 提取设备编码作为标签
                    filepath = os.path.join(root, file)
                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        data = np.load(path)  # 假设 shape 为 [2, L]
        data = torch.tensor(data, dtype=torch.float32)
        return data, label

class UnknownDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.samples = []
        for file in os.listdir(root):
            if file.endswith('.npy'):
                self.samples.append(os.path.join(root, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        data = np.load(path)  # shape = [2, L]
        data = torch.tensor(data, dtype=torch.float32)
        return data, -1  # unknown 类标记为 -1


class MixedDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.samples = []

        # 文件名示例：device_09_5549.npy
        # pattern = re.compile(r'device_(\d+)_\d+\.npy')

        pattern = re.compile(r'device_(\d+)_\d+(?:_[\w\d]+)?\.npy')

        for file in os.listdir(root):
            if file.endswith('.npy'):
                match = pattern.match(file)
                if match:
                    label = int(match.group(1))  # 提取设备编码作为标签
                    filepath = os.path.join(root, file)
                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        data = np.load(path)  # shape = [2, L]
        data = torch.tensor(data, dtype=torch.float32)
        if  label > 7:
           label= -1
        return data, label # unknown 类标记为 -1

