import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSEINet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        # 输入形状：[B, 3, 29, 42]（批量×通道×高度×宽度）
        # 注意：Conv2d的第一个参数是输入通道数，这里为3（与你的数据通道一致）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1, padding=(2, 2))  # 2D卷积
        self.gn1 = nn.GroupNorm(4, 32)  # 分组归一化，4个组，输入通道32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=(2, 2))  # 2D卷积
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 2D自适应池化，输出1×1
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 输入x原始形状：[B, 29, 42, 3]（批量×高度×宽度×通道）
        # 转换为PyTorch卷积层要求的格式：[B, C, H, W]（批量×通道×高度×宽度）
        x = x.permute(0, 3, 1, 2)  # 通道维度从最后移到第2位

        # 卷积+激活
        x = F.relu(self.gn1(self.conv1(x)))  # 输出形状：[B, 32, 29, 42]
        x = F.relu(self.gn2(self.conv2(x)))  # 输出形状：[B, 64, 29, 42]

        # 池化并展平
        x = self.pool(x).squeeze()  # 池化后[B, 64, 1, 1]，挤压为[B, 64]
        return self.fc(x)  # 输出分类结果：[B, num_classes]
