# 文件：model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.fuse = nn.Conv1d(in_channels * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fuse(out)
        return self.relu(self.bn(out))

class FFTBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]
        fft_mag = torch.fft.rfft(x, dim=2).abs()
        fft_mag = F.interpolate(fft_mag, size=x.shape[-1], mode='linear', align_corners=True)
        out = self.conv(fft_mag)
        return self.relu(self.bn(out))

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, 3500]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(64, 128, stride=2)             # -> [B, 128, 1750]
        self.layer2 = nn.Sequential(                               # 深度加深
            ResidualBlock(128, 128, stride=2),                     # -> [B, 128, 875]
            ResidualBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            # MultiScaleBlock(128),                                  # 多尺度
            nn.Conv1d(128, 256, kernel_size=1),                    # 通道提升
            nn.BatchNorm1d(256),
            nn.ReLU()
        )                                                          # [B, 256, 875]
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 256, stride=2),                     # -> [B, 256, 438]
            FFTBlock(256),                                        # 频域增强
            ResidualBlock(256, 256, stride=2)                      # -> [B, 256, 219]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)                        # -> [B, 256, 1]
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, 3, 7000]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)  # [B, 256]
        feat = F.normalize(self.fc(x), dim=-1)  # [B, feature_dim]
        return feat


#
# class ClassifierHead(nn.Module):
#     def __init__(self, in_dim=128, num_classes=10):
#         super().__init__()
#         self.classifier = nn.Linear(in_dim, num_classes)
#
#     def forward(self, x):
#         return self.classifier(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

    # open_detector.py





