import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual1D(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(in_planes * expand_ratio)
        self.use_res_connect = stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_planes, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv1d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm1d(out_planes)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV2_1D(nn.Module):
    def __init__(self, feature_dim=128, width_mult=1.0):
        super().__init__()
        setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(32 * width_mult)
        layers = [
            nn.Conv1d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU6(inplace=True)
        ]
        for t, c, n, s in setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual1D(input_channel, output_channel, stride, t))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        feat = F.normalize(self.fc(x), dim=-1)
        return feat

def MobileNetV2_1D_Base(feature_dim=128):
    return MobileNetV2_1D(feature_dim)
