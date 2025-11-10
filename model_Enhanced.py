import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Lightweight SE (Squeeze-Excite)模块
# ---------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        s = F.adaptive_avg_pool2d(x, 1)       # [B, C, 1, 1]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

# ---------------------------
# Residual block (2D)
# ---------------------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

# ---------------------------
# Multi-scale block (2D)
# ---------------------------
class MultiScale2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, dilation=1, bias=False)
        self.b2 = nn.Conv2d(ch, ch, kernel_size=3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(ch, ch, kernel_size=3, padding=3, dilation=3, bias=False)
        self.bn = nn.BatchNorm2d(ch * 3)
        self.fuse = nn.Conv2d(ch * 3, ch, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.fuse(out)
        return self.act(out)

# ---------------------------
# FFTBlock (2D adapted)
# ---------------------------
class FFTBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # flatten spatial dim
        x_flat = x.view(B, C, H * W)                      # [B, C, S]
        # rFFT along spatial dimension
        fft = torch.fft.rfft(x_flat, dim=2)               # complex [B, C, S//2+1]
        mag = fft.abs()                                   # magnitude
        # upsample back to S length (linear interpolation)
        mag_up = F.interpolate(mag, size=H * W, mode='linear', align_corners=False)  # [B, C, S]
        mag2d = mag_up.view(B, C, H, W)
        out = self.conv(mag2d)
        out = self.bn(out)
        return self.act(out)

# ---------------------------
# Enhanced Feature Extractor (scaled)
# ---------------------------
class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512, base_channels=64):
        """
        - in_channels: input channels (3)
        - feature_dim: output embedding dim (建议 256 或 512；目标100类建议 512)
        - base_channels: 基础通道宽度，可取 64 或 96
        """
        super().__init__()
        C = base_channels
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        # stage1
        self.layer1 = nn.Sequential(
            ResidualBlock2D(C, C * 2, stride=2),   # downsample
            SEBlock(C * 2)
        )  # ~ /4
        # stage2
        self.layer2 = nn.Sequential(
            ResidualBlock2D(C * 2, C * 2, stride=2),
            ResidualBlock2D(C * 2, C * 2, stride=1),
            SEBlock(C * 2)
        )  # ~ /8
        # stage3 (wider)
        self.layer3 = nn.Sequential(
            MultiScale2D(C * 2),
            nn.Conv2d(C * 2, C * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(C * 4),
            nn.ReLU(inplace=True),
            SEBlock(C * 4)
        )  # ~ /8
        # stage4 (deep + fft)
        self.layer4 = nn.Sequential(
            ResidualBlock2D(C * 4, C * 4, stride=2),
            FFTBlock2D(C * 4),
            ResidualBlock2D(C * 4, C * 4, stride=1),
            SEBlock(C * 4)
        )  # ~ /32 or so depending input

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(C * 4, C * 4),
            nn.BatchNorm1d(C * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(C * 4, feature_dim)
        )

    def forward(self, x):
        # x expected [B, H, W, C_in] as your pipeline - convert to [B, C, H, W]
        if x.dim() == 4 and x.shape[-1] in (1, 2, 3, 4, 6):  # heuristic
            x = x.permute(0, 3, 1, 2)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).squeeze(-1).squeeze(-1)  # [B, C*4]
        feat = self.fc(out)
        feat = F.normalize(feat, dim=1)  # L2 normalize for angular losses
        return feat

# ---------------------------
# ArcFace (Angular margin) head
# ---------------------------
class ArcMarginProduct(nn.Module):
    """
    Implementation of ArcFace margin (cosine + angular margin).
    s: scale, m: margin in radians
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.3, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(torch.pi) - m)
        self.mm = torch.sin(torch.tensor(torch.pi) - m) * m

    def forward(self, input, label):
        # input: [B, in_features] normalized
        # label: [B] long
        # weight is not normalized automatically; normalize for cosine
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # [B, out_features]
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output  # logits

# ---------------------------
# Classifier wrapper (optionally ArcFace)
# ---------------------------
class ClassifierWithArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # 归一化权重和特征
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # 计算 cosθ
        cosine = F.linear(features, weight)

        if labels is None:
            # 推理阶段
            return self.s * cosine

        # 取出对应类别的角度，添加 margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # 用 margin 替换对应类的 logit
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits *= self.s
        return logits
