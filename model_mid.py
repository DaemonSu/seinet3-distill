import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLayer(nn.Module):
    """
    插入在特征提取器和分类器之间的对比学习层
    输出特征用于对比损失或继续分类
    """
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=1024, use_norm=True):
        """
        in_dim : encoder输出特征维度
        hidden_dim : 对比层内部隐藏维度
        out_dim : 输出特征维度（可继续接分类器）
        use_norm : 是否对输出特征做 L2 归一化（通常对比学习需要）
        """
        super(ContrastiveLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
        self.use_norm = use_norm

    def forward(self, x):
        """
        x: [B, in_dim] encoder输出特征
        返回: [B, out_dim] 经过对比学习层处理后的特征
        """
        out = self.mlp(x)
        if self.use_norm:
            out = F.normalize(out, p=2, dim=1)  # 对比学习通常归一化
        return out


class IncrementalContrastiveModel(nn.Module):
    """
    增量训练用的网络结构：
    encoder -> contrastive layer -> classifier
    """
    def __init__(self, encoder, classifier, in_dim=1024, hidden_dim=512, feat_dim=1024, use_norm=True):
        super(IncrementalContrastiveModel, self).__init__()
        self.encoder = encoder
        self.contrastive_layer = ContrastiveLayer(in_dim, hidden_dim, feat_dim, use_norm)
        self.classifier = classifier

    def forward(self, x, return_feat=False):
        """
        x: [B, C, H, W]
        return_feat: 是否返回对比层输出特征
        """
        feat = self.encoder(x)
        feat = self.contrastive_layer(feat)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        else:
            return logits
