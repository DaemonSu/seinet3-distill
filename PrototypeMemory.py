import torch
import torch.nn.functional as F


class PrototypeMemory:
    def __init__(self, num_classes, feature_dim, device, momentum=0.3):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.momentum = momentum

        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)



    @torch.no_grad()
    def update(self, features, labels):
        """
        features: [B, D]
        labels: [B]
        """
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() == 0:
                continue
            feat_mean = features[mask].mean(dim=0)
            if self.initialized[i]:
                self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * feat_mean
            else:
                self.prototypes[i] = feat_mean
                self.initialized[i] = True

    def get(self, labels):
        """
        labels: [B], return [B, D] matched prototypes
        """
        return self.prototypes[labels]

    def get_all(self):
        return self.prototypes

    import torch.nn.functional as F

    def distance(self, features, metric='euclidean'):
        """
        features: [B, D]
        Return: [B, num_classes] 距离矩阵
        """
        if metric == 'euclidean':
            # [B, 1, D] - [1, C, D] -> [B, C, D] -> L2 norm
            diff = features.unsqueeze(1) - self.prototypes.unsqueeze(0)
            dist = torch.norm(diff, dim=2)  # [B, C]
        elif metric == 'cosine':
            # normalize
            features_norm = F.normalize(features, dim=1)
            prototypes_norm = F.normalize(self.prototypes, dim=1)
            dist = 1 - torch.matmul(features_norm, prototypes_norm.T)  # [B, C], cosine distance
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        return dist
    # def distance(self, features, metric='cosine'):
    #     features_norm = F.normalize(features, dim=1)
    #     prototypes_norm = F.normalize(self.prototypes, dim=1)
    #     return 1 - torch.matmul(features_norm, prototypes_norm.T)


