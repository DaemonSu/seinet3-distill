import matplotlib
matplotlib.use('pdf')  # 使用PDF后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm

import colorcet as cc


# 设置输出为矢量图格式 + 字体配置
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Times New Roman",
    "font.size": 12
})

def visualize_features(features, labels, known_class_count, method='t-SNE', prototypes=None, proto_threshold=None, save_path='feature_vis.pdf'):
    all_data = features
    if prototypes is not None:
        all_data = np.concatenate([features, prototypes], axis=0)

    embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_data)
    feat_emb = embedded[:len(features)]
    if prototypes is not None:
        proto_emb = embedded[len(features):]

    plt.figure(figsize=(10, 8))
    labels = np.array(labels)
    colors = cc.glasbey[:known_class_count]

    for cls in range(known_class_count):
        mask = labels == cls
        # plt.scatter(feat_emb[mask, 0], feat_emb[mask, 1], label=f"Class {cls}", color=cmap(cls), s=20, alpha=0.6)
        plt.scatter(feat_emb[mask, 0], feat_emb[mask, 1],
                    label=f"Class {cls}", color=colors[cls],
                    s=20, alpha=0.6, edgecolors='k', linewidths=0.2)

    open_mask = labels == -1
    if open_mask.sum() > 0:
        plt.scatter(feat_emb[open_mask, 0], feat_emb[open_mask, 1], label="Unknown", color='gray', s=20, alpha=0.6, marker='x')

    if prototypes is not None:
        for i in range(min(known_class_count, len(proto_emb))):
            px, py = proto_emb[i]
            # plt.scatter(px, py, marker='X', s=200, color=cmap(i), edgecolors='black', linewidths=1.5, label=f"Proto {i}")
            plt.scatter(px, py, marker='X', s=200, color=colors[i],
                        edgecolors='black', linewidths=1.5, label=f"Proto {i}")
            if proto_threshold is not None:
                # circle = Circle((px, py), proto_threshold, color=cmap(i), linestyle='--', fill=False, alpha=0.4)
                # plt.gca().add_patch(circle)
                circle = Circle((px, py), proto_threshold, color=colors[i],
                                linestyle='--', fill=False, alpha=0.4)
                plt.gca().add_patch(circle)

    plt.legend(loc='best')
    plt.title(f"{method} Feature Visualization (Known + Unknown)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()  # 可取消注释用于调试
