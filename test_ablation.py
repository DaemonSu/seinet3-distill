import os
import torch
import torch.nn.functional as F
import numpy as np
from thop import profile
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve
from metrics import evaluate_all_metrics, save_metrics_to_csv


def test_mixed(encoder, classifier, test_loader, config):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_feats = []

    known_scores, unknown_scores = [], []
    correct_flags = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(config.device), y.to(config.device)
            feat = encoder(x)
            logits = classifier(feat)
            prob = F.softmax(logits, dim=1)
            max_probs, preds = prob.max(dim=1)

            all_probs.append(prob.cpu().numpy())
            all_feats.extend(feat.cpu().numpy())

            pred_labels = []
            for i in range(len(y)):
                mp = max_probs[i].item()
                pred_cls = preds[i].item()

                # 阈值判定 open/closed
                if mp < config.open_threshold:
                    pred_labels.append(-1)
                else:
                    pred_labels.append(pred_cls)

                # === 收集 OSCR 数据 ===
                if y[i].item() == -1:  # 真实 open 样本
                    unknown_scores.append(mp)
                else:  # 真实 closed 样本
                    known_scores.append(mp)
                    correct_flags.append(1 if pred_cls == y[i].item() else 0)

            all_preds.extend(pred_labels)
            all_labels.extend(y.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs = np.concatenate(all_probs)

    metrics = evaluate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        known_scores=np.array(known_scores),
        unknown_scores=np.array(unknown_scores),
        correct_flags=np.array(correct_flags)
    )

    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 保存指标
    save_metrics_to_csv(metrics, os.path.join(config.save_dir, "results", "metrics.csv"))

    # 保存混淆矩阵
    try:
        cm = confusion_matrix(y_true[y_true!=-1], y_pred[y_true!=-1])
        plt.figure(figsize=(6,6))
        plt.imshow(cm, cmap='Blues')
        plt.title("Confusion Matrix (Closed-set)")
        plt.colorbar()
        plt.savefig(os.path.join(config.save_dir, "results", "confusion_matrix.png"), dpi=300)
        plt.close()
    except:
        print("Warning: Cannot compute confusion matrix for open-set labels.")

    # 绘制 ROC 曲线
    y_scores = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    y_probs = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, _ = roc_curve(y_scores, y_probs)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['AUROC']:.4f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    os.makedirs(os.path.join(config.save_dir, "results", "plots"), exist_ok=True)
    plt.savefig(os.path.join(config.save_dir, "results", "plots", "roc_curve.png"), dpi=300)
    plt.close()

    return metrics

if __name__ == "__main__":
    from dataset import MixedDataset
    from config import parse_args

    config = parse_args()

    mixed_testset = MixedDataset(config.test_mixed)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=True)




    #全尺寸模型训练
    # encoder = torch.load(os.path.join(config.save_dir, 'ablation','encoder_wo_multi.pth')).to(config.device)
    # classifier = torch.load(os.path.join(config.save_dir, 'ablation','classifier_wo_multi.pth')).to(config.device)

    # 去除了multi模块
    encoder = torch.load(os.path.join(config.save_dir, 'ablation-input', 'encoder_iq_abs_fft.pth')).to(config.device)
    classifier = torch.load(os.path.join(config.save_dir, 'ablation-input', 'classifier_iq_abs_fft.pth')).to(config.device)

    # 去除了fft模块
    # encoder = torch.load(os.path.join(config.save_dir, 'ablation', 'encoder_wo_fft.pth')).to(config.device)
    # classifier = torch.load(os.path.join(config.save_dir, 'ablation', 'classifier_wo_fft.pth')).to(config.device)



    # 2. 关键修正：用 batch_size=1 构造输入（匹配论文标准）
    # batch_size = 128  # 你原本的批次大小
    # input_shape = (7000, 3)  # 单条数据的形状（时间步×特征数）
    # input_demo = torch.randn(batch_size, *input_shape).to(config.device)  # 原批次输入
    #
    # # 3. 计算批次总FLOPs（与你原本结果一致）
    # with torch.no_grad():
    #     # 计算Encoder的批次FLOPs和参数量（参数量与batch无关，仅计算一次）
    #     flops_encoder_batch, params_encoder = profile(encoder, inputs=(input_demo,), verbose=False)
    #     encoder_output = encoder(input_demo)  # 得到批次输出
    #
    #     # 计算Classifier的批次FLOPs和参数量
    #     flops_classifier_batch, params_classifier = profile(classifier, inputs=(encoder_output,), verbose=False)
    #
    # # 4. 核心：转换为单条数据的FLOPs（除以batch size）
    # flops_encoder_single = flops_encoder_batch / batch_size  # 单条数据的Encoder计算量
    # flops_classifier_single = flops_classifier_batch / batch_size  # 单条数据的Classifier计算量
    # total_flops_single = flops_encoder_single + flops_classifier_single  # 单条数据的总计算量
    #
    # # 5. 参数量：与batch无关，直接汇总（你的原结果无需修改）
    # total_params = params_encoder + params_classifier
    #
    #
    # # 格式化输出（转换为更易读的单位）
    # def format_num(num):
    #     """将数字格式化为带单位的字符串（K, M, G）"""
    #     if num >= 1e9:
    #         return f"{num / 1e9:.2f}G"
    #     elif num >= 1e6:
    #         return f"{num / 1e6:.2f}M"
    #     elif num >= 1e3:
    #         return f"{num / 1e3:.2f}K"
    #     return f"{num:.0f}"
    #
    #
    # print(f"Encoder参数量: {format_num(params_encoder)}")
    # print(f"Classifier参数量: {format_num(params_classifier)}")
    # print(f"总参数量: {format_num(total_params)}")
    # print(f"总计算量: {format_num(total_flops_single)} FLOPs")


    test_mixed(encoder, classifier, mixed_loader, config)
