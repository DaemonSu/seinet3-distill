import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import KnownDataset

def evaluate_incremental(model_path, encoder_path, projector_path, test_data_path,
                         num_old_classes, device,
                         step,               # 当前增量步骤（0 表示初始训练后的评估）
                         initial_classes=50, # 初始训练类数（step=0 的类数）
                         increment_size=5,   # 每次增量加入的类数
                         metrics_dir="model"):
    """
    Revised evaluator that:
      - computes per-step accuracy on each historical group of classes
      - records A_i^i (accuracy on group i when learned) and A_{t,i} (current accuracy on group i)
      - computes forgetting F_t = mean_i(A_i^i - A_{t,i}) for i=0..t-1
      - computes cumulative forgetting as mean of previous forgetting rates + current (or other variants)
    Args:
      - step: integer step index for this evaluation. Use step=0 for initial-training evaluation.
      - initial_classes, increment_size: used to build class-group ranges.
    """

    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "incremental_metrics.json")

    # load models (safe: these are saved model objects in your framework)
    encoder = torch.load(encoder_path, map_location=device)
    classifier = torch.load(model_path, map_location=device)
    projector = torch.load(projector_path, map_location=device)
    encoder.eval(); classifier.eval(); projector.eval()

    # build test loader
    testset = KnownDataset(test_data_path)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # helper: build class ranges per step (0..step)
    # step 0: classes [0, initial_classes-1]
    # step 1: classes [initial_classes, initial_classes+increment_size-1], ...
    def build_step_ranges(max_step):
        ranges = {}
        start = 0
        ranges[0] = (0, initial_classes - 1)
        start = initial_classes
        for s in range(1, max_step + 1):
            ranges[s] = (start, start + increment_size - 1)
            start += increment_size
        return ranges

    # We'll evaluate per-class accuracy and per-step-group accuracy
    # accumulate predictions and labels to compute group-wise accuracies
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device); y = y.to(device)
            feats = projector(encoder(x))
            logits = classifier(feats)
            preds = logits.argmax(dim=1)
            preds_all.append(preds.cpu())
            labels_all.append(y.cpu())

    preds_all = torch.cat(preds_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    total_correct = (preds_all == labels_all).sum()
    total_num = labels_all.shape[0]
    total_acc = 100.0 * total_correct / total_num

    # Build ranges up to current step (we might evaluate beyond current step classes; limit by labels)
    # Determine maximum step index present in labels by checking max label
    max_label = int(labels_all.max())
    # compute steps that cover up to max_label
    # compute how many increments needed after initial:
    if max_label < initial_classes:
        max_step_present = 0
    else:
        extras = max_label - initial_classes + 1
        extra_steps = (extras + increment_size - 1) // increment_size
        max_step_present = extra_steps

    step_ranges = build_step_ranges(max_step_present)

    # compute per-step-group accuracies for *this* evaluation (A_{t,i})
    per_group_acc_now = {}
    for s, (lo, hi) in step_ranges.items():
        # clamp hi to actual max_label
        hi_clamped = min(hi, max_label)
        if lo > hi_clamped:
            continue
        mask = (labels_all >= lo) & (labels_all <= hi_clamped)
        if mask.sum() == 0:
            acc = None
        else:
            acc = 100.0 * (preds_all[mask] == labels_all[mask]).sum() / mask.sum()
        per_group_acc_now[str(s)] = None if acc is None else float(acc)

    # compute aggregated old/new acc like before (for logging)
    old_mask = labels_all < num_old_classes
    new_mask = ~old_mask
    old_acc = 100.0 * (preds_all[old_mask] == labels_all[old_mask]).sum() / (old_mask.sum() if old_mask.sum() > 0 else 1)
    new_acc = 100.0 * (preds_all[new_mask] == labels_all[new_mask]).sum() / (new_mask.sum() if new_mask.sum() > 0 else 1)

    print(f"Test Acc Total: {total_acc:.2f}% | Old: {old_acc:.2f}% | New: {new_acc:.2f}%")

    # load history
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # Save current per-group accuracies into history[step] so we have A_{t,i} record (for i in 0..t)
    history.setdefault(str(step), {})
    history[str(step)]["total_acc"] = float(total_acc)
    history[str(step)]["old_acc"] = float(old_acc)
    history[str(step)]["new_acc"] = float(new_acc)
    history[str(step)]["per_group_acc_now"] = per_group_acc_now  # A_{t,i}

    # If this is the moment right after learning step (i.e., "when learned"), we also want to record A_i^i.
    # Convention: when we call evaluate_incremental immediately after training step `step`, the A_i^i for i==step is per_group_acc_now[str(step)].
    # So store it as 'acc_when_learned' for that step.
    history[str(step)]["acc_when_learned"] = per_group_acc_now.get(str(step), None)

    # ===== compute forgetting =====
    # For current step t, we need for each past i in [0, t-1]:
    #   A_i^i  = history[str(i)]['acc_when_learned'][str(i)]
    #   A_{t,i} = history[str(step)]['per_group_acc_now'][str(i)]
    forgetting_list = []
    for i in range(0, step):  # all previous steps
        key_i = str(i)
        A_i_i = None
        # get A_i^i from history (acc_when_learned)
        if key_i in history and history[key_i].get("acc_when_learned") is not None:
            A_i_i = history[key_i]["acc_when_learned"]
        else:
            # fallback: try per_group_acc_now recorded at that time
            A_i_i = history.get(key_i, {}).get("per_group_acc_now", {}).get(key_i, None)

        A_t_i = history[str(step)]["per_group_acc_now"].get(key_i, None)

        if (A_i_i is None) or (A_t_i is None):
            # can't compute this group's forgetting (no data), skip it
            continue

        # forgetting for group i at time t
        f_i = float(A_i_i) - float(A_t_i)
        forgetting_list.append(max(0.0, f_i))

    if len(forgetting_list) > 0:
        forget_rate = float(np.mean(forgetting_list))
    else:
        forget_rate = 0.0

    # cumulative forgetting: mean of all previous forget rates plus current (or running mean)
    prev_forgets = [v.get("forget_rate", 0.0) for k, v in history.items() if int(k) < step]
    if len(prev_forgets) > 0:
        cumulative_forget = float(np.mean(prev_forgets + [forget_rate]))
    else:
        cumulative_forget = float(forget_rate)

    history[str(step)]["forget_rate"] = forget_rate
    history[str(step)]["cumulative_forget"] = cumulative_forget

    # ======== 计算 per-class 准确率并保存 ========
    from collections import defaultdict
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        feats = encoder(x)
        feats = projector(feats)
        logits = classifier(feats)
        preds = logits.argmax(dim=1)

        for yi, pi in zip(y.cpu().tolist(), preds.cpu().tolist()):
            per_class_total[yi] += 1
            if yi == pi:
                per_class_correct[yi] += 1

    # 计算每个类的准确率
    per_class_acc = {}
    for cls in per_class_total.keys():
        per_class_acc[str(cls)] = per_class_correct[cls] / per_class_total[cls] * 100

    # 保存到历史记录中
    history[str(step)]["per_class_acc_now"] = per_class_acc


    # save history
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    # nice print
    print(f"Forgetting Rate: {forget_rate:.2f}% | Cumulative Forget: {cumulative_forget:.2f}%")

    return total_acc, old_acc, new_acc, per_group_acc_now
