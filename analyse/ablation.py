# ablation_plot_multi.py
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# IEEE / TIFS-friendly settings
# -----------------------------
mpl.rcParams.update({
    "font.family": ["Times New Roman"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": True,
})
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Helpers: load json safely
# -----------------------------
def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

# -----------------------------
# Build per-class × step accuracy matrix
# Returns: steps (sorted ints), class_ids(sorted ints), acc_mat (n_cls x n_steps, np.nan)
# -----------------------------
def build_acc_matrix_from_history(history):
    steps = sorted([int(k) for k in history.keys()])
    # collect class ids
    class_set = set()
    for t in steps:
        d = history[str(t)]
        if 'per_class_acc_now' in d and isinstance(d['per_class_acc_now'], dict):
            class_set.update(int(k) for k in d['per_class_acc_now'].keys())
        if 'acc_when_learned' in d and isinstance(d['acc_when_learned'], dict):
            class_set.update(int(k) for k in d['acc_when_learned'].keys())
    class_ids = sorted(class_set)
    n_steps = len(steps)
    n_cls = len(class_ids)
    if n_cls == 0:
        return steps, class_ids, np.empty((0, n_steps)), np.array([history[str(s)].get('total_acc', np.nan) for s in steps])
    cls2idx = {c:i for i,c in enumerate(class_ids)}
    acc_mat = np.full((n_cls, n_steps), np.nan, dtype=float)
    for t_idx, t in enumerate(steps):
        d = history[str(t)]
        if 'per_class_acc_now' in d and isinstance(d['per_class_acc_now'], dict):
            for c_str, acc in d['per_class_acc_now'].items():
                try:
                    c = int(c_str)
                    acc_mat[cls2idx[c], t_idx] = float(acc)
                except Exception:
                    continue
    total_acc_list = np.array([history[str(s)].get('total_acc', np.nan) for s in steps], dtype=float)
    return steps, class_ids, acc_mat, total_acc_list

# -----------------------------
# Compute AFR(t) and CF(t) vectors
# AFR(t) = mean_c( max_{k<=t} A_c(k) - A_c(t) )
# CF(t)  = mean_c( A_c(first) - A_c(t) )
# Uses acc_mat and optional acc_when_learned to determine first.
# -----------------------------
def compute_stepwise_AFRCF(history, steps, class_ids, acc_mat):
    n_steps = len(steps)
    n_cls = len(class_ids)
    # first_step_acc per class (use acc_when_learned if present)
    first_acc_per_class = {}
    for c in class_ids:
        first_acc_per_class[c] = np.nan
    # try to fill from acc_when_learned if available
    for t_idx, t in enumerate(steps):
        d = history[str(t)]
        if 'acc_when_learned' in d and isinstance(d['acc_when_learned'], dict):
            for c_str, v in d['acc_when_learned'].items():
                try:
                    c = int(c_str)
                    if c in first_acc_per_class and np.isnan(first_acc_per_class[c]):
                        first_acc_per_class[c] = float(v)
                except:
                    continue
    # fallback: first non-nan in acc_mat
    for idx, c in enumerate(class_ids):
        if np.isnan(first_acc_per_class[c]):
            row = acc_mat[idx,:]
            valid = np.where(~np.isnan(row))[0]
            if valid.size>0:
                first_acc_per_class[c] = float(row[valid[0]])
            else:
                first_acc_per_class[c] = np.nan

    # compute per-class cummax up to t and afr/cf seq
    AFR_t = np.full(n_steps, np.nan)
    CF_t  = np.full(n_steps, np.nan)
    count_afr = np.zeros(n_steps, dtype=int)
    count_cf = np.zeros(n_steps, dtype=int)

    for idx, c in enumerate(class_ids):
        row = acc_mat[idx,:]  # may contain nan
        if np.all(np.isnan(row)):
            continue
        cummax = np.full(n_steps, np.nan)
        running_max = np.nan
        for t_idx in range(n_steps):
            if not np.isnan(row[t_idx]):
                running_max = row[t_idx] if np.isnan(running_max) else max(running_max, row[t_idx])
                cummax[t_idx] = running_max
            else:
                cummax[t_idx] = running_max  # may be nan
        first_acc = first_acc_per_class[c]
        for t_idx in range(n_steps):
            if np.isnan(row[t_idx]):
                continue
            # AFR part
            if not np.isnan(cummax[t_idx]):
                afr = cummax[t_idx] - row[t_idx]
                if np.isnan(AFR_t[t_idx]):
                    AFR_t[t_idx] = afr
                else:
                    AFR_t[t_idx] += afr
                count_afr[t_idx] += 1
            # CF part
            if not np.isnan(first_acc):
                cf = first_acc - row[t_idx]
                if np.isnan(CF_t[t_idx]):
                    CF_t[t_idx] = cf
                else:
                    CF_t[t_idx] += cf
                count_cf[t_idx] += 1
    # finalize averages
    for t_idx in range(n_steps):
        if count_afr[t_idx] > 0:
            AFR_t[t_idx] = AFR_t[t_idx] / count_afr[t_idx]
        else:
            AFR_t[t_idx] = np.nan
        if count_cf[t_idx] > 0:
            CF_t[t_idx] = CF_t[t_idx] / count_cf[t_idx]
        else:
            CF_t[t_idx] = np.nan
    return AFR_t, CF_t, first_acc_per_class

# -----------------------------
# Export per-class summary CSV for debugging (optional)
# -----------------------------
def export_per_class_csv(out_dir, label, class_ids, acc_mat, first_acc_map, per_class_stats):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"per_class_metrics_{label}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # header
        header = ['class_id', 'first_acc', 'max_acc', 'last_acc', 'AFR_final', 'CF_final']
        writer.writerow(header)
        for i, c in enumerate(class_ids):
            row = acc_mat[i,:]
            maxacc = float(np.nanmax(row)) if np.any(~np.isnan(row)) else np.nan
            lastacc = float(row[-1]) if not np.isnan(row[-1]) else (float(row[np.where(~np.isnan(row))[0][-1]]) if np.any(~np.isnan(row)) else np.nan)
            writer.writerow([c, first_acc_map.get(c, np.nan), maxacc, lastacc,
                              per_class_stats.get(c, {}).get('AFR_final', np.nan),
                              per_class_stats.get(c, {}).get('CF_final', np.nan)])
    print(f"[i] per-class CSV saved: {csv_path}")

# -----------------------------
# Main plotting routine
# -----------------------------
def plot_multiple_jsons(file_paths, labels, out_pdf="ablation_compare.pdf"):
    assert len(file_paths) == len(labels), "file_paths and labels must match"
    n = len(file_paths)
    plt.figure(figsize=(6.5, 9))  # we'll use subplots
    # Create 3 subplots: total acc, AFR(t), CF(t)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)

    table_rows = []

    for i, fp in enumerate(file_paths):
        label = labels[i]
        if not os.path.exists(fp):
            print(f"[WARN] file not found: {fp}, skipping.")
            continue
        hist = load_json(fp)
        steps, class_ids, acc_mat, total_acc_list = build_acc_matrix_from_history(hist)
        AFR_t, CF_t, first_acc_map = compute_stepwise_AFRCF(hist, steps, class_ids, acc_mat)

        # compute per-class final stats for table
        # final mean AFR = mean(AF R at final step across classes) == AFR_t[-1]
        afr_mean_final = float(AFR_t[-1]) if not np.isnan(AFR_t[-1]) else float(np.nanmean(AFR_t[np.where(~np.isnan(AFR_t))])) if np.any(~np.isnan(AFR_t)) else np.nan
        cf_mean_final  = float(CF_t[-1])  if not np.isnan(CF_t[-1]) else float(np.nanmean(CF_t[np.where(~np.isnan(CF_t))])) if np.any(~np.isnan(CF_t)) else np.nan

        # plot total acc curve
        ax1.plot(steps, total_acc_list, marker='o', label=label)
        # plot AFR(t) and CF(t)
        ax2.plot(steps, AFR_t, marker='s', label=label)
        ax3.plot(steps, CF_t, marker='^', label=label)

        # table row (use final total_acc from last step if available)
        final_total = float(total_acc_list[-1]) if total_acc_list.size>0 else np.nan
        table_rows.append((label, final_total, afr_mean_final, cf_mean_final))

        # prepare per-class csv and fill per_class_stats for CSV
        per_class_stats = {}
        for idx, c in enumerate(class_ids):
            row = acc_mat[idx,:]
            if np.all(np.isnan(row)):
                continue
            maxacc = float(np.nanmax(row))
            lastacc = float(row[-1]) if not np.isnan(row[-1]) else float(row[np.where(~np.isnan(row))[0][-1]])
            firstacc = first_acc_map.get(c, np.nan)
            afr_final = maxacc - lastacc if (not np.isnan(maxacc) and not np.isnan(lastacc)) else np.nan
            cf_final = firstacc - lastacc if (not np.isnan(firstacc) and not np.isnan(lastacc)) else np.nan
            per_class_stats[c] = {'AFR_final': afr_final, 'CF_final': cf_final}
        # export csv for debug
        # export_per_class_csv(out_dir='per_class_csv', label=label.replace(' ','_'), class_ids=class_ids, acc_mat=acc_mat, first_acc_map=first_acc_map, per_class_stats=per_class_stats)

    # style & labels
    ax1.set_title('Overall Accuracy across Incremental Steps', pad=6)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=False)

    ax2.set_title('Average Forgetting Rate (AFR) across Steps', pad=6)
    ax2.set_ylabel('AFR (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', frameon=False)

    ax3.set_title('Cumulative Forget (CF) across Steps', pad=6)
    ax3.set_ylabel('CF (%)')
    ax3.set_xlabel('Incremental Step')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    print(f"[✓] Vector PDF saved: {os.path.abspath(out_pdf)}")

    # print table
    print("\n===== Incremental Learning Summary =====")
    print(f"{'Configuration':35s} | {'Acc (%)':>8s} | {'AFR (%)':>8s} | {'CF (%)':>8s}")
    print("-" * 70)
    for row in table_rows:
        print(f"{row[0]:35s} | {row[1]:8.2f} | {row[2]:8.2f} | {row[3]:8.2f}")

    # LaTeX rows
    print("\nLaTeX rows (Configuration & Acc & AFR & CF):")
    for row in table_rows:
        print(f"{row[0]} & {row[1]:.2f} & {row[2]:.2f} & {row[3]:.2f} \\\\")

# -----------------------------
# If run as main: example invocation
# -----------------------------
if __name__ == "__main__":
    file_paths = [
        'model-wo-feat.json',
        'model-wo-project.json',
        'model-wo-reweight.json',
        'model-wo-reweight2.json',
        'incremental_metrics.json'
    ]
    labels = [
        'w/o Feature Distillation',
        'w/o Contrastive Module',
        'w/o Weight Balancing',
        'w/o Open-set Pretraining',
        'Full Model (Ours)'
    ]
    plot_multiple_jsons(file_paths, labels, out_pdf='ablation_compare.pdf')
