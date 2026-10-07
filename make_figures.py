"""Draws Fig. 2 (ROC + confusion matrix), Fig. 3a (per-fold macro-F1) and Fig. S3 from the
outputs of run_experiments.py.   python make_figures.py --results results"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["DRQ-N", "RF-SVM", "DT-RF", "DCNN", "FRNN"]
COL = {"FRNN": "#2a78d6", "DCNN": "#eb6834", "DT-RF": "#1baf7a", "RF-SVM": "#eda100", "DRQ-N": "#e87ba4"}
LS = {"FRNN": "-", "DCNN": "--", "DT-RF": "-.", "RF-SVM": ":", "DRQ-N": (0, (5, 1, 1, 1))}
plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42})


def main(res):
    R = pd.read_csv(os.path.join(res, "results_per_fold.csv"))
    roc = np.load(os.path.join(res, "roc_curves.npz"))
    cm = pd.read_csv(os.path.join(res, "confusion_matrix_FRNN.csv"), index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(6.85, 2.5))
    for m in MODELS[::-1]:
        auc = R[R.model == m]["auc_ovr_macro"].mean()
        ax[0].plot(roc["grid"], roc[m], color=COL[m], ls=LS[m], lw=1.2, label=f"{m}  AUC {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="grey", lw=.6); ax[0].legend(loc="lower right", fontsize=6)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate (macro OvR)")
    v = cm.to_numpy(); vn = v / v.sum(1, keepdims=True)
    ax[1].imshow(vn, cmap="Blues", vmin=0, vmax=1)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax[1].text(j, i, v[i, j], ha="center", va="center", fontsize=6.5, color="white" if vn[i, j] > .55 else "black")
    ax[1].set_xticks(range(len(cm))); ax[1].set_xticklabels(cm.columns, rotation=35, ha="right")
    ax[1].set_yticks(range(len(cm))); ax[1].set_yticklabels(cm.index)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig2_roc_cm.png"), dpi=300)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    data = [100 * R[R.model == m]["macro_f1"].to_numpy() for m in MODELS]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for b, m in zip(bp["boxes"], MODELS):
        b.set_facecolor(COL[m]); b.set_alpha(.55)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(MODELS); ax.set_ylabel("Macro-F1 per test fold (%)")
    fig.tight_layout(); fig.savefig(os.path.join(res, "fig3a_folds.png"), dpi=300)
    f = pd.read_csv(os.path.join(res, "mabc_selection_frequency.csv")).sort_values("folds_selected")
    fig, ax = plt.subplots(figsize=(3.4, 2.9)); ax.barh(f.feature, f.folds_selected, color="#2a78d6")
    ax.set_xlabel("Folds in which MABC kept the feature"); fig.tight_layout()
    fig.savefig(os.path.join(res, "figS3_mabc_frequency.png"), dpi=300)
    print("figures written to", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); main(ap.parse_args().results)
