"""Metrics, confidence intervals and paired tests (Eqs. 31-33, Table S2)."""
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize


def fold_metrics(y, proba, n_classes):
    yhat = proba.argmax(1)
    P, R, F, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    Pw, Rw, Fw, _ = precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    except ValueError:
        auc = np.nan
    return dict(accuracy=accuracy_score(y, yhat), macro_precision=P, macro_recall=R, macro_f1=F,
                weighted_f1=Fw, kappa=cohen_kappa_score(y, yhat), auc_ovr_macro=auc)


def mean_sd_ci(x, level=0.95):
    x = np.asarray(x, float); n = len(x); m = x.mean(); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.5 + level / 2, n - 1) * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, m - h, m + h


def corrected_resampled_t(a, b, test_train_ratio=0.25):
    """Nadeau-Bengio corrected resampled t-test, Eq. (33)."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.inf, 0.0
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def wilcoxon(a, b):
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return np.nan


def macro_roc(y, proba, n_classes, grid=np.linspace(0, 1, 401)):
    Y = label_binarize(y, classes=list(range(n_classes)))
    tprs = []
    for c in range(n_classes):
        if Y[:, c].sum() == 0:
            continue
        f, t, _ = roc_curve(Y[:, c], proba[:, c]); tprs.append(np.interp(grid, f, t))
    return grid, np.mean(tprs, 0)


def per_class(y, yhat, n_classes):
    P, R, F, S = precision_recall_fscore_support(y, yhat, labels=list(range(n_classes)), zero_division=0)
    return P, R, F, S, confusion_matrix(y, yhat, labels=list(range(n_classes)))
