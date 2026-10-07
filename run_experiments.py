"""Full experiment: repeated stratified group 5-fold CV, baselines, ablation, PRAM, statistics.

python run_experiments.py --data agro_dataset.csv --out results            # article protocol
python run_experiments.py --data synthetic_agro.csv --out smoke --quick    # fast smoke test

All data-dependent steps (median imputation, IqN, MABC, fuzzy rules, membership centres) are
fitted on the training series of each fold. Outputs are CSV tables that map one-to-one to
Tables 4, 5, 6, S4 and S5 of the article.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

import config as C
from preprocessing import prepare_frame, MedianImputer, add_tshr, add_sdir, IqN, MinMax, build_sequences
from mabc import MABC
from frnn import FRNN
from baselines import make_baseline
from evaluation import fold_metrics, mean_sd_ci, corrected_resampled_t, wilcoxon, per_class, macro_roc
from pram import evaluate_allocation, summarise

RAW_IMPUTE = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall", "modal_price", "arrivals",
              "Q_p", "Q_n", "Q_s", "Q_d", "Q_u"]
BASE14 = list(C.BASE_FEATURES)          # 7 soil/climate + 2 calendar + 5 market/stock


def fold_features(df, tr_idx, feats, scaler="iqn"):
    imp = MedianImputer().fit(df.loc[tr_idx], RAW_IMPUTE)
    d = add_sdir(add_tshr(imp.transform(df)))
    d = d.loc[df.index]
    X = d[feats].to_numpy(float)
    sc = IqN(C.IQN["fence"]) if scaler == "iqn" else MinMax()
    sc.fit(X[df.index.get_indexer(tr_idx)])
    return d, sc.transform(X)


def pram_units(df, idx, probs_by_row):
    """One unit per (series, season): the last month of the season in the test fold."""
    sub = df.loc[idx].copy()
    sub["_p"] = list(probs_by_row)
    units, probs = [], []
    crops = C.CROPS
    for (s, se), g in sub.groupby(["series_id", "season_id"]):
        row = g.sort_values("month_index").iloc[-1]
        hist = df[(df.series_id == s) & (df.season_id <= se - 2)]
        hist = hist.sort_values("month_index").groupby("season_id").tail(1)
        if len(hist) < C.PRAM["min_history"]:
            continue
        units.append(dict(hist_returns=hist[[f"ret_{c}" for c in crops]].to_numpy(float),
                          realised=row[[f"ret_{c}" for c in crops]].to_numpy(float),
                          w_req=row[[f"wat_{c}" for c in crops]].to_numpy(float),
                          l_req=row[[f"lab_{c}" for c in crops]].to_numpy(float),
                          hist_share=row[[f"hist_{c}" for c in crops]].to_numpy(float),
                          feasible=row[[f"feas_{c}" for c in crops]].to_numpy(int),
                          W=float(row["W_avail"]), Lam=float(row["L_avail"])))
        probs.append(row["_p"])
    return units, np.array(probs)


def tune_frnn(df, Xsel, y, tr_, va_, cfg, grid, nC, seed):
    """One-factor-at-a-time search over the FRNN grid (Table S3) on the validation series of
    the current fold only; returns the tuned configuration and window length."""
    best_cfg, best_w = dict(cfg), C.PROTOCOL["window"]
    def score(c, w):
        S = build_sequences(df, Xsel, w)
        m = FRNN(seed=seed, **c).fit(S[tr_], y[tr_], S[va_], y[va_], nC)
        return m.best_val_f1_
    best = score(best_cfg, best_w)
    for key in ["window", "hidden", "n_mf", "lr", "n_rules"]:
        for v in grid[key]:
            c, w = dict(best_cfg), best_w
            if key == "window":
                if v == best_w: continue
                w = v
            else:
                if c[key] == v: continue
                c[key] = v
            f = score(c, w)
            if f > best:
                best, best_cfg, best_w = f, c, w
    return best_cfg, best_w


def timed_proba(model, S):
    model.predict_proba(S)                                        # warm-up
    t0 = time.perf_counter()
    for _ in range(C.TIMING_RUNS):
        p = model.predict_proba(S)
    return p, 1000.0 * (time.perf_counter() - t0) / C.TIMING_RUNS


def run(args):
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    df = prepare_frame(df).reset_index(drop=True)
    y_all = df["crop_label"].map({c: i for i, c in enumerate(C.CROPS)}).to_numpy()
    groups = df["series_id"].to_numpy()
    nC = len(C.CROPS)
    reps = C.QUICK["n_repeats"] if args.quick else C.PROTOCOL["n_repeats"]
    folds = C.QUICK["n_folds"] if args.quick else C.PROTOCOL["n_folds"]
    mabc_cfg = dict(C.MABC); frnn_cfg = dict(C.FRNN); base_cfg = {k: dict(v) for k, v in C.BASELINES.items()}
    if args.quick:
        mabc_cfg.update(max_iter=C.QUICK["mabc_iter"]); frnn_cfg.update(max_epochs=C.QUICK["frnn_epochs"])
        for k in ["drqn", "dcnn"]:
            base_cfg[k].update(max_epochs=C.QUICK["frnn_epochs"])
        base_cfg["rfsvm"]["n_trees"] = base_cfg["dtrf"]["n_trees"] = C.QUICK["n_trees"]
    w = C.PROTOCOL["window"]
    models = ["DRQ-N", "RF-SVM", "DT-RF", "DCNN", "FRNN"]
    keymap = {"DRQ-N": "drqn", "RF-SVM": "rfsvm", "DT-RF": "dtrf", "DCNN": "dcnn"}
    ablations = {"FRNN only (min-max, 14 base)": dict(feats=BASE14, scaler="minmax", mabc=False, fuzzy=True),
                 "IqN + FRNN": dict(feats=BASE14, scaler="iqn", mabc=False, fuzzy=True),
                 "IqN + SDIR + TSHR + FRNN (18)": dict(feats=C.CANDIDATE_FEATURES, scaler="iqn", mabc=False, fuzzy=True),
                 "MABC + FRNN (min-max, 14 base)": dict(feats=BASE14, scaler="minmax", mabc=True, fuzzy=True),
                 "Full pipeline, plain RNN": dict(feats=C.CANDIDATE_FEATURES, scaler="iqn", mabc=True, fuzzy=False)}
    rec, pooled, sel_freq, alloc = [], {m: [] for m in models}, np.zeros(len(C.CANDIDATE_FEATURES)), {}
    for r in range(reps):
        seed = C.PROTOCOL["seeds"][r]
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(sgkf.split(df, y_all, groups)):
            gss = GroupShuffleSplit(1, test_size=C.PROTOCOL["val_fraction"], random_state=seed + k)
            tr_i, va_i = next(gss.split(tr, y_all[tr], groups[tr]))
            tr_, va_ = tr[tr_i], tr[va_i]
            if args.train_fraction < 1.0:                       # learning curves (Fig. S2)
                rs = np.random.default_rng(seed + k)
                tr_ = np.sort(rs.choice(tr_, int(round(args.train_fraction * len(tr_))), replace=False))
            # full pipeline features
            d, X = fold_features(df, df.index[tr_], C.CANDIDATE_FEATURES, "iqn")
            t0 = time.perf_counter()
            mab = MABC(seed=seed + k, **mabc_cfg).fit(X[tr_], y_all[tr_], groups[tr_])
            t_mabc = time.perf_counter() - t0
            sel_freq += mab.mask_
            fcfg, w = (frnn_cfg, C.PROTOCOL["window"]) if (args.quick or args.no_tune) else \
                tune_frnn(df, mab.transform(X), y_all, tr_, va_, frnn_cfg, C.FRNN_GRID, nC, seed + k)
            S = build_sequences(df, mab.transform(X), w)
            for m in models:
                t0 = time.perf_counter()
                if m == "FRNN":
                    model = FRNN(seed=seed + k, **fcfg).fit(S[tr_], y_all[tr_], S[va_], y_all[va_], nC)
                else:
                    model = make_baseline(m, base_cfg[keymap[m]], seed + k)
                    model.fit(S[tr_], y_all[tr_], S[va_], y_all[va_], nC)
                t_train = time.perf_counter() - t0
                proba, t_inf = timed_proba(model, S[te])
                met = fold_metrics(y_all[te], proba, nC)
                rec.append(dict(rep=r, fold=k, model=m, train_s=t_train, infer_ms=t_inf, mabc_s=t_mabc,
                                n_selected=int(mab.mask_.sum()), window=w,
                                frnn_cfg=json.dumps({kk: fcfg[kk] for kk in ["hidden", "n_mf", "lr", "n_rules"]}), **met))
                if r == 0:
                    pooled[m].append((te, proba))
                units, probs = pram_units(d, df.index[te], proba)
                if units:
                    for strat in (["pram"] + (["historical", "equal", "top1"] if m == "FRNN" else [])):
                        key = {"pram": f"PRAM+{m}", "historical": "Historical mix", "equal": "Equal share",
                               "top1": "FRNN top-1 only"}[strat]
                        alloc.setdefault(key, []).extend(evaluate_allocation(units, probs, C.CROPS, C.PRAM, strat))
                print(f"rep {r} fold {k} {m:7s} macro-F1 {met['macro_f1']:.4f} acc {met['accuracy']:.4f}", flush=True)
            if not args.skip_ablation:
                for name, cfg in ablations.items():
                    d2, X2 = fold_features(df, df.index[tr_], cfg["feats"], cfg["scaler"])
                    if cfg["mabc"]:
                        X2 = MABC(seed=seed + k, **mabc_cfg).fit(X2[tr_], y_all[tr_], groups[tr_]).transform(X2)
                    S2 = build_sequences(df, X2, w)
                    mdl = FRNN(seed=seed + k, use_fuzzy=cfg["fuzzy"], **fcfg).fit(S2[tr_], y_all[tr_], S2[va_], y_all[va_], nC)
                    met = fold_metrics(y_all[te], mdl.predict_proba(S2[te]), nC)
                    rec.append(dict(rep=r, fold=k, model="ABL: " + name, **met))
    R = pd.DataFrame(rec); R.to_csv(os.path.join(args.out, "results_per_fold.csv"), index=False)
    # Table 4 / Table 5
    rows = []
    for m, g in R.groupby("model", sort=False):
        row = dict(model=m)
        for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "kappa", "auc_ovr_macro", "infer_ms", "train_s"]:
            if col in g and g[col].notna().any():
                mu, sd, lo, hi = mean_sd_ci(g[col].dropna())
                row[col] = f"{mu:.4f} ± {sd:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "table4_table5_summary.csv"), index=False)
    # statistics
    st = []
    ref = R[R.model == "FRNN"].sort_values(["rep", "fold"])["macro_f1"].to_numpy()
    for m in ["DRQ-N", "RF-SVM", "DT-RF", "DCNN"]:
        other = R[R.model == m].sort_values(["rep", "fold"])["macro_f1"].to_numpy()
        t, p = corrected_resampled_t(ref, other, 1.0 / (folds - 1))
        st.append(dict(comparison=f"FRNN vs {m}", mean_gain=float((ref - other).mean()), t=t, p=p, wilcoxon_p=wilcoxon(ref, other)))
    pd.DataFrame(st).to_csv(os.path.join(args.out, "stats_tests.csv"), index=False)
    # pooled confusion matrix and per-class (repetition 1)
    roc = {}
    for m in models:
        te_all = np.concatenate([t for t, _ in pooled[m]]); pr_all = np.concatenate([p for _, p in pooled[m]])
        P, Rc, F, Sup, cm = per_class(y_all[te_all], pr_all.argmax(1), nC)
        if m == "FRNN":
            pd.DataFrame(dict(crop=C.CROPS, records=Sup, precision=P, recall=Rc, f1=F)).to_csv(
                os.path.join(args.out, "tableS4_perclass_FRNN.csv"), index=False)
            pd.DataFrame(cm, index=C.CROPS, columns=C.CROPS).to_csv(os.path.join(args.out, "confusion_matrix_FRNN.csv"))
        roc[m] = macro_roc(y_all[te_all], pr_all, nC)[1]
    np.savez(os.path.join(args.out, "roc_curves.npz"), grid=np.linspace(0, 1, 401), **roc)
    pd.DataFrame(dict(feature=C.CANDIDATE_FEATURES, folds_selected=sel_freq.astype(int))).to_csv(
        os.path.join(args.out, "mabc_selection_frequency.csv"), index=False)
    # Table 6
    if alloc:
        pd.DataFrame([dict(strategy=k, **summarise(v)) for k, v in alloc.items()]).to_csv(
            os.path.join(args.out, "table6_allocation.csv"), index=False)
    print("finished; tables written to", args.out)


def chronological(args):
    """Table S5: train on months <= cut, test on later months; every step refitted."""
    df = prepare_frame(pd.read_csv(args.data)).reset_index(drop=True)
    y = df["crop_label"].map({c: i for i, c in enumerate(C.CROPS)}).to_numpy()
    cut = args.cut
    # training months 1..cut, a one-season gap, test months > cut + gap (labels look one season ahead)
    tr_all = np.flatnonzero(df.month_index <= cut); te = np.flatnonzero(df.month_index > cut + args.gap)
    gss = GroupShuffleSplit(1, test_size=C.PROTOCOL["val_fraction"], random_state=0)
    a, b = next(gss.split(tr_all, y[tr_all], df.series_id.to_numpy()[tr_all]))
    tr, va = tr_all[a], tr_all[b]
    d, X = fold_features(df, df.index[tr], C.CANDIDATE_FEATURES, "iqn")
    mcfg = dict(C.MABC); fcfg = dict(C.FRNN); bcfg = {k: dict(v) for k, v in C.BASELINES.items()}
    if args.quick:
        mcfg.update(max_iter=C.QUICK["mabc_iter"]); fcfg.update(max_epochs=C.QUICK["frnn_epochs"])
        for k in ["drqn", "dcnn"]:
            bcfg[k].update(max_epochs=C.QUICK["frnn_epochs"])
    mab = MABC(seed=0, **mcfg).fit(X[tr], y[tr], df.series_id.to_numpy()[tr])
    S = build_sequences(df, mab.transform(X), C.PROTOCOL["window"])
    out = []
    for m, key in [("DRQ-N", "drqn"), ("RF-SVM", "rfsvm"), ("DT-RF", "dtrf"), ("DCNN", "dcnn"), ("FRNN", None)]:
        model = FRNN(seed=0, **fcfg) if m == "FRNN" else make_baseline(m, bcfg[key], 0)
        model.fit(S[tr], y[tr], S[va], y[va], len(C.CROPS))
        out.append(dict(model=m, **fold_metrics(y[te], model.predict_proba(S[te]), len(C.CROPS))))
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(out).to_csv(os.path.join(args.out, "tableS5_chronological.csv"), index=False)
    print(pd.DataFrame(out)[["model", "accuracy", "macro_f1"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="smoke-test settings, not for reporting")
    ap.add_argument("--skip_ablation", action="store_true")
    ap.add_argument("--chronological", action="store_true")
    ap.add_argument("--cut", type=int, default=30)
    ap.add_argument("--gap", type=int, default=3)
    ap.add_argument("--no_tune", action="store_true", help="use config.FRNN without per-fold tuning")
    ap.add_argument("--train_fraction", type=float, default=1.0, help="0.25/0.5/0.75 for learning curves")
    a = ap.parse_args()
    chronological(a) if a.chronological else run(a)
