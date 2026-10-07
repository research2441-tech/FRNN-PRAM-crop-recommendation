"""Leakage-safe preprocessing: imputation, causal SDIR/TSHR indicators, IqN scaling, sequences.

Everything that depends on data statistics is split into fit (training series) and
transform (any series). SDIR and TSHR use only past and current months of the same series.
"""
import numpy as np
import pandas as pd


def add_calendar(df):
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def forward_fill_within_series(df, cols):
    """Gaps are filled from earlier months of the same series only (no look-ahead)."""
    df = df.sort_values(["series_id", "month_index"]).copy()
    df[cols] = df.groupby("series_id")[cols].ffill()
    return df


class MedianImputer:
    """Leading gaps that forward fill cannot reach take the training-fold median."""
    def fit(self, df, cols):
        self.cols, self.med = cols, df[cols].median()
        return self

    def transform(self, df):
        df = df.copy()
        df[self.cols] = df[self.cols].fillna(self.med)
        return df


def add_tshr(df):
    """Eqs. (8)-(11): T = SU/SA*100, SA = Q_p + Q_n, SU = Q_s + Q_d + Q_u, L = SA - SU."""
    df = df.copy()
    df["SA"] = df["Q_p"] + df["Q_n"]
    df["SU"] = df["Q_s"] + df["Q_d"] + df["Q_u"]
    df["TSHR_T"] = np.where(df["SA"] > 0, 100.0 * df["SU"] / df["SA"].replace(0, np.nan), 0.0)
    df["TSHR_L"] = df["SA"] - df["SU"]
    return df


def add_sdir(df):
    """Eqs. (4)-(7), computed causally inside every series.

    P_s   : arrivals summed over the elapsed months of the current season (Eq. 6)
    P_avg : mean demand of the seasons completed before the current month (Eq. 5)
    D_s   : (P_s - P_avg) / P_avg * 100 (Eq. 4); 0 when no season has been completed yet
    Z_s   : P_s / sum of completed-season demand * 100 (Eq. 7)
    """
    df = df.sort_values(["series_id", "month_index"]).copy()
    D, Z = np.zeros(len(df)), np.zeros(len(df))
    for _, g in df.groupby("series_id", sort=False):
        idx = g.index.to_numpy()
        seasons = g["season_id"].to_numpy()
        arr = g["arrivals"].to_numpy(dtype=float)
        completed, cur_season, cur_sum = [], None, 0.0
        for k, (s, a) in enumerate(zip(seasons, arr)):
            if s != cur_season:
                if cur_season is not None:
                    completed.append(cur_sum)
                cur_season, cur_sum = s, 0.0
            cur_sum += a
            pos = df.index.get_loc(idx[k])
            if completed:
                p_avg = np.mean(completed)
                D[pos] = 100.0 * (cur_sum - p_avg) / p_avg if p_avg > 0 else 0.0
                tot = np.sum(completed)
                Z[pos] = 100.0 * cur_sum / tot if tot > 0 else 0.0
    df["SDIR_D"], df["SDIR_Z"] = D, Z
    return df


class IqN:
    """Interquartile normalization (Eqs. 1-3) fitted on training rows only."""
    def __init__(self, fence=1.5):
        self.fence = fence

    def fit(self, X):
        X = np.asarray(X, float)
        self.q1 = np.nanpercentile(X, 25, axis=0)
        self.q3 = np.nanpercentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        self.iqr[self.iqr == 0] = 1.0
        self.lb = self.q1 - self.fence * self.iqr
        self.ub = self.q3 + self.fence * self.iqr
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, float), self.lb, self.ub)       # Eq. (2): cap, never delete
        return (X - self.q1) / self.iqr                             # Eq. (3)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMax:
    """Used only by the ablation rows without IqN."""
    def fit(self, X):
        X = np.asarray(X, float)
        self.lo, self.hi = X.min(0), X.max(0)
        self.rng = np.where(self.hi - self.lo == 0, 1.0, self.hi - self.lo)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.lo) / self.rng


def build_sequences(df, X, window):
    """Window of months t-w+1..t of the same series for every record t.

    Returns array (n_records, window, d + 1); the last channel is a padding mask (1 = real month).
    Records are returned in the row order of df.
    """
    n, d = X.shape
    out = np.zeros((n, window, d + 1), dtype=np.float32)
    pos = {i: k for k, i in enumerate(df.index)}
    for _, g in df.groupby("series_id", sort=False):
        g = g.sort_values("month_index")
        rows = [pos[i] for i in g.index]
        for j, r in enumerate(rows):
            start = max(0, j - window + 1)
            seq = rows[start:j + 1]
            out[r, window - len(seq):, :d] = X[seq]
            out[r, window - len(seq):, d] = 1.0
    return out


def prepare_frame(df):
    """Feature engineering that needs no fitted statistics."""
    df = add_calendar(df)
    raw = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall", "modal_price", "arrivals",
           "Q_p", "Q_n", "Q_s", "Q_d", "Q_u"]
    df = forward_fill_within_series(df, raw)
    return df
