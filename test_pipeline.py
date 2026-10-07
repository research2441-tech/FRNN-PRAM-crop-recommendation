"""Unit and smoke tests. Run with:  python -m pytest -q test_pipeline.py"""
import numpy as np
import pandas as pd
import pytest

import config as C
from make_synthetic_data import make
from preprocessing import IqN, add_sdir, add_tshr, build_sequences, prepare_frame, MedianImputer
from mabc import MABC, dominates
from frnn import FRNN
from pram import solve_pram, evaluate_allocation, summarise
from evaluation import corrected_resampled_t, fold_metrics


@pytest.fixture(scope="module")
def small():
    return prepare_frame(make(n_series=8, n_months=15, seed=1)).reset_index(drop=True)


def test_iqn_uses_training_quartiles_only():
    tr = np.array([[0.], [1.], [2.], [3.], [4.]])
    te = np.array([[100.], [-100.]])
    s = IqN().fit(tr)
    out = s.transform(te)
    # values are capped at the training fences (Eq. 2), not re-estimated on test data
    assert np.allclose(out[:, 0], [(s.ub[0] - s.q1[0]) / s.iqr[0], (s.lb[0] - s.q1[0]) / s.iqr[0]])


def test_tshr_identities(small):
    d = add_tshr(MedianImputer().fit(small, ["Q_p", "Q_n", "arrivals"]).transform(small))
    assert np.allclose(d["TSHR_L"], d["SA"] - d["SU"])
    assert np.allclose(d["SU"], d["Q_s"] + d["Q_d"] + d["Q_u"])


def test_sdir_is_causal(small):
    a = add_sdir(MedianImputer().fit(small, ["arrivals"]).transform(small))
    future = small.copy()
    future.loc[future.month_index > 10, "arrivals"] *= 50     # change the future only
    b = add_sdir(MedianImputer().fit(small, ["arrivals"]).transform(future))
    m = a.month_index <= 10
    assert np.allclose(a.loc[m, "SDIR_D"], b.loc[m, "SDIR_D"])


def test_sequences_shape_and_padding(small):
    X = np.arange(len(small) * 2, dtype=float).reshape(-1, 2)
    S = build_sequences(small, X, 6)
    assert S.shape == (len(small), 6, 3)
    first = small.index[small.month_index == 1][0]
    assert S[first, :5, -1].sum() == 0 and S[first, 5, -1] == 1


def test_pareto_dominance():
    assert dominates(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.3, 0.3]))
    assert not dominates(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))


def test_mabc_returns_nonempty_mask():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6)); y = (X[:, 0] + 0.2 * rng.normal(size=150) > 0).astype(int)
    m = MABC(max_iter=4, seed=0).fit(X, y)
    assert m.mask_.any() and m.mask_[0]


def test_frnn_learns_simple_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4, 3)).astype(np.float32)
    y = (X[:, -1, 0] > 0).astype(int)
    S = np.concatenate([X, np.ones((300, 4, 1), np.float32)], -1)
    f = FRNN(max_epochs=30, n_rules=8, hidden=(16, 8), seed=0).fit(S[:200], y[:200], S[200:250], y[200:250], 2)
    acc = (f.predict(S[250:]) == y[250:]).mean()
    assert acc > 0.8


def test_pram_respects_constraints():
    p = np.array([100., 90., 80.]); Sig = np.diag([400., 100., 50.])
    w = np.array([900., 500., 400.]); l = np.array([150., 100., 90.])
    a = solve_pram(p, Sig, w, l, W=600, Lam=120, gamma=200, a_max=0.6)
    assert abs(a.sum() - 1) < 1e-6 and (a <= 0.6 + 1e-6).all()
    assert a @ w <= 600 + 1e-3 and a @ l <= 120 + 1e-3 and a @ Sig @ a <= 200 + 1e-3


def test_aer_not_above_100_for_pram():
    rng = np.random.default_rng(2)
    units = []
    for _ in range(10):
        units.append(dict(hist_returns=rng.normal(80, 8, (4, 6)), realised=rng.normal(80, 8, 6),
                          w_req=rng.uniform(250, 900, 6), l_req=rng.uniform(90, 160, 6),
                          hist_share=np.array([.34, .18, .17, .14, .10, .07]), feasible=np.ones(6, int),
                          W=700., Lam=125.))
    probs = rng.dirichlet(np.ones(6), 10)
    s = summarise(evaluate_allocation(units, probs, C.CROPS, C.PRAM, "pram"))
    assert s["AER"] <= 100.0 + 1e-6


def test_corrected_t_and_metrics():
    t, p = corrected_resampled_t(np.array([.9, .91, .92, .9, .93]), np.array([.8, .82, .81, .8, .83]))
    assert t > 0 and p < 0.05
    y = np.array([0, 1, 2, 2]); pr = np.eye(3)[[0, 1, 2, 1]]
    m = fold_metrics(y, pr, 3)
    assert abs(m["accuracy"] - 0.75) < 1e-9
