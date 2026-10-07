"""Portfolio resource allocation model (PRAM), Eqs. (26)-(30).

For a unit-season, the candidate crops are those with classifier probability >= min_prob
(at most max_candidates). Expected net returns and their covariance come from the seasons
that precede the evaluated season in the same unit. PRAM maximises expected return subject to
risk <= gamma (risk of the unit's historical mix), land shares summing to one with an upper cap,
and water and labour availability (Eq. 29). AER (Eq. 30) compares realised returns with the
oracle that solves Eq. (29) with realised returns known.
"""
import numpy as np
from scipy.optimize import minimize


def solve_pram(p_bar, Sigma, w_req, l_req, W, Lam, gamma, a_max=0.6, tol=1e-8, max_iter=200, minimise_risk_theta=None):
    """Solves Eq. (29) (or Eq. S2) with SLSQP from several starting points.

    If no start satisfies every constraint, the water, labour and risk limits are relaxed by the
    smallest common factor that makes the problem feasible, so that a solution always exists."""
    p_bar, Sigma = np.asarray(p_bar, float), np.asarray(Sigma, float)
    w_req, l_req = np.asarray(w_req, float), np.asarray(l_req, float)
    k = len(p_bar)
    a_max = max(a_max, 1.0 / k)                                   # keep the simplex feasible
    starts = [np.full(k, 1.0 / k)] + [np.eye(k)[i] * a_max + (1 - a_max) / max(k - 1, 1) * (1 - np.eye(k)[i])
                                      for i in range(k)] if k > 1 else [np.ones(1)]
    for relax in [1.0, 1.05, 1.15, 1.3, 1.6, 2.0, 3.0]:
        Wr, Lr, gr = W * relax, Lam * relax, gamma * relax ** 2
        cons = [{"type": "eq", "fun": lambda a: a.sum() - 1.0},
                {"type": "ineq", "fun": lambda a, Wr=Wr: Wr - a @ w_req},
                {"type": "ineq", "fun": lambda a, Lr=Lr: Lr - a @ l_req}]
        if minimise_risk_theta is None:                           # Eq. (29)
            cons.append({"type": "ineq", "fun": lambda a, gr=gr: gr - a @ Sigma @ a})
            obj = lambda a: -(a @ p_bar)
        else:                                                     # Eq. (S2)
            cons.append({"type": "ineq", "fun": lambda a: a @ p_bar - minimise_risk_theta})
            obj = lambda a: a @ Sigma @ a
        best = None
        for a0 in starts:
            res = minimize(obj, a0, method="SLSQP", bounds=[(0, a_max)] * k, constraints=cons,
                           options={"ftol": tol, "maxiter": max_iter})
            a = np.clip(res.x, 0, None); a = a / a.sum()
            ok = all(c["fun"](a) >= -1e-6 if c["type"] == "ineq" else abs(c["fun"](a)) < 1e-6 for c in cons)
            if ok and (best is None or obj(a) < obj(best)):
                best = a
        if best is not None:
            return best
    return np.full(k, 1.0 / k)


def candidates(prob, crops, min_prob=0.10, max_candidates=3):
    order = np.argsort(prob)[::-1][:max_candidates]
    keep = [i for i in order if prob[i] >= min_prob] or [order[0]]
    return keep


def evaluate_allocation(units, probs, crops, cfg, strategy="pram"):
    """units: list of dicts with keys
         hist_returns (n_prev, C) net returns of previous seasons,
         realised (C,) net returns of the evaluated season,
         w_req (C,), l_req (C,), W, Lam, hist_share (C,)
       probs: (n_units, C) classifier probabilities (ignored by 'historical' and 'equal').
       Returns per-unit realised return, water, labour and the oracle return."""
    out = []
    for u, pr in zip(units, probs):
        H = np.asarray(u["hist_returns"], float)
        C = H.shape[1]
        Sigma_full = np.cov(H.T) if len(H) > 1 else np.eye(C)
        hs = np.asarray(u["hist_share"], float)
        gamma = float(hs @ Sigma_full @ hs)
        if strategy == "historical":
            a_full = hs
        elif strategy == "equal":
            idx = [c for c in range(C) if u["feasible"][c]]
            a_full = np.zeros(C); a_full[idx] = 1.0 / len(idx)
        elif strategy == "top1":
            a_full = np.zeros(C); a_full[int(np.argmax(pr))] = 1.0
        else:
            idx = candidates(pr, crops, cfg["min_prob"], cfg["max_candidates"])
            a = solve_pram(H.mean(0)[idx], Sigma_full[np.ix_(idx, idx)], np.asarray(u["w_req"])[idx],
                           np.asarray(u["l_req"])[idx], u["W"], u["Lam"], gamma, cfg["a_max"])
            a_full = np.zeros(C); a_full[idx] = a
        # oracle: Eq. (29) with realised returns over all feasible crops
        feas = [c for c in range(C) if u["feasible"][c]]
        a_or = solve_pram(np.asarray(u["realised"])[feas], Sigma_full[np.ix_(feas, feas)],
                          np.asarray(u["w_req"])[feas], np.asarray(u["l_req"])[feas], u["W"], u["Lam"],
                          gamma, cfg["a_max"])
        oracle = float(a_or @ np.asarray(u["realised"])[feas])
        out.append(dict(ret=float(a_full @ u["realised"]), water=float(a_full @ u["w_req"]),
                        labour=float(a_full @ u["l_req"]), oracle=oracle, shares=a_full))
    return out


def summarise(rows):
    ret = np.array([r["ret"] for r in rows]); orc = np.array([r["oracle"] for r in rows])
    return dict(return_mean=ret.mean(), risk_sd=ret.std(ddof=1), water=np.mean([r["water"] for r in rows]),
                labour=np.mean([r["labour"] for r in rows]), AER=100.0 * ret.sum() / orc.sum())   # Eq. (30)
