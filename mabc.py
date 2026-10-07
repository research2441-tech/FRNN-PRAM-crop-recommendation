"""Multi-objective ant-bee colony (MABC) feature selection, Eqs. (12)-(19).

Decision variable: binary mask b over d candidate features. Objectives (minimised):
  f1 = 1 - inner macro-F1 (l2 logistic regression, inner 3-fold grouped by series)
  f2 = |b| / d
  f3 = mean absolute Spearman correlation among selected features
Ants build masks from inclusion/exclusion pheromones (Eq. 13); employed bees perturb
continuous positions (Eq. 15); onlookers pick sources by fitness (Eq. 16); the Pareto archive
(Eq. 17) reinforces pheromones (Eq. 14); the final mask minimises the weighted score (Eq. 18);
the search stops when the generational distance between fronts (Eq. 19) stays small.
Run it on the training series of one fold only.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)                       # Eq. (17)


def generational_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    A, B = np.asarray(A), np.asarray(B)
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.sqrt((d ** 2).sum()) / len(A))               # Eq. (19)


class MABC:
    def __init__(self, colony=30, n_ants=15, n_employed=15, n_onlookers=15, max_iter=100, rho=0.10,
                 alpha=1.0, beta=2.0, scout_limit=20, weights=(0.70, 0.15, 0.15), gd_tol=1e-3,
                 gd_patience=10, inner_folds=3, lr_C=1.0, tau0=1.0, seed=0, verbose=False, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    # ---------- objectives, Eq. (12)
    def _objectives(self, b):
        key = b.tobytes()
        if key in self.cache:
            return self.cache[key]
        sel = np.flatnonzero(b)
        if len(sel) == 0:
            val = np.array([1.0, 0.0, 0.0])
        else:
            if self.groups is not None:
                splits = StratifiedGroupKFold(self.inner_folds, shuffle=True, random_state=self.seed).split(self.X, self.y, self.groups)
            else:
                splits = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed).split(self.X, self.y)
            f1s = []
            for tr, va in splits:
                clf = LogisticRegression(C=self.lr_C, max_iter=500)
                clf.fit(self.X[tr][:, sel], self.y[tr])
                f1s.append(f1_score(self.y[va], clf.predict(self.X[va][:, sel]), average="macro"))
            f1 = 1.0 - float(np.mean(f1s))
            f2 = len(sel) / self.d
            f3 = 0.0
            if len(sel) > 1:
                r = np.abs(self.rho_mat[np.ix_(sel, sel)])
                f3 = (r.sum() - len(sel)) / (len(sel) * (len(sel) - 1))
            val = np.array([f1, f2, f3])
        self.cache[key] = val
        return val

    def _scalar(self, F):
        """Eq. (18): weighted sum of objectives min-max normalised over the archive."""
        A = np.array([a[1] for a in self.archive]) if self.archive else F[None]
        lo, hi = A.min(0), A.max(0)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        return float(np.dot(self.weights, (F - lo) / rng))

    def _update_archive(self, b, F):
        if any(dominates(a[1], F) or np.array_equal(a[1], F) for a in self.archive):
            return
        self.archive = [a for a in self.archive if not dominates(F, a[1])] + [(b.copy(), F)]

    def fit(self, X, y, groups=None):
        """groups: series identifiers, so that the inner folds are grouped by series."""
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        self.groups = None if groups is None else np.asarray(groups)
        self.d = X.shape[1]
        self.rho_mat = np.nan_to_num(spearmanr(self.X).correlation if self.d > 1 else np.ones((1, 1)))
        if np.ndim(self.rho_mat) == 0:
            self.rho_mat = np.array([[1.0, self.rho_mat], [self.rho_mat, 1.0]])
        mi = mutual_info_classif(self.X, self.y, random_state=self.seed)
        self.eta = np.clip(mi / (mi.max() + 1e-12), 0.05, 0.95)
        tau1 = np.full(self.d, self.tau0); tau0 = np.full(self.d, self.tau0)
        self.archive = []
        pos = self.rng.uniform(0, 1, (self.n_employed, self.d))       # bee food sources
        trials = np.zeros(self.n_employed, int)
        to_mask = lambda p: (p > 0.5).astype(np.int8)
        fronts, stall = [], 0
        for it in range(self.max_iter):
            # ants, Eq. (13)
            num = tau1 ** self.alpha * self.eta ** self.beta
            pi = num / (num + tau0 ** self.alpha * (1 - self.eta) ** self.beta)
            for _ in range(self.n_ants):
                b = (self.rng.uniform(size=self.d) < pi).astype(np.int8)
                self._update_archive(b, self._objectives(b))
            # employed bees, Eq. (15)
            for x in range(self.n_employed):
                s = self.rng.choice([k for k in range(self.n_employed) if k != x])
                y_ = self.rng.integers(self.d)
                cand = pos[x].copy()
                cand[y_] = np.clip(pos[x, y_] + self.rng.uniform(-1, 1) * (pos[x, y_] - pos[s, y_]), 0, 1)
                Fc, Fx = self._objectives(to_mask(cand)), self._objectives(to_mask(pos[x]))
                self._update_archive(to_mask(cand), Fc)
                if self._scalar(Fc) < self._scalar(Fx):
                    pos[x], trials[x] = cand, 0
                else:
                    trials[x] += 1
            # onlookers, Eq. (16)
            fit = np.array([1.0 / (1.0 + self._scalar(self._objectives(to_mask(p)))) for p in pos])
            prob = fit / fit.sum()
            for _ in range(self.n_onlookers):
                x = self.rng.choice(self.n_employed, p=prob)
                s = self.rng.choice([k for k in range(self.n_employed) if k != x])
                y_ = self.rng.integers(self.d)
                cand = pos[x].copy()
                cand[y_] = np.clip(pos[x, y_] + self.rng.uniform(-1, 1) * (pos[x, y_] - pos[s, y_]), 0, 1)
                Fc = self._objectives(to_mask(cand)); self._update_archive(to_mask(cand), Fc)
                if self._scalar(Fc) < self._scalar(self._objectives(to_mask(pos[x]))):
                    pos[x], trials[x] = cand, 0
            # scouts
            for x in np.flatnonzero(trials > self.scout_limit):
                pos[x], trials[x] = self.rng.uniform(0, 1, self.d), 0
            # pheromone update, Eq. (14)
            dep1, dep0 = np.zeros(self.d), np.zeros(self.d)
            for b, F in self.archive:
                w = 1.0 / (1.0 + self._scalar(F))
                dep1 += w * (b == 1); dep0 += w * (b == 0)
            tau1 = (1 - self.rho) * tau1 + dep1
            tau0 = (1 - self.rho) * tau0 + dep0
            # convergence, Eq. (19)
            front = [F for _, F in self.archive]
            if fronts:
                gd = generational_distance(front, fronts[-1])
                stall = stall + 1 if gd < self.gd_tol else 0
                if self.verbose:
                    print(f"iter {it:3d} archive {len(front):3d} GD {gd:.5f}")
                if stall >= self.gd_patience:
                    break
            fronts.append(front)
        self.n_iter_ = it + 1
        scores = [self._scalar(F) for _, F in self.archive]
        best = self.archive[int(np.argmin(scores))]
        self.mask_ = best[0].astype(bool)
        if not self.mask_.any():
            self.mask_[np.argmax(self.eta)] = True
        self.objectives_ = best[1]
        self.front_ = np.array([F for _, F in self.archive])
        return self

    def transform(self, X):
        return np.asarray(X)[..., self.mask_]
