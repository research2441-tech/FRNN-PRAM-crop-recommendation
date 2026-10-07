"""Re-implemented baselines trained on the same folds, features and MABC subset as the FRNN.

DRQ-N  : LSTM with a Q-head, epsilon-greedy, reward +1/-1 (single-step episodes)
RF-SVM : random-forest ranking of flattened-window inputs + RBF-SVM
DT-RF  : decision tree and random forest stacked by logistic regression
DCNN   : 1-D convolutional network over the window
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def flat(S):
    return S.reshape(len(S), -1)


class RFSVM:
    def __init__(self, n_trees=300, top_frac=0.75, C=10.0, gamma="scale", seed=0, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def fit(self, S_tr, y_tr, *a, **k):
        X = flat(S_tr)
        rf = RandomForestClassifier(self.n_trees, random_state=self.seed, n_jobs=-1).fit(X, y_tr)
        k = max(1, int(round(self.top_frac * X.shape[1])))
        self.idx = np.argsort(rf.feature_importances_)[::-1][:k]
        self.svm = SVC(C=self.C, gamma=self.gamma, probability=True, class_weight="balanced",
                       random_state=self.seed).fit(X[:, self.idx], y_tr)
        return self

    def predict_proba(self, S):
        return self.svm.predict_proba(flat(S)[:, self.idx])


class DTRF:
    def __init__(self, depth=8, n_trees=300, seed=0, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def fit(self, S_tr, y_tr, *a, **k):
        self.m = StackingClassifier(
            [("dt", DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed)),
             ("rf", RandomForestClassifier(self.n_trees, max_features="sqrt", random_state=self.seed, n_jobs=-1))],
            final_estimator=LogisticRegression(max_iter=1000), stack_method="predict_proba", cv=5)
        self.m.fit(flat(S_tr), y_tr)
        return self

    def predict_proba(self, S):
        return self.m.predict_proba(flat(S))


class _TorchModel:
    """Shared training loop with early stopping on validation macro-F1."""
    def _train(self, net, S_tr, y_tr, S_va, y_va, step):
        torch.manual_seed(self.seed)
        self.net = net
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        Xt = torch.tensor(S_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
        best, state, wait = -1, None, 0
        for ep in range(self.max_epochs):
            net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad(); loss = step(Xt[b], yt[b], ep); loss.backward(); opt.step()
            f1 = f1_score(y_va, self.predict_proba(S_va).argmax(1), average="macro")
            if f1 > best + 1e-4:
                best, state, wait = f1, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(state)
        return self

    @torch.no_grad()
    def predict_proba(self, S):
        self.net.eval()
        return torch.softmax(self.net(torch.tensor(S, dtype=torch.float32)), -1).numpy()


class DCNN(_TorchModel):
    def __init__(self, filters=(32, 64), kernel=3, dense=64, dropout=0.3, lr=1e-3, batch=64,
                 max_epochs=150, patience=15, seed=0, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def fit(self, S_tr, y_tr, S_va, y_va, n_classes):
        c = S_tr.shape[-1]
        net = nn.Sequential(
            _Perm(), nn.Conv1d(c, self.filters[0], self.kernel, padding=1), nn.ReLU(),
            nn.Conv1d(self.filters[0], self.filters[1], self.kernel, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.filters[1], self.dense), nn.ReLU(),
            nn.Dropout(self.dropout), nn.Linear(self.dense, n_classes))
        lossf = nn.CrossEntropyLoss()
        return self._train(net, S_tr, y_tr, S_va, y_va, lambda x, y, ep: lossf(net(x), y))


class _Perm(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class _QNet(nn.Module):
    def __init__(self, c, hidden, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, batch_first=True); self.q = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h, _ = self.lstm(x); return self.q(h[:, -1])


class DRQN(_TorchModel):
    def __init__(self, hidden=64, eps_start=1.0, eps_end=0.05, eps_epochs=50, lr=1e-3, batch=64,
                 max_epochs=150, patience=15, seed=0, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def fit(self, S_tr, y_tr, S_va, y_va, n_classes):
        net = _QNet(S_tr.shape[-1], self.hidden, n_classes)
        rng = np.random.default_rng(self.seed)

        def step(x, y, ep):
            q = net(x)
            eps = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * ep / self.eps_epochs)
            greedy = q.argmax(1)
            rand = torch.tensor(rng.integers(0, n_classes, len(y)))
            a = torch.where(torch.tensor(rng.uniform(size=len(y)) < eps), rand, greedy)
            r = torch.where(a == y, 1.0, -1.0)                     # single-step reward, gamma = 0
            return ((q.gather(1, a[:, None]).squeeze(1) - r) ** 2).mean()
        return self._train(net, S_tr, y_tr, S_va, y_va, step)

    @torch.no_grad()
    def predict_proba(self, S):
        self.net.eval()
        return torch.softmax(self.net(torch.tensor(S, dtype=torch.float32)) * 5.0, -1).numpy()


def make_baseline(name, cfg, seed):
    return {"DRQ-N": DRQN, "RF-SVM": RFSVM, "DT-RF": DTRF, "DCNN": DCNN}[name](seed=seed, **cfg)
