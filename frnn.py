"""Fuzzified recurrent neural network (FRNN), Eqs. (20)-(25).

Input: sequences (n, w, d+1); the last channel is the padding mask from build_sequences.
  Gaussian fuzzification (3 MFs per feature, centres/widths learnable)      Eq. (20)
  two stacked tanh RNN layers on [crisp ; fuzzified] inputs                  Eq. (21)
  Wang-Mendel rule layer, firing strength = min of memberships at month T   Eq. (22)
  centroid defuzzification into crop scores                                 Eq. (23)
  softmax over recurrent + rule scores                                      Eq. (24)
  class-weighted cross-entropy + weight decay                               Eq. (25)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def wang_mendel_rules(X_last, y, centres, n_rules, n_classes):
    """Generate rules from training examples: each sample votes for the antecedent made of the
    best-matching linguistic term of every feature; antecedents are ranked by support."""
    sig = np.maximum((centres[:, -1] - centres[:, 0]) / max(centres.shape[1] - 1, 1), 1e-3)
    mu = np.exp(-((X_last[:, :, None] - centres[None]) ** 2) / (2 * sig[None, :, None] ** 2))
    terms = mu.argmax(-1)                                        # (n, d)
    table = {}
    for t, c, s in zip(map(tuple, terms), y, mu.max(-1).prod(1)):
        e = table.setdefault(t, np.zeros(n_classes))
        e[c] += 1.0 + s
    ranked = sorted(table.items(), key=lambda kv: -kv[1].sum())[:n_rules]
    ante = np.array([k for k, _ in ranked], dtype=np.int64)
    cons = np.array([v / v.sum() for _, v in ranked], dtype=np.float32)
    if len(ante) < n_rules:                                       # pad with copies if data are scarce
        reps = n_rules - len(ante)
        ante = np.concatenate([ante, ante[np.arange(reps) % len(ante)]])
        cons = np.concatenate([cons, cons[np.arange(reps) % len(cons)]])
    return ante, cons


class FRNNNet(nn.Module):
    def __init__(self, d, n_classes, centres, widths, ante, cons, hidden=(64, 32), dropout=0.2):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(centres, dtype=torch.float32))         # (d, M)
        self.log_s = nn.Parameter(torch.log(torch.tensor(widths, dtype=torch.float32)))
        M = centres.shape[1]
        self.rnn1 = nn.RNN(d + d * M + 1, hidden[0], batch_first=True, nonlinearity="tanh")
        self.rnn2 = nn.RNN(hidden[0], hidden[1], batch_first=True, nonlinearity="tanh")
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden[1], n_classes)
        self.register_buffer("ante", torch.tensor(ante, dtype=torch.long))       # (R, d)
        self.G = nn.Parameter(torch.tensor(np.log(cons + 1e-3), dtype=torch.float32))  # rule consequents g_rc
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
        nn.init.xavier_uniform_(self.out.weight)

    def memberships(self, x):                                     # x (..., d)
        s = torch.exp(self.log_s)
        return torch.exp(-((x[..., None] - self.g) ** 2) / (2 * s ** 2))          # Eq. (20)

    def forward(self, seq):
        x, mask = seq[..., :-1], seq[..., -1:]
        mu = self.memberships(x)                                  # (n, w, d, M)
        z = torch.cat([x, mu.flatten(-2), mask], -1) * mask      # padded months stay zero
        h1, _ = self.rnn1(z)                                      # Eq. (21), layer 1
        h2, _ = self.rnn2(self.drop(h1))                          # Eq. (21), layer 2
        hT = self.drop(h2[:, -1])
        muT = mu[:, -1]                                           # (n, d, M) at last month T
        d_idx = torch.arange(muT.shape[1], device=seq.device)
        sel = muT[:, d_idx[None, :], self.ante]                   # (n, R, d)
        P = sel.min(-1).values                                    # Eq. (22) firing strength
        Pbar = P / (P.sum(-1, keepdim=True) + 1e-8)
        s = Pbar @ self.G                                         # Eq. (23) centroid defuzzification
        return self.out(hT) + s, Pbar                             # Eq. (24) logits (softmax in loss)


class FRNN:
    def __init__(self, n_mf=3, hidden=(64, 32), dropout=0.2, n_rules=24, lr=1e-3, batch=64,
                 max_epochs=150, patience=15, weight_decay=1e-4, clip=1.0, seed=0, device="cpu",
                 use_fuzzy=True, **_):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def _init_fuzzy(self, Xtr_last):
        q = np.linspace(25, 75, self.n_mf) if self.n_mf > 1 else np.array([50])
        centres = np.percentile(Xtr_last, q, axis=0).T            # (d, M) from training quartiles
        iqr = np.percentile(Xtr_last, 75, 0) - np.percentile(Xtr_last, 25, 0)
        widths = np.repeat(np.maximum(iqr / 1.349, 1e-2)[:, None], self.n_mf, 1)
        return centres, widths

    def fit(self, S_tr, y_tr, S_va, y_va, n_classes):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        d = S_tr.shape[-1] - 1
        last = S_tr[:, -1, :d]
        centres, widths = self._init_fuzzy(last)
        ante, cons = wang_mendel_rules(last, y_tr, centres, self.n_rules, n_classes)
        self.net = FRNNNet(d, n_classes, centres, widths, ante, cons, self.hidden, self.dropout).to(self.device)
        if not self.use_fuzzy:                                    # ablation: plain RNN
            self.net.G.data.zero_(); self.net.G.requires_grad_(False)
        counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=w.to(self.device))     # Eq. (25); L2 via weight_decay
        opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=self.lr,
                               weight_decay=self.weight_decay)
        Xt = torch.tensor(S_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
        best, best_state, wait = -1.0, None, 0
        for ep in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                logits, _ = self.net(Xt[b].to(self.device))
                loss = lossf(logits, yt[b].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                opt.step()
            f1 = f1_score(y_va, self.predict(S_va), average="macro")
            if f1 > best + 1e-4:
                best, best_state, wait = f1, copy.deepcopy(self.net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_, self.best_val_f1_ = ep + 1, best
        return self

    @torch.no_grad()
    def predict_proba(self, S):
        self.net.eval()
        logits, _ = self.net(torch.tensor(S, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, -1).cpu().numpy()

    def predict(self, S):
        return self.predict_proba(S).argmax(1)

    @torch.no_grad()
    def explain(self, S, top=3):
        """Strongest fired rules per record (antecedent term indices and normalised strength)."""
        self.net.eval()
        _, Pbar = self.net(torch.tensor(S, dtype=torch.float32))
        idx = Pbar.argsort(-1, descending=True)[:, :top].numpy()
        return [[(self.net.ante[r].tolist(), float(Pbar[i, r])) for r in row] for i, row in enumerate(idx)]
