"""All experiment settings in one place (values match Table 3 and Table S3 of the article)."""

CROPS = ["Paddy", "Maize", "Groundnut", "Sugarcane", "Tapioca", "Turmeric"]

BASE_FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall",
                 "month_sin", "month_cos", "modal_price", "arrivals", "Q_p", "Q_n", "SU"]
# 14 base features (SU = Q_s + Q_d + Q_u) + 4 derived = 18 candidates
DERIVED_FEATURES = ["SDIR_D", "SDIR_Z", "TSHR_T", "TSHR_L"]
CANDIDATE_FEATURES = BASE_FEATURES + DERIVED_FEATURES  # d = 18

PROTOCOL = dict(n_repeats=5, n_folds=5, val_fraction=0.15, seeds=[11, 23, 37, 41, 53], window=6)

IQN = dict(fence=1.5)

MABC = dict(colony=30, n_ants=15, n_employed=15, n_onlookers=15, max_iter=100, rho=0.10,
            alpha=1.0, beta=2.0, scout_limit=20, weights=(0.70, 0.15, 0.15),
            gd_tol=1e-3, gd_patience=10, inner_folds=3, lr_C=1.0, tau0=1.0)

FRNN = dict(n_mf=3, hidden=(64, 32), dropout=0.2, n_rules=24, lr=1e-3, batch=64, max_epochs=150,
            patience=15, weight_decay=1e-4, clip=1.0)

FRNN_GRID = dict(hidden=[(32, 16), (64, 32), (128, 64)], n_mf=[3, 5], window=[3, 6, 9],
                 lr=[1e-3, 5e-4], n_rules=[16, 24, 32])

BASELINES = dict(
    drqn=dict(hidden=64, eps_start=1.0, eps_end=0.05, eps_epochs=50, replay=5000, target_update=200,
              lr=1e-3, max_epochs=150, patience=15, batch=64),
    rfsvm=dict(n_trees=300, top_frac=0.75, C=10.0, gamma="scale"),
    dtrf=dict(depth=8, n_trees=300),
    dcnn=dict(filters=(32, 64), kernel=3, dense=64, dropout=0.3, lr=1e-3, batch=64, max_epochs=150, patience=15),
)

PRAM = dict(min_prob=0.10, max_candidates=3, a_max=0.6, min_history=2, tol=1e-8, max_iter=200)

TIMING_RUNS = 10

# Quick settings for smoke tests (NOT for reporting)
QUICK = dict(n_repeats=1, n_folds=3, mabc_iter=5, frnn_epochs=5, n_trees=30)
