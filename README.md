# FRNN–PRAM: crop recommendation and portfolio resource allocation

Code for *"Enhancing Crop Recommendations and Resource Allocation using a Fuzzified Recurrent
Neural Network in an Agricultural Industry"* (T. Chella Ramya, K. Umadevi).

The pipeline is: interquartile normalization (IqN) → causal SDIR and TSHR indicators →
multi-objective ant–bee colony (MABC) feature selection → fuzzified recurrent neural network
(FRNN) → portfolio resource allocation model (PRAM). Every data-dependent step is fitted on the
training series of each fold (stratified group 5-fold CV, repeated 5 times).

## Files

| File | Content | Article |
|---|---|---|
| `config.py` | All settings and seeds | Table 3, Table S3 |
| `preprocessing.py` | Imputation, IqN, SDIR, TSHR, sequence builder | Eqs. (1)–(11) |
| `mabc.py` | MABC feature selection | Eqs. (12)–(19), Fig. S1 |
| `frnn.py` | FRNN model, Wang–Mendel rules, training loop | Eqs. (20)–(25) |
| `baselines.py` | DRQ-N, RF-SVM, DT-RF, DCNN re-implementations | Section 5.1 |
| `pram.py` | PRAM solver, oracle and AER | Eqs. (26)–(30), Eq. (S2) |
| `evaluation.py` | Metrics, 95% CIs, corrected resampled t-test, Wilcoxon | Eqs. (31)–(33), Table S2 |
| `run_experiments.py` | Full protocol, ablation, allocation, chronological check | Tables 4–6, S4, S5 |
| `make_figures.py` | Figures from the result files | Figs. 2, 3a, S3 |
| `make_synthetic_data.py` | Synthetic table with the real schema (tests only) | — |
| `test_pipeline.py` | Unit and smoke tests | — |
| `data_schema.md` | Required columns | Table S1 |

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q test_pipeline.py                         # 10 tests

# article protocol (5 x stratified group 5-fold CV, baselines, ablation, PRAM)
python run_experiments.py --data agro_dataset.csv --out results
python run_experiments.py --data agro_dataset.csv --out results --chronological   # Table S5 (train 1-30, gap 31-33, test 34-39)
for f in 0.25 0.5 0.75; do                                   # learning curves, Fig. S2
  python run_experiments.py --data agro_dataset.csv --out lc_$f --train_fraction $f --skip_ablation
done
python make_figures.py --results results
```

FRNN hyperparameters are tuned inside every fold on that fold's validation series (one factor at a
time over `config.FRNN_GRID`); `--no_tune` uses the fixed values in `config.FRNN`.
`--quick` runs a reduced configuration for smoke testing only. Results on
`make_synthetic_data.py` output are meaningless and must not be reported.

## Data

The agro dataset is described in Section 3.1 and Table S1 of the article. See the Data
availability statement for access.

## License

MIT
