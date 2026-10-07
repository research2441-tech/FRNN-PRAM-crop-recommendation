"""Writes a SYNTHETIC table with the exact schema of the agro dataset (see data_schema.md).

It exists only so that the pipeline and the tests can run without the real data. Numbers
obtained on it mean nothing and must never be reported.
"""
import argparse
import numpy as np
import pandas as pd
from config import CROPS


SCORE_MU = np.array([4.0, 4.3, 0.6, 5.2, 2.0, 4.9])
SCORE_SD = np.array([1.2, 0.9, 1.1, 1.3, 0.8, 1.0])


def make(n_series=81, n_months=39, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    C = len(CROPS)
    for s in range(n_series):
        soil = rng.normal([280, 22, 240, 6.8], [60, 6, 50, 0.6])
        bias = rng.normal(0, 1, C)
        stock = rng.uniform(300, 900)
        for t in range(1, n_months + 1):
            m = (t - 1) % 12 + 1
            temp = 27 + 4 * np.sin(2 * np.pi * (m - 4) / 12) + rng.normal(0, 1)
            rain = max(0, 80 + 120 * np.sin(2 * np.pi * (m - 7) / 12) + rng.normal(0, 30))
            hum = 60 + 0.1 * rain + rng.normal(0, 4)
            price = 2200 + 400 * np.sin(2 * np.pi * m / 12) + rng.normal(0, 150) * (1 + (rng.uniform() < 0.03) * 5)
            arrivals = max(1, 120 + 60 * np.cos(2 * np.pi * m / 12) + rng.normal(0, 20))
            q_n = rng.uniform(80, 200); q_s, q_d, q_u = rng.uniform(40, 120), rng.uniform(10, 50), rng.uniform(5, 30)
            q_p = stock
            stock = max(50, q_p + q_n - (q_s + q_d + q_u))
            # crop returns for the following season (INR 10^3 per ha)
            score = np.array([rain / 60 + soil[0] / 200, temp / 10 + soil[3] - 6, -abs(soil[3] - 6.5) * 2 + hum / 30,
                              rain / 50 + soil[2] / 150, price / 900 - rain / 200, soil[1] / 8 + temp / 15]) + bias
            score = (score - SCORE_MU) / SCORE_SD
            ret = 80 + 6 * score + 4 * bias + rng.normal(0, 3, C)
            feas = np.ones(C, int)
            row = dict(series_id=s, month_index=t, month=m, season_id=(t - 1) // 3, N=soil[0], P=soil[1], K=soil[2],
                       pH=soil[3], temperature=temp, humidity=hum, rainfall=rain, modal_price=price, arrivals=arrivals,
                       Q_p=q_p, Q_n=q_n, Q_s=q_s, Q_d=q_d, Q_u=q_u, W_avail=rng.uniform(650, 760), L_avail=rng.uniform(105, 120))
            for c, crop in enumerate(CROPS):
                row[f"ret_{crop}"] = ret[c]
                row[f"wat_{crop}"] = [1200, 550, 450, 1800, 600, 700][c] * rng.uniform(0.9, 1.1) / 2
                row[f"lab_{crop}"] = [140, 95, 90, 160, 100, 130][c] * rng.uniform(0.9, 1.1)
                row[f"hist_{crop}"] = [0.34, 0.18, 0.17, 0.14, 0.10, 0.07][c]
                row[f"feas_{crop}"] = feas[c]
            row["crop_label"] = CROPS[int(np.argmax(ret))]
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ["Q_p", "Q_n", "arrivals"]:                       # a few missing cells, as in the real data
        df.loc[rng.uniform(size=len(df)) < 0.02, col] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_agro.csv")
    ap.add_argument("--series", type=int, default=81)
    ap.add_argument("--months", type=int, default=39)
    a = ap.parse_args()
    make(a.series, a.months).to_csv(a.out, index=False)
    print("synthetic data written to", a.out)
