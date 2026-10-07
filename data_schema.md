# Data schema expected by `run_experiments.py`

One CSV row per region–market unit and month (81 series × 39 months = 3,159 rows in the article).

| Column | Meaning |
|---|---|
| `series_id` | Identifier of the region–market unit (used as the CV group) |
| `month_index` | 1..39, consecutive month number within the series |
| `month` | Calendar month 1..12 |
| `season_id` | Increasing season index within the series (e.g. 3-month seasons); used by SDIR |
| `N`, `P`, `K`, `pH` | Soil nutrients (kg/ha) and pH |
| `temperature`, `humidity`, `rainfall` | Monthly climate (°C, %, mm) |
| `modal_price`, `arrivals` | Market price (INR/quintal) and arrivals (t/month) |
| `Q_p`, `Q_n` | Opening stock carried over and new stock added (t) |
| `Q_s`, `Q_d`, `Q_u` | Stock sold, delivered to distributors and processed (t) |
| `crop_label` | Target: crop with the highest realised net return in the following season (one of the six crops in `config.CROPS`) |
| `ret_<Crop>` | Realised net return of each crop in the following season (INR 10³/ha) |
| `wat_<Crop>`, `lab_<Crop>` | Water (mm/ha) and labour (person-days/ha) required by each crop |
| `hist_<Crop>` | Historical land share of each crop in the unit (sums to 1) |
| `feas_<Crop>` | 1 if the crop is agronomically feasible in the unit, else 0 |
| `W_avail`, `L_avail` | Water (mm/ha) and labour (person-days/ha) available to the unit |

Missing cells are allowed in the raw climate, market and stock columns; they are forward-filled
within each series and remaining leading gaps take the training-fold median.
