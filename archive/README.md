# xs_research — Time‑aware OOF Experiments (Train/Validation Only)

This folder contains **time‑aware research materials** that helped me understand the signal and select ideas.  
It is **not** part of the submission inference path (the test file has `row_id` and features only, **no `timestamp`**).

**What this folder is for**

- Validate models with **Daily Cross‑Sectional IC** (per‑day Pearson) and month‑by‑month stability.
- Try ideas safely (different rolling windows, per‑day label standardization, ensembling rules, LOMO weighting).
- Run **leakage checks** (e.g., shuffle labels *within day* should collapse IC to ~0).

For how these OOF findings translate into the final inference line (which does **not** use timestamps), see the top‑level **README** and the long‑form **REPORT**. 

---

## Inputs

- Processed train table used in research, e.g. `data/processed/train_core_plus6.parquet`  
  (built from `data/train.parquet`; the test file is **not** needed here).
- Feature list, e.g. `reports/feature_selection_postcheck/core_plus6.txt`.

> All transforms in this folder must be **fit on train folds only** and applied to validation folds (causal evaluation).

---

## What’s inside

```

xs\_research/
├─ scripts/
│  ├─ 06\_lgbm\_xs.py                # XS training (240/360/480 windows, optional micro, etc.)
│  ├─ 06\_oof\_ensemble.py           # OOF ensembling (day\_zscore / invnorm\_rank / etc.)
│  ├─ 06\_eval\_oof.py               # OOF metrics (meanIC/medianIC/+rate); shuffle check
│  ├─ 07\_oof\_monthly\_eval.py       # Monthly aggregation and distributions
│  ├─ 06\_oof\_weight\_search\_blocked.py  # LOMO by month weight search
│  └─ 06\_eval\_compare\_oof.py       # Pairwise per‑day deltas / significance
└─ results/
├─ oof\_runs/                    # per‑window, per‑seed OOF parquet + ensembles
├─ monthly\_eval/                # monthly summary CSV
└─ weight\_search/               # LOMO/Grid weights & logs

````

---

## Quick reproduce (example commands)

1) **Run XS models** (3 windows × seeds; adjust paths/flags as needed)
```bash
# Example: window=240, no micro in features, per‑day label z for research only
python -u xs_research/scripts/06_lgbm_xs.py \
  --train data/processed/train_core_plus6.parquet \
  --cv_dir reports/cv_plan \
  --feature_file reports/feature_selection_postcheck/core_plus6.txt \
  --include_micro 0 \
  --ts_norm_win 240 --ts_norm_by_day 1 \
  --learning_rate 0.05 --num_leaves 63 --n_estimators 2000 \
  --min_data_in_leaf 200 --feature_fraction 0.8 --bagging_fraction 0.8 \
  --bagging_freq 1 --lambda_l2 1.0 --early_stopping_rounds 200 \
  --seeds 7 77 770 --save_oof 1 --embargo_days 3 \
  --outdir xs_research/results/oof_runs/ts240
````

2. **Ensemble OOF (day z‑score), then evaluate**

```bash
python -u xs_research/scripts/06_oof_ensemble.py --mode day_zscore \
  --inputs xs_research/results/oof_runs/ts240/oof_seed7.parquet \
           xs_research/results/oof_runs/ts240/oof_seed77.parquet \
           xs_research/results/oof_runs/ts240/oof_seed770.parquet \
  --out xs_research/results/oof_runs/ts240/oof_ens_dayZ.parquet

python -u xs_research/scripts/06_eval_oof.py \
  --input xs_research/results/oof_runs/ts240/oof_ens_dayZ.parquet
# optional leakage check:
python -u xs_research/scripts/06_eval_oof.py \
  --input xs_research/results/oof_runs/ts240/oof_ens_dayZ.parquet \
  --shuffle_within day
```

3. **Monthly stability**

```bash
python -u xs_research/scripts/07_oof_monthly_eval.py \
  --oof xs_research/results/oof_runs/ts240/oof_ens_dayZ.parquet \
  --outdir xs_research/results/monthly_eval
```

4. **LOMO by month (optional)**

```bash
python -u xs_research/scripts/06_oof_weight_search_blocked.py \
  --oof_240 xs_research/results/oof_runs/ts240/oof_ens_dayZ.parquet \
  --oof_360 xs_research/results/oof_runs/ts360/oof_ens_dayZ.parquet \
  --oof_480 xs_research/results/oof_runs/ts480/oof_ens_dayZ.parquet \
  --step 0.05 --agg mean \
  --weights_csv xs_research/results/weight_search/wopt_lomo_by_month.csv \
  --out xs_research/results/oof_runs/oof_ens_dayZ_wopt_lomo.parquet \
  --metric_out xs_research/results/weight_search/oof_ens_dayZ_wopt_lomo_metrics.json
```

---

## Metrics used here

* **Daily Cross‑Sectional IC**: for each calendar day, Pearson corr between `pred` and `label` **within that day**; then summarize across days (mean/median and the fraction of **positive days**, i.e., days with IC > 0).
* **Shuffle‑within‑day**: randomly permute labels within each day; IC should → \~0. This is a quick leakage sanity check.

---

## What was kept vs. dropped (high‑level)

* **Kept (research insights used later):** multiple **window views** (e.g., 240/360/480) and **day‑zscore ensembling** improved mean/median daily IC and stability; **per‑day label z‑score** (research only) helped the model focus on **relative** ranking rather than day‑scale volatility.
* **Dropped (not robust for this dataset):** microstructure add‑ons and minute‑of‑day normalization consistently reduced OOF IC or were unstable month‑to‑month; per‑day sample balancing also hurt.


```

