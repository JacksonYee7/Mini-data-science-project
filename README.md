````markdown
# Mini Data Science Project — Reproducible Baseline

**What this repo does**
I built a small but complete pipeline to predict an anonymized minute‑level target from tabular features. The work is split on purpose: a **research line** (train/validation only) where I judge ideas by **Daily IC** (per‑day cross‑sectional correlation), and a **submission line** (train→test) that **doesn’t rely on timestamps** so it runs on `test.parquet` as it is.

The final blend uses **MLP + XGBoost + LightGBM**. I learn the stacking weights on the **last month in training (2024‑02)** and then apply the same weights to the test predictions. Any statistic that could leak—winsor thresholds, scalers, the AutoEncoder, and the stacker—is fit on training (or on the holdout month) and reused for validation/test. Test is streamed by DuckDB in chunks to keep memory stable.


---

## Environment

```bash
conda create -n crypto_baseline python=3.10 -y
conda activate crypto_baseline
pip install -r requirements.txt
```

Key packages: `duckdb`, `pandas`, `numpy`, `scikit-learn`, `torch`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`.

Commands use POSIX slashes; Windows works equivalently.

---

## Data

Put the provided files under `data/`:

* `data/train.parquet` — historical rows with `timestamp`, features, and `label`.
* `data/test.parquet`  — **no timestamp**, `label=0`, contains `row_id` and the same feature columns as train.

The pipeline **does not** rely on test timestamps.

---

## What I did 

I first explored the signal on `train` using **daily cross‑sectional correlation** (“daily IC”): for each day, compute Pearson correlation between predictions and labels **within that day**, then summarize across days (mean/median and the fraction of positive days). This told me which ideas were stable **through time**.

Once I realized `test.parquet` has **no timestamp**, I simplified the submission to a **global recipe** that does not need a day key: fit **column‑wise winsorization (1%/99%)** and **standardization** on training only; train a shallow **AutoEncoder** on training to obtain **8 nonlinear codes**; then train three families (MLP / XGBoost / LightGBM) and **stack their predictions** using weights learned on the last month of training.

Definitions:

* **Daily IC**: per‑day correlation within the day, then aggregated across days.
* **Positive‑day rate**: fraction of days whose daily IC is above zero.
---

## Quickstart 
### **Option A — one command**
 If you prefer a single runner, add a thin `pipeline.py` that calls the same scripts

```bash
python -u src/pipeline.py --config configs/v2.yaml --stage all
```

### **Option B — three steps**

 1. build core, 2) add AE codes, 3) train (MLP/XGB/LGBM) + stack + check.
    See the exact commands right below.

1. **Build core tables (keeps `row_id` in test)**

```bash
# train → data/processed/train_core_plus6.parquet
python -u src/05_export_core_table.py \
  --input data/train.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6.txt \
  --out data/processed/train_core_plus6.parquet

# test  → data/processed/test_core_plus6.parquet
python -u src/10_build_test_core.py \
  --input data/test.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6.txt \
  --out data/processed/test_core_plus6.parquet
```

2. **Add 8‑dim AutoEncoder features (train‑fit; transform test)**

```bash
python -u src/13_autoencoder_feats.py \
  --train data/processed/train_core_plus6.parquet \
  --test  data/processed/test_core_plus6.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6.txt \
  --code_dim 8 --hidden 128 64 --epochs 30 --batch_size 4096 --lr 1e-3 --seed 7 \
  --out_train data/processed/train_core_plus6_ae.parquet \
  --out_test  data/processed/test_core_plus6_ae.parquet \
  --out_feat_file reports/feature_selection_postcheck/core_plus6_plusAE.txt
```

3. **Train three families (holdout = 2024‑02), then stack**

```bash
# MLP (Pearson-aware loss; saves holdout/test preds)
for s in 7 77 770 ; do
python -u src/12_mlp_global.py \
  --train data/processed/train_core_plus6_ae.parquet \
  --test  data/processed/test_core_plus6_ae.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6_plusAE.txt \
  --label_perday_z 1 --use_holdout_last_month 1 \
  --hidden 128 64 --dropout 0.1 --epochs 60 --batch_size 4096 --lr 1e-3 \
  --seed $s --outdir reports/infer_global_v2_mlp_ae
done

# XGBoost (hist; early-stop on holdout)
for s in 7 77 770 ; do
python -u src/10_xgb_global_v2.py \
  --train data/processed/train_core_plus6_ae.parquet \
  --test  data/processed/test_core_plus6_ae.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6_plusAE.txt \
  --label_perday_z 1 --use_holdout_last_month 1 \
  --learning_rate 0.05 --max_depth 6 --n_estimators 3000 \
  --subsample 0.8 --colsample_bytree 0.8 --early_stopping_rounds 200 \
  --seed $s --outdir reports/infer_global_v2_xgb
done

# LightGBM (holdout early-stop)
for s in 7 77 770 ; do
python -u src/08c_lgbm_global_holdout.py \
  --train data/processed/train_core_plus6_ae.parquet \
  --test  data/processed/test_core_plus6_ae.parquet \
  --feature_file reports/feature_selection_postcheck/core_plus6_plusAE.txt \
  --label_perday_z 1 --use_holdout_last_month 1 \
  --learning_rate 0.05 --num_leaves 63 --n_estimators 2000 \
  --min_data_in_leaf 200 --feature_fraction 0.8 --bagging_fraction 0.8 \
  --bagging_freq 1 --early_stopping_rounds 200 \
  --seed $s --outdir reports/infer_global_v2_lgbm_ae_fullholdout
done

# learn stack weights on holdout; blend test
python -u src/09b_weight_search_holdout.py --mode ridge \
  --valpreds \
    reports/infer_global_v2_mlp_ae/valpred_global_mlp_seed7.parquet \
    reports/infer_global_v2_mlp_ae/valpred_global_mlp_seed77.parquet \
    reports/infer_global_v2_mlp_ae/valpred_global_mlp_seed770.parquet \
    reports/infer_global_v2_xgb/valpred_global_xgb_seed7.parquet \
    reports/infer_global_v2_xgb/valpred_global_xgb_seed77.parquet \
    reports/infer_global_v2_xgb/valpred_global_xgb_seed770.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/valpred_global_lgb_seed7.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/valpred_global_lgb_seed77.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/valpred_global_lgb_seed770.parquet \
  --testpreds \
    reports/infer_global_v2_mlp_ae/pred_global_mlp_seed7.parquet \
    reports/infer_global_v2_mlp_ae/pred_global_mlp_seed77.parquet \
    reports/infer_global_v2_mlp_ae/pred_global_mlp_seed770.parquet \
    reports/infer_global_v2_xgb/pred_global_xgb_seed7.parquet \
    reports/infer_global_v2_xgb/pred_global_xgb_seed77.parquet \
    reports/infer_global_v2_xgb/pred_global_xgb_seed770.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/pred_global_lgb_seed7.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/pred_global_lgb_seed77.parquet \
    reports/infer_global_v2_lgbm_ae_fullholdout/pred_global_lgb_seed770.parquet \
  --out submissions/pred_final_global_v6_ridge.parquet \
  --metric_out reports/stack_holdout_metrics_v6_ridge.txt

# sanity checks (shape, NA, duplicates, tail stats)
python -u src/11_check_submission.py --input submissions/pred_final_global_v6_ridge.parquet
```
---

## Results snapshot

* **Research CV (Daily IC)** on OOF ensembles: mean ≈ **0.097**, median ≈ **0.091**, fraction of positive days ≈ **0.68**
  *(these were used to pick feature sets and blending modes in the research line)*
* **Holdout (2024‑02)**: ridge stack on MLP/XGB/LGBM holdout predictions yields **Pearson r ≈ 0.214**.
* **Final test**: predictions are blended row‑wise with the learned weights; timestamps are not needed.

All figures (prediction distribution, model–model correlation on holdout, per‑model holdout IC boxplots, and ensemble IC curves) are in **`notebooks/01_report.ipynb`**.
Further motivation, alternative approaches, and unsuccessful attempts are detailed in REPORT.md.

---

## What’s inside (scripts)

```
src/
├─ 05_export_core_table.py       # build train_core_plus6.parquet
├─ 10_build_test_core.py         # build test_core_plus6.parquet (keeps row_id)
├─ 13_autoencoder_feats.py       # train AE on train; transform train/test (+8 dims)
├─ 12_mlp_global.py              # MLP with Pearson-aware loss (holdout month)
├─ 10_xgb_global_v2.py           # XGBoost (hist, holdout early-stop)
├─ 08c_lgbm_global_holdout.py    # LightGBM (holdout early-stop)
├─ 09b_weight_search_holdout.py  # holdout ridge stack → blend test
├─ 09_ens_test.py                # simple row-wise averaging (optional)
├─ 09e_build_holdout_dates.py    # export holdout day index for notebook plots
└─ 11_check_submission.py        # sanity checks for final parquet
```
---

## Folder layout (inputs & outputs)

```
data/
  train.parquet
  test.parquet
  processed/
    train_core_plus6.parquet
    test_core_plus6.parquet
    train_core_plus6_ae.parquet
    test_core_plus6_ae.parquet
reports/
  feature_selection_postcheck/
    core_plus6.txt
    core_plus6_plusAE.txt
  infer_global_v2_mlp_ae/
    valpred_global_mlp_seed*.parquet
    pred_global_mlp_seed*.parquet
  infer_global_v2_xgb/
    valpred_global_xgb_seed*.parquet
    pred_global_xgb_seed*.parquet
  infer_global_v2_lgbm_ae_fullholdout/
    valpred_global_lgb_seed*.parquet
    pred_global_lgb_seed*.parquet
  figures/
submissions/
  pred_final_global_v6_ridge.parquet
```

---

## Reproducibility

* Random seeds: **7 / 77 / 770**
* Scripts log the paths and shapes of generated files
* The report notebook is **read‑only** with respect to artifacts (no training inside)


