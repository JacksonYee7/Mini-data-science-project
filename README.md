# Mini Data Science Project 

**What this repo does**
This repository contains a baseline to predict an anonymized minute‑level target from tabular features.  
The training file has a real `timestamp` and `label`. The test file has `row_id` and the same feature columns, but **no timestamp** and `label=0`. I learned with time on `train`, but the final inference does **not** depend on time so it runs on `test.parquet` as is.

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

### 1) Feature engineering 
I started from the raw tabular features in `train.parquet`. Many columns are highly collinear, so the first step was to **shrink to a compact, robust set** before any modeling:
* **Correlation clustering → medoids.**
  I built a feature–feature correlation matrix, used `1 − |corr|` as the distance, grouped features with high intra‑correlation (threshold ≈ 0.6), and kept **one “medoid” per cluster** (the column most representative of others). This cuts redundancy while keeping signal.
* **Drop near‑zero target correlation.**
  On training only, I removed columns whose absolute correlation with the label was essentially zero. This is a coarse filter to get rid of obvious noise.
* **Cross‑fold stability check with trees + SHAP.**
  I trained a simple XGBoost on a **time‑aware CV** (see below), collected **top‑20 SHAP features per fold**, and took the **union of features that repeatedly appear** across folds. The idea is not the exact ranking, but **consistency** across time splits.
* **Simple nonlinear mixes (kept very few).**
  I tried a small menu of safe transformations (differences/ratios/max‑min among a few stable pairs). Most did **not** survive CV stability checks; only a handful that improved multiple folds were kept.
* **Add 8 AutoEncoder codes (train‑fit, then frozen).**
  Because the anonymized columns are still collinear, I trained a small symmetric **AutoEncoder (8‑dim bottleneck)** on the **training split only** (global column standardization, MSE reconstruction, early stopping). At inference I only **encode** with the frozen network and **append** these 8 codes to the core features.
  In practice this helped the MLP noticeably; tree models benefitted less (as expected), which is why stacking across families pays off.
* **What I tried and dropped.**
  I tested microstructure heuristics (order‑book/trade imbalance variants, minute‑of‑day normalization) and per‑day sample balancing. They **hurt** out‑of‑fold stability or improved only a couple of months while degrading others, so I removed them.
  
---

### 2) How I validated ideas on train
Early on I explored the signal using **daily cross‑sectional correlation** (“daily IC”): for each day in training, compute the Pearson correlation between predictions and labels **within that day**, then look at its **mean/median over days** and the **fraction of positive days** (the share of days where the correlation is above zero). This metric matches a ranking use‑case and is less sensitive to day‑to‑day scale shifts.

* All transforms that need fitting (winsor cut points, scalers, AE weights) were **fit on training folds only**; validation folds only **apply** them.
* I also tried **per‑day label standardization**(z-scoring labels within each day, used solely during the training/validation phase), which directs the model toward relative cross-sectional strength rather than day-specific scale and improved the out-of-fold (OOF) daily IC.
* As a sanity check, I randomly permuted labels within each day; the resulting daily IC was approximately zero, indicating that the pipeline is free of leakage.

---

### 3) Changing
Once I realized `test.parquet` has **no timestamp** (and `label=0`), I simplified the final submission to a **global recipe** that does **not** need a day key:
1. **Train‑only, column‑wise transforms.**
   Fit **winsorization (1%/99%)** and **standardization** on the training split; apply the same scalers to holdout and test. No per‑day or rolling statistics appear here.
2. **AE codes are train‑fit, test‑transform.**
   The AutoEncoder is trained on training only; I then freeze the encoder and apply it to holdout/test to get 8 extra features.
3. **Three model families + stacking.**
   I trained **MLP / XGBoost / LightGBM** with seeds {7, 77, 770}. For the MLP I used a blended loss `0.6*MSE + 0.4*(1 − Pearson)` so it aligns with correlation.
   I kept **the last month of training (2024‑02)** as a **holdout** for early stopping and for learning **stacking weights**. It is the most recent block, which is the standard conservative choice in time‑series work. To avoid over‑fitting to a single month, I re‑checked the learned weights by **resampling holdout days**; the same blend stayed ahead on mean/median Pearson and on the 90th percentile.
4. **Final inference is memory‑safe.**
   Test is read in **chunks** with DuckDB (only the needed columns), passed through the **training‑fitted** scalers and the **frozen** AE encoder, then through the 9 predictors; I combine predictions with **ridge‑regularized** weights (with small/negative coefficients clipped) to produce the final `row_id, pred` parquet.

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


