# What I built, why it looks this way, and what I learned
This project started as a time‑series modeling exercise on anonymized market data. The training file includes a real `timestamp` and the target `label`. The test file is different: it has the same feature columns, but **no timestamp**, and the `label` column is set to zero. That one line changes a lot. You can use time when you **learn** from training, but you cannot depend on time when you **predict** on test. Since the test file has no timestamp, I add a deterministic `row_id` (0..N-1 in file order) when building the processed test table. It is purely a row identifier: not used by any model, but carried through inference outputs so that different predictors can be merged on an exact key and the final `[row_id, pred]` parquet is consistent. This is especially important because test is predicted in chunks.

---

## 1) Feature engineering (before any modeling)

The raw table is wide and highly collinear, so I first reduced it to a compact, robust set.

- **Correlation clustering → medoids.** I built a feature–feature correlation matrix, used `1 − |corr|` as the distance, grouped columns with high intra‑correlation (≈0.6), and kept **one medoid per cluster**—the column most representative of the others. This trims redundancy without hand‑picking.

- **Drop near‑zero target correlation (train‑only).** On the training split, I removed columns whose absolute correlation with `label` was essentially zero. It is a coarse filter that removes obvious noise without looking at test.

- **Keep only safe nonlinear mixes.** I tried a small menu of hand‑crafted transforms (differences/ratios/max‑min among a few stable pairs). Most did not survive cross‑fold checks; only a handful that helped multiple folds were kept.

- **Add an AutoEncoder code (8 dims).** An anonymized, collinear feature space is a good fit for a compact nonlinear representation. I trained a shallow symmetric **AutoEncoder** (8‑dim bottleneck) on the training split only (global column standardization, MSE reconstruction, early stopping). At inference I **only encode** with the frozen network and **append** the 8‑dim code to the features. In practice this helped the MLP the most; tree models benefited less, which is one reason stacking across families pays off.

- **What I tried and removed.** Microstructure heuristics (order‑book/trade imbalances with minute‑of‑day normalization) and per‑day sample balancing looked reasonable but **hurt** out‑of‑fold stability or flipped sign across months, so I dropped them.

---

## 2) How I validated ideas on train

I treated the problem like a daily cross‑sectional ranking task and validated by **daily cross‑sectional correlation** (“daily IC”). For each calendar day in training, I computed the Pearson correlation between predictions and labels **within that day**, and then summarized those daily values across the whole period (mean, median, and the **positive‑day rate**—the share of days with IC > 0). This metric is less sensitive to day‑to‑day scale shifts than a single global correlation.

Two patterns emerged quickly:

- **Multiple window views help.** If I train variants that look at different rolling windows—240/360/480‑step **views of history**—and **average predictions after z‑scoring by day**, daily IC becomes both higher and smoother.

- **Per‑day label standardization helps learning.** During training/validation only, I standardized labels **within each day** (subtract the day’s mean and divide by the day’s std). This steers models toward **relative strength** instead of chasing the raw scale of that day and consistently improved out‑of‑fold daily IC.

With that setup, an equal blend of the 240/360/480 variants gave a mean daily IC around **0.084**. Adding per‑day label standardization lifted it to roughly **0.097**, with a positive‑day rate near **0.68**. A within‑day permutation test (shuffle labels inside each day) collapsed IC towards zero—exactly what a leak‑free pipeline should show.

> **Note.** Per‑day label standardization is a **research‑only tool**. It improves learning on train/validation, but I cannot use anything that needs a day key at inference because test has no timestamp.

---

## 3) Changing

When I confirmed that `test.parquet` has **no timestamp**,  I add a deterministic `row_id` (0..N-1 in file order)
when building the processed test table. It is purely a row identifier: not used by any model, but carried through inference outputs so that different predictors can be merged on an exact key and the final `[row_id, pred]` parquet is consistent. This is especially important because test is predicted in chunks.
I made the inference recipe deliberately global and simple:

1. **Train‑only, column‑wise transforms.** Fit **winsorization (1%/99%)** and **standardization** on the training split; apply the same scalers to the holdout month and to test. No per‑day or rolling statistics appear here.

2. **Frozen AE code.** Train the AutoEncoder on training features and keep an **8‑dimensional code** as extra features; the encoder is **frozen** at inference (transform only).

3. **Three families + three seeds.** On this feature space I trained a 2‑layer **MLP** (128→64 with dropout) with a blended loss `0.6*MSE + 0.4*(1 − Pearson)`, plus **XGBoost** (hist, depth 6, early stopping) and **LightGBM** (standard tabular settings). Seeds were {7, 77, 770} for each family.

4. **Holdout for model selection and stacking.** I used the **last month in training (2024‑02)** as the holdout for early stopping and for learning **ensemble weights**. Choosing the most recent block is the conservative option in time‑dependent problems. To make sure the winner is not driven by a few lucky days, I also **resampled holdout days** (sample 80% of the days without replacement, recompute ensemble Pearson a few hundred times).

5. **Stacking.** I learned weights over the nine predictors in two ways:
   - a **non‑negative grid** (weights sum to one) — holdout Pearson around **0.19**;
   - a small **ridge regression** (allowing gentle negatives so correlated errors can cancel) — holdout Pearson around **0.21–0.214**.  
   I then **clipped tiny/negative coefficients** and renormalized (“ridge‑clip”). Under day‑resampling, ridge‑clip had higher mean/median Pearson and a better 90th percentile than equal average and the non‑negative grid, so I used **ridge‑clip** to blend the **test** predictions row‑wise.

6. **Final inference & checks.** Test is streamed in chunks with DuckDB (read only the needed columns), passed through the **training‑fitted** scalers and the **frozen AE**, scored by the 9 predictors, and blended to produce `row_id, pred`. I check: no NaNs, unique `row_id`, sensible distribution.

---

## 4) What the AutoEncoder added 
The anonymized columns are highly collinear. Tree models tolerate redundancy; the neural model benefits from a compact nonlinear view. The AE is a shallow symmetric network with a **bottleneck of 8 units**. I standardized columns on the training split, trained the AE to minimize reconstruction error with early stopping (monitored on the holdout reconstruction loss), and then **froze** the encoder. On test, I only **transform** (no refitting, no timestamps). In my runs this consistently improved the MLP’s holdout correlation across seeds (roughly from ~0.13–0.14 to ~0.15–0.18). Gains for trees were smaller, which is fine—stacking lets the families complement one another.

---

## 5) Numbers at a glance

- **Research (OOF)**: mean daily IC ≈ **0.097**, median ≈ **0.091**, positive‑day rate ≈ **0.68**.  
- **Holdout (2024‑02)**: **ridge‑clip** stack (9 predictors = 3 families × 3 seeds) ≈ **0.214 Pearson**; stayed ahead of equal averaging and non‑negative grid under day‑resampling (typical resample means: equal ≈ 0.168, grid ≈ 0.196, ridge‑clip ≈ 0.207).

---

## 6) If I had more time

I would try a **two‑month holdout** (e.g., 2024‑01 + 2024‑02) to reduce variance of weight estimates, or a **leave‑one‑month‑out** averaging of stacking weights to be less sensitive to a single month. Both are supported by the current scripts without touching the inference path. I would also add a small **linear head** on top of `[core | AE code]`; linear models often add a robust, low‑variance view.

---

### Appendix — concise definitions

- **Daily cross‑sectional correlation (daily IC)**: for each calendar day, the Pearson correlation between predictions and labels **within that day**; afterwards aggregate the daily values (mean/median and the fraction of days with IC > 0).
- **Per‑day label standardization**: within each day on train/validation, subtract the day’s mean label and divide by the day’s std; improves learning of relative ranking.
- **Holdout (2024‑02)**: the last month in training, used for early stopping and for learning ensemble weights; I checked robustness by resampling days and also support two‑month/LOMO alternatives.
- **Ridge‑clip stacking**: fit ridge regression on holdout predictions vs. labels; clip very small/negative coefficients and renormalize for a simple, conservative blend.
