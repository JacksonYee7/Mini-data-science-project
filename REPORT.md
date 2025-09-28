# What I built, why it looks this way, and what I learned

This project started as a time‑series modeling exercise on anonymized market data. The training file includes a real `timestamp` and the target `label`. The test file is different: it has a `row_id` and the same feature columns, but **no timestamp**, and the `label` column is set to zero. That one line changes a lot. You can use time when you **learn** from training, but you cannot depend on time when you **predict** on test.

## How I explored the signal on train

I first treated the problem like a daily cross‑sectional ranking task and validated by **daily cross‑sectional correlation** (“daily IC”). For each calendar day in training, I computed the Pearson correlation between predictions and labels **within that day**, and then summarized those daily values across the whole period (mean, median, and the **fraction of positive days**—the share of days whose correlation is above zero). This metric is more stable than a single global correlation and matches the idea of ranking instruments within the same day.

Two patterns emerged quickly:

- If I train several variants that look at different rolling windows—240/360/480‑step **views of history**—and average their predictions **after z‑scoring by day**, daily IC becomes both higher and smoother.
- If I **standardize the labels within each day** during training/validation (subtract the day’s mean label and divide by that day’s standard deviation), models learn **relative strength** rather than chasing the raw scale of that day. That consistently raised daily IC in out‑of‑fold evaluation.

With that setup, an equal blend of the 240/360/480 variants gave a mean daily IC around **0.084**. Adding the per‑day label standardization lifted it to roughly **0.097**, with a positive‑day rate close to **0.68**. As a sanity check, I ran a within‑day permutation test (shuffle labels inside each day); the IC collapsed toward zero, which is what a leak‑free pipeline should show.

Not every idea survived. I tried adding order‑book/trade imbalances and normalizing them by minute‑of‑day; they **reduced** daily IC and were unstable across months. I also tried per‑day sample balancing, which **hurt** correlation. These detours were useful: they clarified which assumptions were fragile.

> Important: the daily label standardization (“per‑day z‑score of label”) is a **research‑only** tool. It helps models learn the ranking signal on train/validation. It is **not** used in final inference, because test has no day key.

## How I turned it into a submission that works on test

Once I confirmed that `test.parquet` has **no timestamp**, I made the inference recipe deliberately global and simple:

- Fit **column‑wise winsorization** (1%/99%) and **standardization** on the training split only; apply the same scalers to the holdout month and to test.
- Train a shallow **AutoEncoder** on the training features and keep the **8‑dimensional code** as extra features; freeze the encoder and only **transform** holdout/test (no refitting, no time logic).
- Train three families on this feature space: a 2‑layer **MLP** (128→64 with dropout) with a blended loss `0.6*MSE + 0.4*(1 − Pearson)`, **XGBoost** (hist, depth 6, early stopping), and **LightGBM** (standard tabular settings). I used random seeds 7/77/770 for each family.

For model selection and stacking I needed a reference block that plays the role of “what comes next.” I held out the **last month in training** and trained everything else on the months before it. In this dataset that month is **2024‑02**. Using the most recent block is a conservative choice in time‑dependent problems because relations drift. Could test be older? Possibly; that’s why I also checked stability by **resampling holdout days** (sample 80% of days without replacement, recompute ensemble Pearson a couple hundred times). The same blend kept winning across resamples.

I learned ensemble weights on the holdout predictions vs. holdout labels in two ways:

- a **non‑negative grid** whose weights sum to one (~0.19 Pearson);
- a small **ridge regression** that allows gentle negatives so correlated errors can cancel (~0.21–0.214 Pearson).

I then **clipped tiny/negative coefficients** (“ridge‑clip”) and renormalized to keep the blend simple. Under day‑resampling, ridge‑clip had higher mean/median Pearson and a better 90th percentile than equal average and the non‑negative grid, so I used ridge‑clip to blend the **test** predictions row‑wise. The final sanity checks confirm there are no missing values, the `row_id` is unique, and the prediction distribution looks reasonable.

## What the AutoEncoder added (and why it doesn’t leak)

The anonymized columns are highly collinear. Tree models handle redundancy fairly well; the neural model benefits from a compact nonlinear view. The AE is a small symmetric network with a bottleneck of **8 units**; I standardize columns on the training split, train the AE to minimize reconstruction error with early stopping, **freeze** the encoder, then append the 8‑dim code to the raw features: `X → [X | AE(X)]`. On test I only **transform** with the training‑fitted scaler and encoder. In my runs this consistently improved the MLP’s holdout Pearson (e.g., from ~0.13–0.14 to ~0.15–0.18), while gains for trees were smaller—which is fine, because stacking lets the families complement one another.

## Numbers at a glance

- On train (research): **mean daily IC ≈ 0.097**, median ≈ 0.091, **positive‑day rate ≈ 0.68**.  
- On the held‑out month (last month in train): **ridge‑clip** stacking across 9 predictors (3 seeds × 3 families) reached **Pearson ≈ 0.214** and remained ahead under day resampling (equal ≈ 0.168, grid ≈ 0.196, ridge‑clip ≈ 0.207 on resample means in my logs).

## If I had more time

I would try a **two‑month holdout** (e.g., 2024‑01+02) to reduce variance of weight estimates, or a **leave‑one‑month‑out** averaging of stacking weights to be even less sensitive to a single month. The code supports both without touching the inference logic. I would also add a small linear head on `[core | AE code]`—linear models sometimes add a surprisingly robust, low‑variance view.

---

### Appendix — concise definitions

- **Daily cross‑sectional correlation (daily IC)**: per‑day Pearson correlation between predictions and labels **within the day**, then aggregated across days (mean/median and the fraction of positive days).
- **Per‑day label standardization** (research only): within each day on train/validation, subtract the day’s mean label and divide by the day’s standard deviation; helps models learn relative ranking.
- **Holdout month**: the last calendar month in training (chosen for recency). I also checked day‑resample robustness and provide scripts to switch to two‑month holdouts or leave‑one‑month‑out if needed.
- **Ridge‑clip stacking**: fit ridge regression on holdout predictions vs. labels; clip tiny/negative coefficients and renormalize for a simple, conservative blend.
````
