# 文件：06_lgbm_xs.py  v4


import argparse
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        "LGBM with cross-sectional IC, per-group normalization, embargo"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--cv_dir", default="reports/cv_plan")
    ap.add_argument("--feature_file", required=True)
    ap.add_argument("--include_micro", type=int, default=1)
    ap.add_argument("--train_max_rows", type=int, default=400_000)
    ap.add_argument("--val_take_all", type=int, default=1)

    # LGBM
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--min_data_in_leaf", type=int, default=200)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l1", type=float, default=0.0)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--n_jobs", type=int, default=8)

    # XS metrics & normalization
    ap.add_argument("--xs_metrics", type=int, default=1)
    ap.add_argument(
        "--perday_norm",
        type=int,
        default=1,
        help="keep for backward-compat; if 1 -> do XS norm",
    )
    ap.add_argument("--winsor_p", type=float, default=0.01)
    ap.add_argument(
        "--xs_group",
        choices=["date", "minute", "timestamp"],
        default="minute",
        help="group key for XS normalization; minute= floor to minute (recommended)",
    )
    # Time-series (causal) normalization for single-series
    ap.add_argument(
        "--ts_norm_win",
        type=int,
        default=0,
        help="If >0, apply causal rolling z-score with this window (rows/minutes), using past data only.",
    )
    ap.add_argument(
        "--ts_norm_by_day",
        type=int,
        default=1,
        help="If 1, reset rolling window each day for ts_norm.",
    )

    # Seeds & outputs
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    ap.add_argument("--save_oof", type=int, default=1)
    ap.add_argument("--save_model", type=int, default=0)
    ap.add_argument("--outdir", default="reports/lgbm_xs")
    ap.add_argument("--verbose_eval", type=int, default=100)

    # Embargo & ablation
    ap.add_argument("--embargo_days", type=int, default=0)
    ap.add_argument("--drop_feats", type=str, default="")

    # Minute-of-day normalization using TRAIN-only stats (no leakage)
    ap.add_argument(
        "--mod_norm",
        type=int,
        default=0,
        help="If 1, normalize by minute-of-day (MoD) using TRAIN-only stats before TS norm.",
    )
    ap.add_argument(
        "--mod_winsor_p",
        type=float,
        default=0.0,
        help="Optional winsor within MoD on TRAIN before computing mean/std.",
    )
    ap.add_argument(
        "--mod_only_micro",
        type=int,
        default=1,
        help="If 1, apply MoD only to microstructure columns (ob_imb, trade_imb, volume).",
    )
    ap.add_argument(
        "--mod_min_count",
        type=int,
        default=50,
        help="Minimum samples in a MoD bucket to apply winsor on TRAIN.",
    )
    ap.add_argument(
        "--mod_shrink_k",
        type=float,
        default=50.0,
        help="Empirical-Bayes-style shrink strength for MoD stats toward global TRAIN stats. 0 to disable.",
    )

    # Training-side balancing and label scaling (low-risk toggles)
    ap.add_argument(
        "--perday_balance",
        type=int,
        default=0,
        help="If 1, apply per-day sample_weight so each day has equal total weight.",
    )
    ap.add_argument(
        "--label_perday_z",
        type=int,
        default=0,
        help="If 1, use TRAIN-only per-day z-scored label for fitting (eval uses raw label).",
    )

    return ap.parse_args()


def read_month_groups(cv_dir: str):
    m2g = pd.read_csv(f"{cv_dir}/month_to_group.csv")
    g2m = {}
    for _, r in m2g.iterrows():
        g2m.setdefault(int(r["group"]), []).append(str(r["month"]))
    folds = pd.read_csv(f"{cv_dir}/cv_folds_groups.csv")
    return g2m, folds


def read_features(path: str):
    return [
        x.strip()
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    if {"bid_qty", "ask_qty"}.issubset(df.columns):
        df["ob_imb"] = (df["bid_qty"] - df["ask_qty"]) / (
            df["bid_qty"] + df["ask_qty"] + eps
        )
    else:
        df["ob_imb"] = np.nan
    if {"buy_qty", "sell_qty", "volume"}.issubset(df.columns):
        df["trade_imb"] = (df["buy_qty"] - df["sell_qty"]) / (df["volume"] + eps)
    else:
        df["trade_imb"] = np.nan
    return df


def cols_to_read(feats, include_micro):
    base = ["timestamp", "label"]
    raw = (
        ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"] if include_micro else []
    )
    return sorted(set(base + feats + raw))


def fetch_by_months(
    path: str, months, columns, sample_n=None, seed=42, take_all=0
) -> pd.DataFrame:
    cols_sql = ", ".join([f'"{c}"' for c in columns])
    placeholders = ", ".join(["?"] * len(months))
    base_sql = f"""
    SELECT {cols_sql}
    FROM read_parquet(?)
    WHERE strftime(timestamp, '%Y-%m') IN ({placeholders})
    """
    if take_all:
        sql = base_sql
        params = [path] + months
    else:
        sql = base_sql + (
            f"\nUSING SAMPLE RESERVOIR({int(sample_n)} ROWS) REPEATABLE({int(seed)})"
            if sample_n
            else ""
        )
        params = [path] + months
    con = duckdb.connect()
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _group_key(series_ts: pd.Series, mode: str):
    t = pd.to_datetime(series_ts)
    if mode == "date":
        return t.dt.normalize()
    elif mode == "minute":
        return t.dt.floor("min")
    else:  # "timestamp"
        return t


def xs_winsorize_zscore(df: pd.DataFrame, cols, winsor_p=0.01, group_mode="minute"):
    """
    向量化版本：对所有列一次性计算各组分位数/均值/方差，然后广播回原表。
    winsor_p <= 0 时跳过剪尾，只做 z-score。
    """
    if not cols:
        return df
    df = df.copy()
    g = _group_key(df["timestamp"], group_mode)
    df["__g"] = g
    X = df[cols]

    if winsor_p and 0 < winsor_p < 0.5:
        qs = X.groupby(df["__g"], sort=False).quantile([winsor_p, 1 - winsor_p])
        q_lo = qs.xs(winsor_p, level=1)
        q_hi = qs.xs(1 - winsor_p, level=1)
        lo = q_lo.reindex(df["__g"]).to_numpy()
        hi = q_hi.reindex(df["__g"]).to_numpy()
        Xc = np.minimum(np.maximum(X.to_numpy(), lo), hi)
        Xc = pd.DataFrame(Xc, columns=cols, index=X.index)
    else:
        Xc = X

    means = Xc.groupby(df["__g"], sort=False).transform("mean")
    stds = Xc.groupby(df["__g"], sort=False).transform("std", ddof=0)
    eps = 1e-12
    Xz = (Xc - means) / (stds.replace(0, np.nan) + eps)
    Xz = Xz.fillna(0.0).astype(np.float32)

    df[cols] = Xz
    del df["__g"]
    return df


def xs_winsorize_zscore_safe(
    df: pd.DataFrame, cols, winsor_p=0.01, group_mode="minute"
):
    """
    Robust XS normalization per group. For tiny/degenerate groups:
    - If group size < 2 or std ~ 0: keep original values (no z-score)
    - Winsorization only when group size >= threshold (here 5)
    """
    if not cols:
        return df
    df = df.copy()
    df["__g"] = _group_key(df["timestamp"], group_mode)
    eps = 1e-12
    any_col = cols[0]
    gsize = df.groupby("__g")[any_col].transform("size")

    for c in cols:
        s = df[c]

        if winsor_p and winsor_p > 0:

            def _clip(u: pd.Series) -> pd.Series:
                uu = u.dropna()
                if uu.size < 5:
                    return u
                lo = uu.quantile(winsor_p)
                hi = uu.quantile(1 - winsor_p)
                return u.clip(lower=lo, upper=hi)

            s = df.groupby("__g", sort=False)[c].transform(_clip)

        gmean = df.groupby("__g", sort=False)[c].transform("mean")
        gstd = df.groupby("__g", sort=False)[c].transform("std")

        bad = (gsize < 2) | (~np.isfinite(gstd)) | (gstd < eps)
        z = (s - gmean) / (gstd.fillna(0.0) + eps)
        z[bad] = s[bad]

        df[c] = z.astype(np.float32)

    del df["__g"]
    return df


def mod_norm_train_apply(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    cols,
    winsor_p: float = 0.0,
    min_count: int = 50,
    shrink_k: float = 0.0,
):
    """
    Minute-of-day (MoD) normalization using TRAIN-only statistics.
    - Compute per-MoD mean/std on TRAIN (optionally winsorized within MoD if winsor_p>0)
    - Apply to both TRAIN/VAL (no leakage).
    - Fallback to global TRAIN stats when MoD bucket missing/degenerate.
    - Optional empirical-Bayes shrink toward global stats controlled by shrink_k.
    """
    if not cols:
        return df_tr, df_va
    df_tr = df_tr.copy()
    df_va = df_va.copy()
    tr_t = pd.to_datetime(df_tr["timestamp"])
    va_t = pd.to_datetime(df_va["timestamp"])
    df_tr["__mod"] = (
        tr_t.dt.hour.astype(int) * 60 + tr_t.dt.minute.astype(int)
    ).astype(int)
    df_va["__mod"] = (
        va_t.dt.hour.astype(int) * 60 + va_t.dt.minute.astype(int)
    ).astype(int)
    eps = 1e-12

    for c in cols:
        s_tr = pd.to_numeric(df_tr[c], errors="coerce")

        # Optional winsor on TRAIN within MoD (only when samples are sufficient)
        if winsor_p and winsor_p > 0:

            def _clip(u: pd.Series) -> pd.Series:
                uu = u.dropna()
                if uu.size < int(min_count):
                    return u
                lo = uu.quantile(winsor_p)
                hi = uu.quantile(1 - winsor_p)
                return u.clip(lower=lo, upper=hi)

            s_tr = df_tr.groupby("__mod", sort=False)[c].transform(_clip)

        g = pd.DataFrame({"x": s_tr, "mod": df_tr["__mod"]}).dropna()
        mean_mod = g.groupby("mod")["x"].mean()
        std_mod = g.groupby("mod")["x"].std(ddof=0)
        cnt_mod = g.groupby("mod")["x"].size()

        glob_mu = float(np.nanmean(s_tr))
        glob_sd = float(np.nanstd(s_tr))
        if not np.isfinite(glob_sd) or glob_sd < eps:
            glob_sd = 1.0

        # Empirical-Bayes shrink toward global TRAIN stats
        if shrink_k and shrink_k > 0:
            k = float(shrink_k)
            # variance shrink
            var_mod = std_mod**2
            var_hat = (cnt_mod * var_mod + k * (glob_sd**2)) / (cnt_mod + k)
            std_hat = np.sqrt(var_hat)
            # mean shrink
            mean_hat = (cnt_mod * mean_mod + k * glob_mu) / (cnt_mod + k)

            use_mu = mean_hat
            use_sd = std_hat
        else:
            use_mu = mean_mod
            use_sd = std_mod

        # TRAIN apply
        mu_tr = df_tr["__mod"].map(use_mu).astype("float32")
        sd_tr = df_tr["__mod"].map(use_sd).astype("float32")
        base_tr = pd.to_numeric(df_tr[c], errors="coerce")
        z_tr = (base_tr - mu_tr) / (sd_tr.replace(0, np.nan) + eps)
        z_tr = z_tr.where(
            mu_tr.notna() & (sd_tr > 0), (base_tr - glob_mu) / (glob_sd + eps)
        )
        df_tr[c] = z_tr.astype(np.float32)

        # VAL apply
        mu_va = df_va["__mod"].map(use_mu).astype("float32")
        sd_va = df_va["__mod"].map(use_sd).astype("float32")
        base_va = pd.to_numeric(df_va[c], errors="coerce")
        z_va = (base_va - mu_va) / (sd_va.replace(0, np.nan) + eps)
        z_va = z_va.where(
            mu_va.notna() & (sd_va > 0), (base_va - glob_mu) / (glob_sd + eps)
        )
        df_va[c] = z_va.astype(np.float32)

        # Debug: share of VAL rows that fell back to global stats
        miss_share = float((~mu_va.notna() | ~(sd_va > 0)).mean())
        print(
            f"[MoD:{c}] val missing/fallback share={miss_share:.3f}  "
            f"(winsor_p={winsor_p}, min_count={min_count}, shrink_k={shrink_k})"
        )

    del df_tr["__mod"]
    del df_va["__mod"]
    return df_tr, df_va


def ts_rolling_zscore(df: pd.DataFrame, cols, window: int, by_day: bool = True):
    """
    Causal (time-series) rolling z-score. Use past window stats (shifted) per feature.
    If by_day=True, reset window within each day.
    """
    if not cols or window <= 0:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    eps = 1e-12
    for c in cols:
        if by_day:
            m = df.groupby("date", sort=False)[c].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            v = df.groupby("date", sort=False)[c].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std(ddof=0)
            )
        else:
            m = df[c].shift(1).rolling(window, min_periods=1).mean()
            v = df[c].shift(1).rolling(window, min_periods=1).std(ddof=0)
        df[c] = (
            ((df[c] - m) / (v.replace(0, np.nan) + eps)).fillna(0.0).astype(np.float32)
        )
    return df


def daily_ic_summary(y, p, dates):
    tmp = pd.DataFrame({"y": y, "p": p, "date": dates})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for d, g in tmp.groupby("date", sort=True):
        if g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        ic = np.corrcoef(g["p"].values, g["y"].values)[0, 1]
        rows.append((d, ic, len(g)))
    df = pd.DataFrame(rows, columns=["date", "ic", "n"])
    if df.empty:
        return {
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "pos_rate": np.nan,
            "days": 0,
        }, df
    arr = df["ic"].values
    return {
        "mean_ic": float(np.nanmean(arr)),
        "median_ic": float(np.nanmedian(arr)),
        "pos_rate": float(np.mean(arr > 0)),
        "days": int(df.shape[0]),
    }, df


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    g2m, folds_df = read_month_groups(args.cv_dir)
    base_feats = read_features(args.feature_file)

    drop = [x.strip() for x in args.drop_feats.split(",") if x.strip()]
    if args.include_micro:
        feat_cols = list(dict.fromkeys(base_feats + ["ob_imb", "trade_imb", "volume"]))
    else:
        feat_cols = base_feats[:]
    feat_cols = [c for c in feat_cols if c not in drop]

    read_cols = cols_to_read(feat_cols, args.include_micro)
    print(
        f"[INFO] n_feat={len(feat_cols)}  include_micro={args.include_micro}  drop_feats={drop}"
    )
    print(
        f"[INFO] XS norm group = {args.xs_group}  | columns to read = {len(read_cols)}"
    )

    overall_seed_summ = []

    for seed in args.seeds:
        fold_summ = []
        oof_records = []

        for _, r in folds_df.iterrows():
            fold = int(r["fold"])
            test_g = int(r["test_group"])
            train_groups = [
                int(x) for x in str(r["train_groups"]).split(",") if x != ""
            ]
            tr_months = sum([g2m[g] for g in train_groups], [])
            va_months = g2m[test_g]

            df_tr = fetch_by_months(
                args.train,
                tr_months,
                read_cols,
                sample_n=args.train_max_rows,
                seed=seed + fold,
                take_all=0,
            )
            df_va = fetch_by_months(
                args.train,
                va_months,
                read_cols,
                sample_n=None,
                seed=seed + 100 + fold,
                take_all=args.val_take_all,
            )

            if args.include_micro:
                df_tr = add_microstructure(df_tr)
                df_va = add_microstructure(df_va)

            df_tr["date"] = pd.to_datetime(df_tr["timestamp"]).dt.normalize()
            df_va["date"] = pd.to_datetime(df_va["timestamp"]).dt.normalize()

            # Ensure strict time order for causal operations
            df_tr = df_tr.sort_values("timestamp").reset_index(drop=True)
            df_va = df_va.sort_values("timestamp").reset_index(drop=True)

            # embargo：剔除靠近验证窗口的训练样本
            if args.embargo_days > 0 and not df_va.empty:
                va_min, va_max = df_va["date"].min(), df_va["date"].max()
                gap = pd.Timedelta(days=args.embargo_days)
                mask = (df_tr["date"] < va_min - gap) | (df_tr["date"] > va_max + gap)
                df_tr = df_tr.loc[mask].copy()

            df_tr = coerce_numeric(df_tr, feat_cols + ["label"])
            df_va = coerce_numeric(df_va, feat_cols + ["label"])

            # MoD normalization (train-only stats) if enabled
            if getattr(args, "mod_norm", 0):
                if getattr(args, "mod_only_micro", 1):
                    mod_cols = [
                        c for c in ["ob_imb", "trade_imb", "volume"] if c in feat_cols
                    ]
                else:
                    mod_cols = feat_cols[:]  # 不建议，但保留开关
                if mod_cols:
                    df_tr, df_va = mod_norm_train_apply(
                        df_tr,
                        df_va,
                        mod_cols,
                        winsor_p=float(getattr(args, "mod_winsor_p", 0.0)),
                        min_count=int(getattr(args, "mod_min_count", 50)),
                        shrink_k=float(getattr(args, "mod_shrink_k", 0.0)),
                    )

            # XS 规范化（minute/timestamp 分组；当启用 MoD 时跳过，避免过度处理）
            if (not getattr(args, "mod_norm", 0)) and args.perday_norm:
                df_tr = xs_winsorize_zscore_safe(
                    df_tr, feat_cols, winsor_p=args.winsor_p, group_mode=args.xs_group
                )
                df_va = xs_winsorize_zscore_safe(
                    df_va, feat_cols, winsor_p=args.winsor_p, group_mode=args.xs_group
                )

            # Causal time-series normalization（可选）
            if (
                hasattr(args, "ts_norm_win")
                and args.ts_norm_win
                and args.ts_norm_win > 0
            ):
                t3 = time.time()
                df_tr = ts_rolling_zscore(
                    df_tr,
                    feat_cols,
                    window=int(args.ts_norm_win),
                    by_day=bool(args.ts_norm_by_day),
                )
                df_va = ts_rolling_zscore(
                    df_va,
                    feat_cols,
                    window=int(args.ts_norm_win),
                    by_day=bool(args.ts_norm_by_day),
                )
                print(
                    f"[seed={seed} fold={fold}] TS causal norm done in {time.time() - t3:.2f}s "
                    f"(win={args.ts_norm_win}, by_day={args.ts_norm_by_day})"
                )

            Xtr = df_tr[feat_cols].astype(np.float32)
            Xva = df_va[feat_cols].astype(np.float32)
            # Label construction: optional TRAIN-only per-day z-score
            if getattr(args, "label_perday_z", 0):

                def _z(s):
                    m = s.mean()
                    v = s.std(ddof=0)
                    return (s - m) / (v + 1e-12)

                df_tr["label_z"] = df_tr.groupby("date", sort=False)["label"].transform(
                    _z
                )
                ytr = df_tr["label_z"].astype(np.float32)
            else:
                ytr = df_tr["label"].astype(np.float32)
            # Eval always uses raw label (correlation invariant to scaling)
            yva = df_va["label"].astype(np.float32)

            # Optional per-day balancing: equal total weight per day on TRAIN
            if getattr(args, "perday_balance", 0):
                n_per_day = (
                    df_tr.groupby("date", sort=False)["label"]
                    .transform("size")
                    .astype(np.float32)
                )
                w = 1.0 / n_per_day
                w = w * (len(w) / float(w.sum()))
            else:
                w = None

            # Debug checks: feature variance on val; singleton share if minute groups
            try:
                nz_ratio = float((Xva.std(axis=0) > 1e-8).mean())
                if args.xs_group == "minute":
                    vc = (
                        pd.to_datetime(df_va["timestamp"])
                        .dt.floor("min")
                        .value_counts()
                        if len(df_va)
                        else pd.Series(dtype=int)
                    )
                    share_singleton = (
                        float((vc == 1).mean()) if len(vc) else float("nan")
                    )
                    print(
                        f"[DBG seed={seed} fold={fold}] val non-zero-std ratio={nz_ratio:.3f} | "
                        f"minute groups={len(vc)} singleton share={share_singleton:.3f}"
                    )
                else:
                    print(
                        f"[DBG seed={seed} fold={fold}] val non-zero-std ratio={nz_ratio:.3f}"
                    )
            except Exception:
                pass

            model = lgb.LGBMRegressor(
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                n_estimators=args.n_estimators,
                min_data_in_leaf=args.min_data_in_leaf,
                feature_fraction=args.feature_fraction,
                bagging_fraction=args.bagging_fraction,
                bagging_freq=args.bagging_freq,
                lambda_l1=args.lambda_l1,
                lambda_l2=args.lambda_l2,
                n_jobs=args.n_jobs,
                random_state=seed,
            )
            model.fit(
                Xtr,
                ytr,
                sample_weight=w,
                eval_set=[(Xva, yva)],
                eval_metric="l2",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=args.early_stopping_rounds,
                        verbose=args.verbose_eval > 0,
                    ),
                    lgb.log_evaluation(period=args.verbose_eval),
                ],
            )
            pred = model.predict(Xva, num_iteration=model.best_iteration_)
            val_r = float(np.corrcoef(pred, yva)[0, 1])
            val_mse = float(np.mean((pred - yva) ** 2))

            if args.xs_metrics:
                summ, df_daily = daily_ic_summary(
                    yva.values, pred, df_va["date"].values
                )
                print(
                    f"[seed={seed} fold={fold}] r={val_r:.5f} mse={val_mse:.5f} "
                    f"meanIC={summ['mean_ic']:.5f} medIC={summ['median_ic']:.5f} "
                    f"+rate={summ['pos_rate']:.3f} days={summ['days']}"
                )
                df_daily.to_csv(
                    Path(args.outdir) / f"daily_ic_seed{seed}_fold{fold}.csv",
                    index=False,
                )
            else:
                summ = {"mean_ic": np.nan}

            fold_summ.append((val_r, val_mse, summ["mean_ic"]))

            if args.save_model:
                bst_path = Path(args.outdir) / f"lgbm_seed{seed}_fold{fold}.txt"
                model.booster_.save_model(
                    str(bst_path), num_iteration=model.best_iteration_
                )
                print(f"[INFO] model saved: {bst_path}")

            if args.save_oof:
                df_oof = pd.DataFrame(
                    {
                        "seed": seed,
                        "fold": fold,
                        "row_id": np.arange(len(df_va), dtype=np.int64),
                        "timestamp": df_va["timestamp"].values,
                        "date": df_va["date"].values,
                        "pred": pred.astype(np.float32),
                        "label": yva.values.astype(np.float32),
                    }
                )
                oof_records.append(df_oof)

        fs = np.array(fold_summ, dtype=float)
        mean_r, std_r = fs[:, 0].mean(), fs[:, 0].std(ddof=0)
        mean_mse, std_mse = fs[:, 1].mean(), fs[:, 1].std(ddof=0)
        mean_ic = np.nanmean(fs[:, 2])
        print(
            "\n[SEED %d summary] r=%.6f±%.6f  mse=%.6f±%.6f  mean_daily_IC=%.6f\n"
            % (seed, mean_r, std_r, mean_mse, std_mse, mean_ic)
        )

        if args.save_oof and oof_records:
            oof_df = pd.concat(oof_records, ignore_index=True)
            outp = Path(args.outdir) / f"oof_seed{seed}.parquet"
            oof_df.to_parquet(outp, index=False)
            print(f"[INFO] OOF saved: {outp}")

        overall_seed_summ.append(
            {
                "seed": seed,
                "mean_r": mean_r,
                "std_r": std_r,
                "mean_mse": mean_mse,
                "std_mse": std_mse,
                "mean_daily_ic": mean_ic,
            }
        )

    print("\n====== OVERALL SUMMARY (LGBM+XS) ======")
    if overall_seed_summ:
        df = pd.DataFrame(
            overall_seed_summ,
            columns=["seed", "mean_r", "std_r", "mean_mse", "std_mse", "mean_daily_ic"],
        )
        print(df.to_string(index=False))
        out_csv = Path(args.outdir) / "cv_summary_by_seed.csv"
        out_csv.write_text(df.to_csv(index=False), encoding="utf-8")
    else:
        print("No seeds summarized (empty).")


if __name__ == "__main__":
    main()
