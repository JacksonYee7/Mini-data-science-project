# 文件：06_oof_ensemble.py  （增加 --mode: mean | day_zscore | invnorm_rank）
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Ensemble OOF predictions with several modes")
    ap.add_argument(
        "--inputs", nargs="+", required=True, help="Paths to OOF parquet files"
    )
    ap.add_argument("--out", required=True, help="Output ensembled OOF parquet")
    ap.add_argument("--metric_out", default=None, help="Path to save metrics json")
    ap.add_argument(
        "--mode",
        choices=["mean", "day_zscore", "invnorm_rank"],
        default="mean",
        help="Ensembling mode",
    )
    return ap.parse_args()


def load_oof(p: str) -> pd.DataFrame:
    df = pd.read_parquet(p)
    # 必备列：timestamp, pred, label；最佳：还有 date
    need = {"timestamp", "pred", "label"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{p} missing columns: {missing}")
    if "date" not in df.columns:
        # 兜底：从 timestamp 提取
        df["date"] = pd.to_datetime(df["timestamp"]).normalize()
    return df[["timestamp", "date", "pred", "label"]].copy()


def outer_join_on_timestamp_date(dfs):
    # 依次外连接，按 timestamp & date 对齐
    base = dfs[0].rename(columns={"pred": "pred_0"})
    for i, d in enumerate(dfs[1:], start=1):
        d = d.rename(columns={"pred": f"pred_{i}"})
        # 对齐键
        base = base.merge(
            d, on=["timestamp", "date"], how="outer", suffixes=("", f"_{i}")
        )
        # label 以左为准，若左缺再用右
        if "label_y" in base.columns and "label_x" in base.columns:
            base["label"] = base["label_x"].fillna(base["label_y"])
            base.drop(columns=["label_x", "label_y"], inplace=True)
    # 保留列顺序：timestamp, date, label, preds...
    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    cols = ["timestamp", "date", "label"] + pred_cols
    base = base[cols]
    # 按时间排序
    base = base.sort_values(["date", "timestamp"]).reset_index(drop=True)
    return base


def invnorm(u):
    """
    Approximate N(0,1) inverse CDF (probit) using Acklam's rational approximation.
    Avoids scipy/numpy.special dependency.
    """
    import numpy as np

    p = np.asarray(u, dtype=np.float64)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow
    x = np.empty_like(p)

    # lower region
    mask_low = p < plow
    if np.any(mask_low):
        q = np.sqrt(-2 * np.log(p[mask_low]))
        x[mask_low] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        x[mask_low] = -x[mask_low]

    # central region
    mask_cen = (p >= plow) & (p <= phigh)
    if np.any(mask_cen):
        q = p[mask_cen] - 0.5
        r = q * q
        x[mask_cen] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    # upper region
    mask_high = p > phigh
    if np.any(mask_high):
        q = np.sqrt(-2 * np.log(1 - p[mask_high]))
        x[mask_high] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    return x


def ens_mean(pred_mat: pd.DataFrame) -> np.ndarray:
    return np.nanmean(pred_mat.values, axis=1)


def ens_day_zscore(df: pd.DataFrame, pred_cols):
    # 对每个 pred 列，按日做 z-score，再行均值
    zcols = []
    for c in pred_cols:
        g = df.groupby("date", sort=False)[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        z = (df[c] - mu) / (sd + 1e-12)
        zcols.append(z)
    Z = np.vstack([z.values for z in zcols]).T
    return np.nanmean(Z, axis=1)


def ens_invnorm_rank(df: pd.DataFrame, pred_cols):
    # 每列：按日排名 -> rank/(n+1) -> 正态分位 -> 行均值
    out = np.zeros(len(df), dtype=np.float64)
    for d, idx in df.groupby("date", sort=False).groups.items():
        part = df.loc[idx, pred_cols]
        # 对每一列单独求秩
        ranks = []
        for c in pred_cols:
            s = part[c]
            # rank: 1..n, method='average' to handle ties
            r = s.rank(method="average")
            u = r / (r.size + 1.0)
            z = invnorm(u)
            ranks.append(z)
        Z = np.vstack(ranks).T  # (n, m)
        out[idx] = np.nanmean(Z, axis=1)
    return out


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
    args = parse_args()
    dfs = [load_oof(p) for p in args.inputs]
    base = outer_join_on_timestamp_date(dfs)
    pred_cols = [c for c in base.columns if c.startswith("pred_")]
    if not pred_cols:
        raise SystemExit("No pred_* columns after join.")

    if args.mode == "mean":
        ens = ens_mean(base[pred_cols])
    elif args.mode == "day_zscore":
        ens = ens_day_zscore(base, pred_cols)
    else:  # invnorm_rank
        ens = ens_invnorm_rank(base, pred_cols)

    out_df = pd.DataFrame(
        {
            "timestamp": base["timestamp"].values,
            "date": base["date"].values,
            "pred": ens.astype(np.float32),
            "label": base["label"].values.astype(np.float32),
        }
    )
    # 评估
    summ, df_daily = daily_ic_summary(
        out_df["label"].values, out_df["pred"].values, out_df["date"].values
    )
    print(
        f"[Ensemble daily IC] mode={args.mode} "
        f"mean={summ['mean_ic']:.5f}  median={summ['median_ic']:.5f}  "
        f"+rate={summ['pos_rate']:.3f}  days={summ['days']}"
    )
    # 保存
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    df_daily.to_csv(Path(args.out).with_suffix(".daily_ic.csv"), index=False)
    if args.metric_out:
        Path(args.metric_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metric_out).write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"[INFO] Ensembled OOF saved: {args.out}")


if __name__ == "__main__":
    main()
