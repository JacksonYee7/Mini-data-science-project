# 文件：06_eval_oof.py
import argparse

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        "Evaluate OOF parquet (daily IC), with optional within-day label shuffle"
    )
    ap.add_argument("--input", required=True, help="oof_*.parquet")
    ap.add_argument("--shuffle_within", choices=["none", "day"], default="none")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def daily_ic(df):
    rows = []
    for d, g in df.groupby("date", sort=True):
        if g["pred"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        ic = np.corrcoef(g["pred"].values, g["label"].values)[0, 1]
        rows.append((d, ic, len(g)))
    out = pd.DataFrame(rows, columns=["date", "ic", "n"])
    if out.empty:
        return {
            "mean_ic": np.nan,
            "median_ic": np.nan,
            "pos_rate": np.nan,
            "days": 0,
        }, out
    arr = out["ic"].values
    return {
        "mean_ic": float(np.nanmean(arr)),
        "median_ic": float(np.nanmedian(arr)),
        "pos_rate": float(np.mean(arr > 0)),
        "days": int(out.shape[0]),
    }, out


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)
    need = {"date", "pred", "label"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"missing columns: {miss}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    if args.shuffle_within == "day":
        rng = np.random.RandomState(args.seed)

        def _perm(s):
            idx = np.arange(len(s))
            rng.shuffle(idx)
            return s.values[idx]

        df = df.copy()
        df["label"] = df.groupby("date")["label"].transform(_perm)

    summ, perday = daily_ic(df)
    print(
        f"[OOF Eval] meanIC={summ['mean_ic']:.5f}  medianIC={summ['median_ic']:.5f}  +rate={summ['pos_rate']:.3f}  days={summ['days']}"
    )
    perday.to_csv(
        str(args.input).replace(".parquet", ".eval_daily_ic.csv"), index=False
    )


if __name__ == "__main__":
    main()
