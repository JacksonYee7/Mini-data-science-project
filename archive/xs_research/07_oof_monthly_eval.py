# 文件：07_oof_monthly_eval.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Monthly IC breakdown for OOF")
    ap.add_argument("--oof", required=True)
    ap.add_argument("--outdir", default="reports/oof_monthly_eval")
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.oof)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # 日 IC
    grp = df.groupby("date")
    daily = (
        grp.apply(
            lambda g: np.corrcoef(g["pred"], g["label"])[0, 1]
            if g["pred"].nunique() > 1 and g["label"].nunique() > 1
            else np.nan
        )
        .rename("ic")
        .to_frame()
        .reset_index()
    )
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    # 月归并
    m = (
        daily.groupby("month")["ic"]
        .agg(
            mean="mean",
            median="median",
            p25=lambda x: np.nanpercentile(x, 25),
            p75=lambda x: np.nanpercentile(x, 75),
            pos_rate=lambda x: np.nanmean(x > 0),
            days="count",
        )
        .reset_index()
    )
    m.to_csv(Path(args.outdir) / "monthly_ic_summary.csv", index=False)
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
