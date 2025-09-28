# 11_check_submission.py — sanity check for prediction parquet(s)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def describe_one(p):
    df = pd.read_parquet(p)
    print(f"\n[INFO] file = {p}")
    print(f"shape = {df.shape}")
    print(f"columns = {list(df.columns)}")
    need_cols = {"row_id", "pred"}
    if not need_cols.issubset(df.columns):
        raise SystemExit(f"{p} must contain columns {need_cols}")

    # dtypes
    print("[dtypes]")
    print(df.dtypes)

    # NA / dup
    na = df.isna().mean().to_dict()
    print(f"\n[NA ratio] {na}")
    dups = df.duplicated(subset=["row_id"]).sum()
    print(f"[Dup row_id] {dups}")

    # basic stats
    s = (
        df["pred"]
        .astype(float)
        .describe(percentiles=[0.001, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 0.999])
    )
    print("\n[pred describe()]")
    print(s)

    # extreme tails
    q001 = df["pred"].quantile(0.001)
    q999 = df["pred"].quantile(0.999)
    print(f"[tails] q0.1%={q001:.6f}  q99.9%={q999:.6f}")

    # monotonic row_id check (非硬性，但方便定位异常)
    is_sorted = (df["row_id"].values == np.sort(df["row_id"].values)).all()
    print(f"[row_id sorted] {is_sorted}")

    return df[["row_id", "pred"]].copy()


def compare_align(dfs, names):
    print("\n[ALIGNMENT across files]")
    base = dfs[0][["row_id"]].copy()
    ok = True
    for i in range(1, len(dfs)):
        m = base.merge(dfs[i], on="row_id", how="inner")
        ok_i = len(m) == len(base) == len(dfs[i])
        print(f"align with {names[i]}: {ok_i}  common={len(m)}")
        ok = ok and ok_i
    if not ok:
        print("WARNING: row_id misalignment detected.")
    else:
        print("row alignment across files: True")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Check prediction parquet(s)")
    ap.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="one or multiple parquet with [row_id,pred]",
    )
    args = ap.parse_args()

    Path(".").mkdir(exist_ok=True)
    dfs = []
    for p in args.files:
        dfs.append(describe_one(p))
    if len(dfs) >= 2:
        compare_align(dfs, args.files)
