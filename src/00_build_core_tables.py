# 00_build_core_tables.py — select minimal columns for downstream (train/test)
import argparse
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser("Build minimal core tables from raw train/test")
    ap.add_argument("--train_raw", required=True, help="data/train.parquet")
    ap.add_argument("--test_raw", required=True, help="data/test.parquet")
    ap.add_argument("--feature_file", required=True, help="assets/features/core_plus6.txt")
    ap.add_argument("--include_micro", type=int, default=0, help="if 1 add micro-derived cols (optional)")
    ap.add_argument("--out_train", default="data/processed/train_core_plus6_rawmicro.parquet")
    ap.add_argument("--out_test", default="data/processed/test_core_plus6.parquet")
    return ap.parse_args()


def read_features(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def add_micro_inplace(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    if {"bid_qty", "ask_qty"}.issubset(df.columns):
        df["ob_imb"] = (
            pd.to_numeric(df["bid_qty"], errors="coerce")
            - pd.to_numeric(df["ask_qty"], errors="coerce")
        ) / (
            pd.to_numeric(df["bid_qty"], errors="coerce")
            + pd.to_numeric(df["ask_qty"], errors="coerce")
            + eps
        )
    if {"buy_qty", "sell_qty", "volume"}.issubset(df.columns):
        v = pd.to_numeric(df["volume"], errors="coerce")
        df["trade_imb"] = (
            pd.to_numeric(df["buy_qty"], errors="coerce")
            - pd.to_numeric(df["sell_qty"], errors="coerce")
        ) / (v + eps)
    return df


def main():
    a = parse_args()
    Path(Path(a.out_train).parent).mkdir(parents=True, exist_ok=True)

    feats = read_features(a.feature_file)
    # Train wants timestamp,label + feats (+ raw micro columns if we want to compute imbalances)
    train_cols = ["timestamp", "label"] + feats + (
        ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"] if a.include_micro else []
    )
    # Test wants row_id + feats (+ raw micro for optional derivations). No timestamp by design
    test_cols = ["row_id"] + feats + (
        ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"] if a.include_micro else []
    )

    con = duckdb.connect()
    try:
        tr = con.execute(
            "SELECT " + ", ".join([f'"{c}"' for c in train_cols if c]) + " FROM read_parquet(?)",
            [a.train_raw],
        ).df()
        te = con.execute(
            "SELECT " + ", ".join([f'"{c}"' for c in test_cols if c]) + " FROM read_parquet(?)",
            [a.test_raw],
        ).df()
    finally:
        con.close()

    if a.include_micro:
        tr = add_micro_inplace(tr)
        te = add_micro_inplace(te)

    # enforce numeric for feature columns
    used_feat_cols = feats + (["ob_imb", "trade_imb", "volume"] if a.include_micro else [])

    def coerce_num(df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    tr = coerce_num(tr, used_feat_cols + ["label"])  # label numeric for downstream
    te = coerce_num(te, used_feat_cols)

    tr.to_parquet(a.out_train, index=False)
    te.to_parquet(a.out_test, index=False)
    print(f"[INFO] train core saved: {a.out_train} shape={tr.shape} cols={len(tr.columns)}")
    print(f"[INFO] test  core saved: {a.out_test}  shape={te.shape} cols={len(te.columns)}")


if __name__ == "__main__":
    main()

