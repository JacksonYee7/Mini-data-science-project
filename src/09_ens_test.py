# 文件：09_ens_test.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Row-wise ensemble for test predictions")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="pred parquet(s) with columns [row_id,pred]",
    )
    ap.add_argument(
        "--weights",
        default="",
        help="comma weights, same length as inputs; default=equal",
    )
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    dfs = [pd.read_parquet(p) for p in args.inputs]
    for i, df in enumerate(dfs):
        if "row_id" not in df.columns or "pred" not in df.columns:
            raise SystemExit(f"{args.inputs[i]} must have columns [row_id,pred]")
        dfs[i] = df[["row_id", "pred"]].copy().rename(columns={"pred": f"pred_{i}"})
    base = dfs[0]
    for i in range(1, len(dfs)):
        base = base.merge(dfs[i], on="row_id", how="inner")
    cols = [c for c in base.columns if c.startswith("pred_")]
    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")], dtype=float)
        if len(w) != len(cols):
            raise SystemExit("weights length != #models")
        w = w / w.sum()
    else:
        w = np.ones(len(cols), dtype=float) / len(cols)
    P = base[cols].to_numpy()
    ens = (P * w).sum(axis=1).astype(np.float32)
    out = base[["row_id"]].copy()
    out["pred"] = ens
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[INFO] ensembled saved: {args.out}  shape={out.shape}")


if __name__ == "__main__":
    main()
