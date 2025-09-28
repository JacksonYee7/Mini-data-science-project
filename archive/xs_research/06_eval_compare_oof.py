# 06_eval_compare_oof.py
# 比较两个 OOF（A vs B）的日 IC，做配对差异统计（均值差、sign test）
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Compare two OOFs by daily IC (paired)")
    ap.add_argument("--a", required=True, help="OOF A parquet")
    ap.add_argument("--b", required=True, help="OOF B parquet")
    ap.add_argument("--out_csv", default=None)
    return ap.parse_args()


def daily_ic(y, p, d):
    tmp = (
        pd.DataFrame({"y": y, "p": p, "date": d})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    rows = []
    for dd, g in tmp.groupby("date", sort=True):
        if g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        rows.append((dd, np.corrcoef(g["p"].values, g["y"].values)[0, 1]))
    return pd.DataFrame(rows, columns=["date", "ic"])


def load_oof(p):
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    return df


def main():
    args = parse_args()
    A = load_oof(args.a)
    B = load_oof(args.b)
    M = A.merge(B, on=["timestamp", "date", "label"], suffixes=("_a", "_b"))
    icA = daily_ic(M["label"].values, M["pred_a"].values, M["date"].values)
    icB = daily_ic(M["label"].values, M["pred_b"].values, M["date"].values)
    D = icA.merge(icB, on="date", how="inner", suffixes=("_a", "_b"))
    D["delta"] = D["ic_b"] - D["ic_a"]
    mean_delta = float(D["delta"].mean())
    med_delta = float(D["delta"].median())
    pos_rate = float((D["delta"] > 0).mean())
    n = int(D.shape[0])
    # 正态近似的 t 统计（只是参考）
    se = float(D["delta"].std(ddof=1)) / np.sqrt(max(n, 1))
    t = mean_delta / se if se > 0 else np.nan
    print(
        f"[Compare] n={n}  meanΔ={mean_delta:.6f}  medianΔ={med_delta:.6f}  +rate={pos_rate:.3f}  t≈{t:.2f}"
    )
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        D.to_csv(args.out_csv, index=False)
        print(f"[INFO] per-day delta saved: {args.out_csv}")


if __name__ == "__main__":
    main()
