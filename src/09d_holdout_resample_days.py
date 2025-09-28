# 09d_holdout_resample_days.py — day-level resampling to check stacking robustness
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument(
    "--valpreds", nargs="+", required=True, help="each parquet has [pred,y]"
)
ap.add_argument(
    "--dates",
    required=True,
    help='parquet with a single column "date" aligned with valpred rows',
)
ap.add_argument(
    "--w_grid",
    default="",
    help="comma weights for grid (length = #models); if empty, skip",
)
ap.add_argument(
    "--w_ridge_clip", default="", help="comma weights for ridge-clip; if empty, skip"
)
ap.add_argument("--n_rounds", type=int, default=200)
ap.add_argument("--sample_frac", type=float, default=0.8)
ap.add_argument("--out", default="")
args = ap.parse_args()

dfs = [pd.read_parquet(p) for p in args.valpreds]
n = len(dfs[0])
for i, df in enumerate(dfs):
    if df.shape[0] != n:
        raise SystemExit(f"len mismatch at {i}")
    if not {"pred", "y"}.issubset(df.columns):
        raise SystemExit(f"columns missing in {i}")
y = dfs[0]["y"].values.astype(np.float32)
P = np.column_stack([d["pred"].values.astype(np.float32) for d in dfs])

dates = pd.read_parquet(args.dates)["date"].values
if len(dates) != n:
    raise SystemExit("dates length mismatch")

m = P.shape[1]
weq = np.ones(m, dtype=float) / m


def parsew(s):
    if not s:
        return None
    w = np.array([float(x) for x in s.split(",")], dtype=float)
    if len(w) != m:
        raise SystemExit("weights length != #models")
    if (w < 0).any():  # allow negative for ridge-raw; for ridge-clip应该非负
        # 不归一，直接按原样；下方会 /sum(|w|) 再比较
        pass
    return w


w_grid = parsew(args.w_grid)
w_rc = parsew(args.w_ridge_clip)


def score(w, idx):
    w0 = w.copy()
    s = np.sum(np.abs(w0))
    if s > 0:
        w0 = w0 / s
    pred = (P[idx] @ w0).astype(np.float32)
    return float(np.corrcoef(pred, y[idx])[0, 1])


uniq_days = np.unique(dates)
rng = np.random.default_rng(7)

res = {"eq": []}
if w_grid is not None:
    res["grid"] = []
if w_rc is not None:
    res["ridge_clip"] = []

for _ in range(args.n_rounds):
    take = rng.choice(
        uniq_days, size=int(args.sample_frac * len(uniq_days)), replace=False
    )
    idx = np.where(np.isin(dates, take))[0]
    res["eq"].append(score(weq, idx))
    if w_grid is not None:
        res["grid"].append(score(w_grid, idx))
    if w_rc is not None:
        res["ridge_clip"].append(score(w_rc, idx))

for k, v in res.items():
    arr = np.array(v)
    print(
        f"{k:10s} mean={arr.mean():.5f}  median={np.median(arr):.5f}  p90={np.percentile(arr, 90):.5f}"
    )

if args.out:
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(args.out, index=False)
