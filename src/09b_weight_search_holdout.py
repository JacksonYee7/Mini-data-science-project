# -*- coding: utf-8 -*-
# 09b_weight_search_holdout.py — search weights on holdout, apply to test preds
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Weight search on holdout, fuse test")
    ap.add_argument(
        "--valpreds",
        nargs="+",
        required=True,
        help="val preds parquet(s) with columns [pred,y]",
    )
    ap.add_argument(
        "--testpreds",
        nargs="+",
        required=True,
        help="test preds parquet(s) with columns [row_id,pred]",
    )
    ap.add_argument("--mode", choices=["grid", "ridge"], default="grid")
    ap.add_argument(
        "--step", type=float, default=0.05, help="grid step (for grid mode)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric_out", default="")
    return ap.parse_args()


def pearson(y, p):
    y = y - y.mean()
    p = p - p.mean()
    den = np.sqrt((y * y).sum()) * np.sqrt((p * p).sum())
    return float((y * p).sum() / (den + 1e-12))


def main():
    a = parse_args()
    assert len(a.valpreds) == len(a.testpreds), "valpreds and testpreds must align"

    # load holdout preds
    V = []
    for p in a.valpreds:
        df = pd.read_parquet(p)
        if not {"pred", "y"}.issubset(df.columns):
            raise SystemExit(f"{p} must have [pred,y]")
        V.append(df["pred"].to_numpy(dtype=np.float64))
        y = df["y"].to_numpy(dtype=np.float64)
    V = np.vstack(V).T  # [n, m]
    m = V.shape[1]

    # load test preds & align by row_id
    T = None
    row_id = None
    for i, p in enumerate(a.testpreds):
        df = (
            pd.read_parquet(p)[["row_id", "pred"]]
            .copy()
            .rename(columns={"pred": f"pred_{i}"})
        )
        if T is None:
            T = df
            row_id = df["row_id"].values
        else:
            T = T.merge(df, on="row_id", how="inner")
    test_mat = T[[c for c in T.columns if c.startswith("pred_")]].to_numpy(
        dtype=np.float64
    )

    if a.mode == "grid":
        step = a.step
        grid = np.arange(0.0, 1.0 + 1e-9, step)
        best = (-1.0, None)
        if m == 2:
            for w0 in grid:
                w = np.array([w0, 1.0 - w0])
                r = pearson(y, V @ w)
                if r > best[0]:
                    best = (r, w)
        elif m == 3:
            for w0 in grid:
                for w1 in grid:
                    if w0 + w1 > 1.0:
                        continue
                    w = np.array([w0, w1, 1.0 - w0 - w1])
                    r = pearson(y, V @ w)
                    if r > best[0]:
                        best = (r, w)
        else:
            # simple dirichlet-like normalized random search
            rng = np.random.default_rng(42)
            for _ in range(10000):
                raw = rng.random(m)
                w = raw / raw.sum()
                r = pearson(y, V @ w)
                if r > best[0]:
                    best = (r, w)
        best_r, w = best
        print(f"[WOPT] holdout pearson={best_r:.5f}  weights={w}")
    else:
        # ridge stacking
        from sklearn.linear_model import Ridge

        alphas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
        best = (-1.0, None)
        for a0 in alphas:
            mdl = Ridge(alpha=a0, fit_intercept=True).fit(V, y)
            p = mdl.predict(V)
            r = pearson(y, p)
            if r > best[0]:
                best = (r, mdl.coef_ / np.sum(np.abs(mdl.coef_)))
        best_r, w = best
        print(f"[RIDGE] holdout pearson={best_r:.5f}  weights={w}")

    # apply to test
    ens = (test_mat * w).sum(axis=1).astype(np.float32)
    out = pd.DataFrame({"row_id": row_id, "pred": ens})
    Path(Path(a.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"[INFO] stacked test saved: {a.out}  shape={out.shape}")

    if a.metric_out:
        Path(a.metric_out).write_text(
            f"holdout_pearson={best_r:.6f}\nweights={w}\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
