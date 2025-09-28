# 08c_lgbm_global_holdout.py — LGBM global, full last-month holdout, save valpred+testpred
import argparse
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser("Global LGBM (full last-month holdout)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--feature_file", required=True)
    ap.add_argument("--include_micro", type=int, default=0)
    ap.add_argument("--winsor_p", type=float, default=0.01)
    ap.add_argument("--label_perday_z", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--n_estimators", type=int, default=4000)
    ap.add_argument("--min_data_in_leaf", type=int, default=200)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l1", type=float, default=0.0)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--outdir", default="reports/infer_global_v2_lgbm_ae_fullholdout")
    return ap.parse_args()


def read_features(p):
    return [
        x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def add_micro(df):
    eps = 1e-9
    df = df.copy()
    if {"bid_qty", "ask_qty"}.issubset(df.columns):
        df["ob_imb"] = (df["bid_qty"] - df["ask_qty"]) / (
            (df["bid_qty"] + df["ask_qty"]) + eps
        )
    if {"buy_qty", "sell_qty", "volume"}.issubset(df.columns):
        df["trade_imb"] = (df["buy_qty"] - df["sell_qty"]) / (df["volume"] + eps)
    return df


def winsorize(df, cols, p):
    if p <= 0 or p >= 0.5:
        return df.astype(np.float32), None, None
    tmp = df[cols].astype(np.float32)
    lo = tmp.quantile(p)
    hi = tmp.quantile(1 - p)
    df[cols] = tmp.clip(lower=lo, upper=hi, axis=1)
    return df.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)


def main():
    a = parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    feats = read_features(a.feature_file)

    # 读 train
    cols = list(dict.fromkeys(["timestamp", "label"] + feats))
    con = duckdb.connect()
    try:
        tr = con.execute(
            "SELECT "
            + ", ".join([f'"{c}"' for c in cols if c])
            + " FROM read_parquet(?)",
            [a.train],
        ).df()
    finally:
        con.close()

    if a.include_micro:
        tr = add_micro(tr)
        feats = list(dict.fromkeys(feats + ["ob_imb", "trade_imb", "volume"]))

    for c in feats:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    y = pd.to_numeric(tr["label"], errors="coerce").astype(np.float32)

    # y 按日 z-score（与 MLP/XGB 对齐）
    if a.label_perday_z and "timestamp" in tr.columns:
        d = pd.to_datetime(tr["timestamp"]).dt.normalize()
        mu = y.groupby(d).transform("mean")
        sd = y.groupby(d).transform("std").replace(0, np.nan)
        y = ((y - mu) / (sd + 1e-12)).fillna(0.0).astype(np.float32)

    # holdout = 最后一个月（用满所有行）
    m = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m")
    last_m = m.max()
    val_idx = (m == last_m).values

    X_tr = tr.loc[~val_idx, feats].copy()
    y_tr = y[~val_idx].values
    X_va = tr.loc[val_idx, feats].copy()
    y_va = y[val_idx].values

    # winsor（train 统计）
    X_tr, lo, hi = winsorize(X_tr, feats, a.winsor_p)
    X_va = X_va.astype(np.float32)
    if lo is not None:
        X_va = X_va.clip(lower=lo, upper=hi, axis=1)
    X_tr = X_tr.fillna(0.0)
    X_va = X_va.fillna(0.0)

    model = lgb.LGBMRegressor(
        learning_rate=a.learning_rate,
        num_leaves=a.num_leaves,
        n_estimators=a.n_estimators,
        min_data_in_leaf=a.min_data_in_leaf,
        feature_fraction=a.feature_fraction,
        bagging_fraction=a.bagging_fraction,
        bagging_freq=a.bagging_freq,
        lambda_l1=a.lambda_l1,
        lambda_l2=a.lambda_l2,
        n_jobs=a.n_jobs,
        random_state=a.seed,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=a.early_stopping_rounds, verbose=False),
        ],
    )
    # 保存 holdout 预测
    p_va = model.predict(X_va, num_iteration=model.best_iteration_).astype(np.float32)
    val_df = pd.DataFrame({"pred": p_va, "y": y_va.astype(np.float32)})
    vp = Path(a.outdir) / f"valpred_global_lgb_seed{a.seed}.parquet"
    val_df.to_parquet(vp, index=False)
    print(f"[LGB] holdout preds saved: {vp}  shape={val_df.shape}")

    # 流式 test 预测
    con = duckdb.connect()
    try:
        cols_test = (
            con.execute("SELECT * FROM read_parquet(?) LIMIT 1", [a.test])
            .fetchdf()
            .columns
        )
        has_rowid = "row_id" in cols_test
        tcols = [c for c in feats if c in cols_test]
        sel_cols = (["row_id"] if has_rowid else []) + tcols
        sqlb = (
            "SELECT "
            + ", ".join([f'"{c}"' for c in sel_cols])
            + " FROM read_parquet(?)"
        )
        n_test = con.execute(
            "SELECT COUNT(*) AS n FROM read_parquet(?)", [a.test]
        ).fetchone()[0]
        print(
            f"[LGB] test rows={n_test}  read_cols={len(sel_cols)}  has_row_id={has_rowid}"
        )

        parts = []
        offset = 0
        while offset < n_test:
            take = min(a.chunk_rows, n_test - offset)
            df = con.execute(sqlb + f" LIMIT {take} OFFSET {offset}", [a.test]).df()
            offset += take
            X = df[tcols].astype(np.float32)
            if lo is not None:
                X = X.clip(lower=lo, upper=hi, axis=1)
            X = X.fillna(0.0)
            p = model.predict(X, num_iteration=model.best_iteration_).astype(np.float32)
            part = pd.DataFrame({"pred": p})
            if has_rowid:
                part["row_id"] = df["row_id"].values
            parts.append(part)
            print(f"[LGB] chunk {offset}/{n_test} done.")

        pred = pd.concat(parts, ignore_index=True)
        if has_rowid:
            pred = pred[["row_id", "pred"]]
        outp = Path(a.outdir) / f"pred_global_lgb_seed{a.seed}.parquet"
        pred.to_parquet(outp, index=False)
        print(f"[LGB] test preds saved: {outp}  shape={pred.shape}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
