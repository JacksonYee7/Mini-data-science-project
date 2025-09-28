# 10_xgb_global_v2.py — Global XGBoost (train-only transforms, holdout early-stop via callbacks, memory-safe test)
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb


def parse_args():
    ap = argparse.ArgumentParser(
        "Global XGBoost v2 (callbacks early-stop, label_perday_z)"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--feature_file", required=True)
    ap.add_argument("--include_micro", type=int, default=0)
    ap.add_argument("--winsor_p", type=float, default=0.01)
    ap.add_argument(
        "--label_perday_z", type=int, default=1
    )  # <<< 新增：与 MLP/LGBM 对齐
    ap.add_argument("--use_holdout_last_month", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--n_estimators", type=int, default=3000)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--outdir", default="reports/infer_global_v2_xgb")
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
        # 统一 float32，避免后续 setitem 报 dtype 警告
        return df.astype(np.float32)
    tmp = df[cols].astype(np.float32)
    lo = tmp.quantile(p)
    hi = tmp.quantile(1 - p)
    df[cols] = tmp.clip(lower=lo, upper=hi, axis=1)
    return df.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)


def predict_with_best(model, X):
    """兼容各版本 XGB 的 best-iteration 预测"""
    # 优先使用 sklearn API 暴露的 best_iteration
    bi = getattr(model, "best_iteration", None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=(0, int(bi) + 1))
        except Exception:
            pass
    # 尝试 booster 的 best_ntree_limit / best_iteration
    try:
        booster = model.get_booster()
        if hasattr(booster, "best_iteration"):
            bi = int(booster.best_iteration)
            return model.predict(X, iteration_range=(0, bi + 1))
        if hasattr(booster, "best_ntree_limit"):
            return model.predict(X, ntree_limit=int(booster.best_ntree_limit))
    except Exception:
        pass
    # 兜底：用全部树
    return model.predict(X)


def main():
    a = parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    feats = read_features(a.feature_file)

    # ---------- 读 train ----------
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

    # y（可选：按日 z-score，与 MLP/LGBM 对齐）
    y = pd.to_numeric(tr["label"], errors="coerce").astype(np.float32)
    if a.label_perday_z and "timestamp" in tr.columns:
        d = pd.to_datetime(tr["timestamp"]).dt.normalize()
        mu = y.groupby(d).transform("mean")
        sd = y.groupby(d).transform("std").replace(0, np.nan)
        y = ((y - mu) / (sd + 1e-12)).fillna(0.0).astype(np.float32)

    # ---------- holdout 切分：最后一个月 ----------
    if a.use_holdout_last_month and "timestamp" in tr.columns:
        m = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m")
        last_m = m.max()
        val_idx = (m == last_m).values
    else:
        idx = np.arange(len(tr))
        np.random.shuffle(idx)
        cut = int(0.9 * len(idx))
        val_idx = np.zeros(len(tr), dtype=bool)
        val_idx[idx[cut:]] = True

    X_tr = tr.loc[~val_idx, feats].copy()
    y_tr = y[~val_idx].values
    X_va = tr.loc[val_idx, feats].copy()
    y_va = y[val_idx].values

    # ---------- winsor：train 统计，val/test 复用 ----------
    wz = winsorize(X_tr, feats, a.winsor_p)
    if isinstance(wz, tuple):
        X_tr, lo, hi = wz
    else:
        X_tr = wz.astype(np.float32)
        lo = hi = None
    X_va = X_va.astype(np.float32)
    if lo is not None:
        X_va = X_va.clip(lower=lo, upper=hi, axis=1)
    X_va = X_va.fillna(0.0)
    X_tr = X_tr.fillna(0.0)

    # ---------- XGB 训练（早停用 callbacks，版本兼容） ----------
    params = dict(
        learning_rate=a.learning_rate,
        max_depth=a.max_depth,
        n_estimators=a.n_estimators,
        subsample=a.subsample,
        colsample_bytree=a.colsample_bytree,
        min_child_weight=a.min_child_weight,
        reg_lambda=a.lambda_l2,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=a.seed,
        n_jobs=a.n_jobs,
    )
    model = xgb.XGBRegressor(**params)

    try:
        from xgboost.callback import EarlyStopping

        cbs = [EarlyStopping(rounds=a.early_stopping_rounds, save_best=True)]
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, callbacks=cbs)
    except Exception:
        # 极旧版本兜底：无 callbacks 就全树训练（无早停）
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # ---------- 保存 holdout 预测（给 stacking 用） ----------
    p_va = predict_with_best(model, X_va).astype(np.float32)
    val_out = pd.DataFrame({"pred": p_va, "y": y_va.astype(np.float32)})
    val_path = Path(a.outdir) / f"valpred_global_xgb_seed{a.seed}.parquet"
    val_out.to_parquet(val_path, index=False)
    print(f"[XGB] holdout preds saved: {val_path}  shape={val_out.shape}")

    # ---------- 流式预测 test ----------
    con = duckdb.connect()
    try:
        sample_cols = (
            con.execute("SELECT * FROM read_parquet(?) LIMIT 1", [a.test])
            .fetchdf()
            .columns
        )
        has_rowid = "row_id" in sample_cols
        tcols = [c for c in feats if c in sample_cols]
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
            f"[XGB] test rows={n_test}  read_cols={len(sel_cols)}  has_row_id={has_rowid}"
        )

        parts = []
        offset = 0
        while offset < n_test:
            take = min(a.chunk_rows, n_test - offset)
            df = con.execute(sqlb + f" LIMIT {take} OFFSET {offset}", [a.test]).df()
            offset += take

            X = df[tcols].copy().astype(np.float32)
            if lo is not None:
                X = X.clip(lower=lo, upper=hi, axis=1)
            X = X.fillna(0.0)
            p = predict_with_best(model, X).astype(np.float32)
            part = pd.DataFrame({"pred": p})
            if has_rowid:
                part["row_id"] = df["row_id"].values
            parts.append(part)
            print(f"[XGB] chunk {offset}/{n_test} done.")

        pred = pd.concat(parts, ignore_index=True)
        if has_rowid:
            pred = pred[["row_id", "pred"]]
        outp = Path(a.outdir) / f"pred_global_xgb_seed{a.seed}.parquet"
        pred.to_parquet(outp, index=False)
        print(f"[XGB] test preds saved: {outp}  shape={pred.shape}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
