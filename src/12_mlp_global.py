# 12_mlp_global.py — memory-safe global MLP:
# - train-only stats (winsor + standardize)
# - test streaming with DuckDB; generate row_id if missing
# - optional micro features (ob_imb, trade_imb) computed on-the-fly
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    ap = argparse.ArgumentParser(
        "Global MLP (train-only norm, memory-safe test reading)"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)  # 可指向原始 data/test.parquet 或精简版
    ap.add_argument("--feature_file", required=True)
    ap.add_argument("--include_micro", type=int, default=0)
    ap.add_argument("--winsor_p", type=float, default=0.01)
    ap.add_argument("--label_perday_z", type=int, default=1)
    ap.add_argument("--use_holdout_last_month", type=int, default=1)

    # model & train
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="reports/mlp_global")

    # test streaming
    ap.add_argument(
        "--chunk_rows",
        type=int,
        default=200_000,
        help="test chunk rows for streaming predict",
    )
    return ap.parse_args()


def read_features(p):
    return [
        x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()
    ]


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_micro(df: pd.DataFrame) -> pd.DataFrame:
    # derive ob_imb/trade_imb; safe no-op if raw cols missing
    eps = 1e-9
    df = df.copy()
    if {"bid_qty", "ask_qty"}.issubset(df.columns):
        df["ob_imb"] = (df["bid_qty"] - df["ask_qty"]) / (
            df["bid_qty"] + df["ask_qty"] + eps
        )
    if {"buy_qty", "sell_qty", "volume"}.issubset(df.columns):
        df["trade_imb"] = (df["buy_qty"] - df["sell_qty"]) / (df["volume"] + eps)
    return df


def winsor_fit_transform(X: pd.DataFrame, p: float):
    """Fit winsor thresholds on TRAIN and return (X_clipped, lo, hi).
    p<=0 -> no-op, returns original X and None thresholds."""
    if p <= 0 or p >= 0.5:
        return X.copy(), None, None
    lo = X.quantile(p)
    hi = X.quantile(1 - p)
    Xc = X.clip(lower=lo, upper=hi, axis=1)
    return Xc, lo.astype(np.float32), hi.astype(np.float32)


def winsor_apply(X: pd.DataFrame, lo, hi):
    """Apply TRAIN winsor thresholds to TEST."""
    if lo is None or hi is None:
        return X
    # 对缺失列做兜底：仅对存在于 TEST 的列做裁剪
    lo2 = lo[lo.index.intersection(X.columns)]
    hi2 = hi[hi.index.intersection(X.columns)]
    Xc = X.copy()
    for c in lo2.index:
        Xc[c] = Xc[c].clip(lower=float(lo2[c]), upper=float(hi2[c]))
    return Xc


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout=0.1):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def pearson_corr(pred, y, eps=1e-8):
    vx = pred - pred.mean()
    vy = y - y.mean()
    num = (vx * vy).sum()
    den = torch.sqrt((vx * vx).sum() + eps) * torch.sqrt((vy * vy).sum() + eps)
    return num / (den + eps)


def read_parquet_cols(path, cols):
    # 用 duckdb 只选必要列（节省内存）
    col_sql = ", ".join([f'"{c}"' for c in cols])
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT {col_sql} FROM read_parquet(?)", [path]).df()
    finally:
        con.close()


def main():
    a = parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    feats = read_features(a.feature_file)

    # 训练侧：读取最少列（timestamp/label + feats + 可选微结构原始列）
    base_cols = ["timestamp", "label"]
    raw_micro = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    need_raw_micro = bool(a.include_micro)
    train_cols = list(
        dict.fromkeys(base_cols + feats + (raw_micro if need_raw_micro else []))
    )
    tr = read_parquet_cols(a.train, train_cols)

    # 训练：衍生 micro（仅当 include_micro=1 时且原始列存在）
    if a.include_micro:
        tr = add_micro(tr)
        # 若你希望把 volume 一并纳入（与之前 LGBM 对齐），可在 feats 列表里追加
        feats = list(dict.fromkeys(feats + ["ob_imb", "trade_imb", "volume"]))

    # 训练标签：可选按日 z-score（仅 train；test 没 timestamp）
    y = pd.to_numeric(tr["label"], errors="coerce").astype(np.float32)
    if a.label_perday_z and "timestamp" in tr.columns:
        d = pd.to_datetime(tr["timestamp"]).dt.normalize()
        mu = y.groupby(d).transform("mean")
        sd = y.groupby(d).transform("std").replace(0, np.nan)
        y = ((y - mu) / (sd + 1e-12)).fillna(0.0).astype(np.float32)

    # 训练特征：数值化 + winsor(训练拟合) + 标准化(训练拟合)
    tr = ensure_numeric(tr, feats)
    Xtrain_raw = tr[feats].copy()
    Xtrain_w, lo_thr, hi_thr = winsor_fit_transform(Xtrain_raw, a.winsor_p)
    scaler = StandardScaler().fit(Xtrain_w.fillna(0.0))
    Xtrain = scaler.transform(Xtrain_w.fillna(0.0)).astype(np.float32)

    # holdout = 最后一个月（如有 timestamp），否则 90/10 随机
    if a.use_holdout_last_month and "timestamp" in tr.columns:
        m = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m")
        last_m = m.max()
        idx_val = (m == last_m).values
    else:
        idx = np.arange(len(tr))
        np.random.shuffle(idx)
        cut = int(0.9 * len(idx))
        idx_val = np.zeros(len(tr), dtype=bool)
        idx_val[idx[cut:]] = True

    X_tr, X_val = Xtrain[~idx_val], Xtrain[idx_val]
    y_tr, y_val = y.values[~idx_val], y.values[idx_val]
    print(f"[INFO] train shapes: X_tr={X_tr.shape}  X_val={X_val.shape}")

    # 训练 MLP（MSE + Pearson 复合）
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, num_workers=0)

    model = MLP(Xtrain.shape[1], a.hidden, a.dropout).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True)
    mse = nn.MSELoss()
    best = {"pearson": -1e9, "ep": -1}
    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            pred = model(xb)
            pc = pearson_corr(pred, yb)
            loss = 0.6 * mse(pred, yb) + 0.4 * (1.0 - pc)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # 验证
        model.eval()
        with torch.no_grad():
            pv = []
            for xb, _ in dl_va:
                pv.append(model(xb.to(dev)).cpu().numpy())
            pv = np.concatenate(pv)
            yy = y_val
            r = float(np.corrcoef(pv, yy)[0, 1])
        if r > best["pearson"]:
            best = {"pearson": r, "ep": ep}
            torch.save(model.state_dict(), str(Path(a.outdir) / f"mlp_seed{a.seed}.pt"))
        if ep % 5 == 0:
            print(
                f"[seed={a.seed}] ep={ep} val_pearson={r:.5f} best={best['pearson']:.5f}"
            )

    # === after training loop, save holdout val predictions ===
    model.load_state_dict(
        torch.load(str(Path(a.outdir) / f"mlp_seed{a.seed}.pt"), map_location=dev)
    )
    model.eval()
    with torch.no_grad():
        pv = []
        for xb, yb in dl_va:
            pv.append(model(xb.to(dev)).cpu().numpy())
        pv = np.concatenate(pv).astype(np.float32)

    val_df = pd.DataFrame({"pred": pv, "y": y_val.astype(np.float32)})
    val_path = Path(a.outdir) / f"valpred_global_mlp_seed{a.seed}.parquet"
    val_df.to_parquet(val_path, index=False)
    print(f"[INFO] holdout valpred saved: {val_path}  shape={val_df.shape}")

    # ------------------- TEST 流式预测 -------------------
    con = duckdb.connect()
    try:
        # 探测 test 实际列
        sample_cols = (
            con.execute("SELECT * FROM read_parquet(?) LIMIT 1", [a.test])
            .fetchdf()
            .columns
        )
        sample_cols = set(sample_cols)

        # 是否需要在线生成 ob_imb/trade_imb
        need_ob = (
            a.include_micro and ("ob_imb" in feats) and ("ob_imb" not in sample_cols)
        )
        need_trade = (
            a.include_micro
            and ("trade_imb" in feats)
            and ("trade_imb" not in sample_cols)
        )

        # 若需要在线生成 micro，但 test 缺少必要原始列，则报错
        if need_ob and not {"bid_qty", "ask_qty"}.issubset(sample_cols):
            raise SystemExit(
                "include_micro=1 需要 ob_imb，但 test 缺少 bid_qty/ask_qty，"
                "请直接用原始 test.parquet 或重建精简 test 时带上原始 micro 列。"
            )
        if need_trade and not {"buy_qty", "sell_qty", "volume"}.issubset(sample_cols):
            raise SystemExit(
                "include_micro=1 需要 trade_imb，但 test 缺少 buy_qty/sell_qty/volume，"
                "请直接用原始 test.parquet 或重建精简 test 时带上原始 micro 列。"
            )

        # 构造需要读取的列（若需要在线生成 micro 就把原始列读进来）
        read_feat_cols = [
            c for c in feats if c in sample_cols
        ]  # 直接存在于 test 的特征列
        extra_cols = []
        if need_ob:
            extra_cols += ["bid_qty", "ask_qty"]
        if need_trade:
            extra_cols += ["buy_qty", "sell_qty", "volume"]
        extra_cols = [c for c in extra_cols if c in sample_cols]

        # 是否已存在 row_id 列
        has_row_id = "row_id" in sample_cols

        # 基础 SQL：若没有 row_id 就生成；否则直接选现有 row_id
        if has_row_id:
            select_cols = ", ".join(
                ['"row_id"'] + [f'"{c}"' for c in read_feat_cols + extra_cols]
            )
            sql_base = f"SELECT {select_cols} FROM read_parquet(?)"
        else:
            # 生成 row_id，按原始文件顺序编号
            select_cols = ", ".join([f'"{c}"' for c in read_feat_cols + extra_cols])
            sql_base = f"""
                WITH base AS (
                  SELECT row_number() OVER () - 1 AS row_id, {select_cols}
                  FROM read_parquet(?)
                )
                SELECT * FROM base
            """

        # 总行数
        n_test = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [a.test]
        ).fetchone()[0]
        print(
            f"[INFO] test rows = {n_test} | read_feat_cols={len(read_feat_cols)} | extra_cols={extra_cols} | has_row_id={has_row_id}"
        )

        # 加载最佳模型
        model.load_state_dict(
            torch.load(str(Path(a.outdir) / f"mlp_seed{a.seed}.pt"), map_location=dev)
        )
        model.eval()

        all_rowid, all_pred = [], []
        offset = 0
        while offset < n_test:
            limit = min(a.chunk_rows, n_test - offset)
            df = con.execute(
                sql_base + f" LIMIT {limit} OFFSET {offset}", [a.test]
            ).df()
            offset += limit

            # 在线生成 micro（若需要）
            if need_ob or need_trade:
                df = add_micro(df)

            # 组装预测特征矩阵
            need_cols = [c for c in feats if c in df.columns]
            X = df[need_cols].copy()
            X = ensure_numeric(X, need_cols)
            # 用 TRAIN 的 winsor 阈值裁剪 + scaler 标准化
            X = winsor_apply(X, lo_thr, hi_thr)
            X = scaler.transform(X.fillna(0.0)).astype(np.float32)

            with torch.no_grad():
                pt = model(torch.from_numpy(X).to(dev)).cpu().numpy().astype(np.float32)

            all_rowid.append(df["row_id"].values)
            all_pred.append(pt)
            print(f"[PRED] chunk done: {offset}/{n_test}")

        row_id = np.concatenate(all_rowid)
        pred = np.concatenate(all_pred)

        out = pd.DataFrame({"row_id": row_id, "pred": pred})
        outp = Path(a.outdir) / f"pred_global_mlp_seed{a.seed}.parquet"
        out.to_parquet(outp, index=False)
        # 保存验证信息
        (Path(a.outdir) / f"mlp_valid_seed{a.seed}.json").write_text(
            json.dumps({"best_val_pearson": best["pearson"], "epoch": best["ep"]}),
            encoding="utf-8",
        )
        print(
            f"[INFO] test predictions saved: {outp} shape={out.shape} best_val_pearson={best['pearson']:.5f}"
        )
    finally:
        con.close()


if __name__ == "__main__":
    main()
