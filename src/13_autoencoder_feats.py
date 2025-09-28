# 13_autoencoder_feats.py — Train AE on train(core+6), append 8 latent features to train/test
import argparse
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
        "AutoEncoder features (train-only norm, holdout early-stop)"
    )
    ap.add_argument("--train", required=True)
    ap.add_argument(
        "--test", required=True
    )  # e.g. data/processed/test_core_plus6.parquet
    ap.add_argument("--feature_file", required=True)
    ap.add_argument(
        "--include_micro", type=int, default=0
    )  # keep 0 to match your final pipelines
    ap.add_argument("--code_dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout_last_month", type=int, default=1)
    ap.add_argument("--chunk_rows", type=int, default=200_000)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--out_feat_file", required=True)
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


def read_cols_parquet(path, cols):
    sql = "SELECT " + ", ".join([f'"{c}"' for c in cols]) + " FROM read_parquet(?)"
    con = duckdb.connect()
    try:
        return con.execute(sql, [path]).df()
    finally:
        con.close()


class AE(nn.Module):
    def __init__(self, d_in, hidden, code_dim, dropout=0.0):
        super().__init__()
        enc_layers = []
        prev = d_in
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers += [nn.Linear(prev, code_dim)]
        self.enc = nn.Sequential(*enc_layers)
        dec_layers = []
        prev = code_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers += [nn.Linear(prev, d_in)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return z, recon


def main():
    a = parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    Path(Path(a.out_train).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(a.out_test).parent).mkdir(parents=True, exist_ok=True)
    feats = read_features(a.feature_file)
    base_cols = [
        "timestamp",
        "label",
    ]  # train 里可用；test_core_plus6 通常只有 row_id + feats
    train_cols = list(dict.fromkeys(base_cols + feats))
    tr = read_cols_parquet(
        a.train, [c for c in train_cols if c in ["timestamp", "label"] + feats]
    )

    if a.include_micro:  # 这里只给接口，默认不加
        tr = add_micro(tr)
        feats = list(dict.fromkeys(feats + ["ob_imb", "trade_imb", "volume"]))

    # to numeric
    for c in feats:
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    trX = tr[feats].astype(np.float32)

    # split train/val by last month (holdout)
    if a.holdout_last_month and "timestamp" in tr.columns:
        m = pd.to_datetime(tr["timestamp"]).dt.strftime("%Y-%m")
        last_m = m.max()
        is_val = (m == last_m).values
    else:
        idx = np.arange(len(trX))
        np.random.shuffle(idx)
        cut = int(0.9 * len(idx))
        is_val = np.zeros(len(trX), bool)
        is_val[idx[cut:]] = True

    # train-only scaler（不含持出月）
    scaler = StandardScaler().fit(trX[~is_val].fillna(0.0))
    Xtr = scaler.transform(trX[~is_val].fillna(0.0))
    Xva = scaler.transform(trX[is_val].fillna(0.0))

    # torch AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = TensorDataset(torch.from_numpy(Xtr))
    ds_va = TensorDataset(torch.from_numpy(Xva))
    dl_tr = DataLoader(ds_tr, batch_size=a.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=a.batch_size, shuffle=False, num_workers=0)

    model = AE(
        d_in=Xtr.shape[1], hidden=a.hidden, code_dim=a.code_dim, dropout=a.dropout
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.MSELoss()
    best = {"val": 1e9, "ep": -1}
    patience, bad = 6, 0

    for ep in range(1, a.epochs + 1):
        model.train()
        for (xb,) in dl_tr:
            xb = xb.to(device)
            _, recon = model(xb)
            loss = crit(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # val
        model.eval()
        with torch.no_grad():
            lossv = []
            for (xb,) in dl_va:
                xb = xb.to(device)
                _, recon = model(xb)
                lossv.append(crit(recon, xb).item())
            v = float(np.mean(lossv)) if lossv else np.nan
        if v < best["val"] - 1e-5:
            best = {"val": v, "ep": ep}
            torch.save(model.state_dict(), str(Path(a.out_train).with_suffix(".ae.pt")))
            bad = 0
        else:
            bad += 1
        if ep % 5 == 0:
            print(f"[AE] ep={ep} val_recon={v:.6f} best={best['val']:.6f}")
        if bad >= patience:
            print("[AE] early stop.")
            break

    # reload best & encode full train
    model.load_state_dict(
        torch.load(str(Path(a.out_train).with_suffix(".ae.pt")), map_location=device)
    )
    model.eval()
    with torch.no_grad():
        Z = []
        dl_all = DataLoader(
            TensorDataset(
                torch.from_numpy(scaler.transform(trX.fillna(0.0).to_numpy()))
            ),
            batch_size=a.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for (xb,) in dl_all:
            xb = xb.to(device)
            z, _ = model(xb)
            Z.append(z.cpu().numpy())
        Z = np.concatenate(Z, axis=0).astype(np.float32)

    ae_cols = [f"ae_{i + 1}" for i in range(a.code_dim)]
    df_train_out = tr[[]].copy()
    if "timestamp" in tr.columns:
        df_train_out["timestamp"] = tr["timestamp"].values
    if "label" in tr.columns:
        df_train_out["label"] = tr["label"].values
    for c in feats:
        df_train_out[c] = tr[c].values
    for i, c in enumerate(ae_cols):
        df_train_out[c] = Z[:, i]
    df_train_out.to_parquet(a.out_train, index=False)
    print(f"[AE] train saved: {a.out_train}  shape={df_train_out.shape}")

    # encode test in chunks
    con = duckdb.connect()
    try:
        # discover available columns
        sample_cols = (
            con.execute("SELECT * FROM read_parquet(?) LIMIT 1", [a.test])
            .fetchdf()
            .columns
        )
        tcols = [c for c in feats if c in sample_cols]
        has_rowid = "row_id" in sample_cols
        col_sql = (["row_id"] if has_rowid else []) + tcols
        sql_base = (
            "SELECT " + ", ".join([f'"{c}"' for c in col_sql]) + " FROM read_parquet(?)"
        )
        n_test = con.execute(
            "SELECT COUNT(*) AS n FROM read_parquet(?)", [a.test]
        ).fetchone()[0]
        print(
            f"[AE] test rows={n_test}  read_cols={len(col_sql)}  has_row_id={has_rowid}"
        )

        out_parts = []
        offset = 0
        while offset < n_test:
            take = min(a.chunk_rows, n_test - offset)
            df = con.execute(sql_base + f" LIMIT {take} OFFSET {offset}", [a.test]).df()
            offset += take
            X = df[tcols].copy()
            for c in tcols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            X = scaler.transform(X.fillna(0.0))
            with torch.no_grad():
                z = []
                dl = DataLoader(
                    TensorDataset(torch.from_numpy(X.astype(np.float32))),
                    batch_size=a.batch_size,
                    shuffle=False,
                )
                for (xb,) in dl:
                    xb = xb.to(device)
                    zz, _ = model(xb)
                    z.append(zz.cpu().numpy())
                z = np.concatenate(z, axis=0).astype(np.float32)
            part = pd.DataFrame({ae_cols[i]: z[:, i] for i in range(len(ae_cols))})
            if has_rowid:
                part["row_id"] = df["row_id"].values
            for c in tcols:
                part[c] = df[c].values  # keep original feats for downstream
            out_parts.append(part)
            print(f"[AE] test chunk {offset}/{n_test} done.")
        df_test_out = pd.concat(out_parts, ignore_index=True)
        # put row_id first if exists
        cols = ["row_id"] if "row_id" in df_test_out.columns else []
        cols += tcols + ae_cols
        df_test_out = df_test_out[cols]
        df_test_out.to_parquet(a.out_test, index=False)
        print(f"[AE] test saved: {a.out_test}  shape={df_test_out.shape}")
    finally:
        con.close()

    # write feature file (old feats + ae_*)
    Path(a.out_feat_file).write_text("\n".join(feats + ae_cols), encoding="utf-8")
    print(
        f"[AE] feature file saved: {a.out_feat_file}  (#={len(feats) + len(ae_cols)})"
    )


if __name__ == "__main__":
    main()
