# 06_oof_weight_search_blocked.py
# 对 3 条窗口级 OOF 做“按月 LOMO”凸权重搜索，给出每月最优权重、平均/中位权重，并导出最终 OOF
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser("Blocked (LOMO-by-month) convex weight search for 3 window-level OOFs")
    ap.add_argument("--oof_240", required=True)
    ap.add_argument("--oof_360", required=True)
    ap.add_argument("--oof_480", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="final ensembled oof (parquet)")
    ap.add_argument("--metric_out", default=None)
    ap.add_argument("--weights_csv", default=None, help="save per-month best weights & scores")
    ap.add_argument("--agg", choices=["mean","median"], default="mean", help="aggregate monthly weights")
    return ap.parse_args()

def load_oof(p):
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    return df[["timestamp","date","pred","label"]].copy()

def daily_ic_summary(y, p, dates):
    tmp = pd.DataFrame({"y": y, "p": p, "date": dates}).replace([np.inf,-np.inf], np.nan).dropna()
    rows = []
    for d,g in tmp.groupby("date", sort=True):
        if g["p"].nunique()<2 or g["y"].nunique()<2: continue
        rows.append((d, np.corrcoef(g["p"].values, g["y"].values)[0,1]))
    if not rows:
        return np.nan
    ic = pd.Series([r for _,r in rows], dtype=float)
    return float(ic.mean())

def search_weights(base, mask, step):
    y = base.loc[mask, "label"].values
    dates = base.loc[mask, "date"].values
    P240 = base.loc[mask, "p240"].values
    P360 = base.loc[mask, "p360"].values
    P480 = base.loc[mask, "p480"].values
    best = (-1, (0,0,0))
    ws = np.arange(0, 1+1e-9, step)
    for w1 in ws:
        for w2 in ws:
            w3 = 1 - w1 - w2
            if w3 < -1e-9: continue
            if w3 < 0: w3 = 0
            s = w1*P240 + w2*P360 + w3*P480
            ic = daily_ic_summary(y, s, dates)
            if np.isnan(ic): continue
            if ic > best[0]:
                best = (ic, (w1,w2,w3))
    return best  # (meanIC, (w1,w2,w3))

def main():
    args = parse_args()
    d240 = load_oof(args.oof_240).rename(columns={"pred":"p240"})
    d360 = load_oof(args.oof_360).rename(columns={"pred":"p360"})
    d480 = load_oof(args.oof_480).rename(columns={"pred":"p480"})
    base = d240.merge(d360, on=["timestamp","date"], how="inner") \
               .merge(d480, on=["timestamp","date"], how="inner")
    base["month"] = pd.to_datetime(base["date"]).dt.to_period("M").astype(str)

    months = sorted(base["month"].unique())
    recs = []
    for m in months:
        train_mask = base["month"] != m
        test_mask  = base["month"] == m
        best_train_ic, (w1,w2,w3) = search_weights(base, train_mask, args.step)
        # eval on held-out month
        ic_test = daily_ic_summary(
            base.loc[test_mask,"label"].values,
            (w1*base.loc[test_mask,"p240"].values +
             w2*base.loc[test_mask,"p360"].values +
             w3*base.loc[test_mask,"p480"].values),
            base.loc[test_mask,"date"].values
        )
        recs.append({"month": m, "w240": w1, "w360": w2, "w480": w3,
                     "train_meanIC": best_train_ic, "heldout_meanIC": ic_test})
        print(f"[LOMO] month={m}  best_trainIC={best_train_ic:.5f}  heldoutIC={ic_test:.5f}  "
              f"w=(240:{w1:.2f},360:{w2:.2f},480:{w3:.2f})")

    wdf = pd.DataFrame(recs)
    if args.weights_csv:
        Path(args.weights_csv).parent.mkdir(parents=True, exist_ok=True)
        wdf.to_csv(args.weights_csv, index=False)
        print(f"[INFO] per-month weights saved: {args.weights_csv}")

    # aggregate weights
    if args.agg == "mean":
        W = wdf[["w240","w360","w480"]].mean().values
    else:
        W = wdf[["w240","w360","w480"]].median().values
    # normalize to sum=1 in case of numeric drift
    W = np.maximum(W, 0)
    if W.sum() == 0: W = np.array([1/3,1/3,1/3])
    W = W / W.sum()
    w1,w2,w3 = W.tolist()
    print(f"[FINAL] aggregated weights ({args.agg}) -> (240:{w1:.3f}, 360:{w2:.3f}, 480:{w3:.3f})")

    # export final ensembled OOF
    pred = w1*base["p240"].values + w2*base["p360"].values + w3*base["p480"].values
    out = pd.DataFrame({"timestamp": base["timestamp"].values,
                        "date": base["date"].values,
                        "pred": pred.astype(np.float32),
                        "label": base["label"].values.astype(np.float32)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    ic_all = daily_ic_summary(out["label"].values, out["pred"].values, out["date"].values)
    print(f"[FINAL] meanIC on all days = {ic_all:.5f}")

    if args.metric_out:
        Path(args.metric_out).write_text(
            f'{{"mean_ic_all": {ic_all:.6f}, "w240": {w1:.3f}, "w360": {w2:.3f}, "w480": {w3:.3f}}}',
            encoding="utf-8"
        )
        print(f"[INFO] metrics saved: {args.metric_out}")

if __name__ == "__main__":
    main()
