# 09e_build_holdout_dates.py — build holdout (last-month) dates aligned with valpred order
import argparse
from pathlib import Path

import duckdb
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

con = duckdb.connect()
# 找最后一月
last_m = con.execute(
    "SELECT max(strftime(timestamp,'%Y-%m')) FROM read_parquet(?)", [args.train]
).fetchone()[0]
df = con.execute(
    """
    SELECT timestamp
    FROM read_parquet(?)
    WHERE strftime(timestamp,'%Y-%m')=?
""",
    [args.train, last_m],
).df()
con.close()

d = pd.to_datetime(df["timestamp"]).dt.normalize()
out = pd.DataFrame({"date": d.astype("datetime64[ns]")})
Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
out.to_parquet(args.out, index=False)
print(
    f"[INFO] holdout month={last_m}  days={out['date'].nunique()}  rows={len(out)}  saved={args.out}"
)
