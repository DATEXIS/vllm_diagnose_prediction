#!/usr/bin/env python3
"""Convert predictions parquet to CSV for analysis. Run locally where pyarrow is available."""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "predictions_no_notes.parquet"
out  = path.replace(".parquet", ".csv")

df = pd.read_parquet(path)
# Drop heavy columns we don't need for analysis
drop_cols = [c for c in ["think_block", "full_diagnoses"] if c in df.columns]
df = df.drop(columns=drop_cols)
df.to_csv(out, index=False)
print(f"Written {len(df)} rows → {out}")
