"""Summarize MERLIN retrieval path mix from a saved events CSV.

After a Loop-A run, main.py writes data/retrieval_events_last.csv (or the
path in merlin2.retrieval_events_path). Use this script to decide whether
semantic retrieval is doing work vs threshold-only.

Usage:
    python scripts/inspect_retrieval_logs.py
    python scripts/inspect_retrieval_logs.py --events data/retrieval_events_last.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.merlin2.retriever import THRESHOLD_FPR, THRESHOLD_FNR, is_semantic_path


def _bucket(path: str) -> str:
    if path == THRESHOLD_FPR:
        return "threshold_fpr"
    if path == THRESHOLD_FNR:
        return "threshold_fnr"
    if is_semantic_path(path):
        return "semantic"
    return "other"


def summarize(events_path: Path) -> None:
    if not events_path.exists():
        print(f"No file at {events_path}")
        print("Run Loop A first (src/main.py) or pass --events <csv>.")
        sys.exit(1)

    df = pd.read_csv(events_path)
    if "path" not in df.columns:
        raise KeyError(f"{events_path} must have a 'path' column")

    df["bucket"] = df["path"].map(_bucket)
    total = len(df)
    print(f"File: {events_path}")
    print(f"Total retrieval events: {total}")
    if total == 0:
        print("No events — empty DB or all cases halted at t=0 (empty_db)?")
        return

    print("\nBy bucket:")
    for bucket, count in df["bucket"].value_counts().items():
        print(f"  {bucket:16s} {100 * count / total:5.1f}%  ({count})")

    if "iteration" in df.columns:
        print("\nBy iteration (bucket counts):")
        for t, grp in df.groupby("iteration"):
            n = len(grp)
            sem = (grp["bucket"] == "semantic").sum()
            fpr = (grp["bucket"] == "threshold_fpr").sum()
            fnr = (grp["bucket"] == "threshold_fnr").sum()
            print(
                f"  t={t}: n={n}  semantic={sem} ({100*sem/n:.0f}%)  "
                f"fpr={fpr}  fnr={fnr}"
            )

    sem_pct = 100 * (df["bucket"] == "semantic").sum() / total
    thr_pct = 100 * df["bucket"].str.startswith("threshold").sum() / total
    print("\nRecommendation:")
    if sem_pct < 5 and thr_pct > 50:
        print(
            "  Semantic retrieval is nearly idle; threshold paths dominate. "
            "Skip Qdrant for now. Focus on code_stats + cooccurrence + Loop B."
        )
    elif sem_pct < 20:
        print(
            "  Semantic retrieval is weak. Lower sim_note_threshold / sim_icd_threshold "
            "or grow instructions.parquet before hybrid search."
        )
    else:
        print(
            "  Semantic retrieval is active. Hybrid search may help once the "
            "instruction DB is large enough to stress brute-force cosine."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=str,
        default="data/retrieval_events_last.csv",
        help="CSV from merlin2.retrieval_events_path",
    )
    args = parser.parse_args()
    summarize(Path(args.events))


if __name__ == "__main__":
    main()
