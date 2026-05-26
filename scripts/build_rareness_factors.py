"""Build per-case rareness_factor sidecar for MERLIN efficacy updates.

Uses the full training parquet (ignores sample_size) so IDF counts match
the corpus. Writes hadm_id + rareness_factor to merlin2.rareness_factors_path.

Usage:
    python scripts/build_rareness_factors.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.evaluate import normalize_icd, safe_parse_true_labels
from src.utils.rareness import compute_rareness_factors, labels_from_column

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    m2_cfg = config.get("merlin2", {})
    patients_file = data_cfg["patients_file"]
    target_col = data_cfg.get("target_col", "ICD_CODES")
    out_path = m2_cfg.get("rareness_factors_path", "data/rareness_factors.parquet")

    logger.info("Loading full train set from %s (no sample_size cap)", patients_file)
    df = pd.read_parquet(patients_file)
    if "hadm_id" not in df.columns:
        df = df.copy()
        df["hadm_id"] = df.index.astype(str)

    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not in {patients_file}")

    label_lists = labels_from_column(df[target_col])
    factors = compute_rareness_factors(label_lists)

    out = pd.DataFrame({"hadm_id": df["hadm_id"].astype(str), "rareness_factor": factors})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %d rows to %s (mean=%.3f, min=%.3f, max=%.3f)",
        len(out),
        out_path,
        out["rareness_factor"].mean(),
        out["rareness_factor"].min(),
        out["rareness_factor"].max(),
    )


if __name__ == "__main__":
    main()
