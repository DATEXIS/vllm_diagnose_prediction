#!/usr/bin/env python3
"""Download W&B run artifacts used by MERLIN replay UI (no Streamlit)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


TABLE_SAMPLE_GLOB = "sample_predictions"
TABLE_META_GLOB = "meta_verifier_instructions"


def _safe_delete(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _enforce_cache_limit(cache_root: Path, max_runs: int = 5) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    run_dirs = [
        p for p in cache_root.iterdir()
        if p.is_dir() and p.name != ".wandb_cache"
    ]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in run_dirs[max_runs:]:
        _safe_delete(stale)


def _load_wandb_api_key(app_config_path: Path | None) -> str | None:
    raw = os.environ.get("WANDB_API_KEY")
    if raw:
        return raw.strip() or None
    if app_config_path and app_config_path.exists():
        cfg = json.loads(app_config_path.read_text())
        k = cfg.get("WANDB_API_KEY")
        return str(k).strip() if k else None
    return None


def _parse_run_path(run_input: str) -> str:
    text = run_input.strip()
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    if text.count("/") == 2:
        return text
    if "/" not in text and text:
        return f"datexis-phd/ICD-prediction/{text}"
    raise ValueError(f"Unsupported run input: {run_input}")


def _value_for_csv(cell: object) -> str:
    if cell is None:
        return ""
    if isinstance(cell, (dict, list)):
        return json.dumps(cell, ensure_ascii=False)
    return str(cell)


def _write_history_csv(history_rows: list[dict[str, object]], dest: Path) -> None:
    if not history_rows:
        return
    keys_ordered: list[str] = []
    seen: set[str] = set()
    for row in history_rows:
        for k in row:
            sk = str(k)
            if sk not in seen:
                seen.add(sk)
                keys_ordered.append(sk)
    out_path = dest / "history_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys_ordered)
        for row in history_rows:
            w.writerow(_value_for_csv(row.get(k)) for k in keys_ordered)


def download_run_files(
    run_path: str,
    dest: Path,
    *,
    app_config_path: Path | None = None,
) -> None:
    import wandb

    key = _load_wandb_api_key(app_config_path)
    if key:
        os.environ["WANDB_API_KEY"] = key

    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    dest.mkdir(parents=True, exist_ok=True)

    table_files: dict[str, str] = {}
    for f in run.files():
        if f.name.startswith("media/table/") and TABLE_SAMPLE_GLOB in f.name:
            table_files["sample"] = f.name
        elif f.name.startswith("media/table/") and TABLE_META_GLOB in f.name:
            table_files["meta"] = f.name

    required = [
        "output.log",
        "config.yaml",
        "wandb-summary.json",
        table_files.get("sample", ""),
        table_files.get("meta", ""),
    ]
    for rel in required:
        if not rel:
            continue
        run.file(rel).download(root=str(dest), replace=True)

    history_rows = list(run.scan_history())
    _write_history_csv(history_rows, dest)

    artifacts_dir = dest / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for art in run.logged_artifacts():
        if "instructions_db" in art.name or "code_stats" in art.name:
            target = artifacts_dir / art.name.replace(":", "_")
            target.mkdir(parents=True, exist_ok=True)
            art.download(root=str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MERLIN replay files from W&B.")
    parser.add_argument(
        "--run-input",
        required=True,
        help="W&B URL, entity/project/run_id path, or short run id",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Root directory for wandb_cache (writes <cache_root>/<run_id>/)",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional config.json containing WANDB_API_KEY if env unset",
    )
    args = parser.parse_args()

    run_path = _parse_run_path(args.run_input)
    run_id = run_path.split("/")[-1]
    dest = args.cache_root / run_id
    cache_root = args.cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    download_run_files(run_path, dest, app_config_path=args.config_json)

    os.utime(dest, None)
    _enforce_cache_limit(cache_root, 5)


if __name__ == "__main__":
    main()
