# MERLIN 2 — Changelog (staged work)

**Date:** 2026-05-22  
**Branch:** `merlin/merlin2-hrusheekesh`  
**Status:** 19 files staged, not committed (+657 / −37 lines)

This document summarizes implementation changes made for MERLIN 2 Loop A/B robustness, retrieval observability, rareness-weighted efficacy, K8s deploy, and run post-mortems. For forward-looking retrieval priorities, see [`RETRIEVAL_ROADMAP_2026-05-25.md`](RETRIEVAL_ROADMAP_2026-05-25.md).

---

## Motivation

| Problem | Fix in this batch |
|---------|-------------------|
| Loop-A efficacy scores lost after run | Persist to `instructions.parquet` + Wandb artifact |
| No offline view of semantic vs threshold retrieval | `retrieval_events_last.csv`, stdout summary, inspect script |
| `rareness_factor` always 1.0 | Batch IDF from ground-truth codes; optional full-train sidecar |
| Loop B 400 / context overflow on long notes | Note truncation + lower `max_tokens` |
| vLLM errors swallowed as `None` | Fail-fast `RuntimeError` in `run_inference_with_system` |
| Pod deleted → lost CSVs | Wandb file artifacts for retrieval + predictions |
| Wrong namespace/registry for your cluster | `yxsg9647` in `setup.yaml` |
| Client GPU always requested | `gpu_count` gate in K8s template |

---

## Config & deploy

### `configs/setup.yaml`

- Docker registry: `registry.datexis.com/yxsg9647`
- K8s namespace: `yxsg9647`

### `configs/experiment.yaml`

| Key | Before → after |
|-----|----------------|
| `k8s.client.gpu_count` | (implicit 1) → `1` |
| `k8s.client.gpu_type` | `a100` → `h200` |
| `merlin2.min_support` | `2` → `3` |
| `merlin2.rareness_factors_path` | — → `data/rareness_factors.parquet` (optional) |
| `merlin2.compute_rareness_at_load` | — → `true` |
| `merlin2.retrieval_events_path` | — → `data/retrieval_events_last.csv` |
| `meta_verifier.enabled` | `false` → `true` |
| `meta_verifier.max_tokens` | `4096` → `2048` |
| `meta_verifier.max_note_chars` | — → `5000` |

Server block unchanged in staged diff (`b200` × 1).

### `Dockerfile` + `entrypoint.sh` (new)

- **No pip at image build time** — `entrypoint.sh` runs `pip install -r requirements.txt` on container start, then `exec "$@"`.
- Co-occurrence + mimic data still copied at build.
- `rareness_factors.parquet` COPY line commented out (optional build step).

### `scripts/k8s_templates.py`

- Client container uses `args` instead of `command` (compatible with image `ENTRYPOINT`).
- GPU limits/requests and `nodeSelector` only when `k8s.client.gpu_count > 0`.
- `nodeSelector` includes `gpu` and `kubernetes.io/hostname` when GPU client is used (requires `hostname` in config if that template branch is active).

---

## Loop A — pipeline & memory

### `src/main.py`

1. **`_merge_rareness_factors()`** — Merge optional `rareness_factors.parquet`, else `compute_rareness_at_load` from `true_codes`, else default `1.0` with warning.
2. **`persist_efficacy_updates()`** after Loop A when ground truth is available — writes efficacy scores for persistent instructions only; re-uploads `instructions_db` Wandb artifact if any row updated.
3. **`retrieval_events_last.csv`** — Flatten per-hit retrieval events from pipeline results; log path; upload Wandb artifact `retrieval_events`.
4. **`_log_retrieval_path_summary()`** — Stdout bucket counts: `semantic`, `threshold_fpr`, `threshold_fnr`, `other` (fixes prior logging bug using invalid `% of` format).
5. **Predictions artifact** — `wandb_logger.log_file_artifact("predictions", ...)` after saving CSV.

### `src/meta_verifier/store.py`

- **`persist_efficacy_updates(efficacy_by_id, path)`** — Load parquet, update matching `instruction_id` efficacy scores, save, return count updated.

### `src/merlin2/retriever.py`

- **`persistent_instructions`** property — Rows from `instructions.parquet` only (excludes runtime FP/FN warning rows used for retrieval but not stored).

### `src/utils/rareness.py` (new)

- Mean smoothed IDF over each case’s 3-digit ground-truth codes.
- Used by `main.py` at load and by `scripts/build_rareness_factors.py` for full-corpus sidecar.

---

## Loop B — Meta-Verifier

### `src/meta_verifier/meta_verifier.py`

- **`max_note_chars`** (default 5000) — Truncate admission/discharge in audit prompts with explicit suffix.
- **`max_tokens`** default aligned to 2048 (config-driven).
- **`_truncate_note()`** helper.

### `src/inference.py`

- **`run_inference_with_system`**: non-200 vLLM responses raise `RuntimeError` with status + body snippet (no silent `None`, no broad try/except swallowing).

---

## Observability & tooling

### `src/utils/wandb_logger.py`

- **`log_file_artifact(name, type, path)`** — Generic CSV/file upload when pod storage is ephemeral.

### `scripts/inspect_retrieval_logs.py` (new)

```bash
python scripts/inspect_retrieval_logs.py
python scripts/inspect_retrieval_logs.py --events data/retrieval_events_last.csv
```

Prints bucket percentages (semantic / threshold_fpr / threshold_fnr) and a short recommendation (Qdrant defer vs tune thresholds).

### `scripts/build_rareness_factors.py` (new)

```bash
python scripts/build_rareness_factors.py --config configs/experiment.yaml
```

Builds full-train `hadm_id` + `rareness_factor` parquet at `merlin2.rareness_factors_path`. **Not required** when `compute_rareness_at_load: true`.

### `docs/RETRIEVAL_ROADMAP_2026-05-25.md` (new)

Operational roadmap: what “inspect retrieval logs” means, last-run path mix (~63% semantic), prioritized next steps, Qdrant deferral rationale.

---

## Tests

| File | Coverage |
|------|----------|
| `tests/test_store_efficacy.py` | `persist_efficacy_updates` round-trip |
| `tests/test_rareness.py` | Rare case gets higher factor than common-only case |
| `tests/test_parsing.py` | `test_salvage_complete_entries_from_truncated_object` |

**Known gap:** Salvage test expects truncated-JSON recovery in `parse_prediction`, but **`src/utils/parsing_utils.py` is not in this diff** — run `pytest tests/test_parsing.py` before commit; either implement salvage or drop the test.

---

## Other

- **`.gitignore`**: ignore `UI/wandb_cache/`

---

## Not in this change set

| Item | Notes |
|------|--------|
| `src/utils/parsing_utils.py` truncated JSON salvage | Test only; implementation not staged |
| Force CPU embeddings (`embeddings.py`) | Discussed for P100; reverted / not staged |
| Qdrant / hybrid retrieval | Deferred per roadmap |
| PVC for data | Not added; data via Docker image + Wandb artifacts |

---

## File index (staged)

```
.gitignore
Dockerfile
entrypoint.sh
configs/setup.yaml
configs/experiment.yaml
docs/RETRIEVAL_ROADMAP_2026-05-25.md
docs/CHANGELOG_MERLIN2_2026-05-22.md   # this file (add to git when committing)
scripts/build_rareness_factors.py
scripts/inspect_retrieval_logs.py
scripts/k8s_templates.py
src/main.py
src/inference.py
src/merlin2/retriever.py
src/meta_verifier/meta_verifier.py
src/meta_verifier/store.py
src/utils/rareness.py
src/utils/wandb_logger.py
tests/test_parsing.py
tests/test_rareness.py
tests/test_store_efficacy.py
```

---

## Suggested next steps

1. Rebuild and push client image (`entrypoint.sh` + code changes).
2. Run Loop B with truncation limits; confirm no 16k context errors.
3. After Loop A, download Wandb artifacts `retrieval_events`, `predictions`, updated `instructions_db`.
4. `python scripts/inspect_retrieval_logs.py` on saved CSV before threshold ablations.
5. Fix or remove salvage parsing test before merge.
