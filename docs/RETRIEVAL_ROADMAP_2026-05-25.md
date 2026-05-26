# MERLIN 2: Retrieval & Memory — Next Steps

**Date:** 2026-05-25  
**Context:** After first large Loop-A run (5k cases, ~48k instructions, Wandb run `merlin2-opus-code`). Loop B (Meta-Verifier) hit context limit on long notes; fixes applied in code for next deploy.

---

## 1. What “inspect retrieval logs” means

Before investing in **Qdrant / hybrid retrieval**, check whether **semantic** retrieval is doing useful work or whether almost everything comes from **FPR/FNR threshold** rules.

### Retrieval path families

| Family | `path` values in logs | Mechanism |
|--------|----------------------|-----------|
| **Semantic** | `sem_complaint`, `sem_illness`, `sem_icd`, … | PubMedBERT cosine similarity: note section or prior ICD `reason` vs instruction `description` |
| **Threshold FPR** | `threshold_fpr` | Runtime warning from `code_stats`: high false-positive rate on a predicted code |
| **Threshold FNR** | `threshold_fnr` | Runtime warning from `code_stats` + co-occurrence: often-missed code linked to prediction |

“Inspect” = **count how often each family fires** (per iteration and overall), then decide if upgrading the semantic engine (Qdrant, BM25+dense) is worth it.

### Where to look

| Source | What you get |
|--------|----------------|
| Wandb `sample_predictions` table | `retrieval_log` column (human-readable, 30 rows) |
| `data/retrieval_events_last.csv` | One row per hit: `hadm_id`, `iteration`, `path`, `trigger_value`, … |
| Wandb charts | `retrieval_pct/semantic_note`, `retrieval_pct/semantic_icd`, `retrieval_pct/threshold_fpr`, `retrieval_pct/threshold_fnr` |
| Job stdout | `Retrieval path mix (N events):` summary lines |
| Offline | `python scripts/inspect_retrieval_logs.py --events data/retrieval_events_last.csv` |

On future runs, `retrieval_events` and `predictions` CSVs are also uploaded as Wandb artifacts (after 2026-05-25 code changes).

### Decision rule (Qdrant yes/no)

| Observation | Action |
|-------------|--------|
| Semantic ≈ 0%, threshold dominates | **Skip Qdrant.** Fix `code_stats`, co-occurrence, thresholds, Loop B instruction growth. |
| Semantic weak (&lt; ~20%) | Tune `sim_note_threshold` / `sim_icd_threshold` and instruction DB first. |
| Semantic is most events, DB huge, retrieval slow | Consider hybrid search for **scale/latency**, not to “turn on” semantic retrieval. |

### Result from 2026-05-25 run (5k cases)

| Path bucket | Share of retrieval events |
|-------------|---------------------------|
| **semantic** | **62.9%** |
| threshold_fnr | 32.9% |
| threshold_fpr | 4.1% |

**Conclusion:** Semantic retrieval is already active. **Do not prioritize Qdrant** until threshold tuning and instruction hygiene are exhausted.

**Caveat:** % of events ≠ % of useful fixes. Many semantic hits may be redundant or low-efficacy. Path mix only shows *which mechanism fired*, not whether F1 improved.

---

## 2. What would actually improve retrieval (prioritized)

### Priority 1 — Threshold & retrieval tuning (do first)

| Step | Status | Notes |
|------|--------|-------|
| Log which paths fire | **Done** (basic) | `retrieval_events` CSV + Wandb charts |
| Log “could fire but skipped” (budget) | **Todo** | `skipped_for_budget` in retriever; not in CSV yet |
| Ablate `sim_note_threshold` (e.g. 0.7 vs 0.85) | **Todo** | At 0.85 semantic already ~63%; lower may add noise |
| Ablate `sim_icd_threshold` | **Todo** | Default 0.70 in `experiment.yaml` |
| Visualize distribution per path / iteration | **Todo** | Histogram of `trigger_value`, events per `t` |
| Top-K per case (vs cosine cutoff only) | **Todo** | Research ablation; you already cap by token budget + efficacy |
| Tune FPR/FNR gates & `per_iteration_token_budget` | **Todo** | FPR only 4% of events in last run |

Config knobs: `configs/experiment.yaml` → `merlin2:` section.

### Priority 2 — Instruction quality (high value at ~48k+ instructions)

| Step | Status | Notes |
|------|--------|-------|
| Cluster + deduplicate instructions | **Todo** | Spec § Loop B maintenance; treat merge as ablation |
| Efficacy decay / instruction age | **Todo** | Efficacy persisted after Loop A; no pruning yet |
| Prune consistently low-efficacy rules | **Todo** | After 1–2 more phases |
| Merge redundant rows (same `target_codes` + `type`) | **Todo** | Reduces prompt clutter and duplicate semantic hits |

Storage: `data/instructions.parquet` (Wandb artifact `instructions_db`).

### Priority 3 — Infrastructure (only if P1/P2 plateau)

| Trigger | Current situation (2026-05-25) |
|---------|-------------------------------|
| Instruction count &gt; 50k | ~48k — brute-force cosine still OK |
| Retrieval latency &gt; 500ms/case | Profile client; wave-1 embed ~57s for 5k cases acceptable for batch |
| Distributed inference | Not needed yet |
| **Qdrant / hybrid BM25+dense** | **Defer** until semantic *should* fire but similarity never does, or latency blocks iteration |

---

## 3. Recommended execution order

1. **[Done]** Path mix from last run → semantic dominant → skip Qdrant for now.
2. **Analyze outcomes** — Wandb: macro/micro F1, per-iteration metrics (`iter/all/f1_macro`). Ask: does retrieval change code sets in a good way?
3. **Threshold ablations** — `sim_note_threshold`, `per_iteration_token_budget`, `threshold_budget_fraction`; one change per run.
4. **Instruction hygiene** — dedup/prune pass on `instructions.parquet` before next full 5k Loop A.
5. **Loop B** — re-run Meta-Verifier after deploy (note truncation + lower `max_tokens`); append new instructions only.
6. **Qdrant** — only if (3)+(4) plateau and logs show semantic misses with similar instructions already in DB.

---

## 4. Phase workflow (unchanged)

```text
Phase 1: Loop A (all train samples halt) → inspect retrieval logs / metrics
Phase 2: Loop B (meta_verifier.enabled: true) → new instructions + code_stats
Phase 3+: Repeat Loop A with populated DB → residual error mining
```

Manual gates between phases. Frozen `code_stats` rates per MERLIN2_SPEC §3.

---

## 5. Key files & commands

| Item | Path / command |
|------|----------------|
| Instruction library | `data/instructions.parquet` (not the patient `.pq` file) |
| Per-code thresholds | `data/code_stats.parquet` |
| Co-occurrence | `data/cooccurrence.parquet` (build: `python scripts/build_cooccurrence.py`) |
| Inspect retrieval | `python scripts/inspect_retrieval_logs.py` |
| Config | `configs/experiment.yaml`, `configs/setup.yaml` |
| Spec | `MERLIN2_SPEC.md` |

---

## 6. Open issues from 2026-05-25 run

- **Loop B crashed:** prompt &gt; 16k context (full admission + discharge). Fixed: `max_note_chars: 5000`, `max_tokens: 2048`, fail-fast on vLLM 400.
- **Logging bug:** `% of` in format string (fixed).
- **Rareness:** computed from batch labels if `rareness_factors.parquet` missing (warning is harmless).
- **Eval:** Micro F1 ≈ 0.29, Macro F1 ≈ 0.10 on final predictions (5k sample).

---

*Update this doc when a phase completes or ablation results change the Qdrant decision.*
