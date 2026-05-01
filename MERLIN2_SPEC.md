# MERLIN 2: System Architecture Specification

**Project Goal:** A dual-loop, multi-agent framework to iteratively improve ICD coding from clinical admission notes (using MIMIC-IV). The system uses an active inference loop for real-time prediction and an asynchronous memory loop to build a global, contrastive error taxonomy, addressing long-tail coding errors without constant fine-tuning.

**Label Space:** Predictions and metrics operate on the **first 3 digits** of ICD codes (~2k classes), reducing the long tail and giving FPR/FNR enough support per code to be meaningful.

---

## 1. Core Agents & Components

### Generator (G)
* **Role:** The primary clinical coding LLM.
* **Infrastructure:** vLLM with Guided Decoding.
* **Output Schema:** Strict JSON list (Fields: `icd_code`, `reason`).
* **Execution:** Processes the admission note and iterates sequentially through retrieved instructions via a pre-filled `<think>` block before finalizing its JSON prediction. The retrieved instructions *are* the reasoning content — the design choice is deliberate: the memory bank provides better, contrastive thinking than the model would generate unaided, and the resulting traces double as future SFT data.
* **Concurrency:** Iterations run in lockstep across the batch. All samples complete iteration `t` before any sample starts iteration `t+1`. After each iteration, samples that meet a halting condition are filtered out; the remaining samples proceed to `t+1`.

### Meta-Verifier (M)
* **Role:** Asynchronous clinical auditor and taxonomy author. **Runs only on training data.** Never used at test time.
* **Model:** Same vLLM-served model as the Generator (for now). The task is easier than coding from scratch because M sees the answer.
* **Inputs (per case):** `admission_note`, `prediction` (the Generator's final JSON), `ground_truth_codes`, `discharge_note`, `hadm_id`. The discharge note is the new signal not available at inference time.
* **Hard constraint — no leakage into case-level instructions:** Persisted instructions must be useful for a Generator that *never sees the discharge note*. The Meta-Verifier prompt must explicitly forbid LLM outputs of the form *"the discharge note mentions X"* or *"you should have read the discharge note"*. Each persisted instruction must be grounded in cues that exist in the **admission note**. (Synthesised threshold warnings have no leakage risk — their text is generated from `code_stats` by deterministic templates and references the predicted / cooccurring code only.)
* **Prompt:** Lives in `configs/` (not in code) so it can be iterated without rebuilds. Starts basic — accepts the four inputs, requests JSON output conforming to the Instruction Schema, includes the leakage prohibition. Iterated empirically once we see the first batch of generated instructions.
* **Two error-discovery paths, two different sinks:**
  1. **Case-level analysis (semantic instructions):** Compares prediction against ground truth + discharge note, identifies root causes, writes contrastive instructions to be retrieved later by **semantic similarity** to the admission note. May relate to multiple ICD codes. Output: rows appended to `instructions.parquet`.
  2. **Aggregate metrics (per-code threshold rules):** Computes per-code FPR/FNR over the audit batch. When a code's FPR or FNR exceeds a threshold AND meets `min_support`, emits a row to a separate per-code stats table (`code_stats.parquet`) — *not* an Instruction row. The Retriever synthesises the warning text at runtime from the stats; nothing is persisted in the instruction store. A code already present in `code_stats.parquet` is left untouched (frozen-rate semantics, see §3).
* **Instruction Schema (persistent rows in `instructions.parquet`):**
  * `instruction_id` — primary key. Sequential int.
  * `type` — `semantic` or `contrastive_swap`. Threshold-warning types (`fp_warning`, `fn_warning`) are reserved for runtime-synthesised instances and never appear in the persistent store.
  * `instruction_text` — the thinking-style content injected into the Generator's `<think>` block
  * `description` / `quote` — short text used as the embedding target for semantic retrieval (PubMedBERT)
  * `target_codes` — 3-digit ICD codes the instruction relates to (one or more).
  * `source_hadm_ids` — list of `hadm_id`s the instruction was derived from. Non-empty for case-level instructions.
  * `efficacy_score` — running score; see *Efficacy Score Update Rule* below.
* **CodeStats Schema (rows in `code_stats.parquet`):**
  * `code` — 3-digit ICD code (primary key).
  * `fpr` / `fnr` — frozen rate snapshot from the Loop-B pass that first registered this code. Exactly one of the two is set; the other is null.
  * `support_pred` / `support_true` — case counts from the same pass, used for the runtime warning text.
  * No `efficacy_score`. Synthesised threshold warnings have no per-warning efficacy (see *Efficacy Score Update Rule*).
* **FPR / FNR definitions (3-digit code level):**
  * `FPR(c) = (# cases where c was predicted but not in ground truth) / (# cases where c was predicted)`
  * `FNR(c) = (# cases where c was in ground truth but not predicted) / (# cases where c was in ground truth)`
  * Codes below a minimum-support count are not eligible for `code_stats` rows (rates are too noisy).

### Hybrid-Retriever (R)
* **Role:** The memory fetcher during active inference.
* **Embedding model:** **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` or equivalent S-PubMedBERT variant) for both admission notes and instruction `description`/`quote` fields.
* **Inputs at construction time:** the persistent instructions list (from `instructions.parquet`), the per-code stats lookup (from `code_stats.parquet`), and the cooccurrence index (from `cooccurrence.parquet`, see *Cooccurrence Index* below).
* **Three retrieval paths, all OR'd at the per-instruction level:**
  * **Semantic path:** embed the admission note, fetch persistent instructions whose `description`/`quote` embedding has cosine similarity ≥ `sim_threshold`.
  * **FP gate (threshold, runtime-synthesised):** for each 3-digit code in the *previous iteration's* prediction, look up `code_stats[code].fpr`. If it is ≥ `fpr_threshold`, synthesise an `fp_warning` Instruction at retrieval time and emit it. Synthesised IDs are deterministic (md5 hash of `"fp_<code>"`, high-bit set) so the same warning keeps the same ID across iterations.
  * **FN gate (threshold, runtime-synthesised, cooccurrence-driven):** expand the previous prediction via the cooccurrence index — for each predicted code, take the top-`cooccurrence_top_k` codes with `lift ≥ cooccurrence_threshold`. For each candidate in that expanded set, look up `code_stats[code].fnr`. If it is ≥ `fnr_threshold`, synthesise an `fn_warning` Instruction. The asymmetry with the FP gate is deliberate: an FN warning is about a code the model *should have* predicted but didn't, so gating on the predicted set would make it unreachable.
  * Both threshold gates are inactive at `t=0` (no prior prediction yet).
* **Cooccurrence Index:** `lift(a, b) = N · joint(a,b) / (count(a) · count(b))` over training-set 3-digit ground-truth co-occurrence. Built once per train split via `scripts/build_cooccurrence.py`, frozen for retrieval. `min_joint_count` and `min_support` filter noise. Lift naturally biases toward rare codes via the `count(b)` denominator; no separate rare-code re-ranking is applied.
* **Behavior at t=0:** The Generator runs **zero-shot** — no instructions retrieved, no `<think>` block prefilled. This produces a baseline prediction. From `t=1` onward, all three retrieval paths are active.
* **Deduplication:** Instructions retrieved in earlier iterations of the same case are excluded from later iterations (but the originals stay in the prompt — see *Efficacy Score Update Rule*). Synthesised threshold warnings participate in dedup via their deterministic IDs.
* **Budgeting:** Caps retrieval at `max_tokens_budget`, prioritising instructions with the highest `efficacy_score`. Synthesised threshold warnings always have `efficacy_score = 0.0`, so they only beat ineffective semantic instructions in priority order.

### Verifier (V)
* **Role:** Synchronous traffic controller.
* **Task:** Halts the Generator's inference loop if any condition is met: Token budget exhausted, no new instructions retrieved from the database, predictions stabilize (convergence), or maximum iteration steps reached.

### Efficacy Score Update Rule

* **Granularity:** scores update **per iteration**, after the Generator produces prediction `P_t`.
* **Rewarded set:** only the persistent (semantic / contrastive) instructions retrieved **fresh at iteration `t`** receive a score update. Instructions carried over from earlier iterations stay in the prompt but no longer accumulate reward — they had their chance. Synthesised threshold warnings (`fp_warning` / `fn_warning`) are never rewarded: their `efficacy_score` is permanently `0.0`. They fire whenever the gate triggers, full stop. This is a deliberate simplification over per-warning efficacy tracking.
* **Reward signal:** `delta_F1 = F1(P_t) − F1(P_{t−1})`, computed against ground truth at the case level.
* **Frequency-aware F1:** rare codes contribute more to the reward than common codes. Implemented simply: each sample in the patient file carries a precomputed `rareness_factor` (derived from training-set ICD-code frequencies in a separate preprocessing step). The reward signal is multiplied by this factor — no in-loop frequency lookup table needed.
* **Update:** for each freshly retrieved instruction `i` at iteration `t`:
  ```
  efficacy_score[i] += learning_rate * delta_F1 * rareness_factor
  ```
* **Training only:** efficacy updates run only when ground truth is available (training split). At test time, scores are read but never written.
* **t=0 implication:** the zero-shot prediction `P_0` is the baseline that the first reward signal is computed against. No instructions are rewarded at t=0 (none were used).

---

## 2. The Dual-Loop Pipeline

### Loop A: Synchronous Inference (Real-Time)
1. **Initial Pass (t=0, zero-shot):** Generator reads the Admission Note with no retrieved instructions and no prefilled `<think>` block. Produces baseline JSON prediction. *On a completely empty database (very first run), the case halts here — no instructions can ever be retrieved at t=1.*
2. **Refinement Passes (t≥1):** Retriever runs both semantic and threshold paths. Newly retrieved instructions are appended to the running `<think>` block; instructions retrieved in earlier iterations remain in the prompt but are not re-retrieved or re-rewarded.
3. **CoT Integration:** The Generator processes these warnings sequentially in a pre-filled internal monologue:

    ```text
    <think>
    Reviewing previous prediction: E11.9.
    Instruction Check: FPR warning triggered. Rule: Predict E11.4 INSTEAD OF E11.9 if diabetic neuropathy is present.
    Action: Note mentions neuropathy. Dropping E11.9, predicting E11.4.
    </think>
    [JSON Output]
    ```

4. **Halting:** Loop repeats until the Verifier (V) signals convergence.

### Loop B: Asynchronous Knowledge Acquisition (Global Memory)
1. **Audit (case-level):** For each closed case, Meta-Verifier reads `(admission_note, prediction, ground_truth, discharge_note)` and emits contrastive instructions for missed or hallucinated codes. These are appended to `instructions.parquet`.
2. **Audit (aggregate):** Recompute per-code FPR/FNR over the audited cases; for codes that cross the threshold AND meet `min_support`, emit a row to `code_stats.parquet` (one row per code, one of `fpr` / `fnr` set). Codes already present in the table are left untouched — frozen-rate semantics. The Retriever synthesises the warning text at runtime; nothing about the warning is persisted in the instructions store.
3. **Database Maintenance:** A clustering routine groups *persistent* instructions by `target_codes` and `type`. Redundant instructions are candidates for merge/summarisation via LLM, combining their `efficacy_score`. The `code_stats.parquet` table doesn't need clustering — it's already keyed by code.
   * **Open question (to be tested):** LLM-based merging may degrade the `description`/`quote` embedding such that semantic retrieval misses cases the originals would have caught. Treat merging as an ablation, not a default.

---

## 3. Experimental Execution Strategy

**Data:** MIMIC-IV with a fixed train / dev / test split. No distribution shift is expected — this is research, single dataset.

The experiment runs in alternating phases, manually triggered by the user:

* **Phase 1: Knowledge Bootstrapping (one full pass):**
  1. Loop A runs over **all** training samples until each one halts.
  2. Loop B runs once: Meta-Verifier audits all cases, writes/updates instructions, recomputes FPR/FNR, performs database maintenance.
  3. Stop. The user inspects the database and decides whether to launch another phase.
* **Phase 2 (and beyond): Residual Error Mining:** Restart Loop A on the same (or held-out) samples using the populated database. Loop B then runs again on the new predictions.
    * *Crucial Constraint:* Historical FPR/FNR metrics from earlier phases are **frozen** for retrieval purposes. Because successful instructions reduce errors, naively recomputed rates would drop and stop triggering the Retriever. Freezing keeps successful rules active. The Meta-Verifier writes new instructions only for novel errors.

---

## 4. Evaluation

Predictions are evaluated against MIMIC-IV ground truth at the **3-digit ICD level**.

**Headline metrics:**
* **F1 micro** — overall coding quality, dominated by frequent codes
* **F1 macro** — long-tail performance, every code weighted equally (this is the metric MERLIN 2 is designed to move)

**Standard ICD coding metrics also reported:**
* Precision / Recall (micro and macro)
* Per-code F1 for the top-N most frequent and a held-out tail set
* Set-level metrics: exact-match accuracy, Jaccard similarity to ground-truth code set

**Per-iteration tracking:** All metrics logged at each iteration `t` so we can see whether refinement is monotonically improving F1 or merely thrashing.

---

## 5. Future Dataset Generation (Side Note)

While not the primary focus of the active inference architecture, the MERLIN 2 framework naturally produces a highly curated dataset of Admission Notes, sequential `<think>` reasoning traces, and correct JSON outputs. To finalize a "gold standard" dataset for future Supervised Fine-Tuning (SFT), Rejection Sampling will be applied: the Generator will produce Best-of-N candidate traces at higher temperatures, and only the trace yielding the highest F1 score against the MIMIC-IV ground truth will be saved.


## Key Commands

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Build Docker (docker settings in configs/setup.yaml)
python scripts/build_docker.py

# 3. Start vLLM server (must run before client)
python scripts/server_start.py

# 4. Run inference job (MERLIN pipeline)
python scripts/merlin_start.py

# 5. Rebuild + rerun after code changes
python scripts/merlin_restart.py
```

## Architecture

- **Server**: vLLM model server (GPU) - hosts model weights, provides OpenAI-compatible API
- **Client**: K8s Job that loads data, runs inference via HTTP, saves predictions
- **Entry point**: `src/main.py` - loads config, builds prompts, runs async inference, evaluates

## Critical Config Notes

- `configs/setup.yaml`: Docker and wandb settings (rarely changed, requires rebuild)
- `configs/experiment.yaml`: Job, model, data, k8s, inference settings (no rebuild needed)
- `docker.platform`: Must be `linux/amd64` when building on Mac M1/M2/M3
- `inference.guided_decoding`: Enables Pydantic-based JSON schema output (modify `src/prompter.py` to change)
- `data/mimic/`: Gitignored - contains sensitive patient data

## Execution Order

1. Start server (`server_start.py`) - waits for pod ready
2. Run MERLIN pipeline (`merlin_start.py`) - sends inference requests to server
3. Server must be running before MERLIN; use `merlin_restart.py` to rebuild and rerun

## Monitoring

```bash
# Watch client logs
kubectl logs -f job/diagnose-prediction-<job_name> -n <namespace>

# Check server status
kubectl get pods -l app=vllm-server -n <namespace>
```

## Testing

No formal test suite. Validate changes by running a small sample:
- Set `sample_size: 10` in experiment config
- Use `merlin_restart.py` for quick iteration


## debugging
- Results might be uploaded to wandb for inspection (check `merlin_start.py` for logging code)
- Results should contain a table that have important fields. Use field to look if json output is correct. If it was an extraction error or do response did not contain one.
- Do we need to adapt the prompt?

## Code Philosophy

This is **research code**, not production software:

- **Fail fast**: Code crashes loudly on errors instead of silently handling them
- **Explicit is better**: No empty default values, no silent returns
- **Easier debugging**: When something breaks, you see the full traceback immediately
- **Crash is fine**: If the model fails to parse, the API returns an error, or wandb fails - the pipeline should crash so you can fix it

This approach makes the code easier to read and debug during research iterations.

---

## 6. Implementation Reference

### Component → File Mapping

| Component | File |
|-----------|------|
| **Generator (G)** | `src/merlin2/generator.py` |
| **Retriever (R)** | `src/merlin2/retriever.py` |
| **Verifier (V)** | `src/merlin2/verifier.py` |
| **Pipeline (Loop A orchestration)** | `src/merlin2/pipeline.py` |
| **Entry point (Loop A + Loop B driver)** | `src/main.py` |
| **Meta-Verifier (M)** | `src/meta_verifier/meta_verifier.py` |
| **Persistent instruction store** | `src/meta_verifier/store.py` (rows in `data/instructions.parquet`) |
| **Per-code threshold stats** | `src/meta_verifier/code_stats.py` (rows in `data/code_stats.parquet`) |
| **Cooccurrence index loader** | `src/utils/cooccurrence.py` (rows in `data/cooccurrence.parquet`) |
| **Cooccurrence builder (preprocessing)** | `scripts/build_cooccurrence.py` |
| **Wandb artifact roundtrip** | `src/utils/wandb_logger.py` (`instructions_db`, `code_stats` artifacts) |

### Default Configuration (Starting Values)

Tuned empirically from here; locked in as the first run's config.

| Parameter | Value | Used by |
|---|---|---|
| `sim_threshold` | 0.8 | Retriever — semantic path (cosine similarity cutoff) |
| `fpr_threshold` | 0.5 | Retriever — FP gate (synthesise fp_warning when `code_stats[c].fpr ≥ this`) |
| `fnr_threshold` | 0.5 | Retriever — FN gate (synthesise fn_warning when `code_stats[c].fnr ≥ this`) |
| `min_support` | 3 | Meta-Verifier — minimum case count before a code is eligible for a `code_stats` row |
| `cooccurrence_threshold` | 3.0 | Retriever — minimum lift to admit a code into the FN candidate set |
| `cooccurrence_top_k` | 20 | Retriever — top-K cap on cooccurring codes per predicted code (after threshold filter) |
| `cooccurrence_min_joint_count` | 3 | Cooccurrence builder — minimum joint count per pair (noise filter) |
| `convergence_threshold` | 0.9 | Verifier — Jaccard similarity between consecutive predictions to halt |
| `max_iterations` | 5 | Verifier — hard cap on iterations per case |
| `max_tokens_budget` | 2500 | Retriever — token cap for the `<think>` block |
| `learning_rate` | 1.2 | Efficacy update — multiplier on `delta_F1_weighted` |
| `instructions_artifact_name` | `instructions_db` | Wandb artifact lineage for `instructions.parquet` |
| `instructions_artifact_version` | `latest` | Pin to a specific version (e.g. `v3`) for reproducibility ablations |
| `code_stats_artifact_name` | `code_stats` | Wandb artifact lineage for `code_stats.parquet` |
| `code_stats_artifact_version` | `latest` | Same pinning behaviour as the instructions artifact |

### Logging & Observability

Every retrieval event must be logged with enough detail to later tune the thresholds and audit which path is doing the work:

* `hadm_id`, iteration `t`
* For each retrieved instruction: `instruction_id`, `retrieval_path` (`semantic` | `threshold_fpr` | `threshold_fnr`), the value that triggered it (cosine similarity for `semantic`, the looked-up `code_stats[c].fpr` / `fnr` for the threshold paths), and `efficacy_score` at retrieval time (always `0.0` for synthesised threshold warnings)
* Per-iteration aggregates: count and ratio of instructions retrieved by each path
* Halting reason (one of `max_iterations`, `budget_exhausted`, `no_new_instructions`, `convergence`, `empty_db`, `parse_failure`)
* `delta_F1` and the resulting score update (`delta_F1 * rareness_factor * learning_rate`) — training only, applied only to persistent (non-threshold) instructions

These logs are the basis for later adjusting `sim_threshold`, `fpr_threshold`, `fnr_threshold`, and `learning_rate` — in particular, if one retrieval path dominates or never fires, the thresholds are mis-tuned.

### Storage

Three Parquet files, three different lifecycles:

* **`data/instructions.parquet`** — persistent semantic / contrastive instructions. Columns match the Instruction Schema plus an `embedding` column of float32 vectors. Loaded into memory at the start of each Loop A run; semantic retrieval is brute-force cosine similarity. At the end of each Loop B run the file is rewritten with updated efficacy scores and newly-minted instructions appended. Round-tripped via the `instructions_db` wandb artifact.
* **`data/code_stats.parquet`** — per-code threshold-warning rows (one row per code that has ever crossed the FPR or FNR threshold). Frozen-rate semantics: codes already present are not modified by later Loop B passes. Round-tripped via the `code_stats` wandb artifact.
* **`data/cooccurrence.parquet`** — long-format `(code_a, code_b, lift, joint_count, count_a, count_b)` table over training-set 3-digit ground-truth co-occurrence. A frozen dataset property; built once per train split via `scripts/build_cooccurrence.py`. Currently *not* round-tripped via wandb — anyone with the train data can recompute, and changing the train split requires rebuilding anyway.

Re-evaluation candidates for the instruction store (open question): SQLite for cheap row-level updates, or DuckDB on top of the same Parquet for SQL access without changing the storage substrate.

### Retrieval Logic (concrete form)

For each persistent semantic / contrastive instruction `i`:

```
fire i IF cosine(note_embedding, i.embedding) >= sim_threshold
```

For each 3-digit code `c` in the previous prediction:

```
synthesise fp_warning(c) IF code_stats[c].fpr >= fpr_threshold
```

For each 3-digit code `c` in the cooccurring set (top-K by lift over the previous prediction, threshold `cooccurrence_threshold`, with the previous prediction itself excluded):

```
synthesise fn_warning(c) IF code_stats[c].fnr >= fnr_threshold
```

Then:
- Deduplicate by `instruction_id` (synthesised IDs are deterministic so dedup is stable across iterations).
- Prioritise by `efficacy_score` descending (synthesised threshold warnings tie at `0.0`).
- Respect `max_tokens_budget`.

### Halting Conditions (concrete form)

Verifier stops the inference loop when ANY condition is met:
1. **max_iterations**: Reached iteration limit
2. **budget_exhausted**: Token budget depleted
3. **no_new_instructions**: Retrieval returned empty
4. **convergence**: Jaccard similarity ≥ `convergence_threshold` between consecutive predictions

### Running MERLIN2

```bash
# Enable in experiment.yaml, then:
python scripts/merlin_start.py
```

### Testing MERLIN2

```bash
# Run MERLIN2 unit tests
pytest tests/test_schemas.py tests/test_retriever.py tests/test_generator.py tests/test_verifier.py tests/test_run_inference.py -v

# Test with small sample
# Set in experiment.yaml:
#   sample_size: 10
#   merlin2.max_iterations: 3
```