# vLLM Diagnose Prediction

ICD-10 code prediction pipeline using vLLM with MERLIN 2 dual-loop refinement.

---

## Architecture

### MERLIN 2 Dual-Loop Framework

```
+------------------------------------------------------------------+
|                    Inference Loop (Loop A)                       |
|                                                                  |
|    Admission                                                      |
|      Note                                                         |
|        |                                                          |
|        v                                                          |
|   +----------+     +-------------+     +-----------+     +------+|
|   |Generator |---->|  Prediction |---->| Retriever |---->|Verifier
|   +----------+     +-------------+     +-----------+     +------+|
|        ^                                    |                   ||
|        |                                    v                   ||
|        |                              +-----------+             ||
|        |                              |Instructions|<------------
|        |                              +-----------+             ||
|        |                                    ^                   ||
|        +------------------------------------+                   ||
|                      Loop until convergence                     ||
+------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------+
|              Knowledge Acquisition Loop (Loop B)                 |
|                                                                  |
|    Discharge      Results                                        |
|      Note         (pred + ground truth)                         |
|        |                |                                        |
|        v                v                                        |
|   +-------------+    +-------------+                            |
|   |MetaVerifier |<---|  Analysis   |                            |
|   +-------------+    +-------------+                            |
|        |                                                          |
|        v                                                          |
|   +-----------+                                                   |
|   |Instructions|                                                  |
|   +-----------+                                                   |
|        |                                                          |
|        v                                                          |
|   +-------------+                                                 |
|   |Upload to   |                                                 |
|   |W&B         |                                                 |
|   +-------------+                                                 |
+------------------------------------------------------------------+
```

**Loop A (Synchronous - Real-Time Inference)**
1. **Generator** predicts ICD codes from admission notes (with guided decoding for structured JSON output)
2. **Retriever** fetches relevant instructions based on predicted codes (semantic similarity + FPR/FNR thresholds)
3. **Verifier** checks halting conditions → repeat until convergence

**Loop B (Asynchronous - Global Memory)**
- **MetaVerifier** reviews closed cases, generates contrastive instructions via LLM-based root cause analysis
- Instructions are uploaded to W&B and can be retrieved in future runs

### Components

| Component | File | Role |
|-----------|------|------|
| **Generator** | `src/merlin2/generator.py` | LLM inference with CoT refinement |
| **Retriever** | `src/merlin2/retriever.py` | Instruction retrieval via semantic search |
| **Verifier** | `src/merlin2/verifier.py` | Halting condition evaluation |
| **MetaVerifier** | `src/meta_verifier/meta_verifier.py` | Instruction generation from results |
| **Pipeline** | `src/merlin2/pipeline.py` | Loop A orchestration |

---

## Quick Start

```bash
# Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure cluster settings in configs/setup.yaml
# Configure experiment in configs/experiment.yaml

# Start vLLM server
python scripts/server_start.py

# Run inference
python scripts/merlin_start.py

# After code changes, rebuild with:
python scripts/build_docker.py
python scripts/merlin_restart.py
```

---

## Configuration

### `configs/setup.yaml` (infrastructure)

| Parameter | Section | Description |
| :--- | :--- | :--- |
| `registry` | `docker` | Docker registry URL |
| `image_name` | `docker` | Client image name |
| `tag` | `docker` | Image tag |
| `platform` | `docker` | Target architecture (`linux/amd64` for cross-platform builds) |
| `server_image` | `docker` | vLLM server image |
| `project` | `wandb` | W&B project name |
| `entity` | `wandb` | W&B entity/team |

### `configs/experiment.yaml` (per experiment)

#### Root Level
| Parameter | Description |
| :--- | :--- |
| `job_name` | Unique run identifier (used for K8s pod naming) |
| `run_name` | W&B run name |
| `log_level` | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### Data
| Parameter | Description |
| :--- | :--- |
| `patients_file` | Path to input Parquet file |
| `target_col` | Column containing ground truth labels |
| `admission_col` | Column containing clinical admission notes |
| `discharge_col` | Column containing clinical discharge notes (optional, falls back to admission_col) |
| `sample_size` | Rows to process (`null` for full dataset) |

#### Model
| Parameter | Description |
| :--- | :--- |
| `name` | HuggingFace model ID |
| `max_model_len` | Max context length |
| `max_num_seqs` | Max concurrent sequences |
| `max_num_batched_tokens` | Max tokens per batch |
| `enable_chunked_prefill` | Enable chunked prompt processing |

#### Inference
| Parameter | Description |
| :--- | :--- |
| `temperature` | Sampling temperature (0.0 = deterministic) |
| `max_tokens` | Max response tokens |
| `concurrency` | Parallel HTTP requests to vLLM |
| `guided_decoding` | Enable Pydantic-based JSON schema output |

#### MERLIN2
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Enable MERLIN2 pipeline |
| `sim_threshold` | `0.7` | Semantic similarity threshold for instruction retrieval |
| `fpr_threshold` | `0.1` | False positive rate threshold |
| `fnr_threshold` | `0.1` | False negative rate threshold |
| `max_tokens_budget` | `512` | Max tokens for retrieved instructions |
| `max_iterations` | `5` | Max refinement passes |
| `convergence_threshold` | `0.95` | Jaccard similarity for halting |

#### Meta-Verifier
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `enabled` | `false` | Generate contrastive instructions from results |
| `embedding_model` | `BiomedNLP-PubMedBERT-...` | Model for semantic embeddings |
| `min_error_count` | `3` | Min errors before generating instruction |
| `efficacy_threshold` | `0.1` | Min efficacy score for instructions |
| `upload_results` | `true` | Upload instructions to W&B |
| `download_run_name` | `null` | Download instructions from previous run |

---

## Config Changes & Rebuilds

| Change Type | Action Required |
| :--- | :--- |
| Inference/Data params (`sample_size`, `temperature`, etc.) | Restart job: `merlin_start.py` |
| Model params (`name`, `max_model_len`) | Restart server: `server_start.py` |
| Setup/Docker params | Rebuild: `build_docker.py` |

---

## Output Schema

With `guided_decoding: true`, output conforms to `ICDsModel`:

```json
{
  "diagnoses": [
    {"icd_code": "I10", "reason": "Patient presents with elevated blood pressure."},
    {"icd_code": "E11.9", "reason": "Documented type 2 diabetes mellitus."}
  ]
}
```

---

## Customization

- **Output Schema:** Modify `ICDPrediction` class in `src/prompter.py`
- **Prompts:** Edit prompt templates in `src/merlin2/prompts.py` or `src/meta_verifier/prompts.py`
- **Instruction Retrieval:** Adjust thresholds in `merlin2` config section

---

## Monitoring

```bash
# Client logs
kubectl logs -f job/diagnose-prediction-<job_name> -n <namespace>

# Server status
kubectl get pods -l app=vllm-server -n <namespace>
```

W&B logs: parameters, F1 micro/macro, precision/recall, prediction samples.