# Project Overview: vLLM Diagnose Prediction

A Kubernetes-deployed pipeline that uses large language models (LLMs) to predict ICD diagnosis codes from clinical admission notes (MIMIC dataset). The LLM is served via vLLM's OpenAI-compatible API; an inference client reads patient data, builds prompts, calls the API concurrently, parses the structured JSON output, and evaluates results against ground-truth labels. Optionally, a RAG step retrieves relevant PubMed abstracts before prompt construction.


## System Architecture

```
[configs/] ──► [scripts/] ──────────────────────────────────────┐
                  │                                               │
          server_start.py          client_start.py               │
                  │                       │                       │
          K8s Deployment            K8s Job ──► src/main.py      │
          (vLLM server pod)          (client pod)                 │
                  │                       │                       │
          GPU node serves          1. data_loader.py              │
          HuggingFace model        2. retriever.py  (optional)    │
                  │                3. prompter.py                 │
                  └──── HTTP ────► 4. inference.py                │
                                   5. evaluate.py                 │
                                   6. wandb_logger.py             │
                                   7. parsing_utils.py            │
```

**Two-pod split:** The vLLM server pod holds the model weights on GPU. The client pod runs CPU-only and sends concurrent HTTP requests to the server. They communicate over Kubernetes internal DNS.

---

## Directory Structure

```
vllm_diagnose_prediction/
├── configs/
│   ├── setup.yaml          # Static: Docker registry, WandB project
│   └── experiment.yaml     # Dynamic: model, inference, data, k8s params
├── src/
│   ├── main.py             # Entry point – orchestrates the full pipeline
│   ├── data_loader.py      # Loads and samples the patient parquet/csv file
│   ├── prompter.py         # Builds LLM prompts + defines output JSON schema
│   ├── retriever.py        # RAG: builds/loads PubMed vector index, retrieves chunks
│   ├── inference.py        # Async HTTP client – sends prompts to vLLM, gathers results
│   ├── parsing_utils.py    # Parses & repairs LLM JSON output into ICD code lists
│   ├── evaluate.py         # Normalizes codes, computes micro/macro F1
│   └── wandb_logger.py     # Logs params, metrics, and sample tables to W&B
├── scripts/
│   ├── utils.py            # Config loader (merges setup + experiment), kubectl runner
│   ├── k8s_templates.py    # Jinja2 templates for server Deployment/Service and client Job
│   ├── build_docker.py     # Builds and pushes the client Docker image
│   ├── server_start.py     # Deploys the vLLM server pod to K8s
│   ├── server_stop.py      # Removes the vLLM server pod from K8s
│   ├── client_start.py     # Submits the inference Job to K8s
│   ├── client_stop.py      # Cancels the inference Job
│   └── client_restart.py   # Stops then restarts the client Job
├── data/
│   ├── mimic/              # Patient parquet files (gitignored – sensitive)
│   └── pubmed/             # PubMed abstracts .txt + persisted vector index
├── tests/
│   ├── test_evaluation.py  # Unit tests for evaluate.py (normalization, metrics)
│   └── test_parsing.py     # Unit tests for parsing_utils.py (JSON repair)
├── Dockerfile              # Client image: python:3.10-slim + requirements
└── requirements.txt        # Python dependencies
```

---

## Source Files (`src/`)

### [main.py](src/main.py)
**Pipeline orchestrator.** Parses CLI args (`--config`), loads the YAML config, and runs six sequential steps inside an `asyncio` event loop:
1. Load patient data
2. (Optional) Run RAG retrieval
3. Build prompts
4. Run inference
5. Evaluate predictions
6. Log to W&B and save results CSV

### [data_loader.py](src/data_loader.py)
**Patient data ingestion.** Reads a `.parquet` or `.csv` file specified in config, applies an optional `sample_size` random sample (fixed seed 42 for reproducibility), and warns if expected columns (`admission_note`, `ICD_CODES`) are missing.

### [prompter.py](src/prompter.py)
**Prompt factory + JSON schema.** Defines the Pydantic output models (`ICDPrediction`, `ICDsModel`) that describe the expected structured JSON. `build_prompt()` assembles the system instruction, optional RAG context block, and the patient's admission note into a single string. `get_schema()` returns the JSON Schema used for vLLM guided decoding so the model is constrained to valid output.

### [retriever.py](src/retriever.py)
**RAG module.** When enabled in config, uses LlamaIndex + `MedCPT` embeddings (NCBI's biomedical encoder) to build a vector store index over ~250k PubMed abstracts. The index is persisted to disk so it is only built once. At inference time it uses the `MedCPT-Query-Encoder` to retrieve the top-k most relevant abstracts for each patient's admission note, which are then injected into the prompt.

### [inference.py](src/inference.py)
**Async HTTP inference client.** Connects to the vLLM OpenAI-compatible endpoint with exponential-backoff health checks. Builds JSON payloads (including optional `response_format` for guided decoding) and fires all requests concurrently via `aiohttp`, bounded by a semaphore (`concurrency` config value). Results are extracted from `choices[0].message.content`.

### [parsing_utils.py](src/parsing_utils.py)
**Robust JSON parser.** Because LLMs sometimes produce truncated or slightly malformed JSON, `safe_parse_json()` attempts parsing in stages: direct Pydantic validation → repair literal newlines/tabs → `repair_json_truncation()` (closes open strings, brackets, and braces via a stack). Returns a flat list of ICD code strings, or `[]` on failure.

### [evaluate.py](src/evaluate.py)
**Evaluation pipeline.** `normalize_icd()` reduces codes to their 3-character category (e.g. `K86.01` → `K86`). `evaluate_predictions()` calls the parser on all predictions and the ground-truth label parser on all labels, then computes micro and macro Precision/Recall/F1 via scikit-learn's `precision_recall_fscore_support`. Also reports the percentage of responses that contained valid JSON.

### [wandb_logger.py](src/wandb_logger.py)
**Weights & Biases integration.** Initializes a W&B run using the `WANDB_API_KEY` environment variable. Logs config hyperparameters, final F1/precision/recall metrics, and an interactive sample table of up to 30 predictions (with per-row F1) for qualitative inspection.

---

## Script Files (`scripts/`)

### [utils.py](scripts/utils.py)
**Shared helpers.** `load_config()` deep-merges `configs/setup.yaml` and `configs/experiment.yaml` (experiment overrides setup). `render_k8s_template()` fills Jinja2 templates with the merged config. `run_kubectl()` pipes the rendered YAML to `kubectl apply` or `kubectl delete`.

### [k8s_templates.py](scripts/k8s_templates.py)
**Kubernetes manifest templates.** Jinja2 strings for:
- **`server_template`**: A `ConfigMap` (holds serialized config), a `Deployment` (vLLM server container on a GPU node), and a `Service` (internal ClusterIP on port 80 → 8000).
- **`client_template`**: A `ConfigMap` and a `Job` (the inference client container, no GPU, `restartPolicy: Never`).

### [build_docker.py](scripts/build_docker.py)
Loads configs, constructs the full image URI (`registry/image_name:tag`), and runs `docker build` + `docker push`. Handles cross-platform builds for Linux/amd64.

### [server_start.py](scripts/server_start.py) / `server_stop.py`
Deploy or remove the vLLM server Deployment and Service on the cluster.

### [client_start.py](scripts/client_start.py) / `client_stop.py` / `client_restart.py`
Submit, cancel, or restart the inference Job on the cluster.

---

## Configuration Files (`configs/`)

### [setup.yaml](configs/setup.yaml)
Rarely changed. Holds Docker registry coordinates and WandB project/entity. Required for `build_docker.py`.

### [experiment.yaml](configs/experiment.yaml)
Changed per run. Controls everything experiment-specific: `job_name`, model name and vLLM batching params, inference temperature/concurrency/guided_decoding, dataset path and sample size, RAG on/off, and Kubernetes resource requests. Passed to Kubernetes pods at runtime as a `ConfigMap`, so most changes only need a client restart, not a Docker rebuild.

---

## Data Flow (end-to-end)

```
patients_1411.pq
       │
 data_loader.py ──► DataFrame (admission_note, ICD_CODES, ...)
       │
 retriever.py ──► DataFrame + retrieved_chunks (if RAG enabled)
       │
 prompter.py ──► List[str] prompts
       │
 inference.py ──► List[str] raw LLM responses (JSON text)
       │
 parsing_utils.py ──► List[List[str]] predicted ICD codes
       │
 evaluate.py ──► micro/macro F1, precision, recall
       │
 wandb_logger.py ──► W&B run + predictions.csv
```

---

## Key Dependencies

| Library | Role |
|---|---|
| `vllm` | LLM inference server (GPU pod) |
| `openai` / `aiohttp` | Async HTTP client to vLLM |
| `llama-index` | Vector store for RAG |
| `sentence-transformers` (via HuggingFace) | MedCPT embeddings for RAG |
| `pydantic` | Output schema definition + validation |
| `pandas` | Tabular data handling |
| `scikit-learn` | Multi-label metrics |
| `wandb` | Experiment tracking |
| `jinja2` | Kubernetes manifest templating |
| `kubectl` (CLI) | Cluster interaction |
