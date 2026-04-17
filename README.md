# vLLM Diagnose Prediction

A scalable, production-ready pipeline for performing multi-label diagnosis (ICD) prediction on clinical datasets (like MIMIC) using Large Language Models (LLMs) via `vLLM`.

This repository provides a modular foundation for deploying LLMs on Kubernetes clusters, featuring guided decoding for structured output, efficient data handling, and automated deployment scripts.

---

## 🛠️ Prerequisites & Cluster Setup

This pipeline is designed to run on a Kubernetes cluster. Ensure you have the following prerequisites met:

### 1. Account & Permissions
- Verify that your campus or organization account has active permissions for cluster access.
- For RIS cluster users, this typically involves induction via a supervisor or the dedicated support channels.

### 2. Environment Configuration
- **VPN:** A secure VPN connection to the organization network is required for all cluster interactions.
- **Kubectl:** Install and configure `kubectl` to point to your target cluster namespace.
- **Cluster Documentation:** Refer to the [RIS Cluster Docs](https://docs.cluster.ris.bht-berlin.de/) for detailed setup steps.

---

## 🚀 Execution Workflow

This section guides you through the complete execution workflow, from setting up your local environment to running inference on Kubernetes.

### 1. Local Environment Setup
Install the required Python dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
Place clinical datasets (e.g., in Parquet format) in the `data/mimic/` directory. By default, this folder is ignored by version control to protect sensitive patient information.

### 3. Containerization
Package the inference client and dataset into a Docker image:
1. Configure the `docker` section in `configs/config.yaml`.
2. Execute the build script:
   ```bash
   python scripts/build_docker.py
   ```
   *Note: Cross-platform builds (e.g., Mac to Linux) are handled automatically based on the `platform` configuration.*

> **Tip:** Config changes (e.g., `sample_size`, `guided_decoding`, model parameters) do NOT require a new Docker build. The config is passed to Kubernetes at runtime via ConfigMap. Simply run `server_start.py` or `client_start.py` to apply new config values.

### 4. Kubernetes Deployment
The pipeline uses dynamic manifest generation driven by `configs/config.yaml`.

#### Start the vLLM Server
The server hosts the model weights and provides the inference API:
```bash
python scripts/server_start.py
```

#### Run the Inference Job
Once the server is initialized, start the processing pipeline:
```bash
python scripts/client_start.py
```

---

## ⚙️ Configuration Reference (`configs/config.yaml`)

| Parameter | Section | Description |
| :--- | :--- | :--- |
| `job_name` | Root | Unique identifier for the run. Used to name Pods, Services, and Jobs. |
| `registry` | `docker` | Base URL/path for the Docker registry. |
| `image_name` | `docker` | Name of the resulting client Docker image. |
| `tag` | `docker` | Version tag for the image (e.g., `latest`). |
| `platform` | `docker` | Target CPU architecture for the build (e.g., `linux/amd64`). |
| `server_image` | `docker` | The specific `vLLM` container image used for the server. |
| `patients_file` | `data` | Path to the input Parquet file relative to the project root. |
| `target_col` | `data` | The column name containing the ground truth labels or text. |
| `sample_size` | `data` | Number of rows to process for testing. Set to `null` for the full dataset. |
| `name` | `model` | HuggingFace model identifier or local path. |
| `max_model_len` | `model` | Maximum context length permitted by the model configuration. |
| `max_num_seqs` | `model` | vLLM: Maximum number of concurrent sequences per batch. |
| `max_num_batched_tokens` | `model` | vLLM: Maximum tokens processed in a single batch. |
| `enable_chunked_prefill`| `model` | vLLM: Enable splitting large prompt prefills into smaller chunks. |
| `temperature` | `inference` | Controls output randomness (0.0 = deterministic). |
| `max_tokens` | `inference` | Maximum length of the generated response per request. |
| `concurrency` | `inference` | Number of parallel HTTP requests sent to the vLLM server. |
| `guided_decoding` | `inference` | Enables structured JSON output using Pydantic schemas. |
| `namespace` | `k8s` | The Kubernetes namespace for all deployments. |
| `image_pull_secrets` | `k8s` | Secret name for private registry authentication. |
| `gpu_count` | `server` | Number of GPUs allocated to the server. |
| `gpu_type` | `server` | Specific GPU model required (e.g., `a100`). |
| `memory_limit` | `server/client`| Maximum RAM permitted for the Pod/Job. |
| `memory_request` | `server/client`| Minimum RAM requested from the cluster. |

---

## 📊 Monitoring

- **Utilization:** Monitor available hardware and cluster health via the [Grafana Dashboard](https://monitoring.cluster.ris.bht-berlin.de/).
- **Logs:** Watch real-time execution logs:
  ```bash
  kubectl logs -f job/diagnose-prediction-{{job_name}} -n {{namespace}}
  ```

---

## 📄 Guided Decoding JSON Structure

When `guided_decoding` is enabled, the model must output JSON that conforms to the `ICDsModel` schema defined in `src/prompter.py`. The expected shape is:

```json
{
  "diagnoses": [
    {
      "icd_code": "I10",
      "reason": "Patient has persistent hypertension noted in the admission note."
    },
    {
      "icd_code": "E11.9",
      "reason": "Elevated blood glucose levels indicating type 2 diabetes mellitus."
    }
  ]
}
```

The `icd_code` field must contain a valid ICD code, and `reason` should provide a brief clinical justification (1‑2 sentences).

---

## 💡 Customization

- **Config Changes:** Modify `configs/config.yaml` and re-run `server_start.py` or `client_start.py` to apply. No rebuild needed.
- **Output Schema:** Modify the `ICDPrediction` class in `src/prompter.py` to change the structured output format. This requires a Docker rebuild.
- **Code Changes:** After modifying Python code, rebuild the image with `python scripts/build_docker.py`.
