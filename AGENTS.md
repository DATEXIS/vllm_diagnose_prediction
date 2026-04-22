# Agent Instructions

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

---

## MERLIN 2 Framework

MERLIN 2 is a dual-loop, multi-agent framework for iteratively improving ICD coding from clinical admission notes.

### Architecture

| Component | Role | File |
|-----------|------|------|
| **Generator (G)** | Primary clinical coding LLM - vLLM with guided decoding + CoT | `src/generator.py` |
| **Retriever (R)** | Memory fetcher - threshold-based boolean selection | `src/retriever.py` |
| **Verifier (V)** | Traffic controller - synchronous halting conditions | `src/verifier.py` |
| **Pipeline** | Loop A orchestration (synchronous inference) | `src/run_inference.py` |

### Dual-Loop Pipeline

**Loop A (Synchronous - Real-Time):**
1. t=0: Initial pass - Generator predicts with baseline instructions
2. t>0: Retriever fetches instructions based on predicted codes (FPR/FNR thresholds)
3. Generator processes warnings sequentially in `<thinking>` blocks
4. Verifier checks halting conditions → repeat until convergence

**Loop B (Asynchronous - Global Memory):**
- Meta-Verifier reviews closed cases, generates contrastive instructions
- Background clustering groups instructions by ICD code and error type

### Configuration

Add to `configs/experiment.yaml`:

```yaml
# MERLIN 2 settings
merlin2:
  enabled: true                    # Enable MERLIN2 pipeline
  sim_threshold: 0.7               # Semantic similarity threshold
  fpr_threshold: 0.1               # False positive rate threshold
  fnr_threshold: 0.1               # False negative rate threshold
  max_tokens_budget: 512           # Max tokens per retrieval
  max_iterations: 5                # Max refinement passes
  convergence_threshold: 0.95      # Jaccard similarity for convergence

# Override Generator settings
generator:
  api_base: "http://localhost:8000/v1"
  model: "Qwen/Qwen3-8B"
  temperature: 0.0
  max_tokens: 1024
```

### Running MERLIN2

```bash
# Enable in experiment.yaml, then:
python scripts/merlin_start.py

# Or test locally (requires vLLM running):
python -c "
from src.run_inference import run_inference
result = run_inference('Patient presents with chest pain and diabetes...')
print(result)
"
```

### Retrieval Logic

Threshold-based boolean selection:
```
fetch instruction IF (semantic_similarity >= X) OR (fpr >= Y) OR (fnr >= Z)
```
- Filters by predicted codes matching instruction's target_code
- Prioritizes by efficacy_score descending
- Respects max_tokens_budget

### Halting Conditions

Verifier stops the inference loop when ANY condition is met:
1. **max_iterations**: Reached iteration limit
2. **budget_exhausted**: Token budget depleted
3. **no_new_instructions**: Retrieval returned empty
4. **convergence**: Jaccard similarity ≥ convergence_threshold between consecutive predictions

### Testing MERLIN2

```bash
# Run MERLIN2 unit tests
pytest tests/test_schemas.py tests/test_retriever.py tests/test_generator.py tests/test_verifier.py tests/test_run_inference.py -v

# Test with small sample
# Set in experiment.yaml:
#   sample_size: 10
#   merlin2.max_iterations: 3
```