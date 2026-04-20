# Agent Instructions

## Key Commands

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Build Docker (docker settings in configs/setup.yaml)
python scripts/build_docker.py

# 3. Start vLLM server (must run before client)
python scripts/server_start.py

# 4. Run inference job
python scripts/client_start.py

# 5. Rebuild + rerun after code changes
python scripts/client_restart.py
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
2. Run client (`client_start.py`) - sends inference requests to server
3. Server must be running before client; use `client_restart.py` to rebuild and rerun

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
- Use `client_restart.py` for quick iteration