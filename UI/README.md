# MERLIN2 Run Replay UI

Next.js dashboard that pulls the same artifacts as the retired Streamlit app (`UI/app.py` kept for reference) and replays Loop A iterations from the W&B `sample_predictions` table.

## Behavior

- Per-patient iteration slider (`[t=…]` predictions + cumulative instructions vs admission note), TP / FP / FN panels
- ICD short descriptions resolved from `UI/ICD_names/*.csv`
- Run-level diagnostics: metrics line chart (`history_metrics.csv` when present), `wandb-summary.json`, tail of `output.log`

## Cache

- Downloads go to `UI/wandb_cache/<run_id>/`
- Last 5 runs retained (TTL enforced server-side via `UI/py/fetch_run.py`)

## Local dev

From repo root (`MERLIN_UI_ROOT` defaults to the parent of `dashboard/`):

```bash
cd UI/dashboard
npm install
export WANDB_API_KEY=...
npm run dev
```

Open http://localhost:3000 .

Optional: `/config.json` (gitignored here) — if present at repo root it is consumed as fallback when `WANDB_API_KEY` is unset (`MERLIN_CONFIG_JSON` overrides path in Docker).

## Docker

Requires a build context that contains `config.json` at repo root (same expectation as before) **or** pass only `WANDB_API_KEY` at runtime via env and omit `MERLIN_CONFIG_JSON` by layering a shim file.

```bash
docker build -f UI/Dockerfile -t merlin-ui:latest .
docker run --rm -e WANDB_API_KEY=... -p 3000:3000 merlin-ui:latest
```

## Kubernetes (`UI/ui-k8s.yaml`)

```bash
kubectl apply -f UI/ui-k8s.yaml
kubectl port-forward -n yxsg9647 svc/merlin-ui 3000:3000
```
