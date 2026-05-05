# MERLIN2 Streamlit UI (MVP)

This UI replays a W&B run and visualizes:

- Loop A patient-level iterations (`[t=0..N]` predictions + retrieved instructions)
- Ground-truth vs final predicted codes
- Loop B summary + top Meta-Verifier instruction rows
- Run-level metric trends and log tail

## Run locally

```bash
pip install -r UI/requirements.txt
streamlit run UI/app.py
```

If your local env already has `wandb` but an older `protobuf`, fix with:

```bash
pip install protobuf==6.32.1
```

Default startup data uses cached run `2bzc4wvd`.  
You can paste another W&B run URL/path in the sidebar.

ICD descriptions are loaded from `UI/ICD_names/*.csv` (ICD-10-CM / ICD-9 files with code + short description columns).

## Cache behavior

- Downloads are stored under `UI/wandb_cache/<run_id>/`
- Keeps the latest 5 runs and deletes older cached runs

## Docker

```bash
docker build -f UI/Dockerfile -t merlin-ui:latest .
docker run --rm -p 8501:8501 merlin-ui:latest
```

## Kubernetes (ClusterIP + port-forward)

```bash
kubectl apply -f UI/ui-k8s.yaml
kubectl port-forward -n jfrick svc/merlin-ui 8501:8501
```
