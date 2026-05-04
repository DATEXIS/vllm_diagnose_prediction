import asyncio
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
BUILD_BATCH_SIZE = 16
QUERY_BATCH_SIZE = 32


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_model(model_name: str, device: str):
    """Load tokenizer + model once. Caller is responsible for cleanup."""
    from transformers import AutoTokenizer, AutoModel
    logger.info(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def _encode_batch(batch: list, tokenizer, model, max_length: int, device: str) -> np.ndarray:
    """Encode one batch → L2-normalised CLS embeddings (float32)."""
    import torch
    with torch.no_grad():
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        # MedCPT requires CLS-token pooling — do NOT use mean pooling
        emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.clip(norms, 1e-8, None)).astype(np.float32)


def build_index(abstracts_path: str, index_persist_dir: str, max_abstracts: int = None):
    """
    Streams PubMed abstracts in batches → encodes each batch with
    MedCPT-Article-Encoder → adds directly to a FAISS IndexFlatIP.

    Peak RAM = one batch of embeddings, not the full matrix.
    Persists faiss.index (binary) + texts.pkl.
    Returns (faiss_index, texts).
    """
    import faiss

    persist_path = Path(index_persist_dir)
    index_file = persist_path / "faiss.index"
    texts_file = persist_path / "texts.pkl"

    if index_file.exists() and texts_file.exists():
        logger.info(f"Loading cached FAISS index from {index_persist_dir}")
        index = faiss.read_index(str(index_file))
        with open(texts_file, "rb") as f:
            texts = pickle.load(f)
        logger.info(f"Index loaded: {index.ntotal} vectors, dim={index.d}")
        return index, texts

    device = _get_device()
    logger.info(f"Device: {device}")
    tokenizer, model = _load_model(ARTICLE_MODEL, device)

    texts = []
    batch: list = []
    index = None

    def flush(batch_texts: list):
        nonlocal index
        emb = _encode_batch(batch_texts, tokenizer, model, max_length=512, device=device)
        if index is None:
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

    logger.info(f"Streaming abstracts from {abstracts_path} ...")
    with open(abstracts_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building FAISS index"):
            line = line.strip()
            if not line:
                continue
            texts.append(line)
            batch.append(line)
            if len(batch) == BUILD_BATCH_SIZE:
                flush(batch)
                batch = []
            if max_abstracts and len(texts) >= max_abstracts:
                break

    if batch:
        flush(batch)

    del model  # free model weights from RAM/VRAM

    persist_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)

    logger.info(f"FAISS index saved: {index.ntotal} vectors, dim={index.d} → {index_persist_dir}")
    return index, texts


_QUERY_REWRITE_PROMPT = (
    "You are a medical information retrieval expert. "
    "Given a clinical admission note, reformulate the key medical information "
    "as a concise PubMed search query. "
    "Capture the primary diagnoses, chief complaints, relevant comorbidities, "
    "and key clinical findings. "
    "Return ONLY the search query — no explanation, no preamble.\n\n"
    "Admission note:\n{note}\n\n"
    "Search query:"
)


async def _rewrite_one(session, url: str, model: str, note: str, semaphore: asyncio.Semaphore) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _QUERY_REWRITE_PROMPT.format(note=note[:3000])}],
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
    }
    async with semaphore:
        try:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed, falling back to raw note: {e}")
            return note[:500]


async def generate_queries(admission_notes: list, config: dict) -> list:
    """
    Rewrites raw admission notes into concise medical search queries
    optimised for MedCPT retrieval by calling the vLLM server.
    Falls back to the truncated raw note on any per-request failure.
    """
    from aiohttp import ClientSession, ClientTimeout

    job_name = config.get("job_name", "default")
    namespace = config.get("k8s", {}).get("namespace", "default")
    api_base = config["model"].get("api_base") or (
        f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"
    )
    url = f"{api_base}/chat/completions"
    model = config["model"]["name"]
    concurrency = config["inference"].get("concurrency", 10)

    logger.info(f"Rewriting {len(admission_notes)} admission notes into retrieval queries ...")
    semaphore = asyncio.Semaphore(concurrency)

    async with ClientSession(timeout=ClientTimeout(total=120)) as session:
        tasks = [_rewrite_one(session, url, model, note, semaphore) for note in admission_notes]
        queries = await asyncio.gather(*tasks)

    logger.info("Query rewriting complete.")
    return list(queries)


def retrieve_for_dataframe(
    df: pd.DataFrame,
    index,
    texts: list,
    k: int = 5,
    query_column: str = "admission_note",
    queries: list = None,
) -> pd.DataFrame:
    """
    Encodes admission notes with MedCPT-Query-Encoder (loaded once) and
    retrieves the top-k most similar PubMed abstracts per row.
    Stores results as 'retrieved_chunks' (list of strings).
    """
    device = _get_device()
    tokenizer, model = _load_model(QUERY_MODEL, device)

    if queries is not None:
        raw_queries = [str(q)[:500] for q in queries]
        logger.info(f"Encoding {len(raw_queries)} rewritten queries ...")
    else:
        raw_queries = [str(val)[:500] for val in df[query_column]]
        logger.info(f"Encoding {len(raw_queries)} queries ...")

    all_embs = []
    for start in range(0, len(raw_queries), QUERY_BATCH_SIZE):
        batch = raw_queries[start : start + QUERY_BATCH_SIZE]
        all_embs.append(_encode_batch(batch, tokenizer, model, max_length=64, device=device))

    del model
    q_embs = np.vstack(all_embs)

    _, indices = index.search(q_embs, k)

    retrieved_chunks_list = [
        [texts[i] for i in row_idx if i != -1 and i < len(texts)]
        for row_idx in indices
    ]

    df = df.copy()
    df["retrieved_chunks"] = retrieved_chunks_list
    logger.info(f"Retrieval complete: top-{k} chunks × {len(retrieved_chunks_list)} rows.")
    return df
