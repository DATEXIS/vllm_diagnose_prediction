import asyncio
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
CROSS_ENCODER_MODEL = "ncbi/MedCPT-Cross-Encoder"
BUILD_BATCH_SIZE = 16
QUERY_BATCH_SIZE = 32
RERANK_BATCH_SIZE = 16


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


def _load_cross_encoder(model_name: str, device: str):
    """Load the cross-encoder with its classification head."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    logger.info(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
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


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_SINGLE_QUERY_PROMPT = (
    "You rewrite clinical admission notes into concise queries for retrieval and ICD coding.\n"
    "Rules:\n"
    "- Use only explicitly stated information (no assumptions).\n"
    "- Keep key clinical facts: symptoms, diagnoses, history, meds, labs.\n"
    "- Preserve negations and temporality (e.g., \"denies\", acute/chronic).\n"
    "- Remove non-clinical and redundant text.\n"
    "- Use standard medical terminology.\n\n"
    "Return ONLY the search query — no explanation, no preamble.\n\n"
    "Admission note:\n{note}\n\n"
    "Search query:"
)

_MULTI_QUERY_PROMPT = (
    "You rewrite clinical admission notes into multiple diverse, concise queries "
    "for PubMed retrieval and ICD coding.\n\n"
    "Generate exactly {n} queries that together cover different aspects of the note:\n"
    "- Query 1: Primary diagnoses and chief complaints\n"
    "- Query 2: Comorbidities, medical history, current medications\n"
    "- Query 3: Procedures, laboratory findings, key clinical observations\n\n"
    "Rules:\n"
    "- Use only explicitly stated information (no assumptions).\n"
    "- Preserve negations and temporality (e.g., \"denies\", acute/chronic).\n"
    "- Remove non-clinical and redundant text.\n"
    "- Use standard medical terminology.\n\n"
    "Return ONLY a JSON array of exactly {n} query strings, no explanation.\n"
    "Example: [\"query one\", \"query two\", \"query three\"]\n\n"
    "Admission note:\n{note}\n\n"
    "JSON array:"
)


async def _rewrite_one(
    session, url: str, model: str, note: str, semaphore: asyncio.Semaphore, n_queries: int
) -> list:
    """Returns a list of n_queries strings for one admission note."""
    if n_queries == 1:
        prompt = _SINGLE_QUERY_PROMPT.format(note=note[:3000])
        max_tokens = 200
    else:
        prompt = _MULTI_QUERY_PROMPT.format(n=n_queries, note=note[:3000])
        max_tokens = 100 * n_queries

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        # Disable Qwen3 thinking mode — query rewriting is a formatting task
        # that doesn't benefit from chain-of-thought, and thinking tokens would
        # consume the entire budget leaving no room for the actual output.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with semaphore:
        try:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Qwen3 (and other thinking models) wrap output in <think>...</think>
                # before the actual answer — strip that block first.
                if "</think>" in content:
                    content = content.split("</think>", 1)[-1].strip()
                if n_queries == 1:
                    return [content]
                parsed = json.loads(content)
                if isinstance(parsed, list) and parsed:
                    return [str(q).strip() for q in parsed[:n_queries]]
                return [note[:500]]
        except Exception as e:
            logger.warning(f"Query generation failed, falling back to raw note: {e}")
            return [note[:500]]


async def generate_queries(admission_notes: list, config: dict) -> list:
    """
    Generates retrieval queries for each admission note by calling the vLLM server.
    Returns list[list[str]] — one inner list of queries per patient.
    n_queries=1 produces a single rewritten query; n_queries>1 produces diverse
    query variants that together cover the note from multiple angles.
    Falls back to the truncated raw note on any per-request failure.
    """
    from aiohttp import ClientSession, ClientTimeout

    n_queries = config.get("rag", {}).get("query_rewriting", {}).get("n_queries", 1)
    job_name = config.get("job_name", "default")
    namespace = config.get("k8s", {}).get("namespace", "default")
    api_base = config["model"].get("api_base") or (
        f"http://vllm-server-{job_name}.{namespace}.svc.cluster.local/v1"
    )
    url = f"{api_base}/chat/completions"
    model = config["model"]["name"]
    concurrency = config["inference"].get("concurrency", 10)

    logger.info(
        f"Generating {n_queries} quer{'y' if n_queries == 1 else 'ies'} "
        f"for {len(admission_notes)} admission notes ..."
    )
    semaphore = asyncio.Semaphore(concurrency)

    async with ClientSession(timeout=ClientTimeout(total=120)) as session:
        tasks = [
            _rewrite_one(session, url, model, note, semaphore, n_queries)
            for note in admission_notes
        ]
        results = await asyncio.gather(*tasks)

    logger.info("Query generation complete.")
    return list(results)  # list[list[str]]


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def _rerank_candidates(
    query: str,
    candidates: list,
    tokenizer,
    model,
    device: str,
    top_k: int,
) -> list:
    """
    Scores (query, candidate) pairs with MedCPT-Cross-Encoder and returns
    the top_k candidates sorted by descending relevance score.
    """
    import torch
    if not candidates:
        return candidates

    # Cross-encoder has a 512-token budget shared between query and passage.
    # Truncate query to 128 chars to leave enough room for the passage.
    truncated_query = query[:512]
    pairs = [[truncated_query, c] for c in candidates]
    scores = []

    with torch.no_grad():
        for i in range(0, len(pairs), RERANK_BATCH_SIZE):
            batch = pairs[i : i + RERANK_BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            # logits shape: [batch, num_labels]. Use the last column:
            # - num_labels=2 (binary): index 1 is the "relevant" class score
            # - num_labels=1: index 0 is the single relevance score
            batch_scores = out.logits[:, -1].cpu().float().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [text for _, text in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Main retrieval
# ---------------------------------------------------------------------------

def retrieve_for_dataframe(
    df: pd.DataFrame,
    index,
    texts: list,
    k: int = 5,
    query_column: str = "admission_note",
    queries: list = None,
    rerank: bool = False,
    fetch_k: int = 20,
) -> pd.DataFrame:
    """
    Retrieves top-k PubMed abstracts per patient.

    queries: list[list[str]] from generate_queries, or None to use raw admission notes.
    rerank:  if True, fetches fetch_k candidates per query and re-ranks with
             MedCPT-Cross-Encoder before trimming to top-k.
    fetch_k: candidates to retrieve per query when re-ranking is enabled.
    """
    device = _get_device()
    tokenizer_q, model_q = _load_model(QUERY_MODEL, device)

    # Build a flat list of all queries and track how many each patient has.
    if queries is not None:
        flat_queries = [q[:500] for patient_qs in queries for q in patient_qs]
        n_per_patient = [len(pqs) for pqs in queries]
    else:
        flat_queries = [str(val)[:500] for val in df[query_column]]
        n_per_patient = [1] * len(df)

    logger.info(
        f"Encoding {len(flat_queries)} quer{'y' if len(flat_queries) == 1 else 'ies'} "
        f"for {len(df)} patients ..."
    )

    all_embs = []
    for start in range(0, len(flat_queries), QUERY_BATCH_SIZE):
        batch = flat_queries[start : start + QUERY_BATCH_SIZE]
        all_embs.append(_encode_batch(batch, tokenizer_q, model_q, max_length=64, device=device))

    del model_q
    q_embs = np.vstack(all_embs)

    # How many candidates to fetch per FAISS query.
    candidates_per_query = fetch_k if rerank else k
    _, all_faiss_indices = index.search(q_embs, candidates_per_query)

    # Group FAISS results by patient, merging and deduplicating across queries.
    candidates_per_patient = []
    offset = 0
    for n_q in n_per_patient:
        seen = set()
        merged = []
        for q_idx in range(n_q):
            for idx in all_faiss_indices[offset + q_idx]:
                if idx != -1 and idx < len(texts) and idx not in seen:
                    seen.add(idx)
                    merged.append(idx)
        candidates_per_patient.append(merged)
        offset += n_q

    # Re-rank or trim to top-k.
    if rerank:
        logger.info(
            f"Re-ranking with {CROSS_ENCODER_MODEL} "
            f"({[len(c) for c in candidates_per_patient]} candidates per patient) ..."
        )
        tokenizer_ce, model_ce = _load_cross_encoder(CROSS_ENCODER_MODEL, device)
        retrieved_chunks_list = []
        for patient_idx, candidate_indices in enumerate(candidates_per_patient):
            note = str(df.iloc[patient_idx][query_column])
            candidates = [texts[i] for i in candidate_indices]
            ranked = _rerank_candidates(note, candidates, tokenizer_ce, model_ce, device, k)
            retrieved_chunks_list.append(ranked)
        del model_ce
    else:
        retrieved_chunks_list = [
            [texts[i] for i in candidate_indices[:k]]
            for candidate_indices in candidates_per_patient
        ]

    df = df.copy()
    df["retrieved_chunks"] = retrieved_chunks_list
    logger.info(f"Retrieval complete: top-{k} chunks × {len(retrieved_chunks_list)} rows.")
    return df
