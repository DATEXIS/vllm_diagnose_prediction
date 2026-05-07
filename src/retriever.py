import asyncio
import json
import logging
import pickle
import re
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


def build_index(
    abstracts_path: str = None,
    index_persist_dir: str = None,
    max_abstracts: int = None,
    precomputed_config: dict = None,
):
    """
    Builds or loads a FAISS index.

    When precomputed_config.enabled is True, loads NCBI pre-computed
    MedCPT embeddings (downloading them if needed) and builds an IVFPQ
    index. Returns (faiss_index, TextAccessor).

    Otherwise encodes abstracts_path from scratch with MedCPT-Article-
    Encoder and builds an IndexFlatIP. Returns (faiss_index, list[str]).
    """
    if precomputed_config and precomputed_config.get("enabled"):
        # chunks: null in config means all 38 chunks (full PubMed)
        chunks = precomputed_config.get("chunks") or list(range(38))
        return _build_index_from_precomputed(
            precomputed_dir=precomputed_config["download_dir"],
            index_persist_dir=index_persist_dir,
            chunks=chunks,
            nlist=precomputed_config.get("nlist", 4096),
            nprobes=precomputed_config.get("nprobes", 128),
            flat_threshold=precomputed_config.get("flat_threshold", 20_000_000),
        )

    return _build_index_from_abstracts(abstracts_path, index_persist_dir, max_abstracts)


def _build_index_from_abstracts(abstracts_path: str, index_persist_dir: str, max_abstracts: int = None):
    """Encodes a plain-text abstracts file and builds a FAISS IndexFlatIP."""
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

    del model

    persist_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)

    logger.info(f"FAISS index saved: {index.ntotal} vectors, dim={index.d} → {index_persist_dir}")
    return index, texts


# ---------------------------------------------------------------------------
# Pre-computed NCBI MedCPT embeddings
# ---------------------------------------------------------------------------

_NCBI_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings"
_IVFPQ_M = 96    # subquantizers; 768 / 96 = 8 dims each → 96 bytes/vector
_IVFPQ_NBITS = 8  # 256 centroids per subquantizer


class TextAccessor:
    """
    Memory-efficient random access into a flat UTF-8 text file.
    One abstract per line; byte offsets are pre-computed so each lookup
    is a single seek + readline — no texts held in RAM.
    Implements the same __getitem__ / __len__ interface as a plain list
    so retrieve_for_dataframe works without changes.
    """

    def __init__(self, file_path: str, offsets: np.ndarray):
        self._path = file_path
        self._offsets = offsets
        self._fh = open(file_path, "rb")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> str:
        self._fh.seek(int(self._offsets[idx]))
        return self._fh.readline().decode("utf-8").rstrip("\n")

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass


def _download_file(url: str, dest: Path) -> None:
    """Stream-download url → dest with resume support."""
    import requests

    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    with requests.get(url, headers=headers, stream=True, timeout=3600) as r:
        if r.status_code == 416:  # Range Not Satisfiable → file already complete
            return
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)


def _download_chunks(precomputed_path: Path, chunks: list) -> None:
    for chunk_id in chunks:
        for prefix, ext in [("embeds", "npy"), ("pubmed", "json"), ("pmids", "json")]:
            name = f"{prefix}_chunk_{chunk_id}.{ext}"
            dest = precomputed_path / name
            if dest.exists():
                logger.info(f"  Already present: {name}")
                continue
            url = f"{_NCBI_FTP_BASE}/{name}"
            logger.info(f"  Downloading {url} ...")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                _download_file(url, tmp)
                tmp.rename(dest)
            except Exception as e:
                if tmp.exists():
                    tmp.unlink()
                raise RuntimeError(f"Download failed for {url}: {e}")


def _build_index_from_precomputed(
    precomputed_dir: str,
    index_persist_dir: str,
    chunks: list,
    nlist: int = 4096,
    nprobes: int = 128,
    flat_threshold: int = 20_000_000,
):
    """
    Builds (or loads) a FAISS index from NCBI pre-computed MedCPT embeddings.

    Index type is chosen automatically based on total vector count:
      ≤ flat_threshold  →  IndexFlatIP  (exact, best recall; needs ~4 GB RAM per 1 M vecs)
      >  flat_threshold  →  IndexIVFPQ  (approximate, ~96 B/vec; ~3.5 GB for 37 M vecs)

    Default flat_threshold = 20 M.  Note: IndexFlatIP for 20 M vecs needs ~61 GB RAM
    in the client pod — raise memory_limit in experiment.yaml if you use FlatIP at scale.

    Returns (faiss_index, TextAccessor) — same interface as _build_index_from_abstracts.
    """
    import faiss

    persist_path = Path(index_persist_dir)
    precomputed_path = Path(precomputed_dir)

    index_file   = persist_path / "faiss.index"
    offsets_file = persist_path / "text_offsets.npy"
    texts_file   = persist_path / "texts.txt"

    if index_file.exists() and offsets_file.exists() and texts_file.exists():
        logger.info(f"Loading cached FAISS index from {index_persist_dir}")
        index = faiss.read_index(str(index_file))
        if hasattr(index, "nprobe"):
            index.nprobe = nprobes
        offsets = np.load(str(offsets_file))
        logger.info(f"Index loaded: {index.ntotal:,} vectors, dim={index.d}")
        return index, TextAccessor(str(texts_file), offsets)

    precomputed_path.mkdir(parents=True, exist_ok=True)
    persist_path.mkdir(parents=True, exist_ok=True)

    # 1. Download missing chunk files
    logger.info(f"Checking/downloading {len(chunks)} chunks from NCBI FTP ...")
    _download_chunks(precomputed_path, chunks)

    # 2. Count total vectors using memory-mapped reads (no data loaded into RAM)
    logger.info("Counting total vectors across chunks ...")
    first_npy = precomputed_path / f"embeds_chunk_{chunks[0]}.npy"
    first_arr = np.load(str(first_npy), mmap_mode="r")
    dim = int(first_arr.shape[1])   # 768
    total_vectors = sum(
        np.load(str(precomputed_path / f"embeds_chunk_{c}.npy"), mmap_mode="r").shape[0]
        for c in chunks
    )
    logger.info(f"Total vectors: {total_vectors:,}  dim={dim}")

    # 3. Choose and (optionally) train the index
    if total_vectors <= flat_threshold:
        ram_gb = total_vectors * dim * 4 / 1e9
        logger.info(
            f"Using IndexFlatIP (exact search) — {total_vectors:,} vectors, "
            f"~{ram_gb:.1f} GB RAM needed in client pod."
        )
        index = faiss.IndexFlatIP(dim)
    else:
        logger.info(
            f"Using IndexIVFPQ — {total_vectors:,} vectors > threshold {flat_threshold:,}. "
            f"RAM ~{total_vectors * _IVFPQ_M / 1e9:.1f} GB."
        )
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, _IVFPQ_M, _IVFPQ_NBITS, faiss.METRIC_INNER_PRODUCT
        )
        logger.info(f"Training IVFPQ (nlist={nlist}) on {first_arr.shape[0]:,} vectors ...")
        index.train(first_arr.astype(np.float32))

    del first_arr

    # 4. Stream all chunks: add embeddings + write texts file with validation
    # Only articles with a non-empty abstract are added to both FAISS and
    # texts.txt so that every retrieved chunk has meaningful content.
    offsets: list = []
    byte_pos = 0
    first_chunk = True
    total_skipped = 0

    with open(texts_file, "wb") as tf:
        for chunk_id in tqdm(chunks, desc="Indexing chunks"):
            emb_path    = precomputed_path / f"embeds_chunk_{chunk_id}.npy"
            pubmed_path = precomputed_path / f"pubmed_chunk_{chunk_id}.json"
            pmids_path  = precomputed_path / f"pmids_chunk_{chunk_id}.json"

            embeddings = np.load(str(emb_path)).astype(np.float32)

            with open(pubmed_path, "r", encoding="utf-8") as pf:
                pubmed_data = json.load(pf)
            with open(pmids_path, "r", encoding="utf-8") as pmf:
                pmids = json.load(pmf)

            # On first chunk: log a sample so key names are visible in logs
            if first_chunk:
                sample = pubmed_data.get(str(pmids[0]), {})
                logger.info(f"  JSON keys in pubmed_chunk: {list(sample.keys())}")
                logger.info(
                    f"  Sample — t='{sample.get('t', '')[:80]}' "
                    f"a='{sample.get('a', '')[:80]}'"
                )
                first_chunk = False

            # Filter: only index articles that have a non-empty abstract.
            keep_mask = []
            for pmid in pmids:
                article  = pubmed_data.get(str(pmid), {})
                abstract = article.get("a", "").strip()
                keep_mask.append(bool(abstract))

            kept    = sum(keep_mask)
            skipped = len(pmids) - kept
            total_skipped += skipped
            logger.info(
                f"  Chunk {chunk_id}: {kept:,} articles with abstracts, "
                f"{skipped:,} skipped (no abstract)."
            )

            # Add only the kept embeddings to FAISS
            keep_indices = np.array([i for i, k in enumerate(keep_mask) if k], dtype=np.int64)
            if keep_indices.size:
                index.add(embeddings[keep_indices])

            # Write kept texts to texts.txt
            for i, pmid in enumerate(pmids):
                if not keep_mask[i]:
                    continue
                article  = pubmed_data.get(str(pmid), {})
                title    = article.get("t", "").strip().rstrip(".")
                abstract = article.get("a", "").strip()
                text     = f"{title}. {abstract}" if title else abstract
                encoded  = (text.replace("\n", " ") + "\n").encode("utf-8")
                offsets.append(byte_pos)
                tf.write(encoded)
                byte_pos += len(encoded)

            del embeddings, pubmed_data, pmids

    offsets_arr = np.array(offsets, dtype=np.int64)

    logger.info(
        f"Indexing complete: {index.ntotal:,} articles with abstracts indexed "
        f"({total_skipped:,} skipped — no abstract). "
        f"texts.txt: {len(offsets_arr):,} lines."
    )

    if hasattr(index, "nprobe"):
        index.nprobe = nprobes

    faiss.write_index(index, str(index_file))
    np.save(str(offsets_file), offsets_arr)

    index_type = "IndexFlatIP" if total_vectors <= flat_threshold else "IndexIVFPQ"
    logger.info(
        f"{index_type} index saved: {index.ntotal:,} vectors, "
        f"dim={index.d} → {index_persist_dir}"
    )
    return index, TextAccessor(str(texts_file), offsets_arr)


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
    "Generate exactly {n} numbered queries that together cover different aspects of the note:\n"
    "1. Primary diagnoses and chief complaints\n"
    "2. Comorbidities, medical history, current medications\n"
    "3. Procedures, laboratory findings, key clinical observations\n\n"
    "Rules:\n"
    "- Use only explicitly stated information (no assumptions).\n"
    "- Preserve negations and temporality (e.g., \"denies\", acute/chronic).\n"
    "- Remove non-clinical and redundant text.\n"
    "- Use standard medical terminology.\n\n"
    "Output exactly {n} lines, each starting with its number and a period.\n"
    "No explanation, no preamble.\n\n"
    "Admission note:\n{note}\n\n"
    "Queries:"
)


async def _rewrite_one(
    session, url: str, model: str, note: str, semaphore: asyncio.Semaphore, n_queries: int
) -> list:
    """Returns a list of n_queries strings for one admission note."""
    if n_queries == 1:
        prompt = _SINGLE_QUERY_PROMPT.format(note=note[:3000])
        max_tokens = 300
    else:
        prompt = _MULTI_QUERY_PROMPT.format(n=n_queries, note=note[:3000])
        max_tokens = 300 * n_queries

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
                if not resp.ok:
                    body = await resp.text()
                    logger.warning(
                        f"Query rewriting HTTP {resp.status}: {body[:300]!r} — falling back to raw note"
                    )
                    return [note[:500]]
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                logger.debug(f"Raw LLM output for query rewriting: {content[:300]!r}")
                # Strip Qwen3-style thinking block if present.
                if "</think>" in content:
                    after_think = content.split("</think>", 1)[-1].strip()
                    if not after_think:
                        logger.warning(
                            "Thinking block consumed all tokens — no content after </think>. "
                            "Falling back to raw note. Consider increasing max_tokens or ensuring "
                            "enable_thinking=False is respected by the server."
                        )
                        return [note[:500]]
                    content = after_think
                if n_queries == 1:
                    return [content] if content else [note[:500]]
                # Parse numbered list: "1. query", "2. query", ...
                # Works even when the response is partially truncated.
                queries = []
                for line in content.splitlines():
                    cleaned = re.sub(r'^\d+\.\s*', '', line.strip())
                    if cleaned:
                        queries.append(cleaned)
                if queries:
                    return queries[:n_queries]
                logger.warning(
                    f"Query rewriting produced no parseable lines. LLM output was: {content[:300]!r}. "
                    "Falling back to raw note."
                )
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
