import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _encode(texts: list, model_name: str, max_length: int, batch_size: int, device: str) -> np.ndarray:
    """Encode a list of texts using CLS-token pooling (required by MedCPT)."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=model_name.split("/")[-1]):
        batch = texts[start : start + batch_size]
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
            # MedCPT uses the [CLS] token — do NOT use mean pooling
            emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embeddings.append(emb)

    return np.vstack(all_embeddings)


def build_index(abstracts_path: str, index_persist_dir: str, max_abstracts: int = None):
    """
    Reads PubMed abstracts (one line = one abstract), encodes them with
    MedCPT-Article-Encoder, and stores a FAISS inner-product index on disk.

    Storage is ~768 MB binary for 250k abstracts (vs. several GB with LlamaIndex JSON).
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

    logger.info(f"Reading abstracts from {abstracts_path} ...")
    texts = []
    with open(abstracts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
            if max_abstracts and len(texts) >= max_abstracts:
                break

    logger.info(f"{len(texts)} abstracts loaded — encoding with {ARTICLE_MODEL} ...")
    device = _get_device()
    logger.info(f"Device: {device}")

    embeddings = _encode(texts, ARTICLE_MODEL, max_length=512, batch_size=64, device=device)

    # L2-normalise so inner product equals cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / np.clip(norms, 1e-8, None)).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    persist_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)

    logger.info(f"FAISS index saved to {index_persist_dir} ({index.ntotal} vectors, dim={dim})")
    return index, texts


def retrieve_for_dataframe(
    df: pd.DataFrame,
    index,
    texts: list,
    k: int = 5,
    query_column: str = "admission_note",
) -> pd.DataFrame:
    """
    Encodes each admission note with MedCPT-Query-Encoder and retrieves
    the top-k most similar PubMed abstracts. Stores results as 'retrieved_chunks'
    (list of strings per row).
    """
    queries = [str(val)[:500] for val in df[query_column]]

    logger.info(f"Encoding {len(queries)} queries with {QUERY_MODEL} ...")
    device = _get_device()
    q_embs = _encode(queries, QUERY_MODEL, max_length=64, batch_size=32, device=device)

    norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
    q_embs = (q_embs / np.clip(norms, 1e-8, None)).astype(np.float32)

    _, indices = index.search(q_embs, k)

    retrieved_chunks_list = [
        [texts[i] for i in row_idx if i != -1 and i < len(texts)]
        for row_idx in indices
    ]

    df = df.copy()
    df["retrieved_chunks"] = retrieved_chunks_list
    logger.info(f"Retrieval complete: top-{k} chunks per row, {len(retrieved_chunks_list)} rows total.")
    return df
