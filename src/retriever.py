# src/retriever.py
import logging
from pathlib import Path

import pandas as pd
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


def build_index(abstracts_path: str, index_persist_dir: str) -> VectorStoreIndex:
    """
    Liest die .txt Datei mit PubMed-Abstracts (eine Zeile = ein Abstract),
    erstellt einen LlamaIndex VectorStoreIndex mit MedCPT Embeddings
    und speichert ihn auf Disk.
    """
    persist_path = Path(index_persist_dir)

    # Wenn Index bereits existiert, einfach laden
    if persist_path.exists() and any(persist_path.iterdir()):
        logger.info(f"Index bereits vorhanden, lade von {index_persist_dir}")
        return _load_index(index_persist_dir)

    logger.info(f"Baue Index aus {abstracts_path} ...")

    # MedCPT als Embedding-Modell setzen
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="ncbi/MedCPT-Article-Encoder",
        max_length=512,
    )
    # Kein LLM nötig für reines Retrieval
    Settings.llm = None

    # Abstracts laden: eine Zeile = ein Dokument
    documents = []
    with open(abstracts_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                documents.append(Document(text=line, id_=str(i)))

    logger.info(f"{len(documents)} Abstracts geladen, erstelle Embeddings ...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    # Index auf Disk speichern (damit du ihn nicht jedes Mal neu bauen musst)
    persist_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=index_persist_dir)
    logger.info(f"Index gespeichert in {index_persist_dir}")

    return index


def _load_index(index_persist_dir: str) -> VectorStoreIndex:
    """Lädt einen bereits gespeicherten Index von Disk."""
    from llama_index.core import StorageContext, load_index_from_storage

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="ncbi/MedCPT-Article-Encoder",
        max_length=512,
    )
    Settings.llm = None

    storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)
    return load_index_from_storage(storage_context)


def retrieve_for_dataframe(
    df: pd.DataFrame,
    index: VectorStoreIndex,
    k: int = 5,
    query_column: str = "admission_note",
) -> pd.DataFrame:
    """
    Für jede Zeile im DataFrame: retrievet Top-k Abstracts
    und speichert sie als neue Spalte 'retrieved_chunks'.

    Der admission_note Text wird direkt als Query verwendet (Phase 1).
    """
    # Query Encoder für Suchanfragen (anders als Article Encoder)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="ncbi/MedCPT-Query-Encoder",
        max_length=64,  # Queries sind kurz, MedCPT empfiehlt max_length=64
    )

    retriever = index.as_retriever(similarity_top_k=k)

    retrieved_chunks_list = []

    for i, row in df.iterrows():
        query = str(row[query_column])

        # Admission Note kürzen – zu lange Queries verschlechtern Dense Retrieval
        query_truncated = query[:500]

        nodes = retriever.retrieve(query_truncated)

        # Chunks als Liste von Strings speichern
        chunks = [node.text for node in nodes]
        retrieved_chunks_list.append(chunks)

        if i % 50 == 0:
            logger.info(f"Retrieval Fortschritt: {i}/{len(df)}")

    df = df.copy()
    df["retrieved_chunks"] = retrieved_chunks_list
    return df