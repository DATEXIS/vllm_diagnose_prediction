import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)

_model = None
_device = None


def get_device() -> str:
    """Determine whether to use CUDA or CPU for embeddings."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedding_model(model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") -> None:
    """Load sentence-transformers model for embeddings.
    
    Args:
        model_name: HuggingFace model name. Default: PubMedBERT for biomedical text.
    """
    global _model, _device
    
    if _model is not None:
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        
        _device = get_device()
        logger.info(f"Loading embedding model '{model_name}' on {_device}")
        _model = SentenceTransformer(model_name, device=_device)
        logger.info(f"Embedding model loaded successfully")
    except ImportError:
        logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
        raise


def encode_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Encode texts to embeddings using loaded model.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.
        
    Returns:
        List of embedding vectors (list of floats).
    """
    global _model
    
    if _model is None:
        load_embedding_model()
    
    embeddings = _model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=len(texts) > 100)
    return [emb.tolist() for emb in embeddings]


def encode_single_text(text: str) -> List[float]:
    """Encode a single text to embedding.
    
    Args:
        text: Text string to embed.
        
    Returns:
        Embedding vector as list of floats.
    """
    return encode_texts([text])[0]


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector.
        embedding2: Second embedding vector.
        
    Returns:
        Cosine similarity score between -1 and 1.
    """
    import numpy as np
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def clear_embedding_model() -> None:
    """Clear loaded model to free memory."""
    global _model, _device
    _model = None
    _device = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()