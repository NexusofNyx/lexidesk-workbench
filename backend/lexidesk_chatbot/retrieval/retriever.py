"""
FAISS Retriever with Lazy Loading
Prevents import-time crashes if index doesn't exist yet.
"""
from pathlib import Path
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# --------------------------------------------------
# Paths
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# --------------------------------------------------
# Lazy-loaded globals (not loaded at import time!)
# --------------------------------------------------
_embed_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_metadata: Optional[List[Dict[str, Any]]] = None


def _load_resources():
    """
    Lazy-load the embedding model, FAISS index, and metadata.
    Only runs once per process and is reload-safe.
    """
    global _embed_model, _index, _metadata

    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if _index is None:
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Please ingest documents first via /upload or run indexer.py."
            )
        _index = faiss.read_index(str(FAISS_INDEX_PATH))

    if _metadata is None:
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. "
                "Please run indexer.py after ingestion."
            )
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def reload_index():
    """
    Force reload of the index (useful after new ingestion).
    Call this after build_index() if needed.
    """
    global _index, _metadata
    _index = None
    _metadata = None
    _load_resources()


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar chunks from FAISS index.

    Args:
        query: User question or search query
        k: Number of results to return

    Returns:
        List of dicts with keys: doc_id, page, text, distance, chunk_id
    """
    _load_resources()

    # Encode query
    q_emb = _embed_model.encode([query]).astype("float32")

    # Search FAISS
    distances, indices = _index.search(q_emb, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0 or idx >= len(_metadata):
            continue  # Skip invalid indices

        chunk = _metadata[idx].copy()
        chunk["distance"] = float(dist)
        chunk["chunk_id"] = int(idx)
        results.append(chunk)

    return results
