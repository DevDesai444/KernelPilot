from __future__ import annotations

from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install it with: pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)


def embed_query(text: str, provider: str, model_name: str) -> list[float]:
    if provider == "sentence-transformers":
        model = _load_sentence_transformer(model_name)
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    raise RuntimeError(f"Unsupported embedding provider: {provider}")
