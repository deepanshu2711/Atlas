from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Loaded on first use, so nothing is downloaded or held in memory while
    reranking is off."""
    return CrossEncoder(settings.reranker_model, max_length=512)
