"""Cross-encoder reranking. The model is loaded once per process."""

from __future__ import annotations

import threading
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import settings

_lock = threading.Lock()
_instance: "Reranker | None" = None


class Reranker:
    """Scores query/passage pairs and returns the highest-scoring passages."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Return up to top_k chunks, each with a cross-encoder score."""
        if not chunks or top_k <= 0:
            return []
        pairs = [(query, chunk["text"]) for chunk in chunks]
        raw = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(chunks, raw, strict=True), key=lambda item: float(item[1]), reverse=True)
        results: list[dict[str, Any]] = []
        for chunk, value in ranked[:top_k]:
            hit = dict(chunk)
            hit["score"] = float(value)
            results.append(hit)
        return results


def get_reranker() -> Reranker:
    """Return the process-wide cross-encoder, loading it on first use."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = Reranker(settings.rerank_model)
    return _instance
