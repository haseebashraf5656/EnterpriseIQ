"""Singleton SentenceTransformer wrapper used for both chunks and questions."""

from __future__ import annotations

import threading

from app.config import settings

_lock = threading.Lock()
_instance: "EmbeddingModel | None" = None


class EmbeddingModel:
    """Loads one embedding model and exposes encode + token counting."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_count(self, text: str) -> int:
        return len(self.token_ids(text))

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def get_embeddings() -> EmbeddingModel:
    """Return the process-wide embedding model, loading it on first use."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = EmbeddingModel(settings.embedding_model)
    return _instance
