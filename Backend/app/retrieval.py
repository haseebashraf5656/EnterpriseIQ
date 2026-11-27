"""Dense search, in-memory BM25, reciprocal rank fusion, and mode dispatch."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any
from uuid import UUID

from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)

MODES = ("dense", "bm25", "hybrid", "hybrid_reranked")

# Lowercase matching is applied by the caller. Dotted sections and hyphenated
# codes stay one token: "4.2.1", "ERR-404".
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+", re.IGNORECASE)

_lock = threading.RLock()
_chunks: list[dict[str, Any]] = []
_index: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping section numbers and codes intact."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _chunk_row(row: dict[str, Any]) -> dict[str, Any]:
    """Copy retrieval fields. Stored text is not modified."""
    return {
        "chunk_id": row.get("chunk_id", row.get("id")),
        "document_id": row["document_id"],
        "chunk_index": row["chunk_index"],
        "text": row["text"],
        "page_number": row["page_number"],
        "heading": row.get("heading"),
        "filename": row["filename"],
    }


def _rebuild_locked(rows: list[dict[str, Any]]) -> None:
    global _chunks, _index
    prepared = [_chunk_row(row) for row in rows]
    tokenized = [tokenize(chunk["text"]) for chunk in prepared]
    keep = [(chunk, tokens) for chunk, tokens in zip(prepared, tokenized, strict=True) if tokens]
    _chunks = [chunk for chunk, _tokens in keep]
    _index = BM25Okapi([tokens for _chunk, tokens in keep]) if keep else None


def rebuild_bm25() -> None:
    """Replace the in-memory index with every ready chunk in the database."""
    from app.db import list_ready_chunks

    rows = list_ready_chunks()
    with _lock:
        _rebuild_locked(rows)
    logger.info("bm25 rebuilt chunks=%s", len(_chunks))


def index_chunks(rows: list[dict[str, Any]]) -> None:
    """Add freshly ingested chunks and rebuild the index."""
    if not rows:
        return
    with _lock:
        _rebuild_locked(_chunks + [_chunk_row(row) for row in rows])


def drop_document(document_id: UUID) -> None:
    """Drop one document's chunks from the in-memory index."""
    key = str(document_id)
    with _lock:
        kept = [chunk for chunk in _chunks if str(chunk["document_id"]) != key]
        _rebuild_locked(kept)


def dense_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed the question with the same model used for chunks and search pgvector."""
    from app.db import search_dense
    from app.embeddings import get_embeddings

    vector = get_embeddings().embed_query(query)
    return search_dense(vector, top_k)


def bm25_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the highest BM25 scores for the tokenized query."""
    tokens = tokenize(query)
    if not tokens or top_k <= 0:
        return []
    with _lock:
        if _index is None:
            return []
        scores = _index.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results: list[dict[str, Any]] = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                break
            hit = dict(_chunks[i])
            hit["score"] = score
            results.append(hit)
            if len(results) >= top_k:
                break
        return results


def rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    *,
    k: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse ranked lists. score = sum 1/(k + rank), rank starting at 1."""
    rrf_k = settings.rrf_k if k is None else k
    cap = settings.fused_candidates if limit is None else limit
    scores: dict[Any, float] = {}
    best: dict[Any, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk_id not in best:
                best[chunk_id] = hit
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:cap]
    fused: list[dict[str, Any]] = []
    for chunk_id in ordered:
        hit = dict(best[chunk_id])
        hit["score"] = scores[chunk_id]
        fused.append(hit)
    return fused


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def retrieve(question: str, mode: str, top_k: int | None = None) -> dict[str, Any]:
    """Run one retrieval mode. Returns chunks plus per-stage latency in ms."""
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}'")
    final_k = settings.top_k if top_k is None else top_k
    latency: dict[str, float] = {}

    if mode == "dense":
        start = time.perf_counter()
        chunks = dense_search(question, final_k)
        latency["dense"] = _elapsed_ms(start)
        return {"chunks": chunks, "latency_ms": latency}

    if mode == "bm25":
        start = time.perf_counter()
        chunks = bm25_search(question, final_k)
        latency["bm25"] = _elapsed_ms(start)
        return {"chunks": chunks, "latency_ms": latency}

    start = time.perf_counter()
    dense_hits = dense_search(question, settings.dense_candidates)
    latency["dense"] = _elapsed_ms(start)

    start = time.perf_counter()
    bm25_hits = bm25_search(question, settings.bm25_candidates)
    latency["bm25"] = _elapsed_ms(start)

    start = time.perf_counter()
    fused = rrf_fuse([dense_hits, bm25_hits])
    latency["rrf"] = _elapsed_ms(start)

    if mode == "hybrid":
        return {"chunks": fused[:final_k], "latency_ms": latency}

    from app.rerank import get_reranker

    start = time.perf_counter()
    reranked = get_reranker().score(question, fused, final_k)
    latency["rerank"] = _elapsed_ms(start)
    return {"chunks": reranked, "latency_ms": latency}
