"""PostgreSQL pool, schema, document/chunk CRUD, and dense vector search."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('uploaded', 'processing', 'ready', 'failed')),
    error_message TEXT,
    storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    heading TEXT,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id UUID PRIMARY KEY,
    configuration TEXT NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
"""


def _configure(conn: Any) -> None:
    register_vector(conn)


def _ensure_vector_extension() -> None:
    """Create the vector extension before any pooled connection registers the type."""
    conn = psycopg.connect(settings.database_url, autocommit=True)
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        conn.close()


def init_pool() -> ConnectionPool:
    """Create the global connection pool."""
    global _pool
    if _pool is None:
        _ensure_vector_extension()
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=_configure,
            open=True,
        )
    return _pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def init_schema() -> None:
    """Create extensions, tables, and indexes if they do not exist."""
    with get_pool().connection() as conn:
        conn.execute(SCHEMA_SQL)


def db_ok() -> bool:
    """Return True if a simple SELECT succeeds."""
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        logger.exception("health check: database unreachable")
        return False


def insert_document(
    *,
    doc_id: UUID,
    filename: str,
    file_type: str,
    status: str,
    storage_path: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, status, error_message, storage_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, filename, file_type, status, error_message, created_at
            """,
            (str(doc_id), filename, file_type, status, error_message, storage_path),
        ).fetchone()
    return dict(row)


def update_document_status(
    doc_id: UUID,
    status: str,
    error_message: str | None = None,
) -> None:
    with get_pool().connection() as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = %s, error_message = %s
            WHERE id = %s
            """,
            (status, error_message, str(doc_id)),
        )


def list_documents() -> list[dict[str, Any]]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, file_type, status, error_message, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: UUID) -> dict[str, Any] | None:
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT id, filename, file_type, status, error_message, storage_path, created_at
            FROM documents
            WHERE id = %s
            """,
            (str(doc_id),),
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: UUID) -> dict[str, Any] | None:
    """Delete a document row (chunks cascade). Returns the deleted row or None."""
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            DELETE FROM documents
            WHERE id = %s
            RETURNING id, filename, file_type, status, error_message, storage_path, created_at
            """,
            (str(doc_id),),
        ).fetchone()
    return dict(row) if row else None


def insert_chunks(doc_id: UUID, chunks: list[dict[str, Any]]) -> None:
    """Insert chunk rows with embeddings for a document."""
    if not chunks:
        return
    with get_pool().connection() as conn:
        with conn.transaction():
            for chunk in chunks:
                chunk_id = chunk.get("id") or uuid4()
                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, text, page_number, heading, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(chunk_id),
                        str(doc_id),
                        chunk["chunk_index"],
                        chunk["text"],
                        chunk["page_number"],
                        chunk.get("heading"),
                        chunk["embedding"],
                    ),
                )


def list_ready_chunks() -> list[dict[str, Any]]:
    """Return every chunk belonging to a ready document, for the BM25 index."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.page_number,
                c.heading,
                d.filename
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready'
            ORDER BY c.created_at, c.chunk_index
            """
        ).fetchall()
    return [dict(r) for r in rows]


def search_dense(embedding: list[float], top_k: int) -> list[dict[str, Any]]:
    """Return the nearest chunks by cosine distance (pgvector <=> )."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.text,
                c.page_number,
                c.heading,
                d.filename,
                (c.embedding <=> %s::vector) AS distance
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND d.status = 'ready'
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, top_k),
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        distance = float(item["distance"])
        item["score"] = 1.0 - distance
        results.append(item)
    return results


def insert_evaluation_run(configuration: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Persist one retrieval-evaluation run. Metrics are numbers, not document text."""
    run_id = uuid4()
    with get_pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO evaluation_runs (id, configuration, metrics)
            VALUES (%s, %s, %s)
            RETURNING id, configuration, metrics, created_at
            """,
            (str(run_id), configuration, Json(metrics)),
        ).fetchone()
    return dict(row)
