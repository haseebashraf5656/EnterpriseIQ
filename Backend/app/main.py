"""FastAPI application: document APIs, health, debug search, grounded chat, and evaluation."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.db import (
    db_ok,
    delete_document,
    get_document,
    init_pool,
    init_schema,
    insert_chunks,
    insert_document,
    list_documents,
    close_pool,
    update_document_status,
)
from app.embeddings import get_embeddings
from app.evaluation import run_configurations
from app.generation import (
    INSUFFICIENT_ANSWER,
    GenerationError,
    build_citations,
    generate,
    should_refuse,
)
from app.rerank import get_reranker
from app.retrieval import drop_document, index_chunks, rebuild_bm25, retrieve
from app.ingestion import (
    IngestionError,
    chunk_pages,
    clean_pages,
    extract_pages,
    prepare_chunk_records,
    save_bytes,
    validate_upload,
)
from app.schemas import (
    ChatIn,
    ChatOut,
    ChunkHit,
    CitationOut,
    DebugSearchIn,
    DebugSearchOut,
    DocumentOut,
    EvaluateIn,
    EvaluateOut,
    EvaluateResult,
    HealthOut,
    RetrievalMeta,
    SourceOut,
    UploadItem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_pool()
    init_schema()
    get_embeddings()
    get_reranker()
    rebuild_bm25()
    yield
    close_pool()


app = FastAPI(title="EnterpriseIQ", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Log method, path, status, and duration. Do not log bodies or secrets."""
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - started) * 1000
        logger.exception(
            "request method=%s path=%s status=500 ms=%.1f",
            request.method,
            request.url.path,
            elapsed,
        )
        raise
    elapsed = (time.perf_counter() - started) * 1000
    logger.info(
        "request method=%s path=%s status=%s ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("http error status=%s detail=%s", exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("request validation failed")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_error(_request: Request, _exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _upload_dir() -> Path:
    return Path(settings.upload_dir)


def _stored_extension(filename: str | None, file_type: str) -> str:
    if filename and Path(filename).suffix.lower() == ".md":
        return ".md"
    if filename and Path(filename).suffix.lower() == ".txt":
        return ".txt"
    if file_type == "pdf":
        return ".pdf"
    if file_type == "md":
        return ".md"
    return f".{file_type}"


def ingest_document(doc_id: UUID) -> None:
    """Background ingestion: extract, chunk, embed, persist. Never logs full text."""
    doc = get_document(doc_id)
    if doc is None:
        logger.error("ingest skipped: document %s missing", doc_id)
        return
    update_document_status(doc_id, "processing")
    logger.info(
        "ingest start document_id=%s filename=%s file_type=%s",
        doc_id,
        doc["filename"],
        doc["file_type"],
    )
    try:
        path = Path(doc["storage_path"])
        pages = clean_pages(extract_pages(path, doc["file_type"]))
        drafts = chunk_pages(pages, file_type=doc["file_type"])
        records = prepare_chunk_records(doc_id, doc["filename"], drafts)
        insert_chunks(doc_id, records)
        update_document_status(doc_id, "ready")
        index_chunks(records)
        logger.info(
            "ingest ready document_id=%s chunks=%s",
            doc_id,
            len(records),
        )
    except IngestionError as exc:
        logger.warning("ingest failed document_id=%s error=%s", doc_id, str(exc))
        update_document_status(doc_id, "failed", str(exc))
    except Exception:
        logger.exception("ingest crashed document_id=%s", doc_id)
        update_document_status(doc_id, "failed", "Internal ingestion error")


async def _ingest_job(doc_id: UUID) -> None:
    await run_in_threadpool(ingest_document, doc_id)


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    ok = db_ok()
    return HealthOut(status="ok" if ok else "degraded", db=ok)


@app.post("/documents/upload", response_model=list[UploadItem])
async def upload_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> list[UploadItem]:
    """Accept many files. One invalid file does not reject the rest."""
    results: list[UploadItem] = []
    for file in files:
        filename = file.filename or "upload"
        data = await file.read()
        try:
            file_type = validate_upload(file.filename, file.content_type, data)
        except IngestionError as exc:
            logger.info("upload rejected filename=%s", filename)
            results.append(UploadItem(filename=filename, error=str(exc)))
            continue
        try:
            stored = save_bytes(data, _stored_extension(file.filename, file_type), _upload_dir())
        except OSError:
            logger.exception("failed to store upload filename=%s", filename)
            results.append(UploadItem(filename=filename, error="Could not store upload"))
            continue
        doc_id = uuid4()
        row = insert_document(
            doc_id=doc_id,
            filename=Path(file.filename or stored.name).name,
            file_type=file_type,
            status="processing",
            storage_path=str(stored),
        )
        logger.info("upload accepted document_id=%s bytes=%s", doc_id, len(data))
        background_tasks.add_task(_ingest_job, doc_id)
        results.append(UploadItem(id=row["id"], filename=row["filename"], status=row["status"]))
    return results


@app.get("/documents", response_model=list[DocumentOut])
def get_documents() -> list[DocumentOut]:
    return [DocumentOut(**row) for row in list_documents()]


@app.delete("/documents/{document_id}", response_model=DocumentOut)
def remove_document(document_id: UUID) -> DocumentOut:
    row = delete_document(document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(row["storage_path"])
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        logger.exception("failed to remove file document_id=%s", document_id)
    drop_document(document_id)
    logger.info("deleted document_id=%s", document_id)
    return DocumentOut(
        id=row["id"],
        filename=row["filename"],
        file_type=row["file_type"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
    )


@app.post("/debug/search", response_model=DebugSearchOut)
def debug_search(body: DebugSearchIn) -> DebugSearchOut:
    found = retrieve(body.query, body.mode, top_k=body.top_k)
    results = [
        ChunkHit(
            chunk_id=h["chunk_id"],
            document_id=h["document_id"],
            filename=h["filename"],
            page=h["page_number"],
            heading=h["heading"],
            chunk_index=h["chunk_index"],
            text=h["text"],
            score=h["score"],
        )
        for h in found["chunks"]
    ]
    return DebugSearchOut(
        query=body.query,
        mode=body.mode,
        latency_ms=found["latency_ms"],
        results=results,
    )


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> ChatOut:
    """Retrieve passages and answer only from their stored metadata."""
    found = retrieve(body.question, body.mode)
    chunks = found["chunks"]
    retrieval = RetrievalMeta(
        method=body.mode,
        top_k=settings.top_k,
        latency_ms=found["latency_ms"],
    )
    sources = [
        SourceOut(filename=chunk["filename"], page=int(chunk["page_number"]), text=chunk["text"])
        for chunk in chunks
    ]
    logger.info(
        "chat mode=%s question_chars=%s hits=%s",
        body.mode,
        len(body.question),
        len(chunks),
    )
    if should_refuse(chunks, body.mode):
        return ChatOut(
            answer=INSUFFICIENT_ANSWER,
            insufficient_evidence=True,
            citations=[],
            retrieval=retrieval,
            sources=sources,
        )
    try:
        generated = generate(body.question, chunks)
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not generated["sufficient"]:
        answer = generated["answer"].strip() or INSUFFICIENT_ANSWER
        return ChatOut(
            answer=answer,
            insufficient_evidence=True,
            citations=[],
            retrieval=retrieval,
            sources=sources,
        )
    citations = [
        CitationOut(**item) for item in build_citations(chunks, generated["used_passage_ids"])
    ]
    return ChatOut(
        answer=generated["answer"],
        insufficient_evidence=False,
        citations=citations,
        retrieval=retrieval,
        sources=sources,
    )


@app.post("/evaluate", response_model=EvaluateOut)
def evaluate(body: EvaluateIn) -> EvaluateOut:
    """Score one retrieval mode, or all four. Does not call the language model."""
    try:
        results = run_configurations(body.configuration)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EvaluateOut(results=[EvaluateResult(**row) for row in results])
