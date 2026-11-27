"""Pydantic request and response models."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import settings

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_reranked"]


class DocumentOut(BaseModel):
    id: UUID
    filename: str
    file_type: str
    status: str
    error_message: str | None = None
    created_at: datetime


class HealthOut(BaseModel):
    status: str
    db: bool


class DebugSearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=30)
    mode: RetrievalMode = "hybrid_reranked"


class ChunkHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page: int
    heading: str | None
    chunk_index: int
    text: str
    score: float


class DebugSearchOut(BaseModel):
    query: str
    mode: RetrievalMode
    latency_ms: dict[str, float]
    results: list[ChunkHit]


class ChatIn(BaseModel):
    question: str = Field(min_length=1)
    mode: RetrievalMode = "hybrid_reranked"

    @field_validator("question")
    @classmethod
    def _question_length(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must not be empty")
        if len(text) > settings.max_question_chars:
            raise ValueError(f"question exceeds {settings.max_question_chars} characters")
        return text


class CitationOut(BaseModel):
    filename: str
    page: int
    chunk_id: UUID


class RetrievalMeta(BaseModel):
    method: str
    top_k: int
    latency_ms: dict[str, float]


class SourceOut(BaseModel):
    filename: str
    page: int
    text: str


class ChatOut(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[CitationOut]
    retrieval: RetrievalMeta
    sources: list[SourceOut]


class UploadItem(BaseModel):
    id: UUID | None = None
    filename: str
    status: str | None = None
    error: str | None = None


class EvaluateIn(BaseModel):
    configuration: str = Field(min_length=1, max_length=64)

    @field_validator("configuration")
    @classmethod
    def _configuration(cls, value: str) -> str:
        return value.strip().lower()


class EvaluateResult(BaseModel):
    configuration: str
    metrics: dict


class EvaluateOut(BaseModel):
    results: list[EvaluateResult]
