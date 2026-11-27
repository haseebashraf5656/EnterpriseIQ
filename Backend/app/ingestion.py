"""Validate uploads, extract text, clean, and chunk with page/heading metadata."""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
ALLOWED_MIME = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {
        "text/markdown",
        "text/x-markdown",
        "text/plain",
        "application/octet-stream",
    },
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"[ \t]+")


class IngestionError(Exception):
    """Raised when a file cannot be accepted or parsed."""


@dataclass
class PageText:
    page_number: int
    text: str
    heading: str | None = None


@dataclass
class ChunkDraft:
    text: str
    page_number: int
    heading: str | None
    chunk_index: int = 0


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_upload(filename: str | None, content_type: str | None, data: bytes) -> str:
    """Validate extension, MIME, size, and basic parseability. Returns file_type."""
    if not filename:
        raise IngestionError("Filename is required")
    ext = extension_of(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestionError("Only PDF, TXT, and Markdown files are allowed")
    if len(data) == 0:
        raise IngestionError("File is empty")
    if len(data) > settings.max_upload_bytes:
        raise IngestionError(
            f"File exceeds size limit of {settings.max_upload_bytes} bytes"
        )
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime and mime not in ALLOWED_MIME[ext]:
        raise IngestionError(f"MIME type '{mime}' does not match extension {ext}")
    if ext == ".pdf":
        if not data.startswith(b"%PDF"):
            raise IngestionError("File is not a valid PDF")
        _assert_pdf_readable(data)
    else:
        if b"\x00" in data:
            raise IngestionError("Text file contains binary data")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IngestionError("Text file is not valid UTF-8") from exc
        if not text.strip():
            raise IngestionError("File has no extractable text")
    return ext.lstrip(".")


def _assert_pdf_readable(data: bytes) -> None:
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, Exception) as exc:
        raise IngestionError("PDF is corrupt or unreadable") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise IngestionError("PDF is encrypted and cannot be read") from exc
        if not unlocked:
            raise IngestionError("PDF is encrypted and cannot be read")
    if len(reader.pages) == 0:
        raise IngestionError("PDF has no pages")


def save_bytes(data: bytes, ext: str, upload_dir: Path) -> Path:
    """Write bytes under a generated UUID name. Never uses the original filename."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid4().hex}{ext}"
    path.write_bytes(data)
    return path


def extract_pages(path: Path, file_type: str) -> list[PageText]:
    """Extract text page-by-page, preserving page numbers."""
    if file_type == "pdf":
        return _extract_pdf(path)
    text = path.read_text(encoding="utf-8")
    pages = [p for p in text.split("\f") if p.strip()] or ([text] if text.strip() else [])
    if not pages:
        raise IngestionError("File has no extractable text")
    return [PageText(page_number=i, text=p) for i, p in enumerate(pages, start=1)]


def _extract_pdf(path: Path) -> list[PageText]:
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, Exception) as exc:
        raise IngestionError("PDF is corrupt or unreadable") from exc
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise IngestionError("PDF is encrypted and cannot be read") from exc
        if not unlocked:
            raise IngestionError("PDF is encrypted and cannot be read")
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            raise IngestionError(f"Failed to extract text from PDF page {i}") from exc
        pages.append(PageText(page_number=i, text=raw))
    if not any(p.text.strip() for p in pages):
        raise IngestionError("PDF has no extractable text")
    return pages


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Normalize whitespace and strip repeated headers/footers when safe."""
    cleaned = [
        PageText(page_number=p.page_number, text=_normalize_ws(p.text), heading=p.heading)
        for p in pages
    ]
    if len(cleaned) < 3:
        return [p for p in cleaned if p.text.strip()]

    first_lines: list[str] = []
    last_lines: list[str] = []
    split_pages: list[list[str]] = []
    for page in cleaned:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        split_pages.append(lines)
        first_lines.append(lines[0] if lines else "")
        last_lines.append(lines[-1] if lines else "")

    headers = _frequent_short(first_lines, n=len(cleaned))
    footers = _frequent_short(last_lines, n=len(cleaned))
    out: list[PageText] = []
    for page, lines in zip(cleaned, split_pages, strict=True):
        if lines and lines[0] in headers:
            lines = lines[1:]
        if lines and lines[-1] in footers:
            lines = lines[:-1]
        text = "\n".join(lines)
        if text.strip():
            out.append(PageText(page_number=page.page_number, text=text, heading=page.heading))
    return out


def _frequent_short(lines: list[str], n: int) -> set[str]:
    counts = Counter(lines)
    return {s for s, c in counts.items() if s and len(s) < 80 and c / n >= 0.5}


def _normalize_ws(text: str) -> str:
    lines = [WHITESPACE_RE.sub(" ", ln).rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    collapsed: list[str] = []
    blank = False
    for ln in lines:
        if not ln.strip():
            if not blank:
                collapsed.append("")
            blank = True
        else:
            collapsed.append(ln.strip())
            blank = False
    return "\n".join(collapsed).strip()


def split_markdown_sections(text: str, page_number: int) -> list[PageText]:
    """Split markdown so ATX headings become section boundaries for chunking."""
    sections: list[PageText] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(PageText(page_number=page_number, text=body, heading=heading))

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            buf = []
        else:
            buf.append(line)
    flush()
    if not sections and text.strip():
        sections.append(PageText(page_number=page_number, text=text.strip(), heading=None))
    return sections


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _units_for_section(
    section: PageText,
    *,
    max_tokens: int,
    token_count,
) -> list[tuple[str, int, str | None]]:
    """Split a section heading -> paragraph -> sentence into token-bounded units."""
    units: list[tuple[str, int, str | None]] = []
    for para in _split_paragraphs(section.text):
        if token_count(para) <= max_tokens:
            units.append((para, section.page_number, section.heading))
            continue
        for sentence in _split_sentences(para):
            if token_count(sentence) <= max_tokens:
                units.append((sentence, section.page_number, section.heading))
                continue
            units.extend(
                _force_split(sentence, section.page_number, section.heading, max_tokens, token_count)
            )
    return units


def _token_windows(
    text: str,
    page: int,
    heading: str | None,
    max_tokens: int,
) -> list[tuple[str, int, str | None]]:
    """Split text on tokenizer windows so each piece is at most max_tokens."""
    embedder = get_embeddings()
    ids = embedder.token_ids(text)
    pieces: list[tuple[str, int, str | None]] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + max_tokens]
        piece = embedder.decode_tokens(window).strip()
        step = len(window)
        while step > 1 and piece and embedder.token_count(piece) > max_tokens:
            step -= 1
            window = ids[start : start + step]
            piece = embedder.decode_tokens(window).strip()
        if piece:
            pieces.append((piece, page, heading))
        start += max(1, step)
    return pieces


def _force_split(
    text: str,
    page: int,
    heading: str | None,
    max_tokens: int,
    token_count,
) -> list[tuple[str, int, str | None]]:
    words = text.split()
    pieces: list[tuple[str, int, str | None]] = []
    buf: list[str] = []
    for word in words:
        if token_count(word) > max_tokens:
            if buf:
                pieces.append((" ".join(buf), page, heading))
                buf = []
            pieces.extend(_token_windows(word, page, heading, max_tokens))
            continue
        trial = " ".join(buf + [word])
        if buf and token_count(trial) > max_tokens:
            pieces.append((" ".join(buf), page, heading))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        pieces.append((" ".join(buf), page, heading))
    return pieces


def _with_overlap(prev: str, nxt: str, overlap: int, max_tokens: int) -> str:
    """Prefix nxt with up to `overlap` tokens from prev without exceeding max_tokens."""
    embedder = get_embeddings()
    if embedder.token_count(nxt) >= max_tokens or overlap <= 0:
        return nxt
    prev_ids = embedder.token_ids(prev)
    take = min(overlap, len(prev_ids), max_tokens - embedder.token_count(nxt))
    combined = nxt
    while take > 0:
        prefix = embedder.decode_tokens(prev_ids[-take:]).strip()
        candidate = f"{prefix} {nxt}".strip() if prefix else nxt
        if embedder.token_count(candidate) <= max_tokens:
            return candidate
        take -= 1
    return combined


def chunk_pages(
    pages: list[PageText],
    *,
    file_type: str,
    chunk_max_tokens: int | None = None,
    chunk_min_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> list[ChunkDraft]:
    """Pack text into chunks of at most chunk_max_tokens using the embedding tokenizer."""
    max_tokens = chunk_max_tokens if chunk_max_tokens is not None else settings.chunk_max_tokens
    min_tokens = chunk_min_tokens if chunk_min_tokens is not None else settings.chunk_min_tokens
    ov = chunk_overlap_tokens if chunk_overlap_tokens is not None else settings.chunk_overlap_tokens

    embedder = get_embeddings()
    token_count = embedder.token_count

    sections: list[PageText] = []
    for page in pages:
        if file_type == "md" or HEADING_RE.search(page.text):
            sections.extend(split_markdown_sections(page.text, page.page_number))
        else:
            sections.append(page)

    units: list[tuple[str, int, str | None]] = []
    for section in sections:
        units.extend(_units_for_section(section, max_tokens=max_tokens, token_count=token_count))

    chunks: list[ChunkDraft] = []
    current_text = ""
    current_page = 1
    current_heading: str | None = None

    def flush() -> None:
        nonlocal current_text
        if current_text.strip():
            chunks.append(
                ChunkDraft(
                    text=current_text.strip(),
                    page_number=current_page,
                    heading=current_heading,
                )
            )

    for text, page, heading in units:
        if not current_text:
            current_text = text
            current_page = page
            current_heading = heading
            continue
        combined = f"{current_text} {text}"
        if token_count(combined) <= max_tokens:
            current_text = combined
            continue
        flush()
        current_text = _with_overlap(current_text, text, ov, max_tokens)
        current_page = page
        current_heading = heading
    flush()

    if len(chunks) >= 2 and token_count(chunks[-1].text) < min_tokens:
        merged = f"{chunks[-2].text} {chunks[-1].text}"
        if token_count(merged) <= max_tokens:
            chunks[-2].text = merged
            chunks.pop()

    bounded: list[ChunkDraft] = []
    for chunk in chunks:
        if token_count(chunk.text) <= max_tokens:
            bounded.append(chunk)
            continue
        for piece, page, heading in _token_windows(
            chunk.text, chunk.page_number, chunk.heading, max_tokens
        ):
            bounded.append(ChunkDraft(text=piece, page_number=page, heading=heading))

    for i, chunk in enumerate(bounded):
        chunk.chunk_index = i
    return bounded


def prepare_chunk_records(
    document_id: UUID,
    filename: str,
    drafts: list[ChunkDraft],
) -> list[dict]:
    """Embed drafts and attach metadata for DB insert."""
    if not drafts:
        raise IngestionError("No chunks produced from document")
    embedder = get_embeddings()
    vectors = embedder.embed_texts([c.text for c in drafts])
    records = []
    for draft, vector in zip(drafts, vectors, strict=True):
        records.append(
            {
                "id": uuid4(),
                "document_id": document_id,
                "filename": filename,
                "chunk_index": draft.chunk_index,
                "text": draft.text,
                "page_number": draft.page_number,
                "heading": draft.heading,
                "embedding": vector,
            }
        )
    return records
