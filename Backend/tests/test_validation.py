"""Upload checks reject a bad extension, an empty file, and an oversized file."""

import pytest

from app.config import settings
from app.ingestion import IngestionError, validate_upload


def test_bad_extension() -> None:
    with pytest.raises(IngestionError, match="Only PDF, TXT, and Markdown"):
        validate_upload("notes.docx", "application/octet-stream", b"hello")


def test_empty_file() -> None:
    with pytest.raises(IngestionError, match="File is empty"):
        validate_upload("notes.txt", "text/plain", b"")


def test_oversized_file(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_upload_bytes", 4)
    with pytest.raises(IngestionError, match="exceeds size limit"):
        validate_upload("notes.txt", "text/plain", b"hello")
