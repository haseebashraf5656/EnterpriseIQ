"""Chunk bounds, overlap, and page numbers. The embedding model is not loaded."""

from app.ingestion import PageText, chunk_pages


class _WordTokens:
    def token_count(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def token_ids(self, text: str) -> list[str]:
        return text.split()

    def decode_tokens(self, token_ids: list[str]) -> str:
        return " ".join(token_ids)


def test_chunks_respect_max_overlap_and_pages(monkeypatch) -> None:
    monkeypatch.setattr("app.ingestion.get_embeddings", lambda: _WordTokens())
    long_page = "w0 w1 w2 w3. w4 w5 w6 w7. w8 w9 w10 w11."
    chunks = chunk_pages(
        [
            PageText(page_number=1, text=long_page),
            PageText(page_number=2, text="only on page two"),
        ],
        file_type="txt",
        chunk_max_tokens=6,
        chunk_min_tokens=1,
        chunk_overlap_tokens=2,
    )
    counter = _WordTokens()
    assert chunks
    assert all(counter.token_count(chunk.text) <= 6 for chunk in chunks)
    assert chunks[0].page_number == 1
    assert any(chunk.page_number == 2 and "page two" in chunk.text for chunk in chunks)
    first_words = chunks[0].text.split()
    assert chunks[1].text.split()[:2] == first_words[-2:]
