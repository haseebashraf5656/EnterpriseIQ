"""Reciprocal rank fusion on a fixed two-list example."""

import pytest

from app.retrieval import rrf_fuse


def test_rrf_rank_starts_at_one() -> None:
    first = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    second = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    fused = rrf_fuse([first, second], k=60, limit=10)
    assert [hit["chunk_id"] for hit in fused] == ["b", "a", "c"]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1]["score"] == pytest.approx(1 / 61)
    assert fused[2]["score"] == pytest.approx(1 / 62)
