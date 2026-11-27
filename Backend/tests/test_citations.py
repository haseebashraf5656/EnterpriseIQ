"""Citations are copied from stored chunk metadata. Unknown ids are dropped."""

from uuid import uuid4

from app.generation import build_citations


def test_citations_use_metadata_and_ignore_unknown_ids() -> None:
    chunk_id = uuid4()
    invented = uuid4()
    passages = [
        {
            "chunk_id": chunk_id,
            "filename": "policy.pdf",
            "page_number": 4,
            "text": "The limit is ten days.",
        }
    ]
    citations = build_citations(
        passages,
        [str(chunk_id), str(invented), "page-99", {"page": 1}],
    )
    assert citations == [
        {"filename": "policy.pdf", "page": 4, "chunk_id": chunk_id}
    ]
