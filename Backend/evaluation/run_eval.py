"""Run all four retrieval configurations and write evaluation/results.md."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import close_pool, init_pool, init_schema
from app.embeddings import get_embeddings
from app.evaluation import failure_summary, render_failures, render_markdown, rerank_score_report, run_configurations
from app.rerank import get_reranker
from app.retrieval import rebuild_bm25


def main() -> None:
    init_pool()
    try:
        init_schema()
        get_embeddings()
        get_reranker()
        rebuild_bm25()
        results = run_configurations("all")
    finally:
        close_pool()
    text = render_markdown(results)
    print(text)
    destination = Path(__file__).resolve().parent / "results.md"
    destination.write_text(text, encoding="utf-8")
    failures = render_failures(results)
    (Path(__file__).resolve().parent / "failures.md").write_text(failures, encoding="utf-8")
    print(failure_summary(results))
    scores = rerank_score_report(results)
    if scores:
        print(scores)


if __name__ == "__main__":
    main()
