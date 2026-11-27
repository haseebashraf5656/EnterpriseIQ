"""Retrieval evaluation. Does not call the language model."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.db import insert_evaluation_run
from app.generation import should_refuse
from app.retrieval import MODES, retrieve

logger = logging.getLogger(__name__)

TOP_K = 5
_REQUIRED = ("question", "type", "expected_filename", "expected_pages", "answerable")


def default_questions_path() -> Path:
    return Path(__file__).resolve().parents[1] / "evaluation" / "questions.json"


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the question set. Unanswerable items are kept for refusal scoring."""
    source = path or default_questions_path()
    if not source.is_file():
        raise FileNotFoundError(f"Questions file not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Questions file must be a JSON list")
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be an object")
        missing = [key for key in _REQUIRED if key not in item]
        if missing:
            raise ValueError(f"Question {index} missing {', '.join(missing)}")
        if not isinstance(item["answerable"], bool):
            raise ValueError(f"Question {index} answerable must be a boolean")
        if not isinstance(item["expected_pages"], list):
            raise ValueError(f"Question {index} expected_pages must be a list")
        questions.append(item)
    return questions


def run_configurations(
    configuration: str,
    questions: list[dict[str, Any]] | None = None,
    *,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Run one mode, or all four. Each mode is stored as its own evaluation run."""
    name = configuration.strip().lower()
    if name == "all":
        modes = MODES
    elif name in MODES:
        modes = (name,)
    else:
        raise ValueError(f"Unknown configuration '{configuration}'")
    loaded = load_questions() if questions is None else questions
    results: list[dict[str, Any]] = []
    for mode in modes:
        metrics = evaluate_mode(loaded, mode)
        failures = metrics.pop("failures")
        rerank_scores = metrics.pop("rerank_scores")
        if persist:
            insert_evaluation_run(mode, metrics)
            logger.info("evaluation saved configuration=%s questions=%s", mode, len(loaded))
        results.append(
            {
                "configuration": mode,
                "metrics": metrics,
                "failures": failures,
                "rerank_scores": rerank_scores,
            }
        )
    return results


def evaluate_mode(questions: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Score retrieval for one mode. Recall and MRR skip unanswerable questions."""
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    end_to_end: list[float] = []
    stage_sums: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    correct_refusals = 0
    wrong_refusals = 0
    answerable = 0
    unanswerable = 0
    failures: list[dict[str, Any]] = []
    rerank_scores: list[dict[str, Any]] = []

    for question in questions:
        started = time.perf_counter()
        found = retrieve(question["question"], mode, top_k=TOP_K)
        end_to_end.append((time.perf_counter() - started) * 1000)
        chunks = found["chunks"]
        for stage, elapsed in found["latency_ms"].items():
            stage_sums[stage] = stage_sums.get(stage, 0.0) + float(elapsed)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        refused = should_refuse(chunks, mode)
        top = chunks[:TOP_K]
        rank = _first_rank(top, question) if question["answerable"] else None
        top_score = _top_rerank_score(top) if mode == "hybrid_reranked" else None
        if mode == "hybrid_reranked":
            rerank_scores.append(
                {
                    "question": question["question"],
                    "type": question["type"],
                    "answerable": question["answerable"],
                    "top_rerank_score": top_score,
                }
            )
        if question["answerable"]:
            answerable += 1
            if refused:
                wrong_refusals += 1
            recalls.append(_recall(top, question))
            reciprocal_ranks.append(_reciprocal_rank(top, question))
        else:
            unanswerable += 1
            if refused:
                correct_refusals += 1
        wrongly_refused = bool(question["answerable"] and refused)
        wrongly_answered = bool(not question["answerable"] and not refused)
        missed = bool(question["answerable"] and rank is None)
        if wrongly_refused or wrongly_answered or missed:
            failures.append(
                {
                    "question": question["question"],
                    "type": question["type"],
                    "expected": _expected_label(question),
                    "rank": rank,
                    "top_rerank_score": top_score,
                    "wrongly_refused": wrongly_refused,
                    "wrongly_answered": wrongly_answered,
                }
            )

    total = len(questions)
    correct_accepts = answerable - wrong_refusals
    accuracy = (correct_refusals + correct_accepts) / total if total else None
    latency: dict[str, float | None] = {"end_to_end": _mean(end_to_end)}
    for stage, total_ms in stage_sums.items():
        count = stage_counts[stage]
        latency[stage] = round(total_ms / count, 3) if count else None
    return {
        "recall_at_5": _mean(recalls),
        "mrr": _mean(reciprocal_ranks),
        "avg_latency_ms": latency,
        "refusal_accuracy": None if accuracy is None else round(accuracy, 3),
        "correct_refusals": correct_refusals,
        "wrong_refusals": wrong_refusals,
        "answerable_questions": answerable,
        "unanswerable_questions": unanswerable,
        "questions": total,
        "failures": failures,
        "rerank_scores": rerank_scores,
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Markdown tables for Recall@5, MRR, latency, and refusal accuracy."""
    lines = [
        "# Evaluation results",
        "",
        "Retrieval only. The language model is not called.",
        "",
        "| configuration | Recall@5 | MRR | avg latency ms | refusal accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        metrics = row["metrics"]
        lines.append(
            "| {configuration} | {recall} | {mrr} | {latency} | {refusal} |".format(
                configuration=row["configuration"],
                recall=_cell(metrics["recall_at_5"]),
                mrr=_cell(metrics["mrr"]),
                latency=_cell(metrics["avg_latency_ms"]["end_to_end"]),
                refusal=_cell(metrics["refusal_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "### Average latency by stage (ms)",
            "",
            "| configuration | end_to_end | dense | bm25 | rrf | rerank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        stages = row["metrics"]["avg_latency_ms"]
        lines.append(
            "| {configuration} | {end} | {dense} | {bm25} | {rrf} | {rerank} |".format(
                configuration=row["configuration"],
                end=_cell(stages.get("end_to_end")),
                dense=_cell(stages.get("dense")),
                bm25=_cell(stages.get("bm25")),
                rrf=_cell(stages.get("rrf")),
                rerank=_cell(stages.get("rerank")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_failures(results: list[dict[str, Any]]) -> str:
    """Per-mode questions that missed, were wrongly refused, or were wrongly answered."""
    lines = ["# Evaluation failures", ""]
    for row in results:
        lines.append(f"## {row['configuration']}")
        lines.append("")
        failures = row.get("failures") or []
        if not failures:
            lines.append("No failures.")
            lines.append("")
            continue
        for item in failures:
            lines.append(f"- question: {item['question']}")
            lines.append(f"  - type: {item['type']}")
            lines.append(f"  - expected: {item['expected']}")
            lines.append(f"  - first correct rank: {_rank_label(item['rank'])}")
            if row["configuration"] == "hybrid_reranked":
                lines.append(f"  - top rerank score: {_score_label(item['top_rerank_score'])}")
            lines.append(f"  - outcome: {_outcome(item)}")
        lines.append("")
    scored = _hybrid_scores(results)
    if scored:
        lines.append("## hybrid_reranked rerank scores")
        lines.append("")
        lines.append("| answerable | type | top rerank score | question |")
        lines.append("|---|---|---:|---|")
        for item in scored:
            lines.append(
                "| {flag} | {kind} | {score} | {question} |".format(
                    flag="yes" if item["answerable"] else "no",
                    kind=item["type"],
                    score=_score_label(item["top_rerank_score"]),
                    question=str(item["question"]).replace("|", "\\|"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def failure_summary(results: list[dict[str, Any]]) -> str:
    """One line of failure counts per mode."""
    parts = [f"{row['configuration']} {len(row.get('failures') or [])}" for row in results]
    return "Failures: " + ", ".join(parts)


def rerank_score_report(results: list[dict[str, Any]]) -> str:
    """Top rerank score for every hybrid_reranked question, split by answerable."""
    scored = _hybrid_scores(results)
    if not scored:
        return ""
    lines = ["hybrid_reranked top rerank scores"]
    for flag, title in ((True, "answerable"), (False, "unanswerable")):
        lines.append(title)
        for item in scored:
            if item["answerable"] is not flag:
                continue
            lines.append(f"  {_score_label(item['top_rerank_score'])}  {item['question']}")
    return "\n".join(lines)


def _hybrid_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in results:
        if row["configuration"] == "hybrid_reranked":
            return list(row.get("rerank_scores") or [])
    return []


def _first_rank(chunks: list[dict[str, Any]], question: dict[str, Any]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant(chunk, question):
            return rank
    return None


def _top_rerank_score(chunks: list[dict[str, Any]]) -> float | None:
    if not chunks:
        return None
    return round(max(float(chunk["score"]) for chunk in chunks), 3)


def _expected_label(question: dict[str, Any]) -> str:
    name = question.get("expected_filename") or "(none)"
    pages = question.get("expected_pages") or []
    if not pages:
        return f"{name} pages none"
    return f"{name} pages {', '.join(str(page) for page in pages)}"


def _rank_label(rank: int | None) -> str:
    if rank is None:
        return "none"
    return str(rank)


def _score_label(score: float | None) -> str:
    if score is None:
        return "none"
    return f"{score:.3f}"


def _outcome(item: dict[str, Any]) -> str:
    labels: list[str] = []
    if item["wrongly_refused"]:
        labels.append("wrongly refused")
    if item["wrongly_answered"]:
        labels.append("wrongly answered")
    if item["rank"] is None and not item["wrongly_answered"]:
        labels.append("no correct hit")
    return "; ".join(labels)


def _recall(chunks: list[dict[str, Any]], question: dict[str, Any]) -> float:
    expected = _expected_pages(question)
    if not expected:
        return 0.0
    found = {int(chunk["page_number"]) for chunk in chunks if _is_relevant(chunk, question)}
    return len(found & expected) / len(expected)


def _reciprocal_rank(chunks: list[dict[str, Any]], question: dict[str, Any]) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant(chunk, question):
            return 1.0 / rank
    return 0.0


def _is_relevant(chunk: dict[str, Any], question: dict[str, Any]) -> bool:
    filename = question.get("expected_filename") or ""
    if not filename or chunk.get("filename") != filename:
        return False
    try:
        page = int(chunk["page_number"])
    except (KeyError, TypeError, ValueError):
        return False
    return page in _expected_pages(question)


def _expected_pages(question: dict[str, Any]) -> set[int]:
    pages: set[int] = set()
    for page in question["expected_pages"]:
        try:
            pages.add(int(page))
        except (TypeError, ValueError):
            continue
    return pages


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _cell(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
