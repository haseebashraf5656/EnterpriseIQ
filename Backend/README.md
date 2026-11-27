# EnterpriseIQ backend

EnterpriseIQ answers questions from an uploaded document collection. Retrieval is measured separately from the language model. Answers cite only stored chunk metadata.

## Architecture

- FastAPI serves upload, search, chat, and evaluation.
- PostgreSQL with pgvector stores documents and chunk embeddings.
- Ingestion extracts PDF, TXT, and Markdown, normalizes whitespace, and chunks to about 200 tokens with the MiniLM tokenizer. Page numbers and headings are kept.
- Retrieval modes: dense (pgvector), BM25 (in memory), hybrid (reciprocal rank fusion), and hybrid plus a cross-encoder rerank.
- Chat calls Gemini or Ollama only after retrieval. Low rerank scores and `sufficient: false` return an insufficient-evidence answer and no citations.
- Evaluation scores retrieval only. It does not call the language model.

## Setup

From the repository root:

```bash
cp Backend/.env.example Backend/.env
docker compose up --build
```

The API listens on port 8000. Put a Gemini key in `Backend/.env` before using `LLM_PROVIDER=gemini`.

## Environment variables

Values below match `Backend/.env.example`. `app/config.py` supplies the same defaults when a variable is unset, except `GEMINI_MODEL` (code default `gemini-2.0-flash`).

| Variable | Example | Role |
|---|---|---|
| `DATABASE_URL` | `postgresql://eiq:eiq@db:5432/enterpriseiq` | Postgres connection |
| `UPLOAD_DIR` | `uploads` | Stored upload directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Chunk and query embeddings |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `EMBEDDING_DIM` | `384` | Vector size |
| `CHUNK_MAX_TOKENS` | `200` | Chunk cap |
| `CHUNK_MIN_TOKENS` | `120` | Merge tail shorter than this when it still fits |
| `CHUNK_OVERLAP_TOKENS` | `40` | Token overlap between chunks |
| `TOP_K` | `5` | Passages returned |
| `DENSE_CANDIDATES` | `20` | Dense candidates before fusion |
| `BM25_CANDIDATES` | `20` | BM25 candidates before fusion |
| `FUSED_CANDIDATES` | `25` | Fusion list size |
| `RRF_K` | `60` | Reciprocal rank constant |
| `RERANK_MIN_SCORE` | `-3.0` | Refuse when the best rerank score is below this |
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `GEMINI_API_KEY` | | Gemini key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model |
| `LLM_TIMEOUT_SECONDS` | `30` | Provider timeout |
| `LLM_RETRIES` | `2` | Retries on timeout, 429, and 5xx |
| `MAX_UPLOAD_BYTES` | `10485760` | Per-file upload cap |
| `MAX_QUESTION_CHARS` | `2000` | Chat question cap |
| `CORS_ORIGINS` | `http://localhost:8501` | Browser origins (Streamlit) |

## Endpoints

| Method | Path | Body |
|---|---|---|
| `GET` | `/health` | Database reachability |
| `POST` | `/documents/upload` | Multipart `files`. One result per file: `id`, `filename`, and `status` or `error` |
| `GET` | `/documents` | Document list |
| `DELETE` | `/documents/{document_id}` | Delete one document |
| `POST` | `/debug/search` | `{query, mode, top_k}` |
| `POST` | `/chat` | `{question, mode}`. `mode` defaults to `hybrid_reranked` |
| `POST` | `/evaluate` | `{configuration}`: one of `dense`, `bm25`, `hybrid`, `hybrid_reranked`, or `all` |

Modes for search and chat: `dense`, `bm25`, `hybrid`, `hybrid_reranked`.

Errors are JSON `{"detail": ...}`. A provider failure on `/chat` is HTTP 502.

## Tests

```bash
docker compose exec backend pytest
```

Tests mock the embedder and do not call the language model, the database, or a model download.

## Evaluation

Edit `evaluation/questions.json` (about 40 questions). Fields: `question`, `type` (`direct`, `paraphrase`, `exact_term`, `unanswerable`), `expected_filename`, `expected_pages`, `answerable`.

Ingest the real documents first. Then:

```bash
docker compose exec backend python evaluation/run_eval.py
```

The command runs all four modes, prints a markdown table, writes `evaluation/results.md`, and inserts each mode into `evaluation_runs`. `POST /evaluate` does the same for one mode or `all`. Recall@5 and MRR skip `answerable: false`. Refusal accuracy uses the rerank threshold on `hybrid_reranked` (other modes refuse only when retrieval returns nothing).

## Known limitations

- No OCR. Scanned PDFs fail ingestion when no text is extracted.
- One shared collection. There are no users or per-tenant indexes.
- Chat is single-turn. There is no conversation memory.
- MiniLM's limit is 256 tokens, so chunks are capped at about 200 tokens.

## Results

Fill this table from a real `evaluation/run_eval.py` run. Leave it blank until then.

| configuration | Recall@5 | MRR | avg latency ms | refusal accuracy |
|---|---:|---:|---:|---:|
| dense | | | | |
| bm25 | | | | |
| hybrid | | | | |
| hybrid_reranked | | | | |
