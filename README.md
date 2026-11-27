# EnterpriseIQ

EnterpriseIQ is a retrieval-augmented (RAG) document question-answering service. You upload documents, it extracts and cleans the text, splits it into chunks, embeds the chunks, and stores them in PostgreSQL with pgvector. When you ask a question, it retrieves candidate passages with dense vector search and BM25 keyword search, fuses the two rankings, reranks the best candidates with a cross-encoder, and asks an LLM to answer using only those passages. Answers include page citations taken from stored chunk metadata, and questions with weak evidence are refused instead of answered. A small evaluation harness measures retrieval quality without calling the LLM.

## Features

- Hybrid retrieval: dense (pgvector) + BM25 keyword search, combined with reciprocal rank fusion (RRF).
- Cross-encoder reranking of fused candidates.
- Grounded answers: the LLM may only use retrieved passages.
- Page citations built from chunk metadata (filename, page, chunk id).
- Refusal on weak evidence using a reranker score threshold.
- Upload, list, delete documents; PDF, TXT, and Markdown support.
- Retrieval-only evaluation harness with Recall@5, MRR, latency, and refusal accuracy.
- Streamlit web UI and a documented FastAPI HTTP API.

## Architecture

Upload flow:

```
Upload -> Extract (PDF/TXT/MD) -> Clean -> Chunk (~200 tokens) -> Embed (MiniLM)
       -> pgvector (dense index) + BM25 (in-memory keyword index)
```

Question flow:

```
Question -> Dense search (pgvector)  ┐
         -> BM25 search             ┴-> RRF fusion -> Cross-Encoder rerank
         -> LLM (Gemini/Ollama) -> Answer + Citations  (refuse if weak evidence)
```

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Reranker | Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Keyword search | `rank_bm25` |
| LLM | Google Gemini API (via `httpx`) |
| UI | Streamlit |
| Orchestration | Docker Compose |

## Quick start

Prerequisites:

- Docker Desktop
- A Gemini API key from Google AI Studio

Steps:

```bash
# 1. Create your local env file and add your key
cp Backend/.env.example Backend/.env
# edit Backend/.env and set GEMINI_API_KEY=...

# 2. Build and start everything
docker compose up --build
```

Notes:

- The first build is slow (large Python/ML dependencies download once).
- Backend startup takes a few minutes while the embedding and reranker models load.
- Open the UI at http://localhost:8501 and the API docs at http://localhost:8000/docs.
- Warning: never run `docker compose down -v` unless you want to delete the database (the `-v` flag removes the data volume).

## Environment variables

All values live in `Backend/.env` (git-ignored). Names match `Backend/.env.example`.

| Variable | Default / example | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://eiq:eiq@db:5432/enterpriseiq` | Postgres connection string |
| `UPLOAD_DIR` | `uploads` | Upload storage directory |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `EMBEDDING_DIM` | `384` | Vector width (matches MiniLM) |
| `CHUNK_MAX_TOKENS` | `200` | Max tokens per chunk |
| `CHUNK_MIN_TOKENS` | `120` | Short-tail merge threshold |
| `CHUNK_OVERLAP_TOKENS` | `40` | Token overlap between chunks |
| `TOP_K` | `5` | Passages returned |
| `DENSE_CANDIDATES` | `20` | Dense candidates before fusion |
| `BM25_CANDIDATES` | `20` | BM25 candidates before fusion |
| `FUSED_CANDIDATES` | `25` | Fused candidates before rerank |
| `RRF_K` | `60` | Reciprocal rank fusion constant |
| `RERANK_MIN_SCORE` | `-2.6` | Refusal threshold on reranker score |
| `LLM_PROVIDER` | `gemini` | `gemini` or `ollama` |
| `GEMINI_API_KEY` | `your_key_here` | Gemini credential (secret) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL (when `LLM_PROVIDER=ollama`) |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `LLM_TIMEOUT_SECONDS` | `30` | LLM request timeout |
| `LLM_RETRIES` | `2` | LLM retry count |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap (10 MB) |
| `MAX_QUESTION_CHARS` | `2000` | Max question length |
| `CORS_ORIGINS` | `http://localhost:8501` | Allowed browser origins |

The frontend reads one variable: `BACKEND_URL` (see `Frontend/.env.example`; Compose sets it automatically).

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check (status, database) |
| `POST` | `/documents/upload` | Upload one or more files (multipart `files`) |
| `GET` | `/documents` | List documents |
| `DELETE` | `/documents/{document_id}` | Delete a document and its chunks |
| `POST` | `/debug/search` | Retrieval-only search (`query`, `mode`, `top_k`) |
| `POST` | `/chat` | Ask a question (`question`, `mode`, default `hybrid_reranked`) |
| `POST` | `/evaluate` | Run retrieval evaluation for a mode or `all` |

Retrieval modes: `dense`, `bm25`, `hybrid`, `hybrid_reranked`.

## Using the UI

1. Open http://localhost:8501.
2. Upload documents. Uploaded files are validated and stored under random names.
3. Ask a question in the chat box.
4. Read the answer and the cited sources (filename + page).
5. If the evidence is too weak, the app returns a refusal message ("The documents do not contain enough information to answer this question.") and shows no citations.

## Run tests and evaluation

```bash
# Unit tests
docker compose exec backend pytest -q

# Retrieval evaluation (does not call the LLM)
docker compose exec backend python evaluation/run_eval.py
```

## Results

Retrieval-only evaluation on 27 questions over 2 synthetic documents (Acme_HR_Policy.pdf, NimbusDB_Admin_Manual.pdf, 6 pages each). Measured on a CPU-only laptop. The refusal threshold (-2.6) was tuned on this same set, so refusal accuracy is optimistic. Small test set: results show trends, not guarantees.

| Mode | Recall@5 | MRR | Avg latency (ms) |
|---|---:|---:|---:|
| dense | 1.000 | 0.935 | 36 |
| bm25 | 1.000 | 0.891 | 0.3 |
| hybrid | 1.000 | 0.957 | 32 |
| hybrid_reranked | 1.000 | 0.978 | 788 |

Refusal accuracy (hybrid_reranked only; the other modes cannot refuse): 0.963 (26 of 27).

The reranker stage takes about 765 ms of the 788 ms.

## Failure analysis

- "When do I need a doctor's note for being ill?" was wrongly refused: the correct page ranked first, but the reranker score was -7.1 because the document says "medical certificate" (vocabulary mismatch).
- "What is the price of a NimbusDB enterprise license?" scored -3.9, close to the threshold: on-topic but not covered by the documents.
- Dense, BM25, and hybrid have no refusal mechanism, so refusal accuracy applies only to `hybrid_reranked`.

## Known limitations

- Text-based PDF/TXT/MD only; no OCR for image-only PDFs.
- One shared collection with no user accounts or per-user isolation.
- Single-turn Q&A with no conversation memory.
- MiniLM reads 256 tokens, so chunks are about 200 tokens.
- Startup takes minutes on CPU while models load.
- Small evaluation set.

## Project structure

```
EnterpriseIQ/
├── docker-compose.yml
├── README.md
├── .gitignore
├── Backend/
│   ├── Dockerfile
│   ├── .env.example
│   ├── requirements.txt
│   ├── requirements-heavy.txt
│   ├── pytest.ini
│   ├── app/
│   │   ├── main.py          # FastAPI app and routes
│   │   ├── config.py        # settings from environment
│   │   ├── db.py            # Postgres / pgvector access
│   │   ├── ingestion.py     # extract, clean, chunk
│   │   ├── embeddings.py    # MiniLM embeddings
│   │   ├── retrieval.py     # dense, BM25, RRF
│   │   ├── rerank.py        # cross-encoder
│   │   ├── generation.py    # Gemini/Ollama + refusal
│   │   ├── schemas.py       # request/response models
│   │   └── evaluation.py    # retrieval metrics
│   ├── evaluation/
│   │   ├── questions.json
│   │   ├── run_eval.py
│   │   ├── results.md
│   │   └── failures.md
│   ├── sample_docs/         # synthetic test documents
│   ├── uploads/             # runtime uploads (git-ignored)
│   └── tests/
└── Frontend/
    ├── Dockerfile
    ├── .env.example
    ├── requirements.txt
    └── streamlit_app.py
```

## Security notes

- Secrets live only in `Backend/.env`, which is git-ignored. No real key is committed.
- Uploads are validated (type and size) and stored under random file names.
- Database access uses parameterized SQL.

## Roadmap

- OCR for scanned PDFs.
- DOCX support.
- Conversation memory for multi-turn Q&A.
- Per-user access control.
- Automated faithfulness scoring for answers.
