# Evaluation results

Retrieval only. The language model is not called.

| configuration | Recall@5 | MRR | avg latency ms | refusal accuracy |
|---|---:|---:|---:|---:|
| dense | 1.000 | 0.935 | 36.161 | 0.852 |
| bm25 | 1.000 | 0.891 | 0.173 | 0.852 |
| hybrid | 1.000 | 0.957 | 19.769 | 0.852 |
| hybrid_reranked | 1.000 | 0.978 | 788.233 | 0.963 |

### Average latency by stage (ms)

| configuration | end_to_end | dense | bm25 | rrf | rerank |
|---|---:|---:|---:|---:|---:|
| dense | 36.161 | 36.148 | n/a | n/a | n/a |
| bm25 | 0.173 | n/a | 0.167 | n/a | n/a |
| hybrid | 19.769 | 19.440 | 0.229 | 0.077 | n/a |
| hybrid_reranked | 788.233 | 23.207 | 0.250 | 0.058 | 764.661 |
