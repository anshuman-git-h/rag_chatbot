# RAG System Report

## Overview
A Retrieval-Augmented Generation (RAG) pipeline built on LangChain, FAISS, and Groq LLMs. It ingests PDFs and text files, chunks and embeds them, retrieves the top-k relevant passages, and generates grounded answers. Evaluation uses RAGAS with LLM-as-judge (Groq) and a sentence-transformer embedding judge.

## Pipeline
1. **Ingestion** (`src/data_loader.py`) — loads PDF, TXT, CSV, Excel, Word, and JSON files from `data/`.
2. **Chunking** — two strategies implemented (see below).
3. **Embedding** — `sentence-transformers/all-MiniLM-L6-v2`.
4. **Indexing** — FAISS (`IndexFlatL2`) with metadata pickled alongside (`src/vectorstore.py`).
5. **Retrieval** — top-k = 3 passages.
6. **Generation** — Llama 3.1 8B Instant via Groq (`src/search.py`).

## Baseline vs. Improved
| Component | Baseline | Improved |
|---|---|---|
| Chunking | Recursive character splitter (1000/200) | Semantic, sentence-boundary splitting (cosine threshold 0.6) |
| Query strategy | Raw user query | HyDE — LLM drafts a hypothetical answer, embedded for retrieval |
| Vector store | `faiss_store_recursive` | `faiss_store` |

## Improvements Made
- **Semantic chunking** (`src/embedding.py`) — splits only at sentence boundaries and groups sentences whose cosine similarity stays above a threshold, producing more coherent, topically-consistent chunks than fixed-size recursive splitting.
- **HyDE** (`src/search.py:generate_hyde_document`) — expands the query into a synthetic document before embedding, improving retrieval of context that is phrased differently than the question.
- **Dual vector stores** — both indexes built side-by-side (`build_vectorstores.py`) so baseline and improved runs can be compared on identical data.
- **Robust evaluation harness** (`evaluate.py` / `evaluate1.py`) — RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall), Groq rate-limit backoff parsing, retry handling for truncated outputs, and disk caching so interrupted runs don't re-spend tokens.

## RAGAS Results (top-k = 3)
| Metric | Baseline | Improved | Change |
|---|---|---|---|
| Faithfulness | 0.28 | 0.48 | +0.20 |
| Answer Relevancy | 0.95 | 0.96 | +0.02 |
| Context Precision | 0.50 | 0.56 | +0.06 |
| Context Recall | 0.68 | 0.79 | +0.11 |

## Takeaways
- Semantic chunking + HyDE consistently improves retrieval quality and answer faithfulness over the recursive + plain-query baseline across all four RAGAS metrics.
- Improved context recall indicates that semantic chunks better capture the passages a grounded answer needs.
- Remaining bottleneck is retrieval (Context Precision), suggesting future work on reranking, larger top-k, or a better embedding model.
