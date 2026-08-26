# evaluate.py
import os
import re
import json
import asyncio
import threading
from statistics import mean

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.cache import DiskCacheBackend
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)
from openai import AsyncOpenAI
from src.search import RAGSearch

# 30K TPM on llama-4-scout gives real headroom -> safe to run a few rows concurrently
CONCURRENCY = 1

# SentenceTransformer's underlying Rust tokenizer isn't thread-safe.
# Two separate model instances (rag_search's retriever model vs ragas'
# evaluator embeddings) don't conflict with each other, but each one
# needs its own calls serialized against itself.
_retrieval_encode_lock = threading.Lock()   # guards rag_search's SentenceTransformer (sync/thread calls)
_eval_encode_lock = asyncio.Lock()          # guards AnswerRelevancy's embeddings (async calls)


def build_metrics():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    async_openai_client = AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    cache = DiskCacheBackend(cache_dir=".ragas_cache")  # reruns after a crash won't re-spend tokens

    evaluator_llm = llm_factory(
        "openai/gpt-oss-20b",  
        client=async_openai_client,
        max_tokens=4096,  # Faithfulness needs headroom for the verdicts JSON
        cache=cache,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }


async def call_with_backoff(coro_fn, *args, max_retries=6, **kwargs):
    """Run an ascore() call, sleeping and retrying on 429s using Groq's
    suggested wait time instead of a fixed guess."""
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "429" in msg:
                match = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(match.group(1)) + 1 if match else 15.0
                print(f"  [rate limited] sleeping {wait:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded due to rate limiting")


async def score_row(metrics, question, response, contexts, reference):
    if isinstance(contexts, str):
        contexts = [contexts]

    async def scored_answer_relevancy():
        # Serialize access to the shared SentenceTransformer embeddings object
        async with _eval_encode_lock:
            return await call_with_backoff(
                metrics["answer_relevancy"].ascore,
                user_input=question, response=response,
            )

    faith, ans_rel, ctx_prec, ctx_rec = await asyncio.gather(
        call_with_backoff(
            metrics["faithfulness"].ascore,
            user_input=question, response=response, retrieved_contexts=contexts,
        ),
        scored_answer_relevancy(),
        call_with_backoff(
            metrics["context_precision"].ascore,
            user_input=question, retrieved_contexts=contexts, reference=reference,
        ),
        call_with_backoff(
            metrics["context_recall"].ascore,
            user_input=question, retrieved_contexts=contexts, reference=reference,
        ),
    )

    return {
        "faithfulness": faith.value,
        "answer_relevancy": ans_rel.value,
        "context_precision": ctx_prec.value,
        "context_recall": ctx_rec.value,
    }


async def process_item(sem, rag_search, metrics, item):
    async with sem:
        def search_locked():
            # Serialize access to rag_search's own SentenceTransformer instance
            with _retrieval_encode_lock:
                return rag_search.search_and_summarize(item["question"], top_k=3)

        result = await asyncio.to_thread(search_locked)
        scores = await score_row(
            metrics,
            question=item["question"],
            response=result["answer"],
            contexts=result["context"],
            reference=item["ground_truth_answer"],
        )
        print(f"[scored] {item['question'][:60]!r} -> {scores}")
        return scores


async def evaluate_pipeline(rag_search, eval_data, metrics):
    sem = asyncio.Semaphore(CONCURRENCY)
    per_row_scores = await asyncio.gather(
        *(process_item(sem, rag_search, metrics, item) for item in eval_data)
    )

    aggregated = {
        metric_name: mean(row[metric_name] for row in per_row_scores)
        for metric_name in per_row_scores[0]
    }
    return aggregated, per_row_scores


async def main():
    with open("evaluation.json", "r") as f:
        eval_data = json.load(f)

    metrics = build_metrics()

    # -----------------------------
    # Baseline: Recursive + Normal Query
    # -----------------------------
    baseline = RAGSearch(persist_dir="faiss_store_recursive", use_hyde=False)
    baseline_agg, baseline_rows = await evaluate_pipeline(baseline, eval_data, metrics)

    print("\n========== BASELINE ==========")
    print(baseline_agg)

    # -----------------------------
    # Improved: Semantic + HyDE
    # -----------------------------
    semantic_hyde = RAGSearch(persist_dir="faiss_store", use_hyde=True)
    semantic_agg, semantic_rows = await evaluate_pipeline(semantic_hyde, eval_data, metrics)

    print("\n====== SEMANTIC + HyDE ======")
    print(semantic_agg)

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/results.txt", "w") as f:
        f.write("BASELINE (Recursive + Normal Query)\n")
        f.write(json.dumps(baseline_agg, indent=2))
        f.write("\n\n")
        f.write("SEMANTIC + HyDE\n")
        f.write(json.dumps(semantic_agg, indent=2))
        f.write("\n\nPer-row scores (baseline):\n")
        f.write(json.dumps(baseline_rows, indent=2))
        f.write("\n\nPer-row scores (semantic+hyde):\n")
        f.write(json.dumps(semantic_rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())