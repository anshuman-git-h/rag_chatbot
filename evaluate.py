# evaluate.py
import os
import re
import json
import asyncio
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


def build_metrics():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    async_openai_client = AsyncOpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    cache = DiskCacheBackend(cache_dir=".ragas_cache")

    evaluator_llm = llm_factory(
        "qwen/qwen3.6-27b",
        client=async_openai_client,
        max_tokens=3000,
        reasoning_effort = "none",
        cache=cache,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": ContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }


def parse_wait_seconds(msg: str) -> float:
    """Parses Groq's 'try again in 1m16.3776s' / 'try again in 11.18s'
    format correctly, including the minutes part the old regex dropped."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if not match:
        return 15.0
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 1


async def call_with_backoff(coro_fn, *args, max_retries=6, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "Request too large" in msg or "reduce your message size" in msg:
                raise RuntimeError(
                    "Request exceeds this model's per-request TPM ceiling — "
                    "reduce max_tokens or truncate context, not a transient rate limit."
                ) from e
            if "rate_limit_exceeded" in msg or "429" in msg:
                wait = parse_wait_seconds(msg)
                print(f"  [rate limited] sleeping {wait:.1f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(wait)
                continue
            if "json_validate_failed" in msg or "max completion tokens" in msg:
                print(f"  [truncated output] retrying (attempt {attempt}/{max_retries})")
                await asyncio.sleep(2)
                continue
            raise
    raise RuntimeError("Max retries exceeded")

MAX_CONTENT_CHARS = 3000

async def score_row(metrics, question, response, contexts, reference):
    if isinstance(contexts, str):
        contexts = [contexts]
    contexts = [c[:MAX_CONTENT_CHARS] for c in contexts]

    # Sequential on purpose: the account's rate budget can't absorb
    # concurrent calls, so this trades speed for not crashing.
    faith = await call_with_backoff(
        metrics["faithfulness"].ascore,
        user_input=question, response=response, retrieved_contexts=contexts,
    )
    ans_rel = await call_with_backoff(
        metrics["answer_relevancy"].ascore,
        user_input=question, response=response,
    )
    ctx_prec = await call_with_backoff(
        metrics["context_precision"].ascore,
        user_input=question, retrieved_contexts=contexts, reference=reference,
    )
    ctx_rec = await call_with_backoff(
        metrics["context_recall"].ascore,
        user_input=question, retrieved_contexts=contexts, reference=reference,
    )

    return {
        "faithfulness": faith.value,
        "answer_relevancy": ans_rel.value,
        "context_precision": ctx_prec.value,
        "context_recall": ctx_rec.value,
    }


async def evaluate_pipeline(rag_search, eval_data, metrics):
    per_row_scores = []
    for item in eval_data:
        result = await asyncio.to_thread(
            rag_search.search_and_summarize, item["question"], top_k=3
        )
        scores = await score_row(
            metrics,
            question=item["question"],
            response=result["answer"],
            contexts=result["context"],
            reference=item["ground_truth_answer"],
        )
        print(f"[scored] {item['question'][:60]!r} -> {scores}")
        per_row_scores.append(scores)

    aggregated = {
        metric_name: mean(row[metric_name] for row in per_row_scores)
        for metric_name in per_row_scores[0]
    }
    return aggregated, per_row_scores


async def main():
    with open("evaluations.json", "r") as f:  # confirm this matches your actual filename
        eval_data = json.load(f)

    metrics = build_metrics()

    baseline = RAGSearch(persist_dir="faiss_store_recursive", use_hyde=False)
    baseline_agg, baseline_rows = await evaluate_pipeline(baseline, eval_data, metrics)

    print("\n========== BASELINE ==========")
    print(baseline_agg)

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