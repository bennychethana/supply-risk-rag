# evals/evaluate.py

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import time
from anthropic import Anthropic
from src.step3_query import (
    naive_retrieve,
    hybrid_retrieve,
    match_canonical,
    generate_summary,
)
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()


# ── LLM-as-judge scoring ──────────────────────────────────────
# We use Claude to judge Claude's outputs.
# This is the same pattern RAGAS uses internally —
# an LLM evaluating another LLM's responses.
# Each metric is one focused prompt asking for a 0-1 score.

def score_faithfulness(answer: str, chunks: list[str]) -> float:
    """
    Are all claims in the answer supported by the retrieved chunks?
    High score = no hallucination.
    Low score = answer contains information not in the chunks.
    """
    context = "\n\n".join(chunks)
    prompt = f"""You are an evaluator assessing if an answer is faithful to its source context.

CONTEXT:
{context}

ANSWER:
{answer}

Task: What fraction of claims in the answer are directly supported by the context?
- 1.0 = every claim is supported by the context
- 0.5 = half the claims are supported
- 0.0 = no claims are supported by the context

Output ONLY a decimal number between 0 and 1. Nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0


def score_answer_relevance(question: str, answer: str) -> float:
    """
    Does the answer actually address the question asked?
    High score = directly answers what was asked.
    Low score = off-topic or tangential answer.
    """
    prompt = f"""You are an evaluator assessing if an answer is relevant to a question.

QUESTION:
{question}

ANSWER:
{answer}

Task: How well does this answer address the question?
- 1.0 = directly and completely answers the question
- 0.5 = partially answers the question
- 0.0 = does not address the question at all

Output ONLY a decimal number between 0 and 1. Nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0


def score_context_recall(chunks: list[str], ground_truth: str) -> float:
    """
    Did the retrieved chunks contain the information needed
    to answer correctly?
    High score = chunks contain everything in the ground truth.
    Low score = chunks are missing important information.
    """
    context = "\n\n".join(chunks)
    prompt = f"""You are an evaluator assessing if retrieved context contains enough information.

RETRIEVED CONTEXT:
{context}

GROUND TRUTH ANSWER:
{ground_truth}

Task: What fraction of the information in the ground truth answer 
can be found in the retrieved context?
- 1.0 = all ground truth information is present in the context
- 0.5 = half the ground truth information is in the context
- 0.0 = none of the ground truth information is in the context

Output ONLY a decimal number between 0 and 1. Nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0


def score_context_precision(question: str, chunks: list[str]) -> float:
    """
    Were the retrieved chunks actually useful for answering?
    High score = every retrieved chunk contributed to the answer.
    Low score = many irrelevant chunks were retrieved.
    """
    context = "\n\n".join(
        [f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks)]
    )
    prompt = f"""You are an evaluator assessing retrieval precision.

QUESTION:
{question}

RETRIEVED CHUNKS:
{context}

Task: What fraction of the retrieved chunks are relevant 
and useful for answering the question?
- 1.0 = all chunks are relevant and useful
- 0.5 = half the chunks are relevant
- 0.0 = no chunks are relevant

Output ONLY a decimal number between 0 and 1. Nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 0.0


def evaluate_answer(
    question: str,
    answer: str,
    chunks: list[str],
    ground_truth: str
) -> dict:
    """
    Run all 4 LLM-as-judge metrics on one answer.
    Small delay between calls to avoid rate limiting.
    """
    scores = {}

    scores["faithfulness"] = score_faithfulness(answer, chunks)
    time.sleep(0.5)

    scores["answer_relevance"] = score_answer_relevance(question, answer)
    time.sleep(0.5)

    scores["context_recall"] = score_context_recall(chunks, ground_truth)
    time.sleep(0.5)

    scores["context_precision"] = score_context_precision(question, chunks)
    time.sleep(0.5)

    return scores


# ── Assertion-based evals ─────────────────────────────────────
def run_assertions(
    question: str,
    answer: str,
    ground_truth: str,
    company: str | None
) -> dict:
    """
    Hard pass/fail rules.
    Catches obvious failures that LLM scores might miss.
    Complements the 0-1 scores with binary checks.
    """
    assertions = {}

    # Answer must be substantive
    assertions["answer_not_empty"] = len(answer.strip()) > 100

    # Must contain a risk rating
    risk_ratings = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assertions["has_risk_rating"] = any(
        r in answer.upper() for r in risk_ratings
    )

    # Company name must appear for specific questions
    if company:
        first_word = company.split()[0].lower()
        assertions["mentions_company"] = first_word in answer.lower()
    else:
        assertions["mentions_company"] = True

    # Forced labor must be mentioned if in ground truth
    if "forced labor" in ground_truth.lower():
        assertions["mentions_forced_labor"] = (
            "forced labor" in answer.lower() or
            "forced labour" in answer.lower()
        )
    else:
        assertions["mentions_forced_labor"] = True

    # Sanctions must be mentioned if in ground truth
    if "sanction" in ground_truth.lower():
        assertions["mentions_sanctions"] = (
            "sanction" in answer.lower()
        )
    else:
        assertions["mentions_sanctions"] = True

    # Answer must not be an error message
    assertions["no_error"] = (
        "error" not in answer.lower()[:50]
    )

    return assertions


def assertion_pass_rate(all_assertions: list[dict]) -> float:
    total = 0
    passed = 0
    for assertions in all_assertions:
        for result in assertions.values():
            total += 1
            if result:
                passed += 1
    return passed / total if total > 0 else 0.0


# ── Run pipeline for one strategy ────────────────────────────
def run_pipeline(strategy: str) -> dict:
    """
    Run all 20 golden questions through naive or hybrid pipeline.
    For each question: retrieve chunks, generate answer,
    score with LLM-as-judge, run assertions.
    """
    with open("evals/golden_dataset.json") as f:
        golden = json.load(f)

    print(f"\n{'='*55}")
    print(f"Running {strategy.upper()} pipeline ({len(golden)} questions)")
    print(f"{'='*55}")

    results = []

    for i, item in enumerate(golden):
        question = item["question"]
        ground_truth = item["ground_truth"]
        company = item.get("company")

        print(f"\n[{i+1}/{len(golden)}] {question[:55]}...")

        try:
            # Retrieve chunks
            if strategy == "naive":
                chunks = naive_retrieve(question, n=5)
            else:
                match = match_canonical(question)
                canonical = match[0] if match else ""
                chunks = hybrid_retrieve(question, canonical, n=5)

            if not chunks:
                print(f"  No chunks retrieved — skipping")
                continue

            # Generate answer
            answer, usage = generate_summary(
                question,
                company or "Unknown",
                chunks
            )
            print(f"  Generated {len(answer)} char answer")

            # LLM-as-judge scores
            scores = evaluate_answer(
                question, answer, chunks, ground_truth
            )
            print(
                f"  Scores — F:{scores['faithfulness']:.2f} "
                f"AR:{scores['answer_relevance']:.2f} "
                f"CR:{scores['context_recall']:.2f} "
                f"CP:{scores['context_precision']:.2f}"
            )

            # Assertions
            assertions = run_assertions(
                question, answer, ground_truth, company
            )
            pass_count = sum(assertions.values())
            print(f"  Assertions: {pass_count}/{len(assertions)} passed")

            results.append({
                "question": question,
                "answer": answer,
                "chunks": chunks,
                "ground_truth": ground_truth,
                "company": company,
                "scores": scores,
                "assertions": assertions,
                "usage": usage,
            })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return results


def average_scores(results: list[dict]) -> dict:
    """Average all scores across all questions."""
    if not results:
        return {}

    metrics = ["faithfulness", "answer_relevance",
               "context_recall", "context_precision"]
    averages = {}

    for metric in metrics:
        scores = [r["scores"][metric] for r in results]
        averages[metric] = round(sum(scores) / len(scores), 3)

    return averages


# ── MLflow logging ────────────────────────────────────────────
def log_to_mlflow(
    strategy: str,
    avg_scores: dict,
    assertion_rate: float,
    sample_count: int,
):
    with mlflow.start_run(run_name=f"{strategy}_retrieval"):
        mlflow.log_params({
            "strategy": strategy,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 5,
            "embedding_model": "voyage-2",
            "reranker": "rerank-2" if strategy == "hybrid" else "none",
            "llm_model": "claude-sonnet-4-6",
            "eval_method": "llm-as-judge",
            "judge_model": "claude-sonnet-4-6",
            "questions_evaluated": sample_count,
        })

        mlflow.log_metrics({
            "faithfulness": avg_scores["faithfulness"],
            "answer_relevance": avg_scores["answer_relevance"],
            "context_recall": avg_scores["context_recall"],
            "context_precision": avg_scores["context_precision"],
            "assertion_pass_rate": assertion_rate,
        })

    print(f"MLflow run logged: {strategy}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    mlflow.set_experiment("supply-risk-rag-evals")

    all_results = {}

    for strategy in ["naive", "hybrid"]:
        # Run pipeline
        results = run_pipeline(strategy)

        if not results:
            print(f"No results for {strategy}")
            continue

        # Average scores
        avg_scores = average_scores(results)

        # Assertion pass rate
        all_assertions = [r["assertions"] for r in results]
        assertion_rate = round(
            assertion_pass_rate(all_assertions), 3
        )

        # Log to MLflow
        log_to_mlflow(
            strategy,
            avg_scores,
            assertion_rate,
            len(results),
        )

        all_results[strategy] = {
            "scores": avg_scores,
            "assertion_pass_rate": assertion_rate,
            "questions_evaluated": len(results),
        }

        # Save detailed results per strategy
        with open(f"evals/results_{strategy}.json", "w") as f:
            json.dump(results, f, indent=2)

    # Save summary
    with open("evals/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "="*60)
    print("EVALUATION RESULTS: NAIVE vs HYBRID")
    print("="*60)
    print(f"{'Metric':<25} {'Naive':>10} {'Hybrid':>10}")
    print("-"*60)

    metrics = [
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevance",  "answer_relevance"),
        ("Context Recall",    "context_recall"),
        ("Context Precision", "context_precision"),
    ]

    for label, key in metrics:
        naive_val = (
            all_results.get("naive", {})
            .get("scores", {})
            .get(key, "N/A")
        )
        hybrid_val = (
            all_results.get("hybrid", {})
            .get("scores", {})
            .get(key, "N/A")
        )
        print(f"{label:<25} {str(naive_val):>10} {str(hybrid_val):>10}")

    naive_ar = all_results.get("naive", {}).get("assertion_pass_rate", "N/A")
    hybrid_ar = all_results.get("hybrid", {}).get("assertion_pass_rate", "N/A")
    print(
        f"{'Assertion Pass Rate':<25} "
        f"{str(naive_ar):>10} {str(hybrid_ar):>10}"
    )
    print("="*60)
    print(f"\nDetailed results saved to evals/results_naive.json")
    print(f"and evals/results_hybrid.json")
    print(f"\nTo view MLflow UI:")
    print(f"  mlflow ui")
    print(f"  open http://localhost:5000")