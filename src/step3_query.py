# src/query.py

import json
import time
import datetime
import hashlib
import pickle
from anthropic import Anthropic
import voyageai
import chromadb
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

load_dotenv()

anthropic_client = Anthropic()
voyage_client = voyageai.Client()
chroma_client = chromadb.PersistentClient(path="./chromadb")

naive_collection = chroma_client.get_collection("naive_chunks")
hybrid_collection = chroma_client.get_collection("hybrid_chunks")

# ── Load canonical names for fuzzy matching ───────────────────
with open("data/processed/clusters.json") as f:
    clusters = json.load(f)

canonical_names = [c["canonical"] for c in clusters]

# ── Redis semantic cache ──────────────────────────────────────
redis_client = None
try:
    import redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    redis_client.ping()
    print("Redis connected — semantic caching enabled")
except Exception:
    print("Redis not available — caching disabled")


# ── Logging ───────────────────────────────────────────────────
LOG_FILE = "query_log.jsonl"

def log_event(event: dict):
    """
    Append one log entry per query to query_log.jsonl.
    .jsonl = one JSON object per line.
    Never overwrites — always appends.
    MLflow will read these in Stage 5.
    """
    event["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[LOG] {json.dumps(event)}")


# ── Step 1: Fuzzy match query to canonical entity ─────────────
def match_canonical(query: str, threshold: int = 60) -> tuple[str, int] | None:
    """
    Match user query to the closest canonical company name.
    Returns (canonical_name, score) or None if no match.

    Threshold 60 is lower than clustering threshold (70)
    because users type partial names, abbreviations,
    or misspellings that are less complete than border names.
    e.g. "sunrise" should match "Sunrise Textile Manufacturing Co."
    """
    result = process.extractOne(
        query.lower(),
        [n.lower() for n in canonical_names],
        scorer=fuzz.token_set_ratio
    )

    if result is None or result[1] < threshold:
        return None

    # Map back to original canonical name
    matched_lower = result[0]
    for name in canonical_names:
        if name.lower() == matched_lower:
            return name, result[1]

    return None


# ── Step 2A: Naive retrieval ──────────────────────────────────
def naive_retrieve(query: str, n: int = 5) -> list[str]:
    """
    Pure vector similarity search.
    Embed query → cosine similarity → top n chunks.
    No filtering, no reranking.
    This is the baseline we compare hybrid against.
    """
    query_vector = voyage_client.embed(
        [query], model="voyage-2"
    ).embeddings[0]

    results = naive_collection.query(
        query_embeddings=[query_vector],
        n_results=n
    )

    return results["documents"][0]


# ── Step 2B: Hybrid retrieval ─────────────────────────────────
def reciprocal_rank_fusion(
    semantic_ids: list[str],
    keyword_ids: list[str],
    k: int = 60
) -> list[str]:
    """
    Combine two ranked lists into one using RRF.

    RRF score for a document = sum of 1/(rank + k)
    across all lists it appears in.

    k=60 is the standard constant — dampens the impact
    of very high ranks so position 1 vs position 2
    isn't disproportionately weighted.

    Documents appearing in both lists score higher
    than those in only one — this is the fusion benefit.
    """
    scores = {}

    for rank, doc_id in enumerate(semantic_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)

    for rank, doc_id in enumerate(keyword_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)

    # Sort by combined RRF score descending
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def hybrid_retrieve(
    query: str,
    canonical: str,
    n: int = 5
) -> list[str]:
    """
    Hybrid retrieval pipeline:
    1. Semantic search (vector similarity)
    2. Keyword search (metadata + document contains)
    3. RRF fusion of both result lists
    4. Cross-encoder reranking of top 10
    5. Return top n

    Why hybrid over naive?
    Semantic search finds conceptually related chunks
    but can miss exact keyword matches.
    Keyword search finds exact matches but misses
    semantic variants ("forced labor" vs "debt bondage").
    Hybrid gets the best of both.
    """
    # Semantic search — top 10 for reranking
    query_vector = voyage_client.embed(
        [query], model="voyage-2"
    ).embeddings[0]

    semantic_results = hybrid_collection.query(
        query_embeddings=[query_vector],
        n_results=10
    )
    semantic_ids = semantic_results["ids"][0]
    semantic_docs = dict(zip(
        semantic_results["ids"][0],
        semantic_results["documents"][0]
    ))

    # Keyword search — filter by canonical company name
    # and document text contains query keywords
    try:
        keyword_results = hybrid_collection.query(
            query_embeddings=[query_vector],
            n_results=10,
            where={"canonical": canonical}
        )
        keyword_ids = keyword_results["ids"][0]
        keyword_docs = dict(zip(
            keyword_results["ids"][0],
            keyword_results["documents"][0]
        ))
    except Exception:
        keyword_ids = []
        keyword_docs = {}

    # Merge doc lookups
    all_docs = {**semantic_docs, **keyword_docs}

    # RRF fusion
    fused_ids = reciprocal_rank_fusion(semantic_ids, keyword_ids)
    top_10_ids = fused_ids[:10]
    top_10_chunks = [all_docs[i] for i in top_10_ids if i in all_docs]

    if not top_10_chunks:
        return naive_retrieve(query, n)

    # Cross-encoder reranking via Voyage rerank-2
    # Scores each (query, chunk) pair together
    # More accurate than vector similarity alone
    try:
        reranked = voyage_client.rerank(
            query=query,
            documents=top_10_chunks,
            model="rerank-2",
            top_k=n
        )
        return [r.document for r in reranked.results]
    except Exception as e:
        print(f"Reranking failed ({e}), falling back to RRF order")
        return top_10_chunks[:n]


# ── Step 3: Generate risk summary with retry ──────────────────
def generate_summary(
    query: str,
    canonical: str,
    chunks: list[str]
) -> tuple[str, dict]:
    """
    Generate plain-English risk summary using Claude.
    Retries with exponential backoff on failure.

    Why exponential backoff?
    If Claude is rate-limited or timing out, hammering
    it immediately makes things worse. Waiting 2s, 4s, 8s
    gives the API time to recover between attempts.
    Max 3 attempts — after that we surface the error
    rather than hanging indefinitely.
    """
    context = "\n\n---\n\n".join(chunks)

    system_prompt = """You are a supply chain compliance analyst.
Your job is to assess supplier risk based on violation records.

Rules:
- Answer in plain English, no jargon
- Be specific: name violation types, severities, jurisdictions
- Mention number of active vs resolved violations
- End with a clear risk rating: CRITICAL / HIGH / MEDIUM / LOW
- Never invent information not present in the records
- If records are insufficient, say so clearly"""

    user_message = f"""Based on these violation records, assess the 
supply chain risk for: {canonical}

RETRIEVED RECORDS:
{context}

User question: {query}"""

    # Retry with exponential backoff
    max_attempts = 3
    wait_seconds = 2

    for attempt in range(max_attempts):
        try:
            start_time = time.time()

            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )

            latency_ms = int((time.time() - start_time) * 1000)
            answer = response.content[0].text

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "latency_ms": latency_ms,
            }

            return answer, usage

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                print(f"Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
                wait_seconds *= 2  # exponential backoff
            else:
                return f"Error generating summary after {max_attempts} attempts: {e}", {}


# ── Step 4: HITL checkpoint ───────────────────────────────────
def hitl_checkpoint(canonical: str, summary: str) -> str:
    """
    Human-in-the-loop validation.
    AI generates the risk summary.
    Human makes the final flag/clear decision.

    In production this would be a UI workflow —
    analyst sees the summary and clicks Approve/Flag/Review.
    Here we simulate it with console input.

    Three outcomes:
      yes    → FLAGGED — supplier blocked pending investigation
      no     → CLEARED — supplier approved
      review → PENDING_REVIEW — escalated to senior analyst
    """
    print("\n" + "="*60)
    print(f"RISK SUMMARY: {canonical}")
    print("="*60)
    print(summary)
    print("="*60)
    print("\nHITL CHECKPOINT")
    print("Flag this supplier for investigation?")
    print("  yes    → FLAGGED")
    print("  no     → CLEARED")
    print("  review → PENDING_REVIEW")

    while True:
        decision = input("\nYour decision (yes/no/review): ").strip().lower()
        if decision in ["yes", "no", "review"]:
            break
        print("Please enter yes, no, or review")

    decision_map = {
        "yes": "FLAGGED",
        "no": "CLEARED",
        "review": "PENDING_REVIEW"
    }

    outcome = decision_map[decision]
    print(f"\n→ Supplier {outcome}")
    return outcome


# ── Step 5: Semantic cache helpers ────────────────────────────
def cache_key(query: str) -> str:
    return f"query:{hashlib.md5(query.lower().encode()).hexdigest()}"

def get_cached(query: str) -> str | None:
    if not redis_client:
        return None
    try:
        cached = redis_client.get(cache_key(query))
        return pickle.loads(cached) if cached else None
    except Exception:
        return None

def set_cache(query: str, answer: str, ttl: int = 3600):
    if not redis_client:
        return
    try:
        redis_client.set(cache_key(query), pickle.dumps(answer), ex=ttl)
    except Exception:
        pass


# ── Main query function ───────────────────────────────────────
def run_query(query: str, strategy: str = "hybrid"):
    """
    Full query pipeline.
    strategy: "naive" or "hybrid"
    """
    print(f"\nQuery: '{query}' | Strategy: {strategy}")

    # Check semantic cache
    cached = get_cached(f"{strategy}:{query}")
    if cached:
        print("Cache hit — returning cached answer")
        log_event({
            "query": query,
            "strategy": strategy,
            "cache_hit": True,
        })
        print(cached)
        return cached

    # Step 1 — fuzzy match
    match = match_canonical(query)
    if not match:
        print(f"No matching company found for: '{query}'")
        return None

    canonical, score = match
    print(f"Matched: '{canonical}' (score: {score})")

    # Step 2 — retrieve
    if strategy == "naive":
        chunks = naive_retrieve(query)
    else:
        chunks = hybrid_retrieve(query, canonical)

    print(f"Retrieved {len(chunks)} chunks")

    # Step 3 — generate summary
    summary, usage = generate_summary(query, canonical, chunks)

    # Step 4 — HITL checkpoint
    decision = hitl_checkpoint(canonical, summary)

    # Step 5 — cache result
    set_cache(f"{strategy}:{query}", summary)

    # Step 6 — log everything
    log_event({
        "query": query,
        "matched_canonical": canonical,
        "match_score": score,
        "retrieval_strategy": strategy,
        "chunks_retrieved": len(chunks),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "latency_ms": usage.get("latency_ms", 0),
        "hitl_decision": decision,
        "cache_hit": False,
    })

    return summary


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Supply Chain Risk Query System")
    print("Commands: 'quit' to exit, 'naive:query' for naive retrieval")
    print("Default strategy: hybrid\n")

    while True:
        user_input = input("Search company: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        # Allow strategy override: "naive:sunrise textile"
        if user_input.startswith("naive:"):
            strategy = "naive"
            query = user_input[6:].strip()
        else:
            strategy = "hybrid"
            query = user_input

        run_query(query, strategy)