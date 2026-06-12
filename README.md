# supply-risk-rag

Supply chain risk intelligence pipeline — resolves variant company names across border control records using fuzzy entity clustering, then answers compliance queries via hybrid RAG with LLM-as-judge evaluation.

---

## Architecture

```
Stage 1 — Data Generation
  Realistic border violation records with messy company
  name variants (no canonical name in raw data)

Stage 2 — Entity Resolution
  Fuzzy clustering (rapidfuzz + union-find) groups variant
  names into canonical entities without a pre-existing registry

Stage 3 — Ingestion
  One prose document per canonical entity
  Chunked (1000 chars, 200 overlap), embedded (voyage-2),
  stored in ChromaDB — two collections:
    naive_chunks  → dense vector search only
    hybrid_chunks → dense vectors + metadata for filtered search

Stage 4 — Query Pipeline
  Fuzzy match query → canonical entity
  Naive:  pure cosine similarity
  Hybrid: semantic search + keyword filter → RRF fusion
          → cross-encoder reranking (voyage rerank-2)
  Claude generates plain-English risk summary
  HITL checkpoint — human flags/clears/reviews
  Semantic cache (Redis), retry with exponential backoff
  Full audit log → query_log.jsonl

Stage 5 — Evaluation
  20 golden Q&A pairs
  LLM-as-judge: faithfulness, answer relevance,
                context recall, context precision
  Assertion-based evals: risk rating present,
                         violations mentioned, no errors
  MLflow experiment tracking — naive vs hybrid comparison
```

---

## Results

| Metric | Naive | Hybrid |
|---|---|---|
| Faithfulness | 0.26 | 0.14 |
| Answer Relevance | 0.88 | 0.89 |
| Context Recall | 0.44 | 0.67 |
| Context Precision | 0.26 | 0.43 |
| Assertion Pass Rate | 96.7% | 97.5% |

Hybrid improved context recall by 52% and context precision by 65% over naive dense retrieval. Answer relevance was equivalent across both strategies. Low faithfulness scores across both pipelines indicate Claude draws on general knowledge beyond retrieved chunks — a known production concern in compliance contexts where answers must be grounded exclusively in verified records.

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | Voyage AI voyage-2 |
| Reranking | Voyage AI rerank-2 |
| Vector store | ChromaDB |
| LLM | Anthropic Claude Sonnet |
| Fuzzy matching | rapidfuzz |
| Semantic cache | Redis |
| Experiment tracking | MLflow |
| Evaluation | LLM-as-judge (custom) |

---

## Project Structure

```
supply-risk-rag/
├── data/
│   ├── raw/
│   │   └── violations.json          ← 48 border violation records
│   ├── processed/
│   │   ├── clusters.json            ← fuzzy clustering output
│   │   └── documents.json           ← prose documents per entity
│   └── generate.py                  ← Stage 1
├── src/
│   ├── cluster.py                   ← Stage 2
│   ├── ingest.py                    ← Stage 3
│   └── query.py                     ← Stage 4
├── evals/
│   ├── golden_dataset.json          ← 20 Q&A pairs
│   ├── evaluate.py                  ← Stage 5
│   ├── results.json                 ← summary scores
│   ├── results_naive.json           ← per-question naive results
│   └── results_hybrid.json         ← per-question hybrid results
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

```bash
git clone https://github.com/bennychethana/supply-risk-rag
cd supply-risk-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

Redis (for semantic caching):
```bash
brew install redis
brew services start redis
```

---

## Run

```bash
# Stage 1 — generate data
python3 data/generate.py

# Stage 2 — entity resolution
python3 src/cluster.py

# Stage 3 — embed and store
python3 src/ingest.py

# Stage 4 — query
python3 src/query.py

# Stage 5 — evaluate
python3 evals/evaluate.py

# MLflow UI
mlflow ui --port 5001
# open http://127.0.0.1:5001
```

---

## Key Design Decisions

**Why fuzzy clustering instead of exact matching**
Border records use inconsistent company name spellings across jurisdictions. "SUNRISE TEXTILE MANUFACTURING", "Sunrise Textile Mfg Co", and "SR Textile Co" are the same entity. Exact matching misses these. Union-find with token_set_ratio at threshold 70 groups transitively — if A matches B and B matches C, all three cluster together even if A and C score below threshold directly.

**Why hybrid search over naive vector search**
Naive vector search retrieves by semantic similarity alone. A query for "forced labor violations in USA" may rank chunks by topic similarity but miss chunks from the correct company if the embedding space does not separate them cleanly. Hybrid search adds a metadata filter on canonical company name, combines results via Reciprocal Rank Fusion, then cross-encoder reranking scores each (query, chunk) pair together for final ordering. Context recall improved from 0.44 to 0.67.

**Why LLM-as-judge instead of RAGAS**
RAGAS implements the same LLM-as-judge pattern but with deep LangChain coupling that creates dependency conflicts on Python 3.13. Built equivalent evaluation directly against the Anthropic API — four focused prompts each returning a 0-1 score. Same methodology, no framework overhead.

**Why two ChromaDB collections**
Both collections store identical chunks and vectors. The difference is query strategy — naive queries ignore metadata, hybrid queries filter on metadata before vector search. Keeping them separate makes the baseline comparison clean and allows independent optimization of each strategy.

**Production gaps**
- SQLite-backed ChromaDB → Postgres with pgvector for concurrent access
- In-memory Redis cache → persistent Redis with TTL management
- Console HITL → UI approval workflow with audit trail
- Faithfulness scores suggest stricter system prompt needed: answer only from retrieved records
