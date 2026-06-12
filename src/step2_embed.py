# src/ingest.py

import json
import os
import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

voyage_client = voyageai.Client()  # reads VOYAGE_API_KEY from .env


def get_embedding(text: str) -> list[float]:
    """
    Call Voyage AI embeddings API.
    voyage-2 returns 1024-dimension vectors.
    """
    result = voyage_client.embed(
        [text],
        model="voyage-2"
    )
    return result.embeddings[0]


def chunk_text(text: str) -> list[str]:
    """
    Split document into overlapping chunks.

    CHAR_SIZE = 1000 chars (~512 tokens at ~4 chars/token)
    CHAR_OVERLAP = 200 chars (~50 tokens)

    Why chunk?
    A full document for one company can be 2000+ chars.
    Embedding a long document loses precision —
    the vector tries to represent too many concepts.
    Smaller chunks give more focused, retrievable vectors.

    Why overlap?
    A violation record that spans a chunk boundary
    would be split in half without overlap.
    200 char overlap ensures boundary context
    appears in both adjacent chunks.

    Why break at newlines?
    Avoids splitting mid-sentence or mid-record.
    A chunk that ends at "Details: Workers were con-"
    is less useful than one ending at a full record.
    """
    CHAR_SIZE = 1000
    CHAR_OVERLAP = 200

    if len(text) <= CHAR_SIZE:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHAR_SIZE

        # Try to break at newline rather than mid-sentence
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start + CHAR_SIZE // 2:
                end = newline_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHAR_OVERLAP

    return chunks


# ── Load data ─────────────────────────────────────────────────
with open("data/raw/violations.json") as f:
    records = json.load(f)

with open("data/processed/clusters.json") as f:
    clusters = json.load(f)

record_lookup = {r["id"]: r for r in records}

# ── Step 1: Build documents ───────────────────────────────────
print(f"Building documents for {len(clusters)} clusters...")
documents = []

for cluster in clusters:
    canonical = cluster["canonical"]

    variant_records = [
        record_lookup[rid]
        for rid in cluster["record_ids"]
        if rid in record_lookup
    ]

    if not variant_records:
        continue

    variant_records.sort(key=lambda r: r["date_reported"])

    # Build prose document
    lines = []
    lines.append(f"COMPANY: {canonical}")
    lines.append(
        f"Country of Origin: {variant_records[0]['country_of_origin']}"
    )
    lines.append(f"Industry: {variant_records[0]['industry']}")
    lines.append(f"Total Violation Records: {len(variant_records)}")
    lines.append(
        f"Known Border Names: {', '.join(cluster['variants'])}"
    )
    lines.append("")
    lines.append("VIOLATION RECORDS:")

    severities = []
    jurisdictions = []
    statuses = []
    violation_types = []

    for rec in variant_records:
        severities.append(rec["severity"])
        jurisdictions.append(rec["jurisdiction"])
        statuses.append(rec["status"])
        violation_types.append(rec["violation_type"])

        lines.append(
            f"\n[{rec['id']}] {rec['date_reported']} — {rec['jurisdiction']}"
        )
        lines.append(
            f"  Type: {rec['violation_type'].replace('_', ' ').title()}"
        )
        lines.append(f"  Severity: {rec['severity']}")
        lines.append(f"  Reported by: {rec['reporting_agency']}")
        lines.append(f"  Source list: {rec['source_list']}")
        lines.append(f"  Status: {rec['status']}")
        lines.append(f"  Details: {rec['description']}")

    critical = severities.count("Critical")
    active = statuses.count("Active")
    unique_jurisdictions = list(set(jurisdictions))
    unique_violation_types = list(set(violation_types))

    lines.append(f"\nSUMMARY:")
    lines.append(f"  Critical violations: {critical}")
    lines.append(f"  Active violations: {active}")
    lines.append(
        f"  Jurisdictions: {', '.join(unique_jurisdictions)}"
    )

    doc = {
        "canonical": canonical,
        "country": variant_records[0]["country_of_origin"],
        "industry": variant_records[0]["industry"],
        "record_count": len(variant_records),
        "critical_count": critical,
        "active_count": active,
        "jurisdictions": unique_jurisdictions,
        "violation_types": unique_violation_types,
        "content": "\n".join(lines),
    }
    documents.append(doc)

with open("data/processed/documents.json", "w") as f:
    json.dump(documents, f, indent=2)

print(f"Built {len(documents)} documents")

# ── Steps 2-5: Chunk, embed, store in ChromaDB ────────────────
chroma_client = chromadb.PersistentClient(path="./chromadb")

# Clean slate on every run
for col_name in ["naive_chunks", "hybrid_chunks"]:
    try:
        chroma_client.delete_collection(col_name)
        print(f"Deleted existing collection: {col_name}")
    except Exception:
        pass

naive_collection = chroma_client.create_collection(
    name="naive_chunks",
    metadata={"hnsw:space": "cosine"}
)

hybrid_collection = chroma_client.create_collection(
    name="hybrid_chunks",
    metadata={"hnsw:space": "cosine"}
)

print("\nChunking, embedding, storing...")
total_chunks = 0

for doc in documents:
    chunks = chunk_text(doc["content"])
    print(f"  {doc['canonical']}: {len(chunks)} chunk(s)")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc['canonical']}_{i}"

        # Step 3 — embed chunk
        # One API call per chunk, done once, never repeated
        vector = get_embedding(chunk)

        # Metadata stored alongside vector
        # Used by hybrid collection for filtered search
        # naive collection stores it too but queries ignore it
        metadata = {
            "canonical": doc["canonical"],
            "country": doc["country"],
            "industry": doc["industry"],
            "critical_count": doc["critical_count"],
            "active_count": doc["active_count"],
            "violation_types": ", ".join(doc["violation_types"]),
            "jurisdictions": ", ".join(doc["jurisdictions"]),
            "chunk_index": i,
        }

        # Step 4 — store in naive collection
        naive_collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[metadata]
        )

        # Step 5 — store same chunk in hybrid collection
        # Same vectors, same metadata
        # Query strategy differs in query.py
        hybrid_collection.add(
            ids=[chunk_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[metadata]
        )

        total_chunks += 1

print(f"\nTotal chunks embedded: {total_chunks}")
print(f"Naive collection:  {naive_collection.count()} chunks")
print(f"Hybrid collection: {hybrid_collection.count()} chunks")
print("\nIngestion complete.")
print("ChromaDB persisted to ./chromadb/")
print("Next step: python3 src/query.py")