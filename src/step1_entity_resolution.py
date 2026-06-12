# src/cluster.py

import json
from rapidfuzz import fuzz

with open("data/raw/violations.json") as f:
    records = json.load(f)

unique_names = list(set(r["name_at_border"] for r in records))
print(f"Unique border names to cluster: {len(unique_names)}")

# Normalize names for comparison — lowercase, strip whitespace
# Keep original names for output
normalized = {name: name.lower().strip() for name in unique_names}

parent = {name: name for name in unique_names}

def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    parent[find(x)] = find(y)

# Lower threshold to 70, compare normalized versions
THRESHOLD = 70
comparisons = 0
merges = 0

for i in range(len(unique_names)):
    for j in range(i + 1, len(unique_names)):
        name_a = unique_names[i]
        name_b = unique_names[j]

        # Compare normalized (lowercase) versions
        score = fuzz.token_set_ratio(
            normalized[name_a],
            normalized[name_b]
        )
        comparisons += 1

        if score >= THRESHOLD:
            union(name_a, name_b)
            merges += 1

print(f"Comparisons made: {comparisons}")
print(f"Pairs merged: {merges}")

clusters_map = {}
for name in unique_names:
    root = find(name)
    if root not in clusters_map:
        clusters_map[root] = []
    clusters_map[root].append(name)

clusters = []
for root, variants in clusters_map.items():
    canonical = max(variants, key=len)
    clusters.append({
        "canonical": canonical,
        "variants": variants,
        "record_ids": [
            r["id"] for r in records
            if r["name_at_border"] in variants
        ]
    })

clusters.sort(key=lambda c: c["canonical"])

with open("data/processed/clusters.json", "w") as f:
    json.dump(clusters, f, indent=2)

print(f"\nClusters found: {len(clusters)}")
print(f"Expected: 10\n")

for c in clusters:
    print(f"Canonical: {c['canonical']}")
    print(f"  Variants ({len(c['variants'])}): {c['variants']}")
    print(f"  Records:  {c['record_ids']}")
    print()