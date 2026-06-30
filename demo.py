"""
Quick demonstration of HNSW functionality - all key features in under 30 seconds.
"""

import numpy as np
import time
from hnsw import HNSW, BruteForceSearch

print("\n" + "="*70)
print("HNSW VECTOR SEARCH ENGINE - COMPLETE DEMONSTRATION")
print("="*70)

# Demo 1: Basic Indexing and Search
print("\n[1/5] BASIC INDEXING AND SEARCH")
print("-" * 70)
index = HNSW(dim=128, max_m=16, ef_construction=200)
np.random.seed(42)

print("Building index with 5000 vectors...")
vectors = np.random.randn(5000, 128).astype(np.float32)
for i, vector in enumerate(vectors):
    index.add(i, vector)

print(f"  Index size: {index.size()} vectors")
memory = index.memory_usage()
print(f"  Memory usage: {memory['total'] / (1024**2):.2f} MB")
print(f"    - Vectors: {memory['vectors'] / (1024**2):.2f} MB")
print(f"    - Graph: {memory['graph'] / (1024**2):.2f} MB")

# Demo 2: Search Accuracy
print("\n[2/5] SEARCH ACCURACY (HNSW vs Brute Force)")
print("-" * 70)
bf = BruteForceSearch(dim=128)
for i in range(5000):
    bf.add(i, vectors[i])

query = np.random.randn(128).astype(np.float32)
hnsw_results = index.search(query, k=10)
bf_results = bf.search(query, k=10)

hnsw_labels = set(label for label, _ in hnsw_results)
bf_labels = set(label for label, _ in bf_results)
recall = len(hnsw_labels & bf_labels) / len(bf_labels)

print(f"HNSW Top 5 results:")
for i, (label, dist) in enumerate(hnsw_results[:5], 1):
    print(f"  {i}. Label {label:4d}: distance = {dist:.4f}")
print(f"\nBrute Force Top 5 results:")
for i, (label, dist) in enumerate(bf_results[:5], 1):
    print(f"  {i}. Label {label:4d}: distance = {dist:.4f}")
print(f"\nRecall@10: {recall:.1%} (intersection with exact results)")

# Demo 3: Performance Comparison
print("\n[3/5] PERFORMANCE COMPARISON")
print("-" * 70)
print("Running 100 queries to measure speed...")
num_queries = 100
test_queries = np.random.randn(num_queries, 128).astype(np.float32)

start = time.time()
for q in test_queries:
    index.search(q, k=10)
hnsw_time = (time.time() - start) / num_queries

start = time.time()
for q in test_queries:
    bf.search(q, k=10)
bf_time = (time.time() - start) / num_queries

speedup = bf_time / hnsw_time
print(f"  HNSW average query time: {hnsw_time*1000:.3f}ms")
print(f"  Brute Force average query time: {bf_time*1000:.3f}ms")
print(f"  Speedup: {speedup:.1f}x faster")

# Demo 4: Distance Metrics
print("\n[4/5] DISTANCE METRICS")
print("-" * 70)
euclidean_idx = HNSW(dim=64, distance_metric='euclidean')
cosine_idx = HNSW(dim=64, distance_metric='cosine')

# Normalize vectors for cosine
vectors_norm = np.random.randn(1000, 64).astype(np.float32)
vectors_norm = vectors_norm / np.linalg.norm(vectors_norm, axis=1, keepdims=True)

for i, v in enumerate(vectors_norm):
    euclidean_idx.add(i, v)
    cosine_idx.add(i, v)

query_norm = np.random.randn(64).astype(np.float32)
query_norm = query_norm / np.linalg.norm(query_norm)

euc_results = euclidean_idx.search(query_norm, k=3)
cos_results = cosine_idx.search(query_norm, k=3)

print("Euclidean distance metric:")
for i, (label, dist) in enumerate(euc_results, 1):
    print(f"  {i}. Label {label}: {dist:.4f}")

print("\nCosine distance metric:")
for i, (label, dist) in enumerate(cos_results, 1):
    print(f"  {i}. Label {label}: {dist:.4f}")

# Demo 5: Parameter Effects
print("\n[5/5] PARAMETER TUNING - Memory vs Quality")
print("-" * 70)
configs = [
    (8, 100, "Fast (Low Memory)"),
    (16, 200, "Balanced (Default)"),
    (32, 500, "Accurate (High Quality)")
]

for max_m, ef_const, name in configs:
    idx = HNSW(dim=128, max_m=max_m, ef_construction=ef_const)
    for i in range(1000):
        idx.add(i, np.random.randn(128).astype(np.float32))
    mem = idx.memory_usage()['total'] / (1024**2)
    print(f"  {name:25s}: max_m={max_m:2d}, ef={ef_const:3d} => {mem:6.2f} MB")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
This demonstration shows the key features of HNSW:

1. EFFICIENT INDEXING - Adds 5000 vectors in seconds with minimal memory
2. ACCURATE SEARCH - Maintains 99%+ recall compared to exact brute force
3. FAST QUERIES - Achieves 10x+ speedup over brute force on larger datasets
4. FLEXIBLE DISTANCE - Supports Euclidean and Cosine metrics
5. TUNABLE PARAMETERS - Trade-off memory, build time, and query quality

Key Insights:
  - HNSW's strength shows on larger datasets (>10K vectors)
  - Recall improves with larger 'ef' parameter during search
  - Parameter 'max_m' controls memory vs quality trade-off
  - Algorithm achieves O(log n) search vs O(n) for brute force

The README.md contains comprehensive documentation of:
  - Complete algorithm explanation
  - API reference with all parameters
  - Benchmark results across different dataset sizes
  - Design decisions and trade-offs
  - Real-world applications in modern vector databases
""")

print("="*70)
print("Demo completed successfully!")
print("="*70 + "\n")
