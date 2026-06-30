#!/usr/bin/env python3
"""
Quick verification that the heap fix improves recall.
This demonstrates the ef parameter now has proper effect on results.
"""

import numpy as np
from hnsw import HNSW, BruteForceSearch


def calculate_recall(hnsw_results, bf_results):
    """Calculate recall between HNSW and brute force results."""
    hnsw_labels = set(label for label, _ in hnsw_results)
    bf_labels = set(label for label, _ in bf_results)
    if not bf_labels:
        return 1.0
    return len(hnsw_labels & bf_labels) / len(bf_labels)


def main():
    print("HNSW Algorithm Fix Verification")
    print("=" * 60)
    print()

    # Setup
    np.random.seed(42)
    num_vectors = 1000
    dim = 64

    print(f"Generating {num_vectors} random vectors (dim={dim})...")
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    print("Indexing with HNSW...")
    hnsw = HNSW(dim=dim, max_m=16, ef_construction=200, seed=42,
                distance_metric='cosine')
    for i, v in enumerate(vectors):
        hnsw.add(i, v)

    print("Building brute force index for ground truth...")
    bf = BruteForceSearch(dim=dim, distance_metric='cosine')
    for i, v in enumerate(vectors):
        bf.add(i, v)

    # Generate test queries
    num_queries = 5
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    print()
    print("Testing ef parameter sensitivity...")
    print("=" * 60)
    print()

    # Test with different ef values
    ef_values = [50, 100, 200, 500, 1000]
    results = {}

    for ef in ef_values:
        recalls = []
        for query in queries:
            hnsw_results = hnsw.search(query, k=10, ef=ef)
            bf_results = bf.search(query, k=10)
            recall = calculate_recall(hnsw_results, bf_results)
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        results[ef] = mean_recall
        print(f"ef={ef:4d}  ->  Mean Recall: {mean_recall:.4f}  ({100*mean_recall:5.1f}%)")

    print()
    print("Verification Results:")
    print("=" * 60)

    # Check if recall improves with larger ef
    recall_values = list(results.values())
    is_monotonic = all(recall_values[i] <= recall_values[i+1]
                       for i in range(len(recall_values)-1))

    if is_monotonic:
        print("[PASS] Recall improves monotonically with larger ef")
        print("  --> The heap fix is working correctly!")
    else:
        print("[FAIL] Recall does not improve with larger ef")
        print("  --> The algorithm may still have issues")

    # Check minimum recall threshold
    min_recall = min(recall_values)
    if min_recall > 0.5:
        print(f"[PASS] Minimum recall ({min_recall:.4f}) is reasonable")
    else:
        print(f"[FAIL] Minimum recall ({min_recall:.4f}) is too low")

    print()
    print("Conclusion:")
    print("-" * 60)
    if is_monotonic and min_recall > 0.5:
        print("[PASS] Algorithm is working correctly after the fix!")
        print("  --> ef parameter now properly controls the recall-speed tradeoff")
    else:
        print("[NOTE] Additional issues may still exist in the implementation")


if __name__ == '__main__':
    main()
