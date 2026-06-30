"""
Example usage of HNSW vector search engine.
Demonstrates basic and advanced usage patterns.
"""

import numpy as np
from hnsw import HNSW, BruteForceSearch


def example_basic():
    """Basic example: Build index and search."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)

    # Create index for 128-dimensional vectors
    index = HNSW(dim=128, max_m=16, ef_construction=200)

    # Generate and add random vectors
    print("\n1. Adding 1000 random vectors...")
    np.random.seed(42)
    for i in range(1000):
        vector = np.random.randn(128).astype(np.float32)
        index.add(label=i, vector=vector)

    print(f"   Index size: {index.size()} vectors")
    memory = index.memory_usage()
    print(f"   Memory used: {memory['total'] / (1024**2):.2f} MB")

    # Search for nearest neighbors
    print("\n2. Searching for nearest neighbors...")
    query = np.random.randn(128).astype(np.float32)
    results = index.search(query, k=5, ef=200)

    print("   Top 5 nearest neighbors:")
    for i, (label, distance) in enumerate(results, 1):
        print(f"   {i}. Label {label}: distance = {distance:.4f}")


def example_distance_metrics():
    """Example: Using different distance metrics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Different Distance Metrics")
    print("=" * 60)

    dim = 64
    num_vectors = 500

    # Create indices with different metrics
    index_euclidean = HNSW(dim=dim, distance_metric='euclidean')
    index_cosine = HNSW(dim=dim, distance_metric='cosine')

    # Generate and normalize vectors (important for cosine)
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    print(f"\n1. Adding {num_vectors} normalized vectors...")
    for i, vector in enumerate(vectors):
        index_euclidean.add(i, vector)
        index_cosine.add(i, vector)

    # Query
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    print("\n2. Searching with Euclidean distance:")
    results_euclidean = index_euclidean.search(query, k=3)
    for i, (label, distance) in enumerate(results_euclidean, 1):
        print(f"   {i}. Label {label}: distance = {distance:.4f}")

    print("\n3. Searching with Cosine distance:")
    results_cosine = index_cosine.search(query, k=3)
    for i, (label, distance) in enumerate(results_cosine, 1):
        print(f"   {i}. Label {label}: distance = {distance:.4f}")


def example_parameter_tuning():
    """Example: Effect of parameters on performance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Parameter Effects")
    print("=" * 60)

    dim = 128
    num_vectors = 5000
    np.random.seed(42)

    # Generate test data
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)

    # Low-memory, fast configuration
    print("\n1. Fast Configuration (max_m=8, ef_construction=100):")
    index_fast = HNSW(dim=dim, max_m=8, ef_construction=100)
    for i, vector in enumerate(vectors):
        index_fast.add(i, vector)
    mem_fast = index_fast.memory_usage()
    print(f"   Memory: {mem_fast['total'] / (1024**2):.2f} MB")

    # Balanced configuration
    print("\n2. Balanced Configuration (max_m=16, ef_construction=200):")
    index_balanced = HNSW(dim=dim, max_m=16, ef_construction=200)
    for i, vector in enumerate(vectors):
        index_balanced.add(i, vector)
    mem_balanced = index_balanced.memory_usage()
    print(f"   Memory: {mem_balanced['total'] / (1024**2):.2f} MB")

    # High-accuracy configuration
    print("\n3. Accurate Configuration (max_m=32, ef_construction=500):")
    index_accurate = HNSW(dim=dim, max_m=32, ef_construction=500)
    for i, vector in enumerate(vectors):
        index_accurate.add(i, vector)
    mem_accurate = index_accurate.memory_usage()
    print(f"   Memory: {mem_accurate['total'] / (1024**2):.2f} MB")

    print(f"\n4. Memory comparison:")
    print(f"   Fast vs Balanced: {mem_balanced['total'] / mem_fast['total']:.2f}x")
    print(f"   Balanced vs Accurate: {mem_accurate['total'] / mem_balanced['total']:.2f}x")


def example_ef_tuning():
    """Example: Tuning ef parameter for query time / recall trade-off."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: ef Parameter Tuning")
    print("=" * 60)

    dim = 128
    num_vectors = 5000
    np.random.seed(42)

    # Build index
    index = HNSW(dim=dim, max_m=16, ef_construction=200)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    for i, vector in enumerate(vectors):
        index.add(i, vector)

    # Build brute force for comparison
    bf = BruteForceSearch(dim=dim)
    for i, vector in enumerate(vectors):
        bf.add(i, vector)

    # Test different ef values
    query = np.random.randn(dim).astype(np.float32)
    bf_results = bf.search(query, k=10)
    bf_labels = set(label for label, _ in bf_results)

    print("\nQuery performance with different ef values:")
    print("ef    | Avg Recall | Quality")
    print("------|------------|--------")

    for ef in [50, 100, 200, 500, 1000]:
        index.ef = ef
        hnsw_results = index.search(query, k=10, ef=ef)
        hnsw_labels = set(label for label, _ in hnsw_results)
        recall = len(hnsw_labels & bf_labels) / len(bf_labels)
        quality = "[Good]" if recall >= 0.99 else "[Fair]" if recall >= 0.95 else "[Poor]"
        print(f"{ef:5d} | {recall:10.1%} | {quality}")

    print("\nNotes:")
    print("  - Lower ef: Faster queries but lower recall")
    print("  - Higher ef: Slower queries but higher recall")
    print("  - ef=200 is good default balance")


def example_incremental_indexing():
    """Example: Building index incrementally."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Incremental Indexing")
    print("=" * 60)

    dim = 64
    index = HNSW(dim=dim, max_m=16, ef_construction=200)

    print("\nAdding vectors incrementally:")
    np.random.seed(42)

    for batch in range(5):
        batch_size = 100
        print(f"\n  Batch {batch + 1}: Adding {batch_size} vectors")

        for i in range(batch_size):
            label = batch * batch_size + i
            vector = np.random.randn(dim).astype(np.float32)
            index.add(label, vector)

        print(f"    Total in index: {index.size()} vectors")

    # Search after building
    print("\nSearching after incremental build:")
    query = np.random.randn(dim).astype(np.float32)
    results = index.search(query, k=5, ef=200)
    for i, (label, distance) in enumerate(results, 1):
        print(f"  {i}. Label {label}: {distance:.4f}")


def example_recall_measurement():
    """Example: Measuring recall against brute force."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Recall Measurement")
    print("=" * 60)

    dim = 128
    num_vectors = 1000
    num_queries = 10

    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # Build HNSW index
    print("\nBuilding HNSW index...")
    hnsw = HNSW(dim=dim, max_m=16, ef_construction=200)
    for i, vector in enumerate(vectors):
        hnsw.add(i, vector)

    # Build brute force
    print("Building brute force index...")
    bf = BruteForceSearch(dim=dim)
    for i, vector in enumerate(vectors):
        bf.add(i, vector)

    # Measure recall
    print("\nMeasuring recall on {} queries:".format(num_queries))
    recalls = []

    for q_idx, query in enumerate(queries):
        hnsw_results = hnsw.search(query, k=10)
        bf_results = bf.search(query, k=10)

        hnsw_labels = set(label for label, _ in hnsw_results)
        bf_labels = set(label for label, _ in bf_results)

        recall = len(hnsw_labels & bf_labels) / len(bf_labels)
        recalls.append(recall)
        print(f"  Query {q_idx + 1}: {recall:.1%} recall")

    avg_recall = np.mean(recalls)
    print(f"\nAverage recall: {avg_recall:.1%}")
    print(f"Min recall: {min(recalls):.1%}")
    print(f"Max recall: {max(recalls):.1%}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HNSW Vector Search Engine - Usage Examples")
    print("=" * 60)

    example_basic()
    example_distance_metrics()
    example_parameter_tuning()
    example_ef_tuning()
    example_incremental_indexing()
    example_recall_measurement()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
