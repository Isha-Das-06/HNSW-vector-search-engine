import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
from hnsw import HNSW, BruteForceSearch


def generate_dataset(num_vectors: int, dim: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random vector dataset for testing."""
    np.random.seed(seed)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Generate query vectors from the same distribution
    num_queries = max(10, num_vectors // 100)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    return vectors, queries


def calculate_recall(hnsw_results: List[Tuple[int, float]],
                     brute_results: List[Tuple[int, float]]) -> float:
    """Calculate recall@k between approximate and exact results."""
    hnsw_labels = set(label for label, _ in hnsw_results)
    brute_labels = set(label for label, _ in brute_results)
    if not brute_labels:
        return 1.0
    return len(hnsw_labels & brute_labels) / len(brute_labels)


def benchmark_indexing(vectors: np.ndarray, dim: int) -> Dict:
    """Benchmark index construction time."""
    results = {}

    # HNSW indexing
    hnsw = HNSW(dim=dim, max_m=16, ef_construction=200)
    start = time.time()
    for i, vector in enumerate(vectors):
        hnsw.add(i, vector)
    hnsw_time = time.time() - start
    results['hnsw'] = {
        'time': hnsw_time,
        'memory': hnsw.memory_usage()['total']
    }

    # Brute force (no indexing needed)
    bf = BruteForceSearch(dim=dim)
    start = time.time()
    for i, vector in enumerate(vectors):
        bf.add(i, vector)
    bf_time = time.time() - start
    results['brute_force'] = {
        'time': bf_time,
        'memory': len(vectors) * dim * 4  # 4 bytes per float32
    }

    return hnsw, bf, results


def benchmark_search(hnsw: HNSW, bf: BruteForceSearch, queries: np.ndarray,
                    k: int = 10, ef: Optional[int] = None) -> Dict:
    """Benchmark search performance."""
    results = {
        'hnsw': {'times': [], 'recalls': []},
        'brute_force': {'times': []}
    }

    if ef is None:
        ef = hnsw.ef_construction

    for query in queries:
        # HNSW search
        start = time.time()
        hnsw_results = hnsw.search(query, k=k, ef=ef)
        hnsw_time = time.time() - start
        results['hnsw']['times'].append(hnsw_time)

        # Brute force search
        start = time.time()
        bf_results = bf.search(query, k=k)
        bf_time = time.time() - start
        results['brute_force']['times'].append(bf_time)

        # Calculate recall
        recall = calculate_recall(hnsw_results, bf_results)
        results['hnsw']['recalls'].append(recall)

    # Compute statistics
    results['hnsw']['mean_time'] = np.mean(results['hnsw']['times'])
    results['hnsw']['median_time'] = np.median(results['hnsw']['times'])
    results['hnsw']['mean_recall'] = np.mean(results['hnsw']['recalls'])
    results['brute_force']['mean_time'] = np.mean(results['brute_force']['times'])
    results['brute_force']['median_time'] = np.median(results['brute_force']['times'])
    results['speedup'] = results['brute_force']['mean_time'] / results['hnsw']['mean_time']

    return results


def run_full_benchmark(dataset_sizes: List[int] = None, dims: List[int] = None):
    """Run comprehensive benchmark across multiple configurations."""
    if dataset_sizes is None:
        dataset_sizes = [1000, 10000, 100000]
    if dims is None:
        dims = [32, 128, 256]

    all_results = {}

    for size in dataset_sizes:
        for dim in dims:
            print(f"\nBenchmarking: {size} vectors, {dim} dimensions")
            print("=" * 50)

            # Generate data
            vectors, queries = generate_dataset(size, dim)

            # Benchmark indexing
            hnsw, bf, indexing_results = benchmark_indexing(vectors, dim)
            print(f"Indexing time - HNSW: {indexing_results['hnsw']['time']:.3f}s, "
                  f"Brute Force: {indexing_results['brute_force']['time']:.3f}s")
            print(f"Memory - HNSW: {indexing_results['hnsw']['memory'] / (1024**2):.2f}MB, "
                  f"Brute Force: {indexing_results['brute_force']['memory'] / (1024**2):.2f}MB")

            # Benchmark search
            search_results = benchmark_search(hnsw, bf, queries, k=10)
            print(f"\nSearch (k=10):")
            print(f"HNSW - Mean: {search_results['hnsw']['mean_time']*1000:.3f}ms, "
                  f"Median: {search_results['hnsw']['median_time']*1000:.3f}ms, "
                  f"Recall: {search_results['hnsw']['mean_recall']:.4f}")
            print(f"Brute Force - Mean: {search_results['brute_force']['mean_time']*1000:.3f}ms, "
                  f"Median: {search_results['brute_force']['median_time']*1000:.3f}ms")
            print(f"Speedup: {search_results['speedup']:.2f}x")

            key = f"{size}_vectors_dim{dim}"
            all_results[key] = {
                'indexing': indexing_results,
                'search': search_results
            }

    return all_results


def run_ef_sensitivity(num_vectors: int = 10000, dim: int = 128):
    """Test sensitivity to ef parameter."""
    print(f"\nTesting ef sensitivity ({num_vectors} vectors, {dim} dimensions)")
    print("=" * 50)

    vectors, queries = generate_dataset(num_vectors, dim)
    hnsw, bf, _ = benchmark_indexing(vectors, dim)

    results = {}
    ef_values = [50, 100, 200, 500, 1000]

    for ef in ef_values:
        search_results = benchmark_search(hnsw, bf, queries, k=10, ef=ef)
        results[ef] = {
            'mean_time': search_results['hnsw']['mean_time'],
            'mean_recall': search_results['hnsw']['mean_recall'],
            'speedup': search_results['speedup']
        }
        print(f"ef={ef:4d} - Time: {search_results['hnsw']['mean_time']*1000:6.3f}ms, "
              f"Recall: {search_results['hnsw']['mean_recall']:.4f}, "
              f"Speedup: {search_results['speedup']:5.2f}x")

    return results


def save_results(results: Dict, filename: str = 'benchmark_results.json'):
    """Save benchmark results to JSON file."""
    # Convert numpy types to Python native types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    with open(filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == '__main__':
    print("HNSW Benchmark Suite")
    print("=" * 50)

    # Run full benchmark
    benchmark_results = run_full_benchmark(
        dataset_sizes=[1000, 10000],
        dims=[32, 128]
    )

    # Run ef sensitivity test
    ef_results = run_ef_sensitivity(num_vectors=10000, dim=128)

    # Save results
    all_results = {
        'full_benchmark': benchmark_results,
        'ef_sensitivity': ef_results
    }
    save_results(all_results)
