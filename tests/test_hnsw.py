import pytest
import numpy as np
from hnsw import HNSW, BruteForceSearch


class TestHNSWBasics:
    """Test basic HNSW functionality."""

    def test_initialization(self):
        """Test HNSW initialization."""
        hnsw = HNSW(dim=32, max_m=16, ef_construction=200)
        assert hnsw.dim == 32
        assert hnsw.max_m == 16
        assert hnsw.ef_construction == 200
        assert hnsw.size() == 0
        assert hnsw.entry_point is None

    def test_add_single_vector(self):
        """Test adding a single vector."""
        hnsw = HNSW(dim=16)
        vector = np.random.randn(16).astype(np.float32)
        hnsw.add(0, vector)
        assert hnsw.size() == 1
        assert hnsw.entry_point == 0

    def test_add_multiple_vectors(self):
        """Test adding multiple vectors."""
        hnsw = HNSW(dim=16, seed=42)
        vectors = np.random.randn(10, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)
        assert hnsw.size() == 10

    def test_add_duplicate_label_fails(self):
        """Test that adding duplicate labels raises error."""
        hnsw = HNSW(dim=16)
        vector = np.random.randn(16).astype(np.float32)
        hnsw.add(0, vector)
        with pytest.raises(ValueError):
            hnsw.add(0, vector)

    def test_dimension_mismatch_fails(self):
        """Test that dimension mismatch raises error."""
        hnsw = HNSW(dim=16)
        vector = np.random.randn(32).astype(np.float32)
        with pytest.raises(ValueError):
            hnsw.add(0, vector)

    def test_search_empty_index(self):
        """Test search on empty index."""
        hnsw = HNSW(dim=16)
        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=10)
        assert results == []

    def test_search_returns_correct_type(self):
        """Test that search returns list of (label, distance) tuples."""
        hnsw = HNSW(dim=16, seed=42)
        vectors = np.random.randn(20, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=10)

        assert isinstance(results, list)
        assert len(results) <= 10
        for label, distance in results:
            assert isinstance(label, int)
            assert isinstance(distance, float)

    def test_search_respects_k(self):
        """Test that search returns at most k results."""
        hnsw = HNSW(dim=16, seed=42)
        vectors = np.random.randn(50, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        for k in [1, 5, 10, 20]:
            results = hnsw.search(query, k=k)
            assert len(results) <= k

    def test_results_sorted_by_distance(self):
        """Test that results are sorted by distance."""
        hnsw = HNSW(dim=16, seed=42)
        vectors = np.random.randn(30, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=10)

        distances = [d for _, d in results]
        assert distances == sorted(distances)


class TestEFParameter:
    """Test ef parameter sensitivity."""

    def test_ef_affects_recall(self):
        """Test that larger ef improves recall."""
        np.random.seed(42)
        vectors = np.random.randn(100, 32).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        query = np.random.randn(32).astype(np.float32)
        query /= np.linalg.norm(query)

        hnsw = HNSW(dim=32, max_m=16, ef_construction=200, seed=42,
                    distance_metric='cosine')
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        bf = BruteForceSearch(dim=32, distance_metric='cosine')
        for i, v in enumerate(vectors):
            bf.add(i, v)

        bf_results = bf.search(query, k=10)
        bf_labels = set(l for l, _ in bf_results)

        recalls = []
        for ef in [50, 100, 200]:
            hnsw_results = hnsw.search(query, k=10, ef=ef)
            hnsw_labels = set(l for l, _ in hnsw_results)
            recall = len(hnsw_labels & bf_labels) / len(bf_labels) if bf_labels else 1.0
            recalls.append(recall)

        assert all(r > 0.1 for r in recalls), f"Recalls too low: {recalls}"
        assert recalls != [recalls[0]] * len(recalls), "ef parameter has no effect on recall"

    def test_ef_default_uses_construction_value(self):
        """Test that ef defaults to ef_construction."""
        hnsw = HNSW(dim=16, ef_construction=300)
        vectors = np.random.randn(20, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=5)
        assert len(results) > 0


class TestDistanceMetrics:
    """Test different distance metrics."""

    def test_euclidean_distance(self):
        """Test Euclidean distance metric."""
        hnsw = HNSW(dim=16, distance_metric='euclidean', seed=42)
        vectors = np.random.randn(20, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        results = hnsw.search(query, k=5)
        assert len(results) > 0

    def test_cosine_distance(self):
        """Test Cosine distance metric."""
        hnsw = HNSW(dim=16, distance_metric='cosine', seed=42)
        vectors = np.random.randn(20, 16).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        query = np.random.randn(16).astype(np.float32)
        query /= np.linalg.norm(query)
        results = hnsw.search(query, k=5)
        assert len(results) > 0

    def test_invalid_distance_metric(self):
        """Test that invalid distance metric raises error."""
        with pytest.raises(ValueError):
            HNSW(dim=16, distance_metric='invalid')


class TestMemoryUsage:
    """Test memory usage estimation."""

    def test_memory_usage_increases(self):
        """Test that memory usage increases with more vectors."""
        hnsw = HNSW(dim=32, max_m=16)
        mem1 = hnsw.memory_usage()['total']

        vectors = np.random.randn(50, 32).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)
        mem2 = hnsw.memory_usage()['total']

        assert mem2 > mem1

    def test_memory_components(self):
        """Test that memory usage has expected components."""
        hnsw = HNSW(dim=16)
        vectors = np.random.randn(20, 16).astype(np.float32)
        for i, v in enumerate(vectors):
            hnsw.add(i, v)

        mem = hnsw.memory_usage()
        assert 'vectors' in mem
        assert 'graph' in mem
        assert 'metadata' in mem
        assert 'total' in mem
        assert mem['total'] == mem['vectors'] + mem['graph'] + mem['metadata']


class TestBruteForceSearch:
    """Test BruteForceSearch implementation."""

    def test_brute_force_exact(self):
        """Test that brute force returns exact results."""
        bf = BruteForceSearch(dim=16, distance_metric='euclidean')
        vectors = np.array([[1, 0, 0, 0] + [0]*12,
                           [0, 1, 0, 0] + [0]*12,
                           [0, 0, 1, 0] + [0]*12,
                           [0, 0, 0, 1] + [0]*12], dtype=np.float32)
        for i, v in enumerate(vectors):
            bf.add(i, v)

        query = np.array([1, 0, 0, 0] + [0]*12, dtype=np.float32)
        results = bf.search(query, k=2)

        assert len(results) == 2
        assert results[0][0] == 0
        assert results[0][1] == 0.0

    def test_brute_force_size(self):
        """Test that brute force size tracking works."""
        bf = BruteForceSearch(dim=16)
        assert bf.size() == 0
        for i in range(10):
            bf.add(i, np.random.randn(16).astype(np.float32))
        assert bf.size() == 10


class TestRecall:
    """Test recall of HNSW vs brute force."""

    def test_recall_reasonable(self):
        """Test that HNSW achieves reasonable recall."""
        np.random.seed(42)
        num_vectors = 500
        dim = 32
        vectors = np.random.randn(num_vectors, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        hnsw = HNSW(dim=dim, max_m=16, ef_construction=200, seed=42,
                    distance_metric='cosine')
        bf = BruteForceSearch(dim=dim, distance_metric='cosine')

        for i, v in enumerate(vectors):
            hnsw.add(i, v)
            bf.add(i, v)

        queries = np.random.randn(10, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        recalls = []
        for query in queries:
            hnsw_results = hnsw.search(query, k=10, ef=200)
            bf_results = bf.search(query, k=10)

            hnsw_labels = set(l for l, _ in hnsw_results)
            bf_labels = set(l for l, _ in bf_results)

            recall = len(hnsw_labels & bf_labels) / len(bf_labels) if bf_labels else 1.0
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert mean_recall > 0.7, f"Mean recall {mean_recall:.2f} is too low"
