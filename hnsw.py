import heapq
import math
import random
from typing import List, Tuple, Set, Dict, Optional
import numpy as np


class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) implementation.

    A fast approximate nearest neighbor search algorithm that organizes data points
    in a hierarchical graph structure for efficient similarity queries.
    """

    def __init__(self, dim: int, max_m: int = 16, ef_construction: int = 200,
                 seed: int = 42, distance_metric: str = 'euclidean'):
        """
        Initialize HNSW index.

        Args:
            dim: Dimensionality of vectors
            max_m: Maximum number of connections per node (default 16)
            ef_construction: Search width during construction (default 200)
            seed: Random seed for reproducibility
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        self.dim = dim
        self.max_m = max_m
        self.ef_construction = ef_construction
        self.ef = ef_construction
        self.distance_metric = distance_metric
        self.ml = 1.0 / math.log(2.0)

        random.seed(seed)
        np.random.seed(seed)

        # Data structures
        self.data: Dict[int, np.ndarray] = {}  # label -> vector
        self.graph: Dict[int, Dict[int, Set[int]]] = {}  # level -> {node -> neighbors}
        self.node_levels: Dict[int, int] = {}  # node -> max level
        self.entry_point: Optional[int] = None

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        if self.distance_metric == 'euclidean':
            return float(np.linalg.norm(a - b))
        elif self.distance_metric == 'cosine':
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return float('inf')
            return float(1.0 - np.dot(a, b) / (norm_a * norm_b))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _get_random_level(self) -> int:
        """Randomly determine level for a new node."""
        return int(-math.log(random.uniform(0, 1)) * self.ml)

    def add(self, label: int, vector: np.ndarray) -> None:
        """Add a new vector to the index."""
        if label in self.data:
            raise ValueError(f"Label {label} already exists")

        vector = np.array(vector, dtype=np.float32)
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match {self.dim}")

        self.data[label] = vector

        if self.entry_point is None:
            self.entry_point = label
            self.node_levels[label] = 0
            self.graph[0] = {label: set()}
            return

        # Determine layer for new element
        level = self._get_random_level()
        self.node_levels[label] = level

        # Initialize graph structure for new node
        for lc in range(level + 1):
            if lc not in self.graph:
                self.graph[lc] = {}
            self.graph[lc][label] = set()

        # Search for neighbors from top to layer 0
        nearest = [self.entry_point]
        for lc in range(max(self.node_levels[self.entry_point], level), level, -1):
            nearest = self._search_layer(vector, nearest, 1, lc)

        # Insert element from level to 0
        for lc in range(level, -1, -1):
            candidates = self._search_layer(vector, nearest, self.ef_construction, lc)

            # Select M neighbors
            m = self.max_m if lc > 0 else self.max_m * 2
            neighbors = self._get_neighbors(vector, candidates, m)

            # Add bidirectional links
            for neighbor in neighbors:
                self.graph[lc][label].add(neighbor)
                # Ensure neighbor exists in this layer's graph
                if neighbor not in self.graph[lc]:
                    self.graph[lc][neighbor] = set()
                self.graph[lc][neighbor].add(label)

                # Prune neighbors of neighbor if needed
                max_conn = self.max_m if lc > 0 else self.max_m * 2
                if len(self.graph[lc][neighbor]) > max_conn:
                    prune_list = self._get_neighbors(
                        self.data[neighbor],
                        list(self.graph[lc][neighbor]),
                        max_conn
                    )
                    self.graph[lc][neighbor] = set(prune_list)

            nearest = neighbors

        # Update entry point if new node is on higher level
        if level > self.node_levels[self.entry_point]:
            self.entry_point = label

    def _search_layer(self, query: np.ndarray, entry_points: List[int],
                     ef: int, layer: int) -> List[int]:
        """Search for nearest neighbors on a specific layer."""
        visited = set()
        candidates = []
        w = []

        for ep in entry_points:
            dist = self._distance(query, self.data[ep])
            heapq.heappush(candidates, (-dist, ep))
            heapq.heappush(w, (-dist, ep))
            visited.add(ep)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > -w[0][0]:
                break

            for neighbor in self.graph[layer].get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query, self.data[neighbor])

                    if dist < -w[0][0] or len(w) < ef:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (-dist, neighbor))

                        if len(w) > ef:
                            heapq.heappop(w)

        return [item[1] for item in w]

    def _get_neighbors(self, vector: np.ndarray, candidates: List[int],
                      m: int) -> List[int]:
        """Select M nearest neighbors using a heuristic."""
        distances = [(self._distance(vector, self.data[c]), c) for c in candidates]
        distances.sort()
        return [c for _, c in distances[:m]]

    def search(self, query: np.ndarray, k: int = 10, ef: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector
            k: Number of neighbors to return
            ef: Search width (default uses ef_construction)

        Returns:
            List of (label, distance) tuples
        """
        if self.entry_point is None:
            return []

        if ef is None:
            ef = self.ef_construction

        query = np.array(query, dtype=np.float32)

        # Search from top layer to target layer
        nearest = [self.entry_point]
        for layer in range(self.node_levels[self.entry_point], 0, -1):
            nearest = self._search_layer(query, nearest, 1, layer)

        # Search at layer 0
        candidates = self._search_layer(query, nearest, ef, 0)

        # Return k nearest neighbors
        results = [(label, self._distance(query, self.data[label])) for label in candidates]
        results.sort(key=lambda x: x[1])
        return results[:k]

    def size(self) -> int:
        """Return number of vectors in index."""
        return len(self.data)

    def memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage in bytes."""
        data_size = sum(v.nbytes for v in self.data.values())

        graph_size = 0
        for level_dict in self.graph.values():
            for neighbors in level_dict.values():
                graph_size += len(neighbors) * 8  # 8 bytes per integer reference

        return {
            'vectors': data_size,
            'graph': graph_size,
            'metadata': len(self.data) * 16 + len(self.node_levels) * 16,
            'total': data_size + graph_size + len(self.data) * 16 + len(self.node_levels) * 16
        }


class BruteForceSearch:
    """Brute force exact nearest neighbor search for comparison."""

    def __init__(self, dim: int, distance_metric: str = 'euclidean'):
        self.dim = dim
        self.data: Dict[int, np.ndarray] = {}
        self.distance_metric = distance_metric

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance_metric == 'euclidean':
            return float(np.linalg.norm(a - b))
        elif self.distance_metric == 'cosine':
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return float('inf')
            return float(1.0 - np.dot(a, b) / (norm_a * norm_b))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def add(self, label: int, vector: np.ndarray) -> None:
        vector = np.array(vector, dtype=np.float32)
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch")
        self.data[label] = vector

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        query = np.array(query, dtype=np.float32)
        results = [(label, self._distance(query, vector)) for label, vector in self.data.items()]
        results.sort(key=lambda x: x[1])
        return results[:k]

    def size(self) -> int:
        return len(self.data)
