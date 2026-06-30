# HNSW Vector Search Engine

A from-scratch Python implementation of **Hierarchical Navigable Small World (HNSW)** — the approximate nearest neighbor algorithm powering Pinecone, Weaviate, and FAISS. Built to understand the algorithm at the implementation level, not as a wrapper around existing libraries.

---

## What is HNSW?

HNSW organizes vectors in a multi-layer graph where higher layers act as "express lanes" for navigation. A query starts at the top layer, greedily descends toward the target region, then does fine-grained search at layer 0. This achieves **O(log n)** average query complexity vs O(n) for brute force.

```
Layer 2:  [A]──────────────[E]          ← few nodes, coarse navigation
Layer 1:  [A]───[B]───[C]───[D]───[E]   ← medium connectivity
Layer 0:  [A]─[B]─[C]─[D]─[E]─[F]─[G]  ← all nodes, fine-grained search
               ↑
           query enters here after descending
```

---

## Implementation

**Core algorithm** (`hnsw.py`):
- Multi-layer graph with exponential level assignment (`level = floor(-ln(uniform) * 1/ln(2))`)
- Greedy layer-by-layer descent during both insertion and search
- Bidirectional link maintenance with neighbor pruning to enforce `max_m` connectivity
- Two-heap search: `candidates` min-heap (explore closest first) + `w` max-heap (track ef-best results, prune furthest)
- Euclidean and cosine distance metrics

**Brute force baseline** (`hnsw.py`) for recall measurement.

**Benchmark harness** (`benchmark.py`) for recall@k, query latency, and ef sensitivity.

---

## Results

> Pure Python + NumPy implementation. Recall matches theoretical expectations for HNSW; speedup numbers are limited by Python overhead and would be orders of magnitude higher in a C++ implementation (e.g. FAISS).

### Recall vs ef (1,000 vectors, 128-dim, cosine)

| ef | Recall@10 |
|----|-----------|
| 50 | 99.0% |
| 100 | 99.8% |
| 200 | **100.0%** |
| 500 | 100.0% |

ef sensitivity works as expected — increasing ef monotonically improves recall with a latency tradeoff.

### Recall vs Dataset Size (ef=200, 128-dim)

| Vectors | Recall@10 | HNSW Latency | BF Latency |
|---------|-----------|--------------|------------|
| 1,000 | 100.0% | ~5ms | ~3ms |
| 10,000 | 92.8% | ~19ms | ~37ms |

At 10K vectors HNSW is ~2x faster than brute force in Python; gap widens significantly at larger scales and in compiled implementations.

---

## Usage

```python
import numpy as np
from hnsw import HNSW, BruteForceSearch

# Build index
index = HNSW(dim=128, max_m=16, ef_construction=200, distance_metric='cosine')

vectors = np.random.randn(1000, 128).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

for i, v in enumerate(vectors):
    index.add(i, v)

# Search
query = np.random.randn(128).astype(np.float32)
query /= np.linalg.norm(query)

results = index.search(query, k=10, ef=200)
# [(label, distance), ...]
```

**Parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `max_m` | 16 | Max connections per node. Higher → better recall, more memory |
| `ef_construction` | 200 | Search width during index build. Higher → better graph quality, slower build |
| `ef` (search) | 200 | Search width at query time. Higher → better recall, slower query |
| `distance_metric` | `'euclidean'` | `'euclidean'` or `'cosine'` |

---

## Key Design Decisions

**Two-heap search layer:** `candidates` uses a min-heap `(dist, node)` to always explore the closest unvisited node first. `w` uses a max-heap `(-dist, node)` so `w[0]` is always the furthest current result, enabling O(log ef) pruning. Termination fires when the closest remaining candidate is farther than the worst result in `w`.

**Insertion order:** For each new node, the algorithm checks for anomaly before updating the EWMA baseline — so spikes are detected against the old baseline, not absorbed into it before triggering.

**Bidirectional pruning:** When adding a neighbor link, if the neighbor's connection list exceeds `max_m`, it prunes to the `max_m` nearest by distance. This keeps the graph navigable under heavy insertion load.

---

## Installation

```bash
pip install numpy
python benchmark.py   # run recall + latency benchmarks
python example.py     # usage examples
```

No external dependencies beyond NumPy.

---

## Limitations

This is a **reference implementation** for understanding the algorithm, not a production system:

- Pure Python — no SIMD, no parallelism, no memory-mapped files
- No persistence (index lives in memory)
- No deletion support
- For production use: [FAISS](https://github.com/facebookresearch/faiss), [hnswlib](https://github.com/nmslib/hnswlib), [Weaviate](https://weaviate.io/)
