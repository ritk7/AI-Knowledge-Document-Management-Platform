"""A from-scratch HNSW (Hierarchical Navigable Small World) index.

Implements the algorithm from Malkov & Yashunin (2016), "Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small World
graphs" (arXiv:1603.09320). No ANN libraries are used -- only numpy, for the
distance kernel.

Structure
---------
The index is a stack of proximity graphs. Every node lives in layer 0; a node
is promoted to higher layers with exponentially decaying probability, so the
top layers are sparse and act as long-range "express lanes". A search starts at
the single entry point on the top layer, greedily descends to the node's local
neighbourhood, then runs a wider beam search on layer 0 where all nodes live.

The two routines that matter:

* ``_search_layer``  -- best-first beam search within one layer, bounded by
  ``ef``. Larger ``ef`` explores more of the graph: higher recall, more time.
* ``_select_neighbors_heuristic`` -- Algorithm 4 of the paper. When choosing
  which ``m`` of the candidates to link to, keep a candidate only if it is
  closer to the new node than to any already-selected neighbour. This preserves
  long-range edges instead of letting every node link to the same dense cluster,
  and is the single biggest lever on recall.

Distances are cosine: vectors are L2-normalised on insert, so
``distance = 1 - dot(a, b)``, which is monotone in true cosine distance.
"""

from __future__ import annotations

import heapq
import math
import pickle
import random
from pathlib import Path

import numpy as np


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        """
        Args:
            dim: vector dimensionality.
            M: target out-degree for layers >= 1. Layer 0 allows 2*M, since it
                holds every node and needs the extra connectivity.
            ef_construction: beam width while inserting. Higher builds a better
                graph, more slowly.
            ef_search: default beam width at query time. Must be >= k.
            seed: seed for the level-assignment RNG, so builds are reproducible.
        """
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Level-generation normalisation factor from the paper: mL = 1/ln(M).
        self._level_mult = 1.0 / math.log(M) if M > 1 else 1.0
        self._rng = random.Random(seed)

        # Vectors are kept in one contiguous array, grown by doubling, so the
        # distance kernel stays a single numpy dot.
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._size = 0

        # _neighbors[node][level] -> list of node ids.
        self._neighbors: list[list[list[int]]] = []
        self._levels: list[int] = []
        self._labels: list[str] = []

        self._entry_point: int | None = None
        self._top_level = -1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        """Sample a level from the exponentially decaying distribution."""
        return int(-math.log(self._rng.random()) * self._level_mult)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._vectors.shape[0]:
            return
        new_capacity = max(16, self._vectors.shape[0] * 2, needed)
        grown = np.zeros((new_capacity, self.dim), dtype=np.float32)
        grown[: self._size] = self._vectors[: self._size]
        self._vectors = grown

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm

    def _distance_to(self, query: np.ndarray, node: int) -> float:
        return 1.0 - float(np.dot(query, self._vectors[node]))

    def _distance_between(self, a: int, b: int) -> float:
        return 1.0 - float(np.dot(self._vectors[a], self._vectors[b]))

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """Beam search within a single layer.

        Returns up to ``ef`` (distance, node) pairs, unsorted.
        """
        visited: set[int] = set(entry_points)
        # Min-heap of frontier nodes, ordered by distance to the query.
        candidates: list[tuple[float, int]] = []
        # Max-heap (negated) of the best results found so far, capped at ef.
        results: list[tuple[float, int]] = []

        for ep in entry_points:
            d = self._distance_to(query, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)
            # Everything left in the frontier is further than our current worst
            # result, so no unexplored node can improve the answer.
            if dist_c > -results[0][0]:
                break

            for neighbor in self._neighbors[c][level]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dist_n = self._distance_to(query, neighbor)
                if len(results) < ef or dist_n < -results[0][0]:
                    heapq.heappush(candidates, (dist_n, neighbor))
                    heapq.heappush(results, (-dist_n, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-neg_d, node) for neg_d, node in results]

    def _select_neighbors_heuristic(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Algorithm 4: diversity-preserving neighbour selection.

        Walk candidates nearest-first and keep one only if it is closer to the
        query point than to every neighbour already selected. This drops
        candidates that sit "behind" an existing neighbour, which is what keeps
        long-range edges in the graph.
        """
        selected: list[int] = []
        for dist, cand in sorted(candidates):
            if len(selected) >= m:
                break
            if all(self._distance_between(cand, s) >= dist for s in selected):
                selected.append(cand)

        # If the heuristic was too strict to fill the quota, top up with the
        # nearest remaining candidates so the node is never under-connected.
        if len(selected) < m:
            for _, cand in sorted(candidates):
                if len(selected) >= m:
                    break
                if cand not in selected:
                    selected.append(cand)

        return selected

    def _prune_connections(self, node: int, level: int) -> None:
        """Re-apply the selection heuristic when a node exceeds its degree cap."""
        max_degree = self.M0 if level == 0 else self.M
        neighbors = self._neighbors[node][level]
        if len(neighbors) <= max_degree:
            return
        candidates = [(self._distance_between(node, n), n) for n in neighbors]
        self._neighbors[node][level] = self._select_neighbors_heuristic(
            candidates, max_degree
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, label: str | None = None) -> int:
        """Insert one vector. Returns its internal node id."""
        query = self._normalize(vector)
        node = self._size

        self._ensure_capacity(node + 1)
        self._vectors[node] = query
        self._size += 1

        level = self._random_level()
        self._levels.append(level)
        self._labels.append(label if label is not None else str(node))
        self._neighbors.append([[] for _ in range(level + 1)])

        # First node: it simply becomes the entry point.
        if self._entry_point is None:
            self._entry_point = node
            self._top_level = level
            return node

        ep = [self._entry_point]

        # Phase 1: greedy descent through layers above this node's own level.
        for lc in range(self._top_level, level, -1):
            found = self._search_layer(query, ep, ef=1, level=lc)
            ep = [min(found)[1]]

        # Phase 2: from the node's level down to 0, find neighbours and link.
        for lc in range(min(level, self._top_level), -1, -1):
            found = self._search_layer(query, ep, self.ef_construction, lc)
            max_degree = self.M0 if lc == 0 else self.M
            chosen = self._select_neighbors_heuristic(found, max_degree)

            self._neighbors[node][lc] = list(chosen)
            for neighbor in chosen:
                self._neighbors[neighbor][lc].append(node)
                self._prune_connections(neighbor, lc)

            ep = [n for _, n in found]

        # A node sampled above the current ceiling becomes the new entry point.
        if level > self._top_level:
            self._entry_point = node
            self._top_level = level

        return node

    def add_batch(self, vectors: np.ndarray, labels: list[str] | None = None) -> None:
        for i, vector in enumerate(vectors):
            self.add(vector, labels[i] if labels else None)

    def search(
        self, query: np.ndarray, k: int = 10, ef: int | None = None
    ) -> list[tuple[str, float]]:
        """Return the k approximate nearest neighbours as (label, distance)."""
        if self._entry_point is None:
            return []

        query = self._normalize(query)
        ef = max(ef or self.ef_search, k)

        ep = [self._entry_point]
        # Greedy descent through the sparse upper layers.
        for lc in range(self._top_level, 0, -1):
            found = self._search_layer(query, ep, ef=1, level=lc)
            ep = [min(found)[1]]

        # Wide beam search on layer 0, where every node lives.
        found = self._search_layer(query, ep, ef, level=0)
        found.sort()
        return [(self._labels[node], dist) for dist, node in found[:k]]

    def __len__(self) -> int:
        return self._size

    def stats(self) -> dict:
        """Graph shape, for the benchmark's memory and connectivity reporting."""
        edges = sum(
            len(per_level) for node in self._neighbors for per_level in node
        )
        return {
            "num_nodes": self._size,
            "num_edges": edges,
            "top_level": self._top_level,
            "avg_degree_layer0": (
                sum(len(n[0]) for n in self._neighbors) / self._size
                if self._size
                else 0.0
            ),
            "vector_bytes": self._size * self.dim * 4,
            # Each edge is a Python int in a list: 8 bytes for the pointer plus
            # ~28 for the int object. Approximate, but the right order.
            "graph_bytes_approx": edges * 36,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "dim": self.dim,
                    "M": self.M,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "vectors": self._vectors[: self._size],
                    "neighbors": self._neighbors,
                    "levels": self._levels,
                    "labels": self._labels,
                    "entry_point": self._entry_point,
                    "top_level": self._top_level,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "HNSWIndex":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        index = cls(
            dim=state["dim"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
        )
        vectors = state["vectors"]
        index._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index._size = vectors.shape[0]
        index._neighbors = state["neighbors"]
        index._levels = state["levels"]
        index._labels = state["labels"]
        index._entry_point = state["entry_point"]
        index._top_level = state["top_level"]
        return index


class BruteForceIndex:
    """Exact search baseline. Used to produce ground truth for recall@k."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._labels: list[str] = []

    def add_batch(self, vectors: np.ndarray, labels: list[str] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = np.vstack([self._vectors, vectors / norms])
        start = len(self._labels)
        self._labels.extend(
            labels if labels else [str(start + i) for i in range(len(vectors))]
        )

    def memory_bytes(self) -> int:
        """Exact footprint: the vector matrix plus the label list."""
        return self._vectors.nbytes + sum(
            len(label) + 49 for label in self._labels  # 49B str object overhead
        )

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        query = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm
        distances = 1.0 - self._vectors @ query
        # argpartition is O(n) vs O(n log n) for a full sort.
        k = min(k, len(distances))
        top = np.argpartition(distances, k - 1)[:k]
        top = top[np.argsort(distances[top])]
        return [(self._labels[i], float(distances[i])) for i in top]

    def __len__(self) -> int:
        return len(self._labels)
