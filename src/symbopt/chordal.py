from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Set, Tuple
import numpy as np


@dataclass
class SparsityGraph:
    n: int
    edges: Set[Tuple[int, int]]

    def neighbors(self, i: int) -> Set[int]:
        out = set()
        for a, b in self.edges:
            if a == i:
                out.add(b)
            elif b == i:
                out.add(a)
        return out


@dataclass
class ChordalDecomposition:
    n: int
    cliques: List[List[int]]
    separators: List[List[int]]
    message: str

    def pretty(self) -> str:
        return "\n".join(
            [
                f"n: {self.n}",
                f"cliques: {self.cliques}",
                f"separators: {self.separators}",
                f"message: {self.message}",
            ]
        )


def matrix_sparsity_graph(M: np.ndarray, tol: float = 1e-12) -> SparsityGraph:
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(M[i, j]) > tol or abs(M[j, i]) > tol:
                edges.add((i, j))
    return SparsityGraph(n=n, edges=edges)


def _mcs_order(graph: SparsityGraph) -> List[int]:
    weights = {i: 0 for i in range(graph.n)}
    rem = set(range(graph.n))
    order = []
    while rem:
        v = max(rem, key=lambda x: (weights[x], x))
        rem.remove(v)
        order.append(v)
        for u in graph.neighbors(v):
            if u in rem:
                weights[u] += 1
    return order


def chordal_extension(graph: SparsityGraph) -> ChordalDecomposition:
    order = _mcs_order(graph)
    pos = {v: i for i, v in enumerate(order)}
    adj = {i: set(graph.neighbors(i)) for i in range(graph.n)}
    fill = set(graph.edges)
    for v in order:
        later = [u for u in adj[v] if pos[u] > pos[v]]
        for a, b in combinations(later, 2):
            e = (min(a, b), max(a, b))
            fill.add(e)
            adj[a].add(b)
            adj[b].add(a)
    full = {i: set() for i in range(graph.n)}
    for a, b in fill:
        full[a].add(b)
        full[b].add(a)
    cliques = []
    for v in order:
        clique = sorted([v] + [u for u in full[v] if pos[u] > pos[v]])
        if not any(set(clique).issubset(set(c)) for c in cliques):
            cliques.append(clique)
    seps = []
    for i in range(1, len(cliques)):
        seps.append(
            sorted(set(cliques[i]).intersection(set().union(*map(set, cliques[:i]))))
        )
    return ChordalDecomposition(graph.n, cliques, seps, "Heuristic chordal extension.")


def suggest_psd_blocks(M: np.ndarray, tol: float = 1e-12) -> ChordalDecomposition:
    return chordal_extension(matrix_sparsity_graph(M, tol=tol))
