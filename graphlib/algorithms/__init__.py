"""Algoritmos de la librería ``graphlib``.

Subpaquete que agrupa las 6 familias de algoritmos sobre grafos
ponderados: caminos más cortos (Dijkstra, Bellman-Ford, A*), recorridos
(BFS, DFS), componentes fuertemente conectados (Tarjan) y ranking de
páginas (PageRank).

Cada módulo implementa una familia; este ``__init__`` re-exporta de
forma explícita el API público para que los consumidores puedan hacer
``from graphlib.algorithms import dijkstra`` o directamente
``from graphlib import dijkstra`` sin conocer la estructura interna.
"""

from .astar import astar
from .bellman_ford import bellman_ford, bellman_ford_path
from .bfs import bfs, bfs_shortest_path
from .dfs import dfs, dfs_paths
from .dijkstra import dijkstra, shortest_path
from .pagerank import pagerank
from .tarjan import strongly_connected_components

__all__ = [
    "astar",
    "bellman_ford",
    "bellman_ford_path",
    "bfs",
    "bfs_shortest_path",
    "dfs",
    "dfs_paths",
    "dijkstra",
    "pagerank",
    "shortest_path",
    "strongly_connected_components",
]
