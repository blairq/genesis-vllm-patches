"""Librería ``graphlib``: grafos ponderados dirigidos y no dirigidos.

Paquete top-level que expone el API público completo de la librería:

- **Representación de grafos** (:mod:`graphlib.graph`):
  :class:`DirectedGraph` y :class:`UndirectedGraph`, ambos ponderados.
- **Excepciones** (:mod:`graphlib.exceptions`): jerarquía de 6 excepciones
  derivadas de :class:`GraphError`.
- **Validadores** (:mod:`graphlib.validators`): comprobaciones de tipos y
  valores para nodos, pesos y conjuntos de nodos.
- **Contratos** (:mod:`graphlib.contracts`): precondiciones reutilizables
  para algoritmos (grafo válido, nodo existente, arista existente, pesos
  no negativos, coherencia de distancias).
- **Algoritmos** (:mod:`graphlib.algorithms`): 6 familias — Dijkstra,
  Bellman-Ford, A*, BFS, DFS, componentes fuertemente conectados (Tarjan)
  y PageRank.
"""

from .algorithms import (
    astar,
    bellman_ford,
    bellman_ford_path,
    bfs,
    bfs_shortest_path,
    dfs,
    dfs_paths,
    dijkstra,
    pagerank,
    shortest_path,
    strongly_connected_components,
)
from .contracts import (
    check_distances,
    require_edge,
    require_graph,
    require_node,
    require_non_negative_weights,
)
from .exceptions import (
    GraphError,
    InvalidEdgeError,
    InvalidNodeError,
    NegativeCycleError,
    NoPathError,
    ValidationError,
)
from .graph import DirectedGraph, UndirectedGraph
from .validators import (
    validate_node,
    validate_nodes,
    validate_non_negative,
    validate_weight,
)

__version__ = "0.1.0"

__all__ = [
    "DirectedGraph",
    "GraphError",
    "InvalidEdgeError",
    "InvalidNodeError",
    "NegativeCycleError",
    "NoPathError",
    "UndirectedGraph",
    "ValidationError",
    "astar",
    "bellman_ford",
    "bellman_ford_path",
    "bfs",
    "bfs_shortest_path",
    "check_distances",
    "dfs",
    "dfs_paths",
    "dijkstra",
    "pagerank",
    "require_edge",
    "require_graph",
    "require_node",
    "require_non_negative_weights",
    "shortest_path",
    "strongly_connected_components",
    "validate_node",
    "validate_nodes",
    "validate_non_negative",
    "validate_weight",
]
