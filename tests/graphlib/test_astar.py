"""Tests unitarios de ``graphlib.algorithms.astar``.

Cubre: equivalencia con Dijkstra (heurística ``None`` y heurística nula
explícita), heurística admissible (distancia Manhattan en rejilla 2D) con
garantía de optimalidad, errores de alcanzabilidad y de nodos inexistentes,
y el caso degenerado ``source == target``.
"""

from __future__ import annotations

import pytest

from graphlib.algorithms import astar, dijkstra, shortest_path
from graphlib.exceptions import InvalidNodeError, NoPathError
from graphlib.graph import DirectedGraph


def _grafo_atajo() -> DirectedGraph:
    """Mismo grafo que los tests de Dijkstra: atajo caro A->C (5) frente a
    A->B->C (2)."""
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", 1.0)
    g.add_edge("A", "C", 5.0)
    return g


def _pesos(g: DirectedGraph) -> dict:
    return {(u, v): w for u, v, w in g.edges()}


def _costo(g: DirectedGraph, path: list) -> float:
    """Coste total de ``path``; asume que cada arista consecutiva existe."""
    pesos = _pesos(g)
    total = 0.0
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v), f"falta la arista {u!r} -> {v!r}"
        total += pesos[(u, v)]
    return total


def _rejilla_3x3() -> DirectedGraph:
    """Rejilla 2D 3x3 con aristas bidireccionales de peso 1 entre vecinos
    ortogonales. Nodos son tuplas ``(fila, col)``."""
    g = DirectedGraph()
    for r in range(3):
        for c in range(3):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 3 and 0 <= c2 < 3:
                    g.add_edge((r, c), (r2, c2), 1.0)
    return g


# ---------------------------------------------------------------------------
# Equivalencia con Dijkstra
# ---------------------------------------------------------------------------

def test_mismo_camino_que_dijkstra():
    g = _grafo_atajo()
    assert astar(g, "A", "C") == shortest_path(g, "A", "C")
    assert astar(g, "A", "C") == ["A", "B", "C"]


def test_heuristica_nula_por_defecto_igual_dijkstra():
    """Con ``heuristic=None`` A* debe degenerar exactamente en Dijkstra."""
    g = _grafo_atajo()
    assert astar(g, "A", "C") == shortest_path(g, "A", "C")


def test_heuristica_nula_explicita_igual_dijkstra():
    g = _grafo_atajo()
    assert astar(g, "A", "C", heuristic=lambda n: 0.0) == shortest_path(
        g, "A", "C"
    )


def test_heuristica_admissible_devuelve_camino_optimo_en_rejilla():
    """Distancia Manhattan en rejilla con pesos 1 es admissible: A* debe
    devolver un camino cuyo coste total sea el óptimo (4 en 3x3 de esquina a
    esquina), aunque el camino concreto pueda diferir por empates."""
    g = _rejilla_3x3()
    src, tgt = (0, 0), (2, 2)
    h = lambda n: abs(n[0] - tgt[0]) + abs(n[1] - tgt[1])
    path = astar(g, src, tgt, heuristic=h)
    assert path[0] == src
    assert path[-1] == tgt
    # El coste óptimo es la distancia Manhattan: 4.
    assert _costo(g, path) == pytest.approx(4.0)
    # Y coincide con la distancia óptima de Dijkstra.
    assert dijkstra(g, src, tgt)[tgt] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Errores y casos degenerados
# ---------------------------------------------------------------------------

def test_target_inexistente_lanza_nopatherror():
    """Un destino ausente es inalcanzable: NoPathError, no InvalidNodeError."""
    g = _grafo_atajo()
    with pytest.raises(NoPathError):
        astar(g, "A", "Z")


def test_target_inalcanzable_lanza_nopatherror():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_node("C")
    with pytest.raises(NoPathError):
        astar(g, "A", "C")


def test_source_inexistente_lanza_invalidnodeerror():
    g = _grafo_atajo()
    with pytest.raises(InvalidNodeError):
        astar(g, "X", "C")


def test_source_igual_target_devuelve_lista_trivial():
    g = _grafo_atajo()
    assert astar(g, "A", "A") == ["A"]
