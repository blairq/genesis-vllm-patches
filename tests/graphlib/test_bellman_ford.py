"""Tests unitarios de ``graphlib.algorithms.bellman_ford``.

Cubre: pesos negativos sin ciclo negativo (donde Dijkstra fallaría),
detección de ciclos negativos alcanzables (con identificación del nodo
implicado) y no alcanzables, reconstrucción de caminos con pesos negativos,
errores de alcanzabilidad y de nodos inexistentes, y una propiedad
hipotética de coincidencia con Dijkstra en grafos aleatorios de pesos
positivos.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphlib.algorithms import bellman_ford, bellman_ford_path, dijkstra
from graphlib.exceptions import (
    InvalidNodeError,
    NegativeCycleError,
    NoPathError,
    ValidationError,
)
from graphlib.graph import DirectedGraph


def _grafo_pesos_negativos() -> DirectedGraph:
    """A->B (4), B->C (-2), A->C (5): la ruta óptima A->C pasa por B con
    coste 2, gracias al peso negativo. Sin ciclos negativos."""
    g = DirectedGraph()
    g.add_edge("A", "B", 4.0)
    g.add_edge("B", "C", -2.0)
    g.add_edge("A", "C", 5.0)
    return g


def _grafo_ciclo_negativo_alcanzable() -> DirectedGraph:
    """Ciclo B->C->B de peso -2, alcanzable desde A."""
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", -3.0)
    g.add_edge("C", "B", 1.0)
    return g


def _grafo_ciclo_negativo_no_alcanzable() -> DirectedGraph:
    """Ciclo C->D->C de peso -2, pero sin aristas desde A hacia él."""
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("C", "D", -1.0)
    g.add_edge("D", "C", -1.0)
    return g


# ---------------------------------------------------------------------------
# Pesos negativos sin ciclo
# ---------------------------------------------------------------------------

def test_pesos_negativos_sin_ciclo_negativo():
    g = _grafo_pesos_negativos()
    dist = bellman_ford(g, "A")
    assert dist["A"] == 0.0
    assert dist["B"] == pytest.approx(4.0)
    # La arista negativa hace que A->B->C (2) gane a A->C (5).
    assert dist["C"] == pytest.approx(2.0)


def test_pesos_negativos_donde_dijkstra_fallaria():
    """El mismo grafo hace que Dijkstra rechace la entrada por la
    pre-condición de pesos no negativos."""
    g = _grafo_pesos_negativos()
    with pytest.raises(ValidationError):
        dijkstra(g, "A")


# ---------------------------------------------------------------------------
# Ciclos negativos
# ---------------------------------------------------------------------------

def test_ciclo_negativo_alcanzable_lanza_negativecycleerror():
    g = _grafo_ciclo_negativo_alcanzable()
    with pytest.raises(NegativeCycleError) as exc_info:
        bellman_ford(g, "A")
    # El atributo .node debe identificar un nodo del ciclo {B, C}.
    assert exc_info.value.node in {"B", "C"}


def test_ciclo_negativo_no_alcanzable_no_lanza():
    g = _grafo_ciclo_negativo_no_alcanzable()
    dist = bellman_ford(g, "A")
    assert dist == {"A": 0.0, "B": 1.0}


# ---------------------------------------------------------------------------
# Reconstrucción de caminos
# ---------------------------------------------------------------------------

def test_bellman_ford_path_con_pesos_negativos():
    g = _grafo_pesos_negativos()
    assert bellman_ford_path(g, "A", "C") == ["A", "B", "C"]
    assert bellman_ford_path(g, "A", "A") == ["A"]


def test_bellman_ford_path_target_inalcanzable_lanza_nopatherror():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_node("C")
    with pytest.raises(NoPathError):
        bellman_ford_path(g, "A", "C")


def test_bellman_ford_path_target_inexistente_lanza_nopatherror():
    g = _grafo_pesos_negativos()
    with pytest.raises(NoPathError):
        bellman_ford_path(g, "A", "Z")


def test_source_inexistente_lanza_invalidnodeerror():
    g = _grafo_pesos_negativos()
    with pytest.raises(InvalidNodeError):
        bellman_ford(g, "X")
    with pytest.raises(InvalidNodeError):
        bellman_ford_path(g, "X", "C")


# ---------------------------------------------------------------------------
# Coincidencia con Dijkstra en pesos positivos
# ---------------------------------------------------------------------------

def test_coincide_con_dijkstra_en_pesos_positivos():
    g = DirectedGraph()
    g.add_edge("A", "B", 2.0)
    g.add_edge("B", "C", 3.0)
    g.add_edge("A", "C", 10.0)
    assert bellman_ford(g, "A") == pytest.approx(dijkstra(g, "A"))


@settings(max_examples=50, deadline=None)
@given(st.integers(1, 12))
def test_bf_coincide_dijkstra_grafo_aleatorio_pesos_positivos(n):
    """En un grafo aleatorio con pesos positivos, Bellman-Ford y Dijkstra
    deben devolver exactamente las mismas distancias (mismos alcanzables,
    mismos valores)."""
    rng = random.Random(n)  # determinista por n
    g = DirectedGraph()
    for u in range(n):
        g.add_node(u)
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < 0.3:
                g.add_edge(u, v, rng.uniform(0.1, 10.0))
    bf = bellman_ford(g, 0)
    dj = dijkstra(g, 0)
    assert set(bf) == set(dj)
    for nodo in dj:
        assert bf[nodo] == pytest.approx(dj[nodo])
