# SPDX-License-Identifier: Apache-2.0
"""Tests unitarios del núcleo de ``graphlib.graph``.

Cubre ``DirectedGraph`` y ``UndirectedGraph``: creación, auto-creación de
nodos, conversión de pesos a float, simetría en modo no dirigido,
ordenamiento de ``edges()``, eliminación de nodos/aristas y sus errores,
consultas (``has_node``/``has_edge``/``neighbors``/``predecessors``),
protocolos ``__len__``/``__contains__``/``__repr__`` y la jerarquía de
excepciones. Incluye propiedades de consistencia con Hypothesis.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphlib.exceptions import (
    GraphError,
    InvalidEdgeError,
    InvalidNodeError,
    NoPathError,
    NegativeCycleError,
    ValidationError,
)
from graphlib.graph import DirectedGraph, UndirectedGraph


# ---------------------------------------------------------------------------
# Creación
# ---------------------------------------------------------------------------

def test_directed_graph_creation():
    g = DirectedGraph()
    assert g.directed is True
    assert len(g) == 0
    assert g.nodes() == []
    assert g.edges() == []


def test_undirected_graph_creation():
    g = UndirectedGraph()
    assert g.directed is False
    assert len(g) == 0
    # DirectedGraph(directed=False) es equivalente.
    g2 = DirectedGraph(directed=False)
    assert g2.directed is False
    assert isinstance(g, DirectedGraph)  # UndirectedGraph es subclase


# ---------------------------------------------------------------------------
# add_node / add_edge
# ---------------------------------------------------------------------------

def test_add_node_idempotent():
    g = DirectedGraph()
    g.add_node("a")
    g.add_node("a")
    assert g.nodes() == ["a"]
    assert len(g) == 1


def test_add_edge_autocreates_nodes():
    g = DirectedGraph()
    g.add_edge("a", "b")
    assert g.has_node("a")
    assert g.has_node("b")
    assert g.has_edge("a", "b")
    assert len(g) == 2


def test_add_edge_weight_converted_to_float():
    g = DirectedGraph()
    g.add_edge("a", "b", 2)  # int -> float
    g.add_edge("c", "d", 0.5)
    edges = g.edges()
    assert edges == [("a", "b", 2.0), ("c", "d", 0.5)]
    for _, _, w in edges:
        assert isinstance(w, float)


def test_add_edge_default_weight_is_one():
    g = DirectedGraph()
    g.add_edge("a", "b")
    assert g.edges() == [("a", "b", 1.0)]


def test_add_edge_overwrites_weight_no_multiedges():
    g = DirectedGraph()
    g.add_edge("a", "b", 1.0)
    g.add_edge("a", "b", 9.0)
    assert g.edges() == [("a", "b", 9.0)]


@pytest.mark.parametrize("bad_weight", ["abc", True, False, float("nan"), float("inf"), float("-inf"), None, [1.0]])
def test_add_edge_invalid_weight_raises_validation_error(bad_weight):
    g = DirectedGraph()
    with pytest.raises(ValidationError):
        g.add_edge("a", "b", bad_weight)
    # La arista no debe quedar registrada tras el fallo.
    assert not g.has_edge("a", "b")


# ---------------------------------------------------------------------------
# Simetría en modo no dirigido
# ---------------------------------------------------------------------------

def test_directed_graph_is_asymmetric():
    g = DirectedGraph()
    g.add_edge("a", "b")
    assert g.has_edge("a", "b")
    assert not g.has_edge("b", "a")
    assert g.neighbors("a") == ["b"]
    assert g.neighbors("b") == []
    assert g.predecessors("b") == ["a"]
    assert g.predecessors("a") == []


def test_undirected_has_edge_symmetric():
    g = UndirectedGraph()
    g.add_edge("a", "b")
    assert g.has_edge("a", "b")
    assert g.has_edge("b", "a")


def test_undirected_neighbors_and_predecessors_symmetric():
    g = UndirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    assert sorted(g.neighbors("a")) == ["b", "c"]
    assert g.neighbors("b") == ["a"]
    assert g.neighbors("c") == ["a"]
    # En no dirigido, predecessors == neighbors.
    assert sorted(g.predecessors("a")) == ["b", "c"]
    assert g.predecessors("b") == ["a"]


def test_undirected_remove_edge_both_directions():
    g = UndirectedGraph()
    g.add_edge("a", "b")
    g.remove_edge("a", "b")
    assert not g.has_edge("a", "b")
    assert not g.has_edge("b", "a")
    # También se puede eliminar consultando el sentido contrario.
    g.add_edge("c", "d")
    g.remove_edge("d", "c")
    assert not g.has_edge("c", "d")
    assert not g.has_edge("d", "c")
    # Los nodos siguen existiendo.
    assert g.has_node("a") and g.has_node("b")
    assert g.has_node("c") and g.has_node("d")


def test_undirected_weight_shared_both_directions():
    g = UndirectedGraph()
    g.add_edge("a", "b", 3.5)
    edges = g.edges()
    assert edges == [("a", "b", 3.5)]


# ---------------------------------------------------------------------------
# edges(): ordenado y sin duplicados
# ---------------------------------------------------------------------------

def test_edges_sorted_directed():
    g = DirectedGraph()
    g.add_edge("c", "d", 1.0)
    g.add_edge("a", "b", 2.0)
    g.add_edge("b", "c", 0.5)
    assert g.edges() == [("a", "b", 2.0), ("b", "c", 0.5), ("c", "d", 1.0)]


def test_edges_undirected_no_duplicates():
    g = UndirectedGraph()
    g.add_edge("b", "a", 1.0)
    g.add_edge("c", "b", 2.0)
    g.add_edge("a", "c", 3.0)
    edges = g.edges()
    # Cada arista aparece una única vez, en forma canónica u < v.
    assert edges == [("a", "b", 1.0), ("a", "c", 3.0), ("b", "c", 2.0)]
    assert len(edges) == 3
    pairs = [(u, v) for u, v, _ in edges]
    assert len(pairs) == len(set(pairs))
    for u, v, _ in edges:
        assert u < v


# ---------------------------------------------------------------------------
# remove_node / remove_edge y sus errores
# ---------------------------------------------------------------------------

def test_remove_node_removes_outgoing_and_incoming_edges():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("c", "a")
    g.add_edge("a", "d")
    g.remove_node("a")
    assert not g.has_node("a")
    # Los vecinos siguen existiendo.
    assert g.has_node("b") and g.has_node("c") and g.has_node("d")
    # Ninguna arista que tocaba a "a" sobrevive.
    assert g.edges() == []
    assert not g.has_edge("a", "b")
    assert not g.has_edge("c", "a")
    assert g.neighbors("b") == []
    assert g.predecessors("b") == []
    assert len(g) == 3


def test_remove_node_undirected_cleans_symmetric_side():
    g = UndirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.remove_node("a")
    assert g.edges() == []
    assert g.neighbors("b") == []
    assert g.neighbors("c") == []
    assert len(g) == 2


def test_remove_node_missing_raises_invalid_node_error():
    g = DirectedGraph()
    g.add_node("a")
    with pytest.raises(InvalidNodeError):
        g.remove_node("no-existe")


def test_remove_edge_missing_raises_invalid_edge_error():
    g = DirectedGraph()
    g.add_edge("a", "b")
    with pytest.raises(InvalidEdgeError):
        g.remove_edge("b", "a")  # sentido contrario no existe en dirigido
    with pytest.raises(InvalidEdgeError):
        g.remove_edge("a", "z")


def test_remove_edge_keeps_nodes():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.remove_edge("a", "b")
    assert g.has_node("a") and g.has_node("b")
    assert not g.has_edge("a", "b")
    assert len(g) == 2


# ---------------------------------------------------------------------------
# neighbors / predecessors de nodo inexistente
# ---------------------------------------------------------------------------

def test_neighbors_missing_node_raises():
    g = DirectedGraph()
    with pytest.raises(InvalidNodeError):
        g.neighbors("fantasma")


def test_predecessors_missing_node_raises():
    g = DirectedGraph()
    with pytest.raises(InvalidNodeError):
        g.predecessors("fantasma")


# ---------------------------------------------------------------------------
# Consultas y protocolos
# ---------------------------------------------------------------------------

def test_has_node_and_contains():
    g = DirectedGraph()
    g.add_node("a")
    assert g.has_node("a")
    assert not g.has_node("b")
    assert "a" in g
    assert "b" not in g


def test_has_edge_no_raise_for_missing_nodes():
    g = DirectedGraph()
    assert not g.has_edge("x", "y")  # nodos inexistentes: False, no error


def test_len_and_nodes_consistent():
    g = DirectedGraph()
    g.add_node("a")
    g.add_edge("b", "c")
    assert len(g) == 3
    assert sorted(g.nodes()) == ["a", "b", "c"]


def test_nodes_preserves_insertion_order():
    g = DirectedGraph()
    g.add_node("c")
    g.add_node("a")
    g.add_node("b")
    assert g.nodes() == ["c", "a", "b"]


def test_repr_directed():
    g = DirectedGraph()
    g.add_edge("a", "b")
    r = repr(g)
    assert "DirectedGraph" in r
    assert "nodes=2" in r
    assert "edges=1" in r
    assert "directed=True" in r


def test_repr_undirected():
    g = UndirectedGraph()
    g.add_edge("a", "b")
    r = repr(g)
    assert "UndirectedGraph" in r
    assert "directed=False" in r


# ---------------------------------------------------------------------------
# Jerarquía de excepciones
# ---------------------------------------------------------------------------

def test_exception_hierarchy():
    assert issubclass(GraphError, Exception)
    for exc in (InvalidNodeError, InvalidEdgeError, NegativeCycleError, NoPathError, ValidationError):
        assert issubclass(exc, GraphError)


def test_negative_cycle_error_stores_node():
    err = NegativeCycleError("ciclo negativo detectado", node="x")
    assert err.node == "x"
    assert str(err) == "ciclo negativo detectado"
    # Por defecto, node es None.
    err2 = NegativeCycleError("sin nodo")
    assert err2.node is None


# ---------------------------------------------------------------------------
# Propiedades (Hypothesis)
# ---------------------------------------------------------------------------

NODES = st.sampled_from(["a", "b", "c", "d"])
EDGE = st.tuples(NODES, NODES)


@settings(max_examples=100)
@given(st.lists(EDGE, max_size=30))
def test_property_len_equals_nodes_under_edge_ops(edges):
    """Invariante: tras cualquier mezcla de add_edge/remove_edge,
    len(g) == len(nodes()) y los nodos son únicos."""
    g = DirectedGraph()
    for u, v in edges:
        g.add_edge(u, v)
        assert len(g) == len(g.nodes())
        assert len(set(g.nodes())) == len(g.nodes())
        assert g.has_edge(u, v)
        g.remove_edge(u, v)
        assert not g.has_edge(u, v)
        assert len(g) == len(g.nodes())
        assert len(set(g.nodes())) == len(g.nodes())


@settings(max_examples=100)
@given(st.lists(EDGE, max_size=30))
def test_property_undirected_symmetry(edges):
    """Invariante: en un grafo no dirigido, has_edge/neighbors son
    simétricos y remove_edge borra ambos sentidos a la vez."""
    g = UndirectedGraph()
    for u, v in edges:
        g.add_edge(u, v, 1.0)
        assert g.has_edge(u, v) and g.has_edge(v, u)
        assert v in g.neighbors(u) and u in g.neighbors(v)
    for u, v in edges:
        if g.has_edge(u, v):
            g.remove_edge(u, v)
            assert not g.has_edge(u, v)
            assert not g.has_edge(v, u)
            assert v not in g.neighbors(u)
            assert u not in g.neighbors(v)
