# SPDX-License-Identifier: Apache-2.0
"""Tests unitarios de ``graphlib.validators`` y ``graphlib.contracts``.

Cubre cada validador con casos válidos e inválidos (bool rechazado como
nodo Y como peso, NaN/±inf, tuplas anidadas, list/dict/set/None), los
contratos ``require_*``/``check_*`` con grafos válidos e inválidos, y el
atributo ``.node`` de ``NegativeCycleError``.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphlib.contracts import (
    check_distances,
    require_edge,
    require_graph,
    require_node,
    require_non_negative_weights,
)
from graphlib.exceptions import (
    GraphError,
    InvalidEdgeError,
    InvalidNodeError,
    NegativeCycleError,
    ValidationError,
)
from graphlib.graph import DirectedGraph, UndirectedGraph
from graphlib.validators import (
    validate_node,
    validate_non_negative,
    validate_nodes,
    validate_weight,
)


# ---------------------------------------------------------------------------
# validate_node
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node", ["a", "nodo-1", 0, 42, -7, ("a", 1), ((1, 2), ("x",)), ("a",), ()])
def test_validate_node_accepts_valid(node):
    assert validate_node(node) is node  # devuelve el propio nodo, sin copias


def test_validate_node_accepts_nested_tuple():
    nested = (("a", (1, 2)), 3)
    assert validate_node(nested) is nested


@pytest.mark.parametrize("node", [True, False, 1.5, 0.0, [1], ["a"], {"a": 1}, {1}, set(), None, object(), b"bytes"])
def test_validate_node_rejects_invalid(node):
    with pytest.raises(ValidationError):
        validate_node(node)


def test_validate_node_rejects_tuple_with_bad_element():
    with pytest.raises(ValidationError):
        validate_node((1, [2]))
    with pytest.raises(ValidationError):
        validate_node((True,))


# ---------------------------------------------------------------------------
# validate_weight
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weight, expected", [(1, 1.0), (0, 0.0), (2.5, 2.5), (-3, -3.0), (-0.5, -0.5)])
def test_validate_weight_accepts_finite_numbers(weight, expected):
    result = validate_weight(weight)
    assert result == expected
    assert isinstance(result, float)  # siempre normaliza a float


@pytest.mark.parametrize("weight", [True, False, "1", 1.0j, float("nan"), float("inf"), float("-inf"), None, [1.0]])
def test_validate_weight_rejects_invalid(weight):
    with pytest.raises(ValidationError):
        validate_weight(weight)


# ---------------------------------------------------------------------------
# validate_nodes
# ---------------------------------------------------------------------------

def test_validate_nodes_accepts_valid_iterable():
    assert validate_nodes(["a", 1, (2,)]) == ["a", 1, (2,)]
    assert validate_nodes(iter(["x", "y"])) == ["x", "y"]  # también generadores


def test_validate_nodes_empty_raises():
    with pytest.raises(ValidationError):
        validate_nodes([])
    with pytest.raises(ValidationError):
        validate_nodes(())


def test_validate_nodes_none_raises():
    with pytest.raises(ValidationError):
        validate_nodes(None)


def test_validate_nodes_non_iterable_raises():
    with pytest.raises(ValidationError):
        validate_nodes(42)


def test_validate_nodes_invalid_element_raises():
    with pytest.raises(ValidationError):
        validate_nodes(["a", True])
    with pytest.raises(ValidationError):
        validate_nodes([None])


# ---------------------------------------------------------------------------
# validate_non_negative
# ---------------------------------------------------------------------------

def test_validate_non_negative_accepts_valid():
    validate_non_negative({})  # dict vacío: no hay pesos negativos
    validate_non_negative({"a": 0})
    validate_non_negative({"a": 0.0, ("b", "c"): 1.5})


def test_validate_non_negative_rejects_negative():
    with pytest.raises(ValidationError):
        validate_non_negative({"a": -1})
    with pytest.raises(ValidationError):
        validate_non_negative({"a": 1.0, "b": -0.5})


def test_validate_non_negative_rejects_non_numeric():
    with pytest.raises(ValidationError):
        validate_non_negative({"a": True})
    with pytest.raises(ValidationError):
        validate_non_negative({"a": float("nan")})


def test_validate_non_negative_rejects_non_dict():
    with pytest.raises(ValidationError):
        validate_non_negative([1.0, 2.0])
    with pytest.raises(ValidationError):
        validate_non_negative(None)


# ---------------------------------------------------------------------------
# require_graph
# ---------------------------------------------------------------------------

def test_require_graph_accepts_graphs():
    require_graph(DirectedGraph())
    require_graph(UndirectedGraph())  # subclase: también válido


@pytest.mark.parametrize("bad", [object(), "graph", {}, [], 42, None])
def test_require_graph_rejects_non_graph(bad):
    with pytest.raises(ValidationError):
        require_graph(bad)


# ---------------------------------------------------------------------------
# require_node
# ---------------------------------------------------------------------------

def test_require_node_accepts_present_node():
    g = DirectedGraph()
    g.add_node("a")
    require_node(g, "a")  # no lanza


def test_require_node_rejects_missing_node():
    g = DirectedGraph()
    g.add_node("a")
    with pytest.raises(InvalidNodeError):
        require_node(g, "z")


# ---------------------------------------------------------------------------
# require_edge
# ---------------------------------------------------------------------------

def test_require_edge_accepts_existing_edge():
    g = DirectedGraph()
    g.add_edge("a", "b")
    require_edge(g, "a", "b")  # no lanza


def test_require_edge_rejects_missing_edge():
    g = DirectedGraph()
    g.add_edge("a", "b")
    with pytest.raises(InvalidEdgeError):
        require_edge(g, "b", "a")  # sentido contrario no existe
    with pytest.raises(InvalidEdgeError):
        require_edge(g, "a", "z")


# ---------------------------------------------------------------------------
# require_non_negative_weights
# ---------------------------------------------------------------------------

def test_require_non_negative_weights_accepts_valid_graph():
    g = DirectedGraph()
    g.add_edge("a", "b", 0.0)
    g.add_edge("b", "c", 2.5)
    require_non_negative_weights(g)  # no lanza


def test_require_non_negative_weights_rejects_negative():
    g = DirectedGraph()
    g.add_edge("a", "b", -1.0)
    with pytest.raises(ValidationError):
        require_non_negative_weights(g)


# ---------------------------------------------------------------------------
# check_distances
# ---------------------------------------------------------------------------

def _graph_with(a="a", b="b"):
    g = DirectedGraph()
    g.add_edge(a, b, 1.0)
    return g


def test_check_distances_accepts_valid_table():
    g = _graph_with()
    check_distances({"a": 0.0, "b": 1.0}, g, "a")  # no lanza
    # La tabla puede contener solo la fuente.
    check_distances({"a": 0.0}, g, "a")


def test_check_distances_rejects_missing_source():
    g = _graph_with()
    with pytest.raises(ValidationError):
        check_distances({"b": 1.0}, g, "a")


def test_check_distances_rejects_nonzero_source_distance():
    g = _graph_with()
    with pytest.raises(ValidationError):
        check_distances({"a": 0.5, "b": 1.0}, g, "a")


def test_check_distances_rejects_unknown_key():
    g = _graph_with()
    with pytest.raises(ValidationError):
        check_distances({"a": 0.0, "x": 2.0}, g, "a")


def test_check_distances_rejects_non_dict():
    g = _graph_with()
    with pytest.raises(ValidationError):
        check_distances([("a", 0.0)], g, "a")


# ---------------------------------------------------------------------------
# NegativeCycleError
# ---------------------------------------------------------------------------

def test_negative_cycle_error_node_attribute():
    err = NegativeCycleError("ciclo negativo", node="n1")
    assert err.node == "n1"
    assert str(err) == "ciclo negativo"
    assert isinstance(err, GraphError)


def test_negative_cycle_error_default_node_none():
    err = NegativeCycleError("ciclo negativo")
    assert err.node is None


# ---------------------------------------------------------------------------
# Propiedades (Hypothesis)
# ---------------------------------------------------------------------------

VALID_NODE = st.one_of(st.text(min_size=1, max_size=4), st.integers())
VALID_WEIGHT = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)


@settings(max_examples=100)
@given(VALID_NODE, VALID_WEIGHT)
def test_property_validators_roundtrip(node, weight):
    """Todo nodo válido pasa validate_node sin cambios y todo peso finito
    pasa validate_weight convertido a float con el mismo valor."""
    assert validate_node(node) is node
    assert validate_weight(weight) == float(weight)


@settings(max_examples=50)
@given(st.lists(st.tuples(st.text(min_size=1, max_size=4), st.text(min_size=1, max_size=4)), max_size=10))
def test_property_require_non_negative_matches_sign(edges):
    # Nodos solo str: edges() exige nodos mutuamente ordenables.
    """require_non_negative_weights lanza ValidationError si y solo si
    alguna arista tiene peso negativo."""
    g = DirectedGraph()
    for u, v in edges:
        g.add_edge(u, v, 1.0)
    require_non_negative_weights(g)  # todos 1.0: nunca lanza
    # Sobrescribir una arista a peso negativo debe lanzar siempre.
    if edges:
        u, v = edges[0]
        g.add_edge(u, v, -0.1)
        with pytest.raises(ValidationError):
            require_non_negative_weights(g)
