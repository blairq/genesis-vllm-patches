# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``graphlib.algorithms.tarjan`` (strongly connected components)."""
from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from graphlib.algorithms.tarjan import strongly_connected_components
from graphlib.graph import DirectedGraph


@st.composite
def directed_graphs(draw):
    """Random small directed graph (nodes 0..n-1, self-loops allowed)."""
    n = draw(st.integers(0, 10))
    g = DirectedGraph()
    for i in range(n):
        g.add_node(i)
    if n:
        for _ in range(draw(st.integers(0, 20))):
            g.add_edge(draw(st.integers(0, n - 1)), draw(st.integers(0, n - 1)))
    return g


def reachable_from(g: DirectedGraph, source) -> set:
    """Independent reachability check (own DFS, not the library's)."""
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in g.neighbors(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def test_known_components_two_cycles_and_isolated():
    """{a,b,c} cycle + {d,e} cycle + isolated f -> exactly those 3 SCCs."""
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("d", "e")
    g.add_edge("e", "d")
    g.add_node("f")
    components = strongly_connected_components(g)
    assert {frozenset(c) for c in components} == {
        frozenset({"a", "b", "c"}),
        frozenset({"d", "e"}),
        frozenset({"f"}),
    }


def test_dag_every_node_is_its_own_scc():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a", "c")
    components = strongly_connected_components(g)
    assert {frozenset(c) for c in components} == {
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    }


def test_empty_graph_returns_empty_list():
    assert strongly_connected_components(DirectedGraph()) == []


def test_isolated_node_is_singleton_component():
    g = DirectedGraph()
    g.add_node("solo")
    components = strongly_connected_components(g)
    assert len(components) == 1
    assert components[0] == ["solo"]


@given(g=directed_graphs())
def test_scc_is_valid_partition_with_mutual_reachability(g):
    """Property: partition covers every node once; each SCC is mutually
    reachable (verified with an independent DFS)."""
    components = strongly_connected_components(g)
    flat = [node for comp in components for node in comp]
    assert sorted(flat) == sorted(g.nodes()), "not a partition of the nodes"
    for comp in components:
        for u in comp:
            reach = reachable_from(g, u)
            for v in comp:
                assert v in reach, f"{u!r} cannot reach {v!r} in same SCC"
