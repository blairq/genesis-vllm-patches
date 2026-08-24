# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``graphlib.algorithms.bfs`` (BFS + shortest path by edges)."""
from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from graphlib.algorithms.bfs import bfs, bfs_shortest_path
from graphlib.exceptions import InvalidNodeError, NoPathError
from graphlib.graph import DirectedGraph


@st.composite
def directed_graphs(draw):
    """Random small directed graph (nodes 0..n-1, self-loops allowed)."""
    n = draw(st.integers(0, 12))
    g = DirectedGraph()
    for i in range(n):
        g.add_node(i)
    if n:
        for _ in range(draw(st.integers(0, 25))):
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


def test_bfs_fanout_visits_level_by_level():
    """Fan-out graph: all distance-1 nodes before any distance-2 node."""
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("a", "d")
    g.add_edge("b", "e")
    g.add_edge("c", "f")
    assert bfs(g, "a") == ["a", "b", "c", "d", "e", "f"]


def test_bfs_source_first_and_no_duplicates():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "c")  # c reachable twice; must appear once
    g.add_edge("c", "a")  # cycle back to source
    order = bfs(g, "a")
    assert order[0] == "a"
    assert len(order) == len(set(order))
    assert set(order) == {"a", "b", "c"}


def test_bfs_visits_only_reachable_nodes():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("x", "y")  # disconnected component
    assert set(bfs(g, "a")) == {"a", "b"}


def test_bfs_missing_source_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    try:
        bfs(g, "nope")
        raise AssertionError("expected InvalidNodeError")
    except InvalidNodeError:
        pass


def test_bfs_shortest_path_minimizes_edges_not_weights():
    """Two-edge path with huge weights must beat three cheap edges."""
    g = DirectedGraph()
    g.add_edge("a", "b", weight=100.0)
    g.add_edge("b", "c", weight=100.0)  # 2 edges, total weight 200
    g.add_edge("a", "d", weight=1.0)
    g.add_edge("d", "e", weight=1.0)
    g.add_edge("e", "c", weight=1.0)  # 3 edges, total weight 3
    assert bfs_shortest_path(g, "a", "c") == ["a", "b", "c"]


def test_bfs_shortest_path_source_equals_target():
    g = DirectedGraph()
    g.add_edge("a", "b")
    assert bfs_shortest_path(g, "a", "a") == ["a"]


def test_bfs_shortest_path_unreachable_target_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("c", "d")
    try:
        bfs_shortest_path(g, "a", "d")
        raise AssertionError("expected NoPathError")
    except NoPathError:
        pass


def test_bfs_shortest_path_nonexistent_target_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    try:
        bfs_shortest_path(g, "a", "ghost")
        raise AssertionError("expected NoPathError")
    except NoPathError:
        pass


def test_bfs_shortest_path_nonexistent_source_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    try:
        bfs_shortest_path(g, "ghost", "b")
        raise AssertionError("expected InvalidNodeError")
    except InvalidNodeError:
        pass


@given(data=st.data())
def test_bfs_visits_exactly_reachable_nodes(data):
    """Property: on a random graph, bfs visits exactly the reachable set."""
    g = data.draw(directed_graphs())
    if not g.nodes():
        return
    source = data.draw(st.sampled_from(g.nodes()))
    order = bfs(g, source)
    assert order[0] == source
    assert len(order) == len(set(order))
    assert set(order) == reachable_from(g, source)
