# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``graphlib.algorithms.dfs`` (DFS + all simple paths)."""
from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given

from graphlib.algorithms.dfs import dfs, dfs_paths
from graphlib.exceptions import InvalidNodeError
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


def test_dfs_source_first():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    assert dfs(g, "a")[0] == "a"


def test_dfs_deterministic_first_visit_order():
    """First declared neighbor is explored first (reversed push order)."""
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    assert dfs(g, "a") == ["a", "b", "d", "c"]


def test_dfs_visits_all_reachable_nodes():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")  # cycle: must not loop forever
    g.add_edge("x", "y")  # unreachable component
    order = dfs(g, "a")
    assert len(order) == len(set(order))
    assert set(order) == {"a", "b", "c"}


def test_dfs_missing_source_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    try:
        dfs(g, "nope")
        raise AssertionError("expected InvalidNodeError")
    except InvalidNodeError:
        pass


def test_dfs_paths_enumerates_all_two_routes():
    """Diamond graph: exactly two simple paths a->d, both enumerated."""
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    paths = dfs_paths(g, "a", "d")
    assert len(paths) == 2
    assert {tuple(p) for p in paths} == {("a", "b", "d"), ("a", "c", "d")}


def test_dfs_paths_no_path_returns_empty():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("c", "d")
    assert dfs_paths(g, "a", "d") == []


def test_dfs_paths_missing_source_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    try:
        dfs_paths(g, "nope", "b")
        raise AssertionError("expected InvalidNodeError")
    except InvalidNodeError:
        pass


@given(data=st.data())
def test_dfs_paths_are_simple_and_valid(data):
    """Property: every returned path is simple and a real path in the graph."""
    g = data.draw(directed_graphs())
    if not g.nodes():
        return
    source = data.draw(st.sampled_from(g.nodes()))
    target = data.draw(st.sampled_from(g.nodes()))
    for path in dfs_paths(g, source, target):
        assert len(path) == len(set(path)), "path repeats a node"
        assert path[0] == source
        assert path[-1] == target
        for u, v in zip(path, path[1:]):
            assert g.has_edge(u, v), f"edge {u!r} -> {v!r} not in graph"
