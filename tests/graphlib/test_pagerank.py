# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``graphlib.algorithms.pagerank`` (power iteration)."""
from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

from graphlib.algorithms.pagerank import pagerank
from graphlib.exceptions import ValidationError
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


def test_scores_sum_to_one():
    g = DirectedGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    g.add_edge("a", "c")
    scores = pagerank(g)
    assert set(scores) == {"a", "b", "c"}
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_empty_graph_returns_empty_dict():
    assert pagerank(DirectedGraph()) == {}


@pytest.mark.parametrize("damping", [0, 1, 1.5, -0.1])
def test_invalid_damping_raises(damping):
    g = DirectedGraph()
    g.add_edge("a", "b")
    with pytest.raises(ValidationError):
        pagerank(g, damping=damping)


def test_max_iter_zero_raises():
    g = DirectedGraph()
    g.add_edge("a", "b")
    with pytest.raises(ValidationError):
        pagerank(g, max_iter=0)


def test_node_with_more_incoming_links_has_higher_score():
    """c receives from a, b and d -> c must be the top-ranked node."""
    g = DirectedGraph()
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    g.add_edge("d", "c")
    scores = pagerank(g)
    assert scores["c"] == max(scores.values())
    assert scores["c"] > scores["a"]
    assert scores["c"] > scores["b"]
    assert scores["c"] > scores["d"]


def test_dangling_node_receives_extra_mass():
    """b is dangling (no outgoing edges); its mass is redistributed, and
    b still ends up ranked above its only in-neighbor a."""
    g = DirectedGraph()
    g.add_edge("a", "b")
    scores = pagerank(g)
    assert scores["b"] > scores["a"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_single_node_has_score_one():
    g = DirectedGraph()
    g.add_node("solo")
    scores = pagerank(g)
    assert scores == {"solo": 1.0}


@given(g=directed_graphs())
def test_pagerank_sums_to_one_and_is_nonnegative(g):
    """Property: on a random graph, sum(scores) ~ 1.0 and all scores >= 0."""
    if not g.nodes():
        assert pagerank(g) == {}
        return
    scores = pagerank(g)
    assert set(scores) == set(g.nodes())
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert all(s >= 0.0 for s in scores.values())
