"""Tests unitarios de ``graphlib.algorithms.dijkstra``.

Cubre: elección de la ruta barata, distancia cero de la fuente, errores de
alcanzabilidad y de nodos inexistentes, rechazo de pesos negativos,
reconstrucción del camino, casos degenerados (source == target, grafo de un
nodo) y una propiedad hipotética sobre arborescencias aleatorias.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graphlib.algorithms import dijkstra, shortest_path
from graphlib.exceptions import InvalidNodeError, NoPathError, ValidationError
from graphlib.graph import DirectedGraph


def _grafo_atajo() -> DirectedGraph:
    """Grafo con un atajo caro: A->B->C (coste 2) frente a A->C (coste 5)."""
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", 1.0)
    g.add_edge("A", "C", 5.0)
    return g


def _pesos(g: DirectedGraph) -> dict:
    """Índice {(u, v): peso} para calcular el coste de un camino."""
    return {(u, v): w for u, v, w in g.edges()}


def _costo(g: DirectedGraph, path: list) -> float:
    """Coste total de ``path``; asume que cada arista consecutiva existe."""
    pesos = _pesos(g)
    total = 0.0
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v), f"falta la arista {u!r} -> {v!r}"
        total += pesos[(u, v)]
    return total


# ---------------------------------------------------------------------------
# Comportamiento básico
# ---------------------------------------------------------------------------

def test_elige_la_ruta_barata():
    """Dijkstra debe preferir A->B->C (coste 2) sobre el atajo A->C (coste 5)."""
    g = _grafo_atajo()
    dist = dijkstra(g, "A")
    assert dist["C"] == pytest.approx(2.0)
    assert shortest_path(g, "A", "C") == ["A", "B", "C"]


def test_distancia_de_la_fuente_es_cero():
    g = _grafo_atajo()
    dist = dijkstra(g, "A")
    assert dist["A"] == 0.0


def test_target_inalcanzable_lanza_nopatherror():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_node("C")  # C existe pero es inalcanzable desde A
    with pytest.raises(NoPathError):
        dijkstra(g, "A", "C")


def test_target_inexistente_lanza_nopatherror():
    """Un destino ausente es trivialmente inalcanzable."""
    g = _grafo_atajo()
    with pytest.raises(NoPathError):
        dijkstra(g, "A", "Z")


def test_source_inexistente_lanza_invalidnodeerror():
    g = _grafo_atajo()
    with pytest.raises(InvalidNodeError):
        dijkstra(g, "X")
    with pytest.raises(InvalidNodeError):
        shortest_path(g, "X", "C")


def test_peso_negativo_lanza_validationerror():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_edge("B", "C", -1.0)
    with pytest.raises(ValidationError):
        dijkstra(g, "A")
    with pytest.raises(ValidationError):
        shortest_path(g, "A", "C")


def test_shortest_path_reconstruye_camino_valido_y_optimo():
    """El camino devuelto debe ser una cadena de aristas reales cuyo coste
    coincida con la distancia mínima reportada."""
    g = _grafo_atajo()
    dist = dijkstra(g, "A")
    path = shortest_path(g, "A", "C")
    assert path[0] == "A"
    assert path[-1] == "C"
    assert len(path) >= 2
    assert _costo(g, path) == pytest.approx(dist["C"])


def test_source_igual_target_devuelve_lista_trivial():
    g = _grafo_atajo()
    assert shortest_path(g, "A", "A") == ["A"]


def test_grafo_de_un_nodo():
    g = DirectedGraph()
    g.add_node("A")
    assert dijkstra(g, "A") == {"A": 0.0}
    assert shortest_path(g, "A", "A") == ["A"]


def test_nodos_inalcanzables_no_aparecen_en_la_tabla():
    g = DirectedGraph()
    g.add_edge("A", "B", 1.0)
    g.add_node("C")
    dist = dijkstra(g, "A")
    assert set(dist) == {"A", "B"}


# ---------------------------------------------------------------------------
# Propiedad hipotética: en una arborescencia la ruta es única
# ---------------------------------------------------------------------------

@settings(max_examples=50, deadline=None)
@given(
    st.integers(1, 40).flatmap(
        lambda n: st.lists(
            st.tuples(
                st.integers(0, 1000),
                st.floats(0.0, 100.0, allow_nan=False, allow_infinity=False),
            ),
            min_size=n - 1,
            max_size=n - 1,
        )
    )
)
def test_dijkstra_en_arborescencia_coincide_con_la_ruta_unica(spec):
    """En un árbol dirigido desde la raíz hay un único camino a cada nodo,
    así que la distancia de Dijkstra debe ser exactamente la suma de pesos
    de ese camino."""
    g = DirectedGraph()
    g.add_node(0)
    esperado = {0: 0.0}
    for i, (parent, w) in enumerate(spec, start=1):
        parent = min(parent, i - 1)  # garantiza que el padre ya existe
        g.add_edge(parent, i, w)
        esperado[i] = esperado[parent] + w
    dist = dijkstra(g, 0)
    assert set(dist) == set(esperado)
    for nodo, d in esperado.items():
        assert dist[nodo] == pytest.approx(d)
