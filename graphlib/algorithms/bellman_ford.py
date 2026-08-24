"""Algoritmo de Bellman-Ford para caminos más cortos con pesos negativos.

Implementa el algoritmo clásico de Bellman-Ford sobre
:class:`~graphlib.graph.DirectedGraph`, la única opción de la librería
cuando las aristas pueden tener peso negativo. A diferencia de Dijkstra,
no exige ``require_non_negative_weights``: a cambio de una complejidad
``O(V * E)`` (relajación de TODAS las aristas en ``V - 1`` rondas)
admite cualquier peso finito.

El módulo expone dos funciones públicas:

* :func:`bellman_ford` — tabla de distancias desde una fuente.
* :func:`bellman_ford_path` — camino más corto concreto entre dos nodos.

La detección de ciclos negativos se hace en la ronda ``V``: si tras
``V - 1`` rondas alguna arista saliente de un nodo alcanzable aún puede
relajarse, existe un ciclo de peso negativo alcanzable desde la fuente
y el algoritmo lanza :class:`~graphlib.exceptions.NegativeCycleError`,
intentando identificar un nodo implicado en el ciclo siguiendo las
punteras de predecesor.

Sin dependencias externas: solo usa :mod:`graphlib` y el estándar.
"""

from __future__ import annotations

from ..contracts import check_distances, require_graph, require_node
from ..exceptions import NegativeCycleError, NoPathError
from ..graph import DirectedGraph

__all__ = ["bellman_ford", "bellman_ford_path"]


def _bellman_ford_core(graph: DirectedGraph, source) -> tuple[dict, dict]:
    """Núcleo compartido: distancias y predecesores por Bellman-Ford.

    Ejecuta ``V - 1`` rondas de relajación sobre ``graph.edges()`` y, en
    una ronda extra (la ``V``), comprueba si alguna arista saliente de un
    nodo alcanzable aún mejora su destino: si es así, hay un ciclo
    negativo alcanzable desde ``source``.

    La relajación solo se aplica a aristas cuyo origen ya tiene distancia
    finita; los nodos inalcanzables quedan fuera de ``dist`` y su
    distancia se trata como ``inf`` implícito.

    Identificación del nodo implicado en el ciclo: cuando la ronda ``V``
    relaja la arista ``(u, v)``, ``v`` está a la sombra del ciclo. Al
    seguir las punteras de predecesor ``V`` veces desde ``v`` se aterriza
    necesariamente en un nodo del ciclo (cada nodo alcanzable tiene
    predecesor, y la cadena de predecesores de ``v`` entra en el ciclo y
    no puede salir de él).

    :param graph: grafo ponderado dirigido; sus pesos pueden ser
        negativos, pero no admite ciclos negativos alcanzables desde
        ``source``.
    :param source: nodo origen, que debe existir en ``graph``.
    :returns: tupla ``(dist, pred)`` donde ``dist`` es
        ``{nodo_alcanzable: distancia_float}`` (con ``dist[source] ==
        0.0``) y ``pred`` es ``{nodo: predecesor_en_camino_minimo}``
        para todos los alcanzables distintos de ``source``.
    :rtype: tuple[dict, dict]
    :raises NegativeCycleError: si existe un ciclo de peso negativo
        alcanzable desde ``source``; el atributo ``node`` de la excepción
        contiene un nodo del ciclo si se pudo identificar.
    """
    dist: dict = {source: 0.0}
    pred: dict = {}
    edges = graph.edges()
    v_count = len(graph.nodes())

    # V - 1 rondas de relajación: tras la ronda k, dist contiene el
    # camino mínimo usando como mucho k aristas.
    for _ in range(v_count - 1):
        for u, v, w in edges:
            du = dist.get(u)
            if du is None:
                # u no es alcanzable desde source: la arista no puede
                # contribuir a ninguna distancia.
                continue
            if du + w < dist.get(v, float("inf")):
                dist[v] = du + w
                pred[v] = u

    # Ronda V: si alguna arista alcanzable aún se puede relajar, existe
    # un ciclo negativo alcanzable desde source.
    for u, v, w in edges:
        du = dist.get(u)
        if du is None:
            continue
        if du + w < dist.get(v, float("inf")):
            # v recibe una mejora imposible sin ciclo negativo: v está a
            # la sombra del ciclo. Se sigue la cadena de predecesores V
            # veces para aterrizar en un nodo del propio ciclo.
            culprit = v
            for _ in range(v_count):
                culprit = pred.get(culprit, culprit)
            raise NegativeCycleError(
                f"ciclo de peso negativo alcanzable desde {source!r} "
                f"(mejora imposible en la arista {u!r} -> {v!r})",
                node=culprit,
            )

    # Post-condición de la librería: la tabla es coherente con el grafo.
    check_distances(dist, graph, source)
    return dist, pred


def bellman_ford(graph: DirectedGraph, source) -> dict:
    """Distancias más cortas con pesos POSITIVOS O NEGATIVOS (sin ciclos neg. alcanzables).

    Devuelve dict {nodo_alcanzable: distancia_float} con distances[source] == 0.0.
    Lanza NegativeCycleError si existe un ciclo negativo alcanzable desde source
    (el atributo .node de la excepción contiene un nodo del ciclo si se pudo identificar).
    """
    require_graph(graph)
    require_node(graph, source)
    dist, _ = _bellman_ford_core(graph, source)
    return dist


def bellman_ford_path(graph: DirectedGraph, source, target) -> list:
    """Camino más corto source->target reconstruido por Bellman-Ford.

    Lanza NoPathError si no hay camino (incluido el caso de que ``target`` no
    exista en el grafo: un camino hacia un nodo ausente es trivialmente
    inexistente); InvalidNodeError si ``source`` no existe.
    """
    require_graph(graph)
    require_node(graph, source)
    # Nota: NO se exige que `target` exista con require_node: un destino que
    # no está en el grafo se trata como inalcanzable (NoPathError), que es
    # la respuesta más útil para el consumidor. Solo la fuente, sobre la que
    # pivota todo el cómputo, es una pre-condición dura (InvalidNodeError).
    dist, pred = _bellman_ford_core(graph, source)
    if target not in dist:
        raise NoPathError(
            f"no existe camino desde {source!r} hasta {target!r} "
            "(el destino no es alcanzable desde la fuente)"
        )
    # Reconstrucción: se recorre la cadena de predecesores desde target
    # hasta source (que siempre existe en pred salvo el caso trivial
    # source == target) y se invierte.
    path = [target]
    node = target
    while node != source:
        node = pred[node]
        path.append(node)
    path.reverse()
    return path
