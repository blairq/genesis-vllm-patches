"""Algoritmo A* de camino más corto sobre :class:`~graphlib.graph.DirectedGraph`.

Este módulo implementa la búsqueda A* sobre grafos ponderados dirigidos: una
variante de Dijkstra guiada por una función heurística que estima la distancia
restante hasta el destino, de modo que la exploración se concentra en la
dirección prometedoras en vez de expandirse uniformemente.

Propiedades
-----------
* **Pesos no negativos**: como Dijkstra, A* solo es correcto si todas las
  aristas tienen peso mayor o igual a cero. La pre-condición se verifica con
  :func:`~graphlib.contracts.require_non_negative_weights`.
* **Heurística admissible**: la función ``heuristic(nodo)`` debe ser una
  *subestimación* de la distancia real hasta el destino, es decir,
  ``heuristic(n) <= distancia_real(n, target)`` para todo nodo ``n``. Con una
  heurística admissible (y pesos no negativos) A* garantiza devolver el mismo
  camino óptimo que Dijkstra; si la heurística sobrestima, el resultado puede
  dejar de ser óptimo.
* **Heurística nula**: con ``heuristic=None`` (por defecto) la estimación es
  cero para todo nodo y A* degenera exactamente en Dijkstra, devolviendo el
  mismo camino.

La implementación usa una cola de prioridad (:mod:`heapq`) ordenada por la
función de coste ``f(nodo) = g(nodo) + h(nodo)``, donde ``g`` es la distancia
exacta acumulada desde el origen y ``h`` la estimación hasta el destino. No
depende de ninguna librería externa.
"""

from __future__ import annotations

import heapq

from ..contracts import (
    require_graph,
    require_node,
    require_non_negative_weights,
)
from ..exceptions import NoPathError, ValidationError
from ..graph import DirectedGraph

__all__ = ["astar"]


def astar(graph: DirectedGraph, source, target, heuristic=None) -> list:
    """Camino más corto con A* (pesos no negativos, heurística admissible).

    Busca el camino de menor coste total desde ``source`` hasta ``target``
    usando la función de coste ``f(nodo) = g(nodo) + h(nodo)``: ``g`` es la
    distancia exacta acumulada desde ``source`` y ``h`` la estimación que
    ``heuristic`` hace de la distancia restante hasta ``target``.

    La heurística debe ser **admissible**: no sobrestimar nunca la distancia
    real hasta el destino (``heuristic(n) <= dist_real(n, target)``). Con esa
    garantía y pesos no negativos, A* devuelve el mismo camino óptimo que
    Dijkstra; una heurística que sobrestima puede producir un camino
    subóptimo.

    :param graph: grafo ponderado dirigido sobre el que buscar el camino.
    :type graph: DirectedGraph
    :param source: nodo de origen del camino; debe existir en ``graph``.
    :param target: nodo de destino del camino; debe existir en ``graph``.
    :param heuristic: función ``nodo -> float`` que estima la distancia
        restante hasta ``target``; debe ser admissible (no sobrestimar). Con
        ``None`` (por defecto) se usa la heurística nula ``lambda n: 0.0``,
        que degrada A* a Dijkstra.
    :type heuristic: callable | None
    :returns: lista de nodos ``[source, ..., target]`` con el camino más
        corto. Si ``source == target`` devuelve ``[source]``.
    :rtype: list
    :raises ValidationError: si ``graph`` no es una ``DirectedGraph``, si
        alguna arista tiene peso negativo, o si ``heuristic`` no es ``None``
        ni callable.
    :raises InvalidNodeError: si ``source`` no existe en el grafo.
    :raises NoPathError: si no existe camino desde ``source`` hasta
        ``target``. Esto incluye el caso de que ``target`` no exista en el
        grafo: un destino ausente es inalcanzable, así que la búsqueda se
        agota y se reporta como "sin camino" en lugar de como nodo
        inválido. (Sí se exige que ``source`` exista: partir de un nodo
        ausente es un error de entrada, no ausencia de camino.)
    """
    # Pre-condiciones: grafo válido, origen presente y pesos no negativos.
    # Nota: solo se exige que `source` exista. Un `target` ausente no es un
    # error de entrada sino un destino inalcanzable: la búsqueda se agota y
    # lanza NoPathError más abajo, lo cual es la semántica esperada.
    require_graph(graph)
    require_node(graph, source)
    require_non_negative_weights(graph)

    # Heurística nula por defecto: equivale a Dijkstra.
    if heuristic is None:
        heuristic = lambda n: 0.0
    if not callable(heuristic):
        raise ValidationError(
            "heurística inválida: se esperaba None o un callable "
            f"nodo -> float, se recibió {type(heuristic).__name__}"
        )

    # Caso degenerado: origen y destino coinciden; el camino es trivial.
    if source == target:
        return [source]

    # Tabla de pesos {(u, v): peso} para acceso O(1): neighbors() solo
    # devuelve nombres de nodos, así que el peso de cada arista se resuelve
    # contra esta tabla construida una sola vez desde edges().
    pesos = {(u, v): w for u, v, w in graph.edges()}

    g_score = {source: 0.0}   # distancia exacta acumulada desde source
    came_from = {}            # nodo -> predecesor en el camino encontrado
    cerrado = set()           # nodos ya extraídos con su g definitivo
    contador = 0              # desempate determinista en la cola

    # Cola de prioridad: (f, contador, nodo). El contador evita comparar
    # nodos entre sí (que podrían no ser ordenables) cuando f empata.
    cola = [(heuristic(source), contador, source)]
    contador += 1

    while cola:
        _, _, nodo = heapq.heappop(cola)
        if nodo in cerrado:
            # Entrada obsoleta: el nodo ya se procesó con un g menor.
            continue
        cerrado.add(nodo)

        if nodo == target:
            # A* con heurística admissible: al extraer el destino su g es
            # definitivo, así que el camino reconstruido es óptimo.
            return _reconstruir(came_from, source, target)

        g_actual = g_score[nodo]
        for vecino in graph.neighbors(nodo):
            if vecino in cerrado:
                continue
            g_candidato = g_actual + pesos[(nodo, vecino)]
            if g_candidato < g_score.get(vecino, float("inf")):
                g_score[vecino] = g_candidato
                came_from[vecino] = nodo
                f_candidato = g_candidato + heuristic(vecino)
                heapq.heappush(cola, (f_candidato, contador, vecino))
                contador += 1

    raise NoPathError(
        f"no existe camino desde {source!r} hasta {target!r}"
    )


def _reconstruir(came_from: dict, source, target) -> list:
    """Reconstruye la lista de nodos del camino a partir de ``came_from``.

    Recorre enlazando hacia atrás desde ``target`` hasta ``source`` (cada
    nodo apunta a su predecesor en el camino) y revierte el resultado para
    obtener el orden ``[source, ..., target]``.

    :param came_from: tabla ``nodo -> predecesor`` acumulada durante la
        búsqueda; ``source`` no figura como clave (es la raíz).
    :param source: nodo de origen; punto de parada de la retrotraza.
    :param target: nodo de destino; punto de partida de la retrotraza.
    :returns: lista de nodos en orden desde ``source`` hasta ``target``.
    :rtype: list
    """
    camino = [target]
    nodo = target
    while nodo != source:
        nodo = came_from[nodo]
        camino.append(nodo)
    camino.reverse()
    return camino
