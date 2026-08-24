"""Recorridos BFS (amplitud) sobre :class:`~graphlib.graph.DirectedGraph`.

Este módulo ofrece el recorrido en amplitud (Breadth-First Search) y la
búsqueda del camino más corto por número de aristas en un grafo no ponderado.

El BFS es el algoritmo canónico para grafos no ponderados: explora los nodos
en "ondas", visitando primero todos los vecinos a distancia 1, luego a
distancia 2, y así sucesivamente. Esa expansión por niveles es la que
garantiza que el primer camino que se encuentra a un destino es el mínimo en
número de aristas.
"""

from __future__ import annotations

from collections import deque

from ..contracts import require_graph, require_node
from ..exceptions import NoPathError
from ..graph import DirectedGraph


def bfs(graph: DirectedGraph, source) -> list:
    """Recorre el grafo en amplitud desde ``source``.

    Explora el grafo en "ondas": primero ``source``, luego todos sus vecinos
    directos, y a continuación los vecinos de esos vecinos, y así. Cada nodo
    se visita una única vez, por lo que el resultado no contiene duplicados
    incluso en grafos con ciclos.

    La implementación es iterativa (cola con ``collections.deque``), de modo
    que no depende de la pila de recursión de Python y escala a grafos
    grandes sin riesgo de ``RecursionError``.

    :param graph: grafo sobre el que se ejecuta el recorrido.
    :param source: nodo de partida; debe existir en ``graph``.
    :returns: lista de nodos en orden de visita, con ``source`` primero.
        Solo incluye los nodos alcanzables desde ``source``.
    :rtype: list
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph`.
    :raises InvalidNodeError: si ``source`` no existe en ``graph``.
    """
    require_graph(graph)
    require_node(graph, source)

    visited = {source}
    order = [source]
    frontier = deque([source])
    while frontier:
        current = frontier.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                order.append(neighbor)
                frontier.append(neighbor)
    return order


def bfs_shortest_path(graph: DirectedGraph, source, target) -> list:
    """Camino más corto por número de aristas entre ``source`` y ``target``.

    Usa BFS, que en un grafo no ponderado garantiza que el primer camino que
    llega a ``target`` es el mínimo en cantidad de aristas. Los pesos de las
    aristas se ignoran a propósito: solo cuenta cuántas aristas tiene cada
    camino.

    Se detiene en cuanto ``target`` se visita por primera vez (no hace falta
    agotar todo el grafo) y reconstruye el camino siguiendo los nodos padre
    hacia atrás desde ``target`` hasta ``source``.

    Semántica de los extremos: ``source`` debe existir (un origen inexistente
    es un error del llamador y lanza :class:`~graphlib.exceptions.
    InvalidNodeError`). En cambio, si ``target`` no existe o no es alcanzable
    desde ``source``, se considera que "no hay camino" y se lanza
    :class:`~graphlib.exceptions.NoPathError`: un destino ausente es, desde el
    punto de vista de la búsqueda, indistinguible de uno inalcanzable.

    :param graph: grafo sobre el que se busca el camino.
    :param source: nodo de origen; debe existir en ``graph``.
    :param target: nodo de destino; si no existe o no es alcanzable se lanza
        :class:`~graphlib.exceptions.NoPathError`.
    :returns: lista ``[source, ..., target]`` con el camino mínimo en
        aristas. Si ``source == target`` devuelve ``[source]`` (camino de
        longitud 0).
    :rtype: list
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph`.
    :raises InvalidNodeError: si ``source`` no existe en ``graph``.
    :raises NoPathError: si ``target`` no existe o no es alcanzable desde
        ``source``.
    """
    require_graph(graph)
    require_node(graph, source)

    if source == target:
        return [source]

    # parent[v] = u es el nodo desde el que se llegó por primera vez a v.
    # Reconstruir hacia atrás desde target da el camino mínimo en aristas.
    parent = {source: None}
    frontier = deque([source])
    while frontier:
        current = frontier.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor not in parent:
                parent[neighbor] = current
                if neighbor == target:
                    path = [target]
                    node = target
                    while parent[node] is not None:
                        node = parent[node]
                        path.append(node)
                    path.reverse()
                    return path
                frontier.append(neighbor)
    raise NoPathError(f"no existe camino desde {source!r} hasta {target!r}")
