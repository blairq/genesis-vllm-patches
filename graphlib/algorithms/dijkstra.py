"""Algoritmo de Dijkstra para caminos más cortos con pesos no negativos.

Este módulo implementa la búsqueda de caminos más cortos desde una fuente
única sobre :class:`~graphlib.graph.DirectedGraph` usando el algoritmo de
Dijkstra con cola de prioridad (:mod:`heapq`).  Dijkstra es correcto cuando
todos los pesos de arista son no negativos: la relajación por distancia
creciente garantiza que, al extraer un nodo de la cola, su distancia ya es
final y no puede mejorar.

Se exponen dos funciones públicas:

* :func:`dijkstra` — tabla de distancias desde la fuente a todo lo
  alcanzable; opcionalmente comprueba la alcanzabilidad de un destino.
* :func:`shortest_path` — reconstrucción del camino (lista de nodos) entre
  origen y destino a partir de la tabla de padres que el algoritmo
  mantiene.

El recorrido interno se factoriza en :func:`_dijkstra`, que devuelve tanto
las distancias como el árbol de padres, para que ambas funciones públicas
compartan un único cómputo en vez de reimplementar el bucle.
"""

from __future__ import annotations

import heapq
import itertools

from ..contracts import require_graph, require_node, require_non_negative_weights
from ..exceptions import NoPathError
from ..graph import DirectedGraph
from ..validators import validate_node

__all__ = ["dijkstra", "shortest_path"]


def _build_weight_map(graph: DirectedGraph) -> dict:
    """Construye el índice ``{(u, v): peso}`` a partir de ``graph.edges()``.

    ``DirectedGraph.edges()`` emite cada arista una sola vez (en modo no
    dirigido, en su forma canónica ``u < v``), por lo que el índice resultante
    no contiene las dos direcciones de una arista no dirigida.  La recuperación
    del peso de un par concreto se hace en :func:`_edge_weight`, que cae a la
    forma canónica cuando hace falta.

    :param graph: grafo del que se leen las aristas.
    :returns: diccionario ``{(u, v): peso}`` con una entrada por arista emitida.
    :rtype: dict
    """
    return {(u, v): w for u, v, w in graph.edges()}


def _edge_weight(weight_map: dict, u, v) -> float:
    """Devuelve el peso de la arista entre ``u`` y ``v``.

    Prueba primero la dirección ``(u, v)``; si no está (caso de arista no
    dirigida almacenada en forma canónica ``v < u``), prueba ``(v, u)``.  Como
    ``v`` siempre es un vecino real de ``u`` (proviene de ``neighbors``), una
    de las dos direcciones existe con certeza.

    :param weight_map: índice de pesos construido por :func:`_build_weight_map`.
    :param u: extremo de origen de la consulta.
    :param v: extremo de destino de la consulta (vecino real de ``u``).
    :returns: peso de la arista.
    :rtype: float
    """
    weight = weight_map.get((u, v))
    if weight is None:
        weight = weight_map[(v, u)]
    return weight


def _dijkstra(graph: DirectedGraph, source) -> tuple[dict, dict]:
    """Núcleo de Dijkstra: calcula distancias y árbol de padres desde ``source``.

    Recorre el grafo relajando aristas en orden de distancia creciente usando
    una cola de prioridad.  Cada entrada de la cola es una tripleta
    ``(distancia, contador, nodo)``: el ``contador`` monótono actúa como
    desempate para que ``heapq`` nunca tenga que comparar dos nodos entre sí
    (los nodos pueden ser cualquier objeto hashable, no necesariamente
    ordenable).  Las entradas cuya distancia ya fue mejorada se descartan al
    extraerlas (estilo *lazy deletion*), por lo que cada nodo se procesa una
    sola vez con su distancia final.

    :param graph: grafo sobre el que se ejecuta el algoritmo; se asume que ya
        cumplió las pre-condiciones (nodo fuente presente, pesos no negativos).
    :param source: nodo origen desde el que se miden las distancias.
    :returns: tupla ``(distances, parents)`` donde ``distances`` mapea cada
        nodo alcanzable a su distancia mínima (con ``distances[source] == 0.0``)
        y ``parents`` mapea cada nodo alcanzable (salvo ``source``) al nodo
        predecesor en el árbol de caminos más cortos, con
        ``parents[source] is None``.
    :rtype: tuple[dict, dict]
    """
    weight_map = _build_weight_map(graph)
    distances = {source: 0.0}
    parents = {source: None}
    counter = itertools.count()
    heap = [(0.0, next(counter), source)]

    while heap:
        dist, _, node = heapq.heappop(heap)
        # Entrada obsoleta: la distancia final de `node` ya es menor que esta.
        if dist > distances[node]:
            continue
        for neighbor in graph.neighbors(node):
            new_dist = dist + _edge_weight(weight_map, node, neighbor)
            if new_dist < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_dist
                parents[neighbor] = node
                heapq.heappush(heap, (new_dist, next(counter), neighbor))

    return distances, parents


def dijkstra(graph: DirectedGraph, source, target=None) -> dict:
    """Camino más corto de Dijkstra (pesos no negativos).

    Calcula la distancia mínima desde ``source`` hasta cada nodo alcanzable
    del grafo y las devuelve en un diccionario.  El resultado siempre contiene
    ``distances[source] == 0.0`` y, para cada nodo alcanzable, su distancia
    mínima como ``float``; los nodos no alcanzables no aparecen como claves.

    Si se proporciona ``target`` la función además comprueba su alcanzabilidad:
    si ``target`` no es alcanzable desde ``source`` lanza :class:`NoPathError`
    (esto incluye el caso de que ``target`` no exista en el grafo, que es
    trivialmente inalcanzable).  Si ``target`` es alcanzable, la función
    devuelve la misma tabla de distancias: el camino concreto puede
    recuperarse con :func:`shortest_path`.

    :param graph: grafo ponderado dirigido (o no dirigido) sobre el que operar.
    :param source: nodo origen; debe existir en ``graph``.
    :param target: nodo destino opcional cuya alcanzabilidad se comprueba;
        ``None`` (por defecto) para calcular distancias a todo lo alcanzable.
    :returns: diccionario ``{nodo_alcanzable: distancia_float}`` con
        ``distances[source] == 0.0``.
    :rtype: dict
    :raises ValidationError: si ``graph`` no es un grafo, si ``source`` (o
        ``target``, si se da) no es un identificador de nodo válido, o si
        alguna arista tiene peso negativo.
    :raises InvalidNodeError: si ``source`` no existe en el grafo.
    :raises NoPathError: si ``target`` se da y no es alcanzable desde
        ``source``.
    """
    require_graph(graph)
    validate_node(source)
    require_node(graph, source)
    require_non_negative_weights(graph)
    if target is not None:
        # Solo se valida el tipo: un target inexistente se trata como
        # "no alcanzable" (NoPathError) y no como nodo inválido.
        validate_node(target)

    distances, _ = _dijkstra(graph, source)

    if target is not None and target not in distances:
        raise NoPathError(f"No hay camino de {source!r} a {target!r}")
    return distances


def shortest_path(graph: DirectedGraph, source, target) -> list:
    """Lista de nodos ``[source, ..., target]`` del camino más corto.

    Ejecuta Dijkstra desde ``source`` y reconstruye el camino más corto hacia
    ``target`` siguiendo el árbol de padres hacia atrás hasta llegar a la
    fuente.  Si ``source == target`` el camino degenera en ``[source]``
    (distancia cero, sin aristas).

    :param graph: grafo ponderado dirigido (o no dirigido) sobre el que operar.
    :param source: nodo origen; debe existir en ``graph``.
    :param target: nodo destino; debe existir en ``graph`` y ser alcanzable
        desde ``source``.
    :returns: lista de nodos desde ``source`` hasta ``target`` inclusive, en
        ese orden, representando un camino de mínima distancia total.
    :rtype: list
    :raises ValidationError: si ``graph`` no es un grafo, si ``source`` o
        ``target`` no son identificadores de nodo válidos, o si alguna arista
        tiene peso negativo.
    :raises InvalidNodeError: si ``source`` o ``target`` no existen en el
        grafo.
    :raises NoPathError: si no existe camino desde ``source`` hasta ``target``.
    """
    require_graph(graph)
    validate_node(source)
    require_node(graph, source)
    validate_node(target)
    require_node(graph, target)
    require_non_negative_weights(graph)

    if source == target:
        return [source]

    distances, parents = _dijkstra(graph, source)
    if target not in distances:
        raise NoPathError(f"No hay camino de {source!r} a {target!r}")

    # Reconstrucción: subir la cadena de padres desde `target` hasta la fuente
    # (cuya entrada es None) e invertir para obtener el orden origen -> destino.
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = parents[node]
    path.reverse()
    return path
