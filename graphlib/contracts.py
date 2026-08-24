"""Contratos de pre y post-condición para los algoritmos de :mod:`graphlib`.

Los algoritmos de la librería (Dijkstra, Floyd-Warshall, etc.) trabajan
sobre :class:`~graphlib.graph.DirectedGraph` y asumen propiedades de su
entrada: el grafo existe, los nodos y aristas referenciados están en él,
los pesos son no negativos y las distancias devueltas cumplen la
post-condición esperada.

Este módulo centraliza esas comprobaciones en funciones de propósito
único para que cada algoritmo las invoque al inicio (pre-condiciones) y
al final (post-condiciones) en vez de reimplementarlas.  La convención
de nombres es:

* ``require_*`` — pre-condición de entrada: lanza la excepción
  correspondiente si la entrada no cumple el contrato.
* ``check_*`` — post-condición de salida: lanza
  :class:`~graphlib.exceptions.ValidationError` si el resultado no
  cumple el contrato.
"""

from __future__ import annotations

from graphlib.exceptions import (
    InvalidEdgeError,
    InvalidNodeError,
    ValidationError,
)
from graphlib.graph import DirectedGraph

__all__ = [
    "check_distances",
    "require_edge",
    "require_graph",
    "require_node",
    "require_non_negative_weights",
]


def require_graph(graph) -> None:
    """Comprueba la pre-condición de que ``graph`` es un grafo válido.

    Pre-condición: ``graph`` debe ser una instancia de
    :class:`~graphlib.graph.DirectedGraph`.  No se valida más allá del
    tipo (por ejemplo, que no esté vacía) porque los algoritmos admiten
    grafos vacíos como caso degenerado legítimo.

    :param graph: candidato a grafo sobre el que opera el algoritmo.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph`.
    """
    if not isinstance(graph, DirectedGraph):
        raise ValidationError(
            "grafo inválido: se esperaba una instancia de DirectedGraph, "
            f"se recibió {type(graph).__name__}"
        )


def require_node(graph, node) -> None:
    """Comprueba la pre-condición de que ``node`` existe en ``graph``.

    Pre-condición: ``graph.has_node(node)`` debe ser ``True``.  Sirve
    para que un algoritmo que recibe un nodo por parámetro (fuente de
    Dijkstra, nodo a eliminar, etc.) falle rápido con un mensaje claro
    en vez de con un ``KeyError`` difuso a mitad del cómputo.

    :param graph: grafo en el que se busca el nodo.
    :param node: nodo que debe estar presente en ``graph``.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises InvalidNodeError: si ``graph`` no contiene ``node``.
    """
    if not graph.has_node(node):
        try:
            presentes = sorted(graph.nodes())
        except TypeError:
            # Los nodos no son mutuamente ordenables: se muestra el
            # conjunto en orden de inserción en vez de fallar al
            # construir el mensaje.
            presentes = graph.nodes()
        raise InvalidNodeError(
            f"nodo {node!r} no existe en el grafo "
            f"(nodos presentes: {presentes})"
        )


def require_edge(graph, u, v) -> None:
    """Comprueba la pre-condición de que la arista ``u -> v`` existe.

    Pre-condición: ``graph.has_edge(u, v)`` debe ser ``True``.  A
    diferencia de :func:`require_node`, no se valida por separado que
    ``u`` y ``v`` existan: si la arista no existe, el mensaje ya indica
    el par concreto que falta.

    :param graph: grafo en el que se busca la arista.
    :param u: nodo origen de la arista.
    :param v: nodo destino de la arista.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises InvalidEdgeError: si ``graph`` no contiene la arista ``u -> v``.
    """
    if not graph.has_edge(u, v):
        raise InvalidEdgeError(
            f"la arista {u!r} -> {v!r} no existe en el grafo"
        )


def require_non_negative_weights(graph) -> None:
    """Comprueba la pre-condición de pesos no negativos de ``graph``.

    Pre-condición: el peso de toda arista de ``graph`` debe ser mayor o
    igual a cero.  Esta es la condición que hace correctos los
    algoritmos de camino más corto por relajación sin colas de
    prioridad (Dijkstra y variantes); con pesos negativos el resultado
    no está garantizado y el algoritmo debe rechazar la entrada.

    Se recorre ``graph.edges()`` una sola vez; basta con un peso
    negativo para lanzar la excepción (el mensaje indica la arista
    culpable y su peso).

    :param graph: grafo cuyos pesos se van a comprobar.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises ValidationError: si alguna arista de ``graph`` tiene peso
        negativo.
    """
    for u, v, weight in graph.edges():
        if weight < 0:
            raise ValidationError(
                f"peso negativo en la arista {u!r} -> {v!r}: {weight} "
                "(este algoritmo requiere pesos no negativos)"
            )


def check_distances(distances: dict, graph, source) -> None:
    """Comprueba la post-condición de una tabla de distancias.

    Post-condición de los algoritmos de camino más corto desde una
    fuente:

    1. ``distances[source]`` debe existir y valer exactamente ``0.0``:
       la distancia de un nodo a sí mismo es cero, siempre.
    2. Toda clave de ``distances`` debe ser un nodo del grafo: una
       clave ajena a ``graph`` indica que el algoritmo escribió en la
       tabla un nodo que no existe (casi siempre un bug de
       serialización o de construcción de la tabla).

    No se comprueba que ``distances`` contenga a *todos* los nodos
    alcanzables: los algoritmos de la librería solo incluyen en la tabla
    los nodos alcanzables desde ``source``, y el conjunto de
    alcanzables no es conocido sin ejecutar el propio algoritmo.

    :param distances: tabla ``{nodo: distancia}`` devuelta por el
        algoritmo.
    :param graph: grafo sobre el que se ejecutó el algoritmo.
    :param source: nodo fuente desde el que se calcularon las distancias.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises ValidationError: si ``distances[source] != 0.0`` (incluido
        el caso de que ``source`` no esté en la tabla) o si alguna clave
        de ``distances`` no es un nodo de ``graph``.
    """
    if not isinstance(distances, dict):
        raise ValidationError(
            "distancias inválidas: se esperaba un dict {nodo: distancia}, "
            f"se recibió {type(distances).__name__}"
        )
    try:
        source_distance = distances[source]
    except KeyError:
        raise ValidationError(
            f"distancias inválidas: falta la entrada de la fuente {source!r} "
            "(la distancia de la fuente a sí misma debe estar en la tabla)"
        ) from None
    if source_distance != 0.0:
        raise ValidationError(
            f"distancias inválidas: la distancia de la fuente {source!r} "
            f"a sí misma debe ser 0.0, se encontró {source_distance!r}"
        )
    for node in distances:
        if not graph.has_node(node):
            raise ValidationError(
                f"distancias inválidas: la clave {node!r} no es un nodo "
                "del grafo"
            )
