"""Recorridos DFS (profundidad) sobre :class:`~graphlib.graph.DirectedGraph`.

Este módulo ofrece el recorrido en profundidad (Depth-First Search) iterativo
y la enumeración de TODOS los caminos simples entre dos nodos mediante
backtracking.

El DFS explora "hasta el fondo" antes de retroceder: sigue una rama hasta
agotarla y recién entonces vuelve (backtrack) para explorar las ramas
pendientes. A diferencia del BFS, el orden de visita NO coincide con la
distancia al origen.
"""

from __future__ import annotations

from ..contracts import require_graph, require_node
from ..graph import DirectedGraph


def dfs(graph: DirectedGraph, source) -> list:
    """Recorre el grafo en profundidad desde ``source`` (iterativo).

    Sigue cada rama hasta agotarla antes de retroceder (backtracking). Se
    implementa con una pila explícita en lugar de recursión para evitar el
    límite de profundidad de Python en grafos grandes o con caminos largos.

    :param graph: grafo sobre el que se ejecuta el recorrido.
    :param source: nodo de partida; debe existir en ``graph``.
    :returns: lista de nodos en orden de PRIMERA visita, con ``source``
        primero. Cada nodo aparece una única vez.
    :rtype: list
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph`.
    :raises InvalidNodeError: si ``source`` no existe en ``graph``.
    """
    require_graph(graph)
    require_node(graph, source)

    visited = set()
    order = []
    stack = [source]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        # El último vecino apilado se visita primero; se invierte la lista de
        # vecinos para que el primer vecino declarado sea el primero en
        # explorarse (orden determinista coherente con el de inserción).
        for neighbor in reversed(graph.neighbors(current)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def dfs_paths(graph: DirectedGraph, source, target) -> list[list]:
    """Enumera TODOS los caminos simples de ``source`` a ``target``.

    Un camino simple no repite nodos. Se explora cada vecino no visitado y se
    retrocede (backtracking) cuando una rama se agota, de modo que se
    enumeran todos los caminos posibles, no solo uno.

    La exploración es recursiva: la profundidad de la recursión está acotada
    por el número de nodos del grafo (un camino simple no puede ser más largo
    que el nº de nodos), por lo que no se puede desbordar la pila aunque el
    grafo tenga muchos caminos.

    :param graph: grafo sobre el que se buscan los caminos.
    :param source: nodo de origen; debe existir en ``graph``.
    :param target: nodo de destino; debe existir en ``graph``.
    :returns: lista de listas de nodos, una por camino simple encontrado.
        Devuelve ``[]`` si no existe ningún camino.
    :rtype: list[list]
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph`.
    :raises InvalidNodeError: si ``source`` o ``target`` no existen.
    """
    require_graph(graph)
    require_node(graph, source)
    require_node(graph, target)

    results: list[list] = []

    def backtrack(node, path, on_path):
        if node == target:
            results.append(list(path))
            return
        for neighbor in graph.neighbors(node):
            if neighbor in on_path:
                continue  # ya está en el camino: no sería simple
            on_path.add(neighbor)
            path.append(neighbor)
            backtrack(neighbor, path, on_path)
            path.pop()
            on_path.discard(neighbor)

    path = [source]
    on_path = {source}
    backtrack(source, path, on_path)
    return results
