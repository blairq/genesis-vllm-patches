"""PageRank por iteración de potencia.

Calcula el vector PageRank de un grafo dirigido resolviendo la ecuación
de punto fijo por iteración de potencia (power iteration) con el modelo
estándar de PageRank: un surfer aleatorio sigue una arista saliente con
probabilidad ``damping`` y "teletransporta" a un nodo elegido
uniformemente con probabilidad ``1 - damping``. Los nodos sin aristas
salientes (*dangling*) no pueden seguir aristas, así que su masa se
reparte uniformemente entre todos los nodos, de modo que el vector de
scores sigue sumando 1.0 en cada iteración.
"""

from __future__ import annotations

from ..contracts import require_graph
from ..exceptions import ValidationError
from ..graph import DirectedGraph

__all__ = ["pagerank"]


def pagerank(
    graph: DirectedGraph,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Calcula el PageRank de ``graph`` por iteración de potencia.

    Resuelve la ecuación de punto fijo del modelo estándar de PageRank::

        r'(u) = (1 - d) / n + d * Σ_{v -> u} r(v) / out(v) + d * M / n

    donde ``d`` es el factor de amortiguación, ``n`` el número de nodos,
    ``out(v)`` el grado saliente de ``v`` y ``M`` la masa total
    acumulada en nodos sin salidas (*dangling*): esa masa se reparte
    uniformemente entre todos los nodos, por lo que el vector resultante
    sigue sumando 1.0 en cada iteración.

    La iteración parte del vector uniforme (``1/n`` por nodo) y se detiene
    cuando la variación L1 entre dos vectores consecutivos es menor que
    ``tol`` o se agotan ``max_iter`` iteraciones; en ese último caso se
    devuelve la última aproximación, sin excepción.

    El grafo no se modifica: el algoritmo es puramente de consulta.

    :param graph: grafo dirigido sobre el que se calcula el PageRank.
    :type graph: DirectedGraph
    :param damping: factor de amortiguación ``d``; debe estar en el
        intervalo abierto ``(0, 1)``. Fuera de rango ->
        :class:`~graphlib.exceptions.ValidationError`.
    :type damping: float
    :param max_iter: número máximo de iteraciones; debe ser entero
        ``>= 1``.
    :type max_iter: int
    :param tol: umbral de convergencia en norma L1; la iteración se
        detiene cuando ``Σ_u |r'(u) - r(u)| < tol``.
    :type tol: float
    :returns: dict ``{nodo: score}`` con ``sum(scores) == 1.0`` (±tol).
        Grafo vacío -> ``{}``.
    :rtype: dict
    :raises ValidationError: si ``graph`` no es un
        :class:`~graphlib.graph.DirectedGraph`, si ``damping`` no está en
        ``(0, 1)`` o si ``max_iter < 1``.
    """
    require_graph(graph)

    if not isinstance(damping, (int, float)) or not (0.0 < damping < 1.0):
        raise ValidationError(
            "damping inválido: se esperaba un número en (0, 1) exclusivo, "
            f"se recibió {damping!r}"
        )
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValidationError(
            "max_iter inválido: se esperaba un entero >= 1, "
            f"se recibió {max_iter!r}"
        )

    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}

    # Vecinos salientes y grados, cacheados una sola vez: los vecinos se
    # consultan en cada iteración y no conviene re-crear las listas.
    neighbors = {u: graph.neighbors(u) for u in nodes}
    out_degree = {u: len(neighbors[u]) for u in nodes}

    scores = {u: 1.0 / n for u in nodes}

    for _ in range(max_iter):
        # Masa total en nodos sin salidas: se repartirá uniformemente.
        dangling_mass = sum(scores[u] for u in nodes if out_degree[u] == 0)

        # Masa que cada nodo recibe por aristas entrantes.
        incoming = {u: 0.0 for u in nodes}
        for u in nodes:
            if out_degree[u]:
                share = scores[u] / out_degree[u]
                for v in neighbors[u]:
                    incoming[v] += share

        # Teleportación (1 - d)/n + reparto de la masa dangling + aristas.
        teleport = (1.0 - damping) / n + (damping * dangling_mass) / n
        new_scores = {u: teleport + damping * incoming[u] for u in nodes}

        # Convergencia: variación L1 entre vectores consecutivos.
        delta = sum(abs(new_scores[u] - scores[u]) for u in nodes)
        scores = new_scores
        if delta < tol:
            break

    return scores
