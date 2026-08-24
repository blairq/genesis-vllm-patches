"""Componentes fuertemente conectados (algoritmo de Tarjan).

Implementa la descomposición de un grafo dirigido en sus componentes
fuertemente conectados (SCC) usando el algoritmo clásico de Tarjan:
números de orden asignados en el DFS, pila de nodos activos y enlaces de
bajo recorrido (*lowlink*) que detectan cuándo un nodo es la raíz de un
componente.

La implementación es **iterativa** (pila de trabajo explícita en vez de
recursión de Python) para no depender del límite de recursión del
intérprete en grafos con cadenas de DFS largas. Complejidad: O(V + E) en
tiempo y espacio.
"""

from __future__ import annotations

from ..contracts import require_graph
from ..graph import DirectedGraph

__all__ = ["strongly_connected_components"]


def strongly_connected_components(graph: DirectedGraph) -> list[list]:
    """Descompone ``graph`` en sus componentes fuertemente conectados.

    Un componente fuertemente conectado (SCC) es un subconjunto maximal de
    nodos tal que entre cada par de ellos existe camino en ambas
    direcciones. La partición del grafo en SCC es única, aunque el orden
    en que se devuelven los componentes y el orden de los nodos dentro de
    cada componente **NO están garantizados**: dependen del recorrido del
    grafo (los componentes salen en orden topológico inverso y los nodos
    en orden de extracción de la pila), no de ningún criterio de
    ordenación.

    El grafo no se modifica: el algoritmo es puramente de consulta.

    :param graph: grafo dirigido a descomponer.
    :type graph: DirectedGraph
    :returns: lista de componentes; cada componente es una lista de nodos.
        Grafo vacío -> ``[]``. Un nodo aislado forma un SCC de un elemento.
    :rtype: list[list]
    :raises ValidationError: si ``graph`` no es una instancia de
        :class:`~graphlib.graph.DirectedGraph` (pre-condición).
    """
    require_graph(graph)

    index: dict = {}      # nodo -> número de orden (primera visita en el DFS)
    lowlink: dict = {}    # nodo -> menor orden alcanzable desde su subárbol
    on_stack: dict = {}   # nodo -> True si está en la pila de activos
    active: list = []     # pila de nodos activos del DFS
    components: list[list] = []
    counter = 0

    for start in graph.nodes():
        if start in index:
            continue
        # Marco de trabajo: (nodo, iterador de vecinos aún no consumidos).
        # Al desciender se guarda el iterador del padre para retomarlo más
        # tarde, igual que lo haría el frame de una recursión.
        work: list[tuple] = [(start, iter(graph.neighbors(start)))]
        while work:
            node, it = work[-1]
            if node not in index:
                # Primera visita: asigna número de orden y baja el lowlink.
                index[node] = lowlink[node] = counter
                counter += 1
                active.append(node)
                on_stack[node] = True
            descended = False
            for neighbor in it:
                if neighbor not in index:
                    # Vecino sin visitar: desciende en el DFS y retoma el
                    # iterador actual cuando el hijo termine.
                    work.append((neighbor, iter(graph.neighbors(neighbor))))
                    descended = True
                    break
                if on_stack.get(neighbor):
                    # Arista a un nodo aún activo: el lowlink puede bajar.
                    lowlink[node] = min(lowlink[node], index[neighbor])
            if not descended:
                # Todos los vecinos procesados: cierra el nodo.
                work.pop()
                if work:
                    # Propaga el lowlink al padre (equivalente al retorno
                    # de la llamada recursiva).
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                if lowlink[node] == index[node]:
                    # `node` es la raíz de un SCC: vacía la pila hasta él.
                    component = []
                    while True:
                        w = active.pop()
                        del on_stack[w]
                        component.append(w)
                        if w == node:
                            break
                    components.append(component)
    return components
