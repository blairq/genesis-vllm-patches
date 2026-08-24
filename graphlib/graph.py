"""Grafos ponderados dirigidos y no dirigidos.

Módulo central de la librería ``graphlib``. Define :class:`DirectedGraph`,
un grafo ponderado que opera en modo dirigido por defecto y puede crearse en
modo no dirigido, y :class:`UndirectedGraph`, un alias fijo en modo no
dirigido.

Representación interna
----------------------
Se usa adyacencia con dos índices simétricos para que tanto los vecinos
salientes como los entrantes sean accesibles en O(1):

* ``_adj[u]``  -> ``{v: peso}``  aristas salientes de ``u`` (``u -> v``).
* ``_radj[v]`` -> ``{u: peso}``  aristas entrantes en ``v`` (``u -> v``).

En modo no dirigido cada arista ``(u, v)`` se refleja en los dos sentidos y
en los dos índices, de modo que ``_adj`` y ``_radj`` quedan simétricos y
``neighbors``/``predecessors`` devuelven el mismo conjunto.

El orden de inserción de los nodos se preserva (los ``dict`` de CPython 3.7+
mantienen el orden), por lo que :meth:`DirectedGraph.nodes` es determinista.
"""

from __future__ import annotations

from .exceptions import InvalidEdgeError, InvalidNodeError
from .validators import validate_weight


class DirectedGraph:
    """Grafo ponderado dirigido, con modo no dirigido opcional.

    Un grafo es una colección de nodos y aristas ponderadas. En modo dirigido
    (por defecto) una arista ``(u, v)`` solo va de ``u`` a ``v``; en modo no
    dirigido la arista es simétrica y se comporta como si existieran las dos
    direcciones a la vez, con el mismo peso.

    Los nodos pueden ser cualquier objeto *hashable* (str, int, tuplas, …).
    Los pesos se almacenan siempre como ``float`` y se validan con
    :func:`graphlib.validators.validate_weight` al crear la arista.

    Este grafo NO admite multiaristas: si ya existe una arista ``(u, v)`` y se
    vuelve a llamar a :meth:`add_edge` con el mismo par, el peso se
    *sobrescribe* en lugar de añadir una arista paralela.
    """

    def __init__(self, directed: bool = True) -> None:
        """Crea un grafo vacío.

        :param directed: ``True`` para grafo dirigido (por defecto),
            ``False`` para no dirigido. En modo no dirigido cada arista se
            almacena en ambos sentidos con el mismo peso.
        """
        self._directed = directed
        self._nodes: dict = {}   # nodo -> None; preserva el orden de inserción
        self._adj: dict = {}     # u -> {v: peso}  (salientes)
        self._radj: dict = {}    # v -> {u: peso}  (entrantes)

    @property
    def directed(self) -> bool:
        """Indica si el grafo opera en modo dirigido.

        :returns: ``True`` si es dirigido, ``False`` si es no dirigido.
        :rtype: bool
        """
        return self._directed

    def add_node(self, node) -> None:
        """Añade un nodo al grafo si no existe.

        Es idempotente: añadir un nodo ya presente no tiene efecto y no lanza
        excepción. Los nodos no llevan atributos propios; solo existen como
        vértices del grafo.

        :param node: nodo a añadir; debe ser *hashable*.
        """
        if node not in self._nodes:
            self._nodes[node] = None
            self._adj[node] = {}
            self._radj[node] = {}

    def add_edge(self, u, v, weight: float = 1.0) -> None:
        """Añade (o sobrescribe) la arista ``u -> v`` con el peso dado.

        Auto-crea los nodos ``u`` y ``v`` si no existen. El peso se valida con
        :func:`~graphlib.validators.validate_weight` y se almacena como
        ``float``. En modo no dirigido la arista se refleja en ambos sentidos
        con el mismo peso.

        Si la arista ya existe, su peso se sobrescribe (no hay multiaristas).

        :param u: nodo de origen.
        :param v: nodo de destino.
        :param weight: peso de la arista; debe ser numérico, por defecto
            ``1.0``.
        :raises InvalidEdgeError: si ``weight`` no es numérico.
        """
        w = validate_weight(weight)
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = w
        self._radj[v][u] = w
        if not self._directed:
            # Simetría: la arista también existe en el sentido contrario.
            self._adj[v][u] = w
            self._radj[u][v] = w

    def remove_node(self, node) -> None:
        """Elimina un nodo y TODAS las aristas incidentes.

        Borra el nodo junto con todas sus aristas salientes y entrantes (en
        modo no dirigido, también las aristas simétricas que apuntan desde sus
        vecinos). Los vecinos que pierden la conexión siguen existiendo; solo
        desaparece la arista.

        :param node: nodo a eliminar.
        :raises InvalidNodeError: si el nodo no existe en el grafo.
        """
        if node not in self._nodes:
            raise InvalidNodeError(f"El nodo {node!r} no existe en el grafo.")
        # Nodos conectados a `node` por cualquier lado.
        connected = set(self._adj[node]) | set(self._radj[node])
        for other in connected:
            # Quitar la arista entre `node` y `other` en ambos índices y en
            # ambos extremos; `pop(..., None)` tolera que ya no esté.
            self._adj[other].pop(node, None)
            self._radj[other].pop(node, None)
            self._adj[node].pop(other, None)
            self._radj[node].pop(other, None)
        del self._nodes[node]
        del self._adj[node]
        del self._radj[node]

    def remove_edge(self, u, v) -> None:
        """Elimina la arista ``u -> v``.

        En modo no dirigido elimina la arista simétrica completa (ambos
        sentidos) porque la arista es única. No elimina los nodos.

        :param u: nodo de origen de la arista.
        :param v: nodo de destino de la arista.
        :raises InvalidEdgeError: si la arista no existe.
        """
        if not self.has_edge(u, v):
            raise InvalidEdgeError(
                f"La arista ({u!r}, {v!r}) no existe en el grafo."
            )
        del self._adj[u][v]
        del self._radj[v][u]
        if not self._directed:
            self._adj[v].pop(u, None)
            self._radj[u].pop(v, None)

    def has_node(self, node) -> bool:
        """Indica si un nodo está presente en el grafo.

        :param node: nodo a consultar.
        :returns: ``True`` si existe, ``False`` en caso contrario.
        :rtype: bool
        """
        return node in self._nodes

    def has_edge(self, u, v) -> bool:
        """Indica si existe la arista ``u -> v``.

        En modo no dirigido la consulta es simétrica: ``has_edge(u, v)`` y
        ``has_edge(v, u)`` devuelven lo mismo. No lanza por nodos inexistentes:
        si ``u`` no existe simplemente no puede tener aristas salientes.

        :param u: nodo de origen.
        :param v: nodo de destino.
        :returns: ``True`` si la arista existe, ``False`` en caso contrario.
        :rtype: bool
        """
        return v in self._adj.get(u, {})

    def neighbors(self, node) -> list:
        """Devuelve los vecinos salientes de un nodo.

        En modo dirigido son los nodos ``v`` tales que existe ``node -> v``.
        En modo no dirigido incluye ambos sentidos, es decir, todos los nodos
        conectados a ``node``. El orden es el de inserción de las aristas.

        :param node: nodo del que se piden los vecinos.
        :returns: lista de vecinos.
        :rtype: list
        :raises InvalidNodeError: si el nodo no existe.
        """
        if node not in self._nodes:
            raise InvalidNodeError(f"El nodo {node!r} no existe en el grafo.")
        return list(self._adj[node])

    def predecessors(self, node) -> list:
        """Devuelve los nodos ``w`` tales que existe la arista ``w -> node``.

        En modo dirigido son los nodos con arista entrante en ``node``. En
        modo no dirigido es simétrico a :meth:`neighbors`. El orden es el de
        inserción de las aristas.

        :param node: nodo del que se piden los predecesores.
        :returns: lista de predecesores.
        :rtype: list
        :raises InvalidNodeError: si el nodo no existe.
        """
        if node not in self._nodes:
            raise InvalidNodeError(f"El nodo {node!r} no existe en el grafo.")
        return list(self._radj[node])

    def edges(self) -> list[tuple]:
        """Devuelve todas las aristas como ``(u, v, peso)``, ordenado por ``(u, v)``.

        En modo dirigido cada arista aparece una vez, en su dirección. En modo
        no dirigido cada arista aparece una única vez (forma canónica
        ``u < v``) para no duplicar la simetría. Requiere que los nodos sean
        ordenables entre sí (p. ej. todos ``str`` o todos ``int``).

        :returns: lista de tuplas ``(u, v, peso)``.
        :rtype: list[tuple]
        """
        result = []
        for u in self._nodes:
            for v, w in self._adj[u].items():
                if not self._directed and u > v:
                    # En no dirigido (u,v) y (v,u) son la misma arista: solo
                    # se emite la forma canónica u < v.
                    continue
                result.append((u, v, w))
        result.sort()
        return result

    def nodes(self) -> list:
        """Devuelve la lista de nodos en orden de inserción.

        :returns: lista de nodos.
        :rtype: list
        """
        return list(self._nodes)

    def __len__(self) -> int:
        """Número de nodos del grafo.

        :returns: cantidad de nodos.
        :rtype: int
        """
        return len(self._nodes)

    def __contains__(self, node) -> bool:
        """Soporta el operador ``in``: ``node in grafo``.

        :param node: nodo a buscar.
        :returns: ``True`` si el nodo existe en el grafo.
        :rtype: bool
        """
        return node in self._nodes

    def __repr__(self) -> str:
        """Representación legible con nº de nodos, aristas y modo.

        :returns: cadena tipo ``DirectedGraph(nodes=3, edges=4, directed=True)``.
        :rtype: str
        """
        return (
            f"{type(self).__name__}(nodes={len(self._nodes)}, "
            f"edges={len(self.edges())}, directed={self._directed})"
        )


class UndirectedGraph(DirectedGraph):
    """Alias de :class:`DirectedGraph` fijo en modo no dirigido.

    Útil como forma declarativa de expresar que un grafo es no dirigido:
    ``UndirectedGraph()`` equivale a ``DirectedGraph(directed=False)``. Todas
    las aristas se almacenan simétricamente y las consultas (``has_edge``,
    ``neighbors``, ``remove_edge``) respetan esa simetría.
    """

    def __init__(self) -> None:
        """Crea un grafo no dirigido vacío (``directed=False``)."""
        super().__init__(directed=False)
