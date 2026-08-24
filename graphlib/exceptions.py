"""Jerarquía de excepciones de la librería ``graphlib``.

Todas las excepciones que lanza la librería derivan de :class:`GraphError`,
de modo que un consumidor puede capturar el error de un caso concreto o,
con una sola cláusula ``except GraphError``, cubrir cualquier fallo de la
biblioteca sin atrapar excepciones ajenas a ella.

Jerarquía::

    Exception
    └── GraphError
        ├── InvalidNodeError    # un nodo no existe en el grafo
        ├── InvalidEdgeError    # arista inválida (extremos inexistentes, peso no numérico, …)
        ├── NegativeCycleError  # ciclo negativo detectado (p. ej. Bellman-Ford)
        ├── NoPathError         # no existe camino entre origen y destino
        └── ValidationError     # fallo de un validador de tipos/contratos
"""

from __future__ import annotations


class GraphError(Exception):
    """Excepción base de la librería.

    Raíz de la jerarquía: cualquier error que ``graphlib`` lance a propósito
    deriva de esta clase. Sirve como red de seguridad: ``except GraphError``
    atrapa todos los fallos de la librería sin capturar errores de terceros
    (``ValueError``, ``KeyError``, …).
    """


class InvalidNodeError(GraphError):
    """Se referenció un nodo que no existe en el grafo.

    Se lanza al operar sobre un nodo ausente (``remove_node``,
    ``neighbors``, ``predecessors``, …) en lugar de un ``KeyError`` genérico,
    para que el consumidor pueda distinguir "nodo inexistente" de otros
    errores internos.
    """


class InvalidEdgeError(GraphError):
    """Se intentó crear o eliminar una arista inválida.

    Cubre los casos de arista mal formada: extremos inexistentes que no
    pueden auto-crearse, peso no numérico, o eliminación de una arista que
    no está presente en el grafo.
    """


class NegativeCycleError(GraphError):
    """Un algoritmo de caminos mínimos detectó un ciclo de peso negativo.

    Guarda opcionalmente el nodo implicado en el ciclo para que el
    consumidor pueda inspeccionarlo sin volver a recorrer el grafo.

    :param message: descripción legible del fallo.
    :param node: nodo implicado en el ciclo negativo, o ``None`` si no se
        conoce o no aplica.
    """

    def __init__(self, message: str, node=None) -> None:
        """Construye el error guardando el nodo implicado (si se da).

        :param message: texto del mensaje de error.
        :param node: nodo implicado en el ciclo, opcional.
        """
        super().__init__(message)
        self.node = node


class NoPathError(GraphError):
    """No existe camino entre el origen y el destino solicitados.

    Se lanza en algoritmos de búsqueda de camino cuando el destino no es
    alcanzable desde el origen (grafo no conectado o dirección contraria en
    un grafo dirigido).
    """


class ValidationError(GraphError):
    """Un validador de tipos o contratos rechazó un argumento.

    Lo usan los validadores de la librería (p. ej. :mod:`graphlib.validators`)
    cuando un valor no cumple el contrato esperado (tipo, rango, formato).
    """
