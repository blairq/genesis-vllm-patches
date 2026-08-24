"""Validadores de tipos puros para la librería :mod:`graphlib`.

Este módulo centraliza las comprobaciones de entrada que los algoritmos
de grafos realizan sobre sus parámetros: qué constituye un nodo válido,
qué valores son pesos aceptables y cómo se validan colecciones de ambos.

Todas las funciones son *puras* (no conservan estado ni producen efectos
secundarios) y lanzan :class:`~graphlib.exceptions.ValidationError` con
un mensaje descriptivo en español cuando la entrada no cumple el
contrato.  El resto del código de la librería debe llamar a estas
funciones en la frontera pública de cada API en vez de reimplementar
las comprobaciones.
"""

from __future__ import annotations

import math
from graphlib.exceptions import ValidationError

__all__ = [
    "validate_node",
    "validate_nodes",
    "validate_non_negative",
    "validate_weight",
]


def validate_node(node) -> object:
    """Valida que ``node`` es un identificador de nodo aceptable.

    Un nodo válido es un ``str``, un ``int`` (sin incluir ``bool``) o un
    ``tuple`` cuyos elementos sean a su vez nodos válidos (la regla se
    aplica recursivamente, por lo que se admiten tuplas anidadas de
    profundidad arbitraria).  Se rechazan ``bool``, ``float``, ``list``,
    ``dict``, ``set``, ``None`` y cualquier otro objeto: los identificadores
    de nodo deben ser inmutables y hashables para poder usarse como claves
    en los diccionarios internos del grafo.

    :param node: candidato a nodo.
    :returns: el propio ``node`` validado, sin copias ni conversiones.
    :rtype: object
    :raises ValidationError: si ``node`` no es ``str``, ``int`` (no ``bool``)
        o ``tuple`` de nodos válidos.
    """
    # bool es subclase de int: se rechaza antes de cualquier otra prueba.
    if isinstance(node, bool):
        raise ValidationError(
            "nodo inválido: bool no es un identificador de nodo "
            "(probablemente se pasó True/False donde se esperaba un nodo)"
        )
    if isinstance(node, (str, int)):
        return node
    if isinstance(node, tuple):
        for pos, elem in enumerate(node):
            try:
                validate_node(elem)
            except ValidationError as exc:
                # Se reenvuelve indicando la posición del elemento malo
                # dentro de la tupla para que el mensaje sea accionable.
                raise ValidationError(
                    f"nodo inválido: el elemento {pos} de la tupla "
                    f"{node!r} no es un nodo válido ({exc})"
                ) from None
        return node
    raise ValidationError(
        f"nodo inválido: se esperaba str, int o tuple de elementos "
        f"hashables, se recibió {type(node).__name__} ({node!r})"
    )


def validate_weight(weight) -> float:
    """Valida que ``weight`` es un peso numérico finito.

    Se aceptan ``int`` y ``float`` (sin incluir ``bool``) cuyo valor sea
    finito: ni ``NaN`` ni ``±inf`` son pesos válidos, porque contaminan
    las sumas de distancias y hacen indistinguibles "sin camino" de
    "camino infinito".  El resultado se normaliza a ``float`` para que
    los algoritmos manejen un único tipo numérico.

    :param weight: candidato a peso de arista.
    :returns: el peso validado como ``float``.
    :rtype: float
    :raises ValidationError: si ``weight`` no es ``int``/``float`` (o es
        ``bool``) o no es finito.
    """
    if isinstance(weight, bool):
        raise ValidationError(
            "peso inválido: bool no es un peso numérico "
            "(probablemente se pasó True/False donde se esperaba un número)"
        )
    if not isinstance(weight, (int, float)):
        raise ValidationError(
            f"peso inválido: se esperaba int o float, "
            f"se recibió {type(weight).__name__} ({weight!r})"
        )
    value = float(weight)
    if not math.isfinite(value):
        raise ValidationError(
            f"peso inválido: debe ser finito, se recibió {value!r} "
            "(NaN o infinitos no son pesos válidos)"
        )
    return value


def validate_nodes(nodes) -> list:
    """Valida que ``nodes`` es una colección no vacía de nodos válidos.

    Itera ``nodes`` una sola vez y aplica :func:`validate_node` a cada
    elemento; basta con que uno sea inválido para lanzar la excepción.
    Como ``str`` es iterable, se itera carácter a carácter: para validar
    un único nodo como string hay que usar :func:`validate_node`
    directamente.

    :param nodes: iterable (lista, tupla, generador, ...) de candidatos a nodo.
    :returns: la colección validada como ``list`` de nodos.
    :rtype: list
    :raises ValidationError: si ``nodes`` no es iterable, está vacío o
        contiene algún elemento que no es un nodo válido.
    """
    if nodes is None:
        raise ValidationError(
            "nodos inválidos: se esperaba un iterable de nodos, se recibió None"
        )
    try:
        items = list(nodes)
    except TypeError:
        raise ValidationError(
            "nodos inválidos: se esperaba un iterable de nodos, "
            f"se recibió {type(nodes).__name__} ({nodes!r})"
        ) from None
    if not items:
        raise ValidationError(
            "nodos inválidos: la colección de nodos no puede estar vacía"
        )
    return [validate_node(item) for item in items]


def validate_non_negative(weights: dict) -> None:
    """Valida que todos los pesos de ``weights`` son numéricos y >= 0.

    ``weights`` es un diccionario ``{clave: peso}`` donde la clave
    identifica la arista (o el nodo) al que corresponde el peso.  Cada
    valor se valida con :func:`validate_weight` (por lo que debe ser
    ``int``/``float`` finito) y además debe ser mayor o igual a cero:
    los algoritmos de camino más corto de la librería (Dijkstra y
    similares) no admiten pesos negativos.

    :param weights: diccionario ``{arista_o_nodo: peso}`` a validar.
    :returns: ``None``; la función se usa por su efecto de comprobación.
    :rtype: None
    :raises ValidationError: si ``weights`` no es un ``dict``, si algún
        peso no es numérico finito o si algún peso es negativo (el
        mensaje indica qué arista tiene el peso negativo).
    """
    if not isinstance(weights, dict):
        raise ValidationError(
            "pesos inválidos: se esperaba un dict {arista_o_nodo: peso}, "
            f"se recibió {type(weights).__name__}"
        )
    for key, weight in weights.items():
        value = validate_weight(weight)
        if value < 0:
            raise ValidationError(
                f"peso negativo en la arista {key!r}: {value} "
                "(los pesos de arista no pueden ser negativos)"
            )
