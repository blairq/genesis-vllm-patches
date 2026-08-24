"""CLI interactivo de la librería ``graphlib``.

Módulo ejecutable (``python -m graphlib.cli``) que pone en marcha un REPL
en español para explorar grafos de forma interactiva: crear grafos dirigidos
o no dirigidos, añadir/eliminar nodos y aristas, y ejecutar los algoritmos
de la librería (Dijkstra, A*, BFS, DFS, Bellman-Ford, SCC, PageRank) sobre
el grafo activo.

Uso interactivo::

    $ python -m graphlib.cli
    graphlib> directed
    graphlib> add a b 1
    graphlib> dijkstra a
    graphlib> quit

Uso no interactivo (útil para tests): cada elemento de ``argv`` se trata
como una línea del REPL::

    $ python -m graphlib.cli <<< "directed\nadd a b 1\ndijkstra a\nquit"

Toda excepción :class:`~graphlib.exceptions.GraphError` se captura y se
muestra como ``Error: <mensaje>`` sin interrumpir el bucle. La salida es
siempre limpia (código 0) salvo que el propio intérprete falle.
"""

from __future__ import annotations

from .algorithms.astar import astar
from .algorithms.bellman_ford import bellman_ford, bellman_ford_path
from .algorithms.bfs import bfs, bfs_shortest_path
from .algorithms.dfs import dfs, dfs_paths
from .algorithms.dijkstra import dijkstra, shortest_path
from .algorithms.pagerank import pagerank
from .algorithms.tarjan import strongly_connected_components
from .exceptions import GraphError
from .graph import DirectedGraph, UndirectedGraph

#: Prompt del modo interactivo.
PROMPT = "graphlib> "

#: Mensaje cuando un comando requiere grafo activo y no hay ninguno.
SIN_GRAFO = "Primero crea un grafo: directed o undirected"

#: Texto de ayuda listado por el comando ``help``.
AYUDA = """\
Comandos disponibles:

  Grafos
    directed                          Crea un grafo dirigido vacío y lo activa.
    undirected                        Crea un grafo no dirigido vacío y lo activa.
    clear                             Borra el grafo activo.
    quit / exit                       Sale del CLI.

  Nodos y aristas
    add <u> <v> [peso]                Añade la arista u -> v (peso opcional,
                                      default 1.0).
      ejemplo: add a b 2
    addnode <n>                       Añade un nodo suelto.
      ejemplo: addnode x
    rm <u> <v>                        Elimina la arista u -> v.
      ejemplo: rm a b
    rmnode <n>                        Elimina un nodo y todas sus aristas.
      ejemplo: rmnode x
    nodes                             Lista los nodos del grafo activo.
    edges                             Lista las aristas con su peso.

  Algoritmos
    dijkstra <src> [tgt]              Distancias mínimas desde src (pesos no
                                      negativos). Con tgt, además el camino.
      ejemplo: dijkstra a c
    astar <src> <tgt>                 Camino con A* (heurística nula).
      ejemplo: astar a c
    bfs <src>                         Visitas en anchura desde src.
      ejemplo: bfs a
    bfspath <src> <tgt>               Camino más corto en nº de aristas.
      ejemplo: bfspath a c
    dfs <src>                         Visitas en profundidad desde src.
      ejemplo: dfs a
    dfspaths <src> <tgt>              Todos los caminos de src a tgt.
      ejemplo: dfspaths a c
    bellmanford <src> [tgt]           Distancias mínimas con pesos negativos.
                                      Con tgt, además el camino.
      ejemplo: bellmanford a c
    scc                               Componentes fuertemente conectados.
    pagerank [damping]                PageRank (damping opcional, default 0.85).
      ejemplo: pagerank 0.9

  Otros
    help                              Muestra esta ayuda.
"""


class _Contexto:
    """Estado del REPL: guarda el grafo activo entre líneas.

    :param grafo: grafo activo o ``None`` si no hay ninguno.
    """

    def __init__(self) -> None:
        """Crea el contexto sin grafo activo."""
        self.grafo: DirectedGraph | None = None


def _ordenar(pares: list[tuple]) -> list[tuple]:
    """Ordena pares ``(clave, valor)`` por clave cuando es posible.

    Si las claves no son ordenables entre sí (mezcla de tipos, …) se
    conserva el orden de inserción en lugar de fallar.

    :param pares: pares a ordenar.
    :returns: los pares ordenados (o tal cual si no se pueden ordenar).
    :rtype: list[tuple]
    """
    try:
        return sorted(pares)
    except TypeError:
        return pares


def _mostrar_distancias(distancias: dict) -> None:
    """Imprime una tabla de distancias ordenada por nodo.

    :param distancias: ``{nodo: distancia}`` devuelta por un algoritmo.
    """
    if not distancias:
        print("(sin distancias)")
        return
    ordenado = dict(_ordenar(list(distancias.items())))
    print(ordenado)


def _mostrar_camino(camino: list) -> None:
    """Imprime un camino como ``a -> b -> c``.

    :param camino: lista de nodos desde origen hasta destino.
    """
    if not camino:
        print("(camino vacío)")
        return
    print(" -> ".join(str(n) for n in camino))


def _requerir(ctx: _Contexto) -> bool:
    """Comprueba que hay grafo activo; si no, avisa y devuelve ``False``.

    :param ctx: contexto del REPL.
    :returns: ``True`` si hay grafo activo, ``False`` si no (ya avisado).
    """
    if ctx.grafo is None:
        print(SIN_GRAFO)
        return False
    return True


def _usar(ctx: _Contexto, args: list[str], *esperados: int) -> bool:
    """Valida el número de argumentos de un comando.

    :param ctx: contexto del REPL (solo para firma uniforme).
    :param args: argumentos recibidos.
    :param esperados: números de argumentos válidos.
    :returns: ``True`` si el recuento es válido, ``False`` si ya se avisó.
    """
    if len(args) not in esperados:
        n = " o ".join(str(e) for e in esperados)
        print(f"Error: se esperaba{'n' if len(esperados) > 1 else ''} "
              f"{n} argumento(s) para este comando (prueba 'help').")
        return False
    return True


# ---------------------------------------------------------------------------
# Comandos: grafos
# ---------------------------------------------------------------------------

def _cmd_directed(ctx: _Contexto, args: list[str]) -> bool:
    """Crea un grafo dirigido vacío y lo activa (descarta el anterior)."""
    ctx.grafo = DirectedGraph()
    print("Grafo dirigido creado (vacío).")
    return False


def _cmd_undirected(ctx: _Contexto, args: list[str]) -> bool:
    """Crea un grafo no dirigido vacío y lo activa (descarta el anterior)."""
    ctx.grafo = UndirectedGraph()
    print("Grafo no dirigido creado (vacío).")
    return False


def _cmd_clear(ctx: _Contexto, args: list[str]) -> bool:
    """Borra el grafo activo."""
    if ctx.grafo is None:
        print("No hay grafo activo.")
        return False
    ctx.grafo = None
    print("Grafo borrado.")
    return False


def _cmd_quit(ctx: _Contexto, args: list[str]) -> bool:
    """Sale del CLI."""
    return True


# ---------------------------------------------------------------------------
# Comandos: nodos y aristas
# ---------------------------------------------------------------------------

def _cmd_add(ctx: _Contexto, args: list[str]) -> bool:
    """Añade la arista ``<u> <v> [peso]`` al grafo activo.

    El peso es opcional (default 1.0). Si no se parsea como ``float`` se
    muestra un mensaje de error sin lanzar excepción.
    """
    if not _requerir(ctx) or not _usar(ctx, args, 2, 3):
        return False
    u, v = args[0], args[1]
    peso = 1.0
    if len(args) == 3:
        try:
            peso = float(args[2])
        except ValueError:
            print(f"Error: el peso {args[2]!r} no es un número válido.")
            return False
    ctx.grafo.add_edge(u, v, peso)
    print(f"Arista añadida: {u} -> {v} (peso {peso})")
    return False


def _cmd_addnode(ctx: _Contexto, args: list[str]) -> bool:
    """Añade un nodo suelto al grafo activo."""
    if not _requerir(ctx) or not _usar(ctx, args, 1):
        return False
    ctx.grafo.add_node(args[0])
    print(f"Nodo añadido: {args[0]}")
    return False


def _cmd_rm(ctx: _Contexto, args: list[str]) -> bool:
    """Elimina la arista ``<u> <v>`` del grafo activo."""
    if not _requerir(ctx) or not _usar(ctx, args, 2):
        return False
    ctx.grafo.remove_edge(args[0], args[1])
    print(f"Arista eliminada: {args[0]} -> {args[1]}")
    return False


def _cmd_rmnode(ctx: _Contexto, args: list[str]) -> bool:
    """Elimina un nodo (y sus aristas) del grafo activo."""
    if not _requerir(ctx) or not _usar(ctx, args, 1):
        return False
    ctx.grafo.remove_node(args[0])
    print(f"Nodo eliminado: {args[0]}")
    return False


def _cmd_nodes(ctx: _Contexto, args: list[str]) -> bool:
    """Lista los nodos del grafo activo."""
    if not _requerir(ctx):
        return False
    nodos = ctx.grafo.nodes()
    if not nodos:
        print("(grafo vacío)")
    else:
        print(nodos)
    return False


def _cmd_edges(ctx: _Contexto, args: list[str]) -> bool:
    """Lista las aristas del grafo activo con su peso."""
    if not _requerir(ctx):
        return False
    aristas = ctx.grafo.edges()
    if not aristas:
        print("(sin aristas)")
    else:
        for u, v, w in aristas:
            print(f"{u} -> {v} (peso {w})")
    return False


# ---------------------------------------------------------------------------
# Comandos: algoritmos
# ---------------------------------------------------------------------------

def _cmd_dijkstra(ctx: _Contexto, args: list[str]) -> bool:
    """Distancias de Dijkstra desde ``<src>``; con ``<tgt>`` añade el camino."""
    if not _requerir(ctx) or not _usar(ctx, args, 1, 2):
        return False
    src = args[0]
    distancias = dijkstra(ctx.grafo, src)
    _mostrar_distancias(distancias)
    if len(args) == 2:
        camino = shortest_path(ctx.grafo, src, args[1])
        print(f"Camino {src} -> {args[1]}:")
        _mostrar_camino(camino)
    return False


def _cmd_astar(ctx: _Contexto, args: list[str]) -> bool:
    """Camino A* de ``<src>`` a ``<tgt>`` (heurística nula)."""
    if not _requerir(ctx) or not _usar(ctx, args, 2):
        return False
    src, tgt = args[0], args[1]
    camino = astar(ctx.grafo, src, tgt)
    print(f"Camino {src} -> {tgt}:")
    _mostrar_camino(camino)
    return False


def _cmd_bfs(ctx: _Contexto, args: list[str]) -> bool:
    """BFS desde ``<src>``: orden de visita."""
    if not _requerir(ctx) or not _usar(ctx, args, 1):
        return False
    print(bfs(ctx.grafo, args[0]))
    return False


def _cmd_bfspath(ctx: _Contexto, args: list[str]) -> bool:
    """Camino más corto (nº de aristas) de ``<src>`` a ``<tgt>`` vía BFS."""
    if not _requerir(ctx) or not _usar(ctx, args, 2):
        return False
    src, tgt = args[0], args[1]
    camino = bfs_shortest_path(ctx.grafo, src, tgt)
    print(f"Camino {src} -> {tgt}:")
    _mostrar_camino(camino)
    return False


def _cmd_dfs(ctx: _Contexto, args: list[str]) -> bool:
    """DFS desde ``<src>``: orden de visita."""
    if not _requerir(ctx) or not _usar(ctx, args, 1):
        return False
    print(dfs(ctx.grafo, args[0]))
    return False


def _cmd_dfspaths(ctx: _Contexto, args: list[str]) -> bool:
    """Todos los caminos de ``<src>`` a ``<tgt>`` vía DFS."""
    if not _requerir(ctx) or not _usar(ctx, args, 2):
        return False
    src, tgt = args[0], args[1]
    caminos = dfs_paths(ctx.grafo, src, tgt)
    if not caminos:
        print(f"(sin caminos de {src} a {tgt})")
        return False
    for i, camino in enumerate(caminos, 1):
        print(f"Camino {i}:")
        _mostrar_camino(camino)
    return False


def _cmd_bellmanford(ctx: _Contexto, args: list[str]) -> bool:
    """Distancias de Bellman-Ford desde ``<src>``; con ``<tgt>`` el camino."""
    if not _requerir(ctx) or not _usar(ctx, args, 1, 2):
        return False
    src = args[0]
    distancias = bellman_ford(ctx.grafo, src)
    _mostrar_distancias(distancias)
    if len(args) == 2:
        camino = bellman_ford_path(ctx.grafo, src, args[1])
        print(f"Camino {src} -> {args[1]}:")
        _mostrar_camino(camino)
    return False


def _cmd_scc(ctx: _Contexto, args: list[str]) -> bool:
    """Componentes fuertemente conectados del grafo activo."""
    if not _requerir(ctx):
        return False
    componentes = strongly_connected_components(ctx.grafo)
    if not componentes:
        print("(grafo vacío)")
    else:
        for i, comp in enumerate(componentes, 1):
            print(f"Componente {i}: {comp}")
    return False


def _cmd_pagerank(ctx: _Contexto, args: list[str]) -> bool:
    """PageRank del grafo activo, con damping opcional."""
    if not _requerir(ctx) or not _usar(ctx, args, 0, 1):
        return False
    damping = 0.85
    if args:
        try:
            damping = float(args[0])
        except ValueError:
            print(f"Error: el damping {args[0]!r} no es un número válido.")
            return False
    scores = pagerank(ctx.grafo, damping=damping)
    _mostrar_distancias(scores)
    return False


# ---------------------------------------------------------------------------
# Registro de comandos
# ---------------------------------------------------------------------------

#: Tabla de despacho: nombre de comando (minúsculas) -> manejador.
_COMANDOS = {
    "help": lambda ctx, args: _cmd_help(ctx, args),
    "directed": _cmd_directed,
    "undirected": _cmd_undirected,
    "add": _cmd_add,
    "addnode": _cmd_addnode,
    "rm": _cmd_rm,
    "rmnode": _cmd_rmnode,
    "nodes": _cmd_nodes,
    "edges": _cmd_edges,
    "dijkstra": _cmd_dijkstra,
    "astar": _cmd_astar,
    "bfs": _cmd_bfs,
    "bfspath": _cmd_bfspath,
    "dfs": _cmd_dfs,
    "dfspaths": _cmd_dfspaths,
    "bellmanford": _cmd_bellmanford,
    "scc": _cmd_scc,
    "pagerank": _cmd_pagerank,
    "clear": _cmd_clear,
    "quit": _cmd_quit,
    "exit": _cmd_quit,
}


def _cmd_help(ctx: _Contexto, args: list[str]) -> bool:
    """Muestra la ayuda con sintaxis y ejemplos de todos los comandos."""
    print(AYUDA)
    return False


def _ejecutar(linea: str, ctx: _Contexto) -> bool:
    """Procesa una línea del REPL y devuelve ``True`` si hay que salir.

    Divide la línea en tokens, despacha por el comando (case-insensitive)
    y captura cualquier :class:`GraphError` para mostrarlo como
    ``Error: <mensaje>`` sin romper el bucle. Las líneas vacías se ignoran.

    :param linea: línea cruda tal y como la escribió el usuario.
    :param ctx: contexto del REPL (grafo activo).
    :returns: ``True`` si el comando pidió salir, ``False`` para continuar.
    """
    tokens = linea.split()
    if not tokens:
        return False
    comando = tokens[0].lower()
    manejador = _COMANDOS.get(comando)
    if manejador is None:
        print(f"Comando desconocido: {tokens[0]} (prueba 'help')")
        return False
    try:
        return manejador(ctx, tokens[1:])
    except GraphError as exc:
        print(f"Error: {exc}")
        return False


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del CLI. Devuelve 0 en salida limpia, 1 en error.

    En modo interactivo (``argv`` es ``None`` o vacío) abre un bucle REPL
    con prompt ``graphlib> `` que lee líneas con :func:`input`; ``EOFError``
    (Ctrl-D) y ``KeyboardInterrupt`` (Ctrl-C) producen una despedida amable
    y salida 0.

    En modo no interactivo (``argv`` no vacío) cada elemento de ``argv`` se
    ejecuta como si fuera una línea del REPL, sin prompt, y al agotarlas el
    CLI sale con código 0. Esto permite tuberías como
    ``python -m graphlib.cli <<< "directed\nadd a b 1\nquit"``.

    :param argv: líneas a ejecutar en modo no interactivo; ``None`` o lista
        vacía para modo interactivo.
    :returns: 0 en salida limpia (incluida ``quit``/EOF/Ctrl-C), 1 en error
        inesperado.
    :rtype: int
    """
    ctx = _Contexto()
    lineas = iter(argv) if argv else None
    print("graphlib CLI — escribe 'help' para ver los comandos.")
    try:
        while True:
            if lineas is not None:
                try:
                    linea = next(lineas)
                except StopIteration:
                    break
            else:
                linea = input(PROMPT)
            if _ejecutar(linea, ctx):
                break
    except (EOFError, KeyboardInterrupt):
        pass
    except GraphError as exc:  # red de seguridad: no debe romper el REPL
        print(f"Error: {exc}")
    print("¡Hasta pronto!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
