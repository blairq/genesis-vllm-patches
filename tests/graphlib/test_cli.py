# SPDX-License-Identifier: Apache-2.0
"""Tests del CLI (REPL) de ``graphlib``.

Cubre el modo no interactivo de :func:`graphlib.cli.main`: cada elemento de
``argv`` se ejecuta como una línea del REPL y la salida se captura con
``capsys``. El único test de modo interactivo simula EOF (``EOFError`` en
``input``) para verificar la salida limpia.
"""
from __future__ import annotations

import pytest

from graphlib.cli import main


# ---------------------------------------------------------------------------
# Sesión completa: grafo + algoritmos
# ---------------------------------------------------------------------------

def test_sesion_completa_dijkstra(capsys: pytest.CaptureFixture) -> None:
    """directed + adds + dijkstra imprime las distancias correctas y sale 0."""
    rc = main([
        "directed",
        "add a b 1",
        "add b c 2",
        "add a c 5",
        "dijkstra a",
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # Distancias mínimas: a->a 0.0, a->b 1.0, a->c 3.0 (a->b->c, no a->c=5).
    assert "0.0" in out
    assert "1.0" in out
    assert "3.0" in out
    assert "¡Hasta pronto!" in out


def test_dijkstra_con_destino_imprime_camino(capsys: pytest.CaptureFixture) -> None:
    """dijkstra <src> <tgt> añade el camino más corto a la salida."""
    rc = main([
        "directed",
        "add a b 1",
        "add b c 2",
        "add a c 5",
        "dijkstra a c",
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Camino a -> c:" in out
    assert "a -> b -> c" in out


def test_bfspath_scc_pagerank_nodes_edges(capsys: pytest.CaptureFixture) -> None:
    """bfspath, scc, pagerank, nodes y edges funcionan en una misma sesión."""
    rc = main([
        "directed",
        "add a b 1",
        "add b c 2",
        "add c a 1",  # ciclo: un único SCC con los tres nodos
        "bfspath a c",
        "scc",
        "pagerank",
        "nodes",
        "edges",
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # bfspath: camino más corto en nº de aristas a -> b -> c.
    assert "a -> b -> c" in out
    # scc: un solo componente con los tres nodos.
    assert "Componente 1:" in out
    assert "Componente 2:" not in out
    # pagerank: scores para los tres nodos (dict impreso).
    assert "'a'" in out and "'b'" in out and "'c'" in out
    # nodes: lista en orden de inserción.
    assert "['a', 'b', 'c']" in out
    # edges: cada arista con su peso.
    assert "a -> b (peso 1.0)" in out
    assert "b -> c (peso 2.0)" in out
    assert "c -> a (peso 1.0)" in out


# ---------------------------------------------------------------------------
# Estados de error del REPL (nunca rompen el bucle, siempre rc 0)
# ---------------------------------------------------------------------------

def test_comando_antes_de_crear_grafo(capsys: pytest.CaptureFixture) -> None:
    """Sin grafo activo se avisa y se sale con 0."""
    rc = main(["nodes", "quit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Primero crea un grafo: directed o undirected" in out


def test_comando_desconocido(capsys: pytest.CaptureFixture) -> None:
    """Comando no registrado -> mensaje de aviso, rc 0."""
    rc = main(["frobnicate", "quit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Comando desconocido: frobnicate (prueba 'help')" in out


def test_add_peso_no_numerico(capsys: pytest.CaptureFixture) -> None:
    """Peso no numérico en add: error sin excepción y el REPL sigue vivo."""
    rc = main([
        "directed",
        "add a b xyz",
        "nodes",  # debe responder: el grafo sigue activo y vacío
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Error: el peso 'xyz' no es un número válido." in out
    # La arista no se añadió: el grafo sigue vacío.
    assert "(grafo vacío)" in out


def test_error_de_grafo_capturado(capsys: pytest.CaptureFixture) -> None:
    """GraphError (source inexistente) se imprime como 'Error: ...', rc 0."""
    rc = main([
        "directed",
        "add a b 1",
        "dijkstra zzz",
        "nodes",  # el REPL sigue funcionando tras el error
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Error:" in out
    assert "['a', 'b']" in out


# ---------------------------------------------------------------------------
# Comportamiento general del REPL
# ---------------------------------------------------------------------------

def test_comandos_case_insensitive(capsys: pytest.CaptureFixture) -> None:
    """Los comandos se reconocen sin distinguir mayúsculas."""
    rc = main(["DIRECTED", "Add a b 2", "nodes", "quit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Grafo dirigido creado (vacío)." in out
    assert "Arista añadida: a -> b (peso 2.0)" in out
    assert "['a', 'b']" in out


def test_clear_borra_el_grafo(capsys: pytest.CaptureFixture) -> None:
    """clear descarta el grafo activo; nodes posterior no ve nodos."""
    rc = main([
        "directed",
        "add a b 1",
        "clear",
        "nodes",
        "quit",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Grafo borrado." in out
    # Tras clear no hay grafo activo: nodes avisa (no lista nada).
    assert out.count("Primero crea un grafo: directed o undirected") == 1
    assert "['a', 'b']" not in out


def test_quit_devuelve_cero(capsys: pytest.CaptureFixture) -> None:
    """quit (y exit) producen salida limpia con código 0."""
    assert main(["quit"]) == 0
    assert main(["exit"]) == 0
    out = capsys.readouterr().out
    assert out.count("¡Hasta pronto!") == 2


def test_help_imprime_lista_de_comandos(capsys: pytest.CaptureFixture) -> None:
    """help muestra la tabla de comandos sin requerir grafo."""
    rc = main(["help", "quit"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Comandos disponibles:" in out
    assert "dijkstra" in out


def test_eof_interactivo_devuelve_cero(
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EOF (Ctrl-D) en el REPL interactivo: despedida amable y rc 0."""
    def _input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _input)
    rc = main(None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "¡Hasta pronto!" in out
