# SPDX-License-Identifier: Apache-2.0
"""Tests de PN103 — max_offload_tokens automatico."""

from __future__ import annotations

import os

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BLK = 832


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    from vllm._genesis import kv_prefix_probe as P

    P.reset()
    monkeypatch.delenv("GENESIS_ENABLE_PN103_AUTO_PREFIX", raising=False)
    monkeypatch.delenv("GENESIS_PN103_MIN_OBS", raising=False)
    monkeypatch.delenv("GENESIS_PN103_MARGIN_BLOCKS", raising=False)
    yield
    P.reset()


def _A():
    from vllm._genesis import kv_prefix_auto as A

    return A


def _entrenar(agent, bloques, veces):
    from vllm._genesis import kv_prefix_probe as P

    pre = [b"k%d" % i for i in range(bloques)]
    for i in range(veces):
        P.observar(agent, pre + [b"cola%d" % i])


# ─────────── el arranque, que es lo delicado ───────────


def test_sin_datos_no_pone_limite():
    """Limitar antes de medir seria profecia autocumplida.

    Con un limite chico nunca se veria compartir mas de lo que se dejo
    guardar, asi que el valor aprendido quedaria clavado en el inicial.
    """
    assert _A().limite_para("nuevo", BLK) is None


def test_no_limita_hasta_tener_min_obs(monkeypatch):
    monkeypatch.setenv("GENESIS_PN103_MIN_OBS", "3")
    A = _A()
    _entrenar("coder", 4, 2)  # 2 observaciones (la 1a no compara)
    assert A.limite_para("coder", BLK) is None
    _entrenar("coder", 4, 2)
    assert A.limite_para("coder", BLK) is not None


def test_usa_el_minimo_no_el_maximo(monkeypatch):
    """REGRESION MEDIDA: usar el maximo empeora el hit rate a la mitad.

    Subir el limite de coder de 3 a 12 bloques (su maximo) llevo el hit rate
    del tier de 21,4% a 9,0%. El maximo incluye bloques que coincidieron una
    sola vez: cuestan L2 en CADA invocacion y casi nunca se releen, y con 48
    invocaciones dan vuelta L2 entera. El minimo es lo que se comparte
    SIEMPRE.
    """
    from vllm._genesis import kv_prefix_probe as P

    monkeypatch.setenv("GENESIS_PN103_MIN_OBS", "2")
    pre = [b"k%d" % i for i in range(12)]
    # `observar` compara contra el request ANTERIOR, asi que para que el
    # maximo llegue a 12 hacen falta dos consecutivos que compartan 12.
    for n, i in ((12, 0), (12, 1), (12, 2), (9, 3), (9, 4)):
        P.observar("coder", pre[:n] + [b"cola%d" % i])
    assert P.maximo("coder") == 12
    assert P.minimo("coder") == 9
    assert _A().limite_para("coder", BLK) == 9 * BLK


def test_el_margen_por_defecto_es_cero(monkeypatch):
    """El minimo ya es conservador; sumarle margen reintroduce el problema."""
    monkeypatch.setenv("GENESIS_PN103_MIN_OBS", "2")
    _entrenar("coder", 4, 4)
    assert _A().limite_para("coder", BLK) == 4 * BLK


# ─────────── guardas ───────────


def test_prefijo_roto_no_apaga_el_offload(monkeypatch):
    """Un 0 medido no es razon para que la heuristica apague un agente."""
    from vllm._genesis import kv_prefix_probe as P

    monkeypatch.setenv("GENESIS_PN103_MIN_OBS", "1")
    for i in range(4):
        P.observar("raro", [b"var%d" % i, b"comun"])
    assert P.maximo("raro") == 0
    assert _A().limite_para("raro", BLK) is None


def test_apagado_por_env(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN103_AUTO_PREFIX", "0")
    monkeypatch.setenv("GENESIS_PN103_MIN_OBS", "1")
    _entrenar("coder", 4, 4)
    assert _A().limite_para("coder", BLK) is None


def test_sin_agente_o_block_size_invalido():
    A = _A()
    assert A.limite_para(None, BLK) is None
    assert A.limite_para("coder", 0) is None


# ─────────── ancla ───────────


def _tree():
    c = []
    try:
        import vllm

        c.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    c.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    rel = "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    for r in c:
        if r and os.path.exists(os.path.join(r, rel)):
            return r, rel
    return None, rel


ROOT, REL = _tree()


@pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")
def test_ancla_unica_y_compila():
    from vllm._genesis.wiring.hybrid import patch_N103_auto_prefix_limit as P

    with open(os.path.join(ROOT, REL), encoding="utf-8") as fh:
        t = fh.read()
    if P.GENESIS_PN103_MARKER in t:
        pytest.skip("PN103 ya aplicado")
    assert t.count(P.ANCHOR_OLD) == 1
    compile(t.replace(P.ANCHOR_OLD, P.ANCHOR_NEW), REL, "exec")


@pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")
def test_un_valor_explicito_del_cliente_gana():
    """PN103 solo actua si max_offload_tokens quedo en None."""
    from vllm._genesis.wiring.hybrid import patch_N103_auto_prefix_limit as P

    assert "if self.max_offload_tokens is None:" in P.ANCHOR_NEW


# ─────────── el filtro de invocacion nueva ───────────


def test_ignora_los_turnos_de_la_misma_charla():
    """Solo cuenta como observacion un cruce entre INVOCACIONES.

    En un turno siguiente el prompt CRECE; en una invocacion nueva no. Sin
    este filtro, min y max miden historia de conversacion en vez de
    preambulo. Verificado con un proxy que grabo los prompts crudos: dos
    invocaciones frescas de `coder` comparten 9 bloques (system prompt + 12
    esquemas de herramientas), mientras que los turnos intermedios comparten
    hasta 13.
    """
    from vllm._genesis import kv_prefix_probe as P

    pre = [b"p%d" % i for i in range(9)]
    for extra in (1, 3, 5):  # invocacion 1, tres turnos que CRECEN
        P.observar("coder", pre + [b"i1-%d" % j for j in range(extra)])
    P.observar("coder", pre + [b"i2-0"])  # invocacion NUEVA (prompt mas corto)
    for extra in (3, 5):
        P.observar("coder", pre + [b"i2-%d" % j for j in range(extra)])
    P.observar("coder", pre + [b"i3-0"])  # otra invocacion nueva

    assert P.observaciones("coder") == 2, "solo los dos cruces son validos"
    assert P.minimo("coder") == 9
    assert P.maximo("coder") == 9


def test_un_agente_que_solo_crece_no_genera_observaciones():
    """El hilo principal es una sola charla: no tiene preambulo reusable.

    Medido: primary_high comparte 41 tokens (0 bloques) entre invocaciones.
    Lo que PN101 le veia eran 49-71 bloques de historia de conversacion.
    """
    from vllm._genesis import kv_prefix_probe as P

    for n in range(1, 6):
        P.observar("principal", [b"k%d" % i for i in range(n * 5)])
    assert P.observaciones("principal") == 0
    assert _A().limite_para("principal", BLK) is None
