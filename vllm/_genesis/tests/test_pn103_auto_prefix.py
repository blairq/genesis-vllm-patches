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
    for n, i in ((9, 0), (12, 1), (9, 2), (10, 3)):
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
