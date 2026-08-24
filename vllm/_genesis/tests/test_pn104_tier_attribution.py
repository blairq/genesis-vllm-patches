# SPDX-License-Identifier: Apache-2.0
"""Tests de PN104 — atribucion exacta de tier."""

from __future__ import annotations

import types

import pytest


def _A():
    from vllm._genesis import kv_tier_attribution as A

    return A


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    A = _A()
    A.reset()
    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    from vllm._genesis import kv_tier_metrics as M

    monkeypatch.setattr(M, "_SINK", M.TierStatsSink(), raising=False)
    yield
    A.reset()


def _ctx(rid="r1"):
    return types.SimpleNamespace(req_id=rid)


def _counters():
    from vllm._genesis import kv_tier_metrics as M

    return (M._SINK.drain() or {}).get("counters", {})


def test_reparte_en_proporcion_a_los_bloques():
    A = _A()
    c = _ctx()
    for _ in range(9):
        A.anotar(c, "ram")
    for _ in range(3):
        A.anotar(c, "fs")
    assert A.publicar_y_limpiar(c, 1000) == {"disk": 250, "ram": 750}


def test_la_suma_cierra_exacta_aunque_no_divida():
    """El ultimo tier se lleva el resto: no se pueden perder tokens."""
    A = _A()
    c = _ctx()
    for _ in range(1):
        A.anotar(c, "ram")
    for _ in range(2):
        A.anotar(c, "fs")
    rep = A.publicar_y_limpiar(c, 1000)
    assert sum(rep.values()) == 1000


def test_normaliza_el_nombre_del_tier():
    """`fs` y `disk` son el mismo medio: sin normalizar salen dos series."""
    A = _A()
    c = _ctx()
    A.anotar(c, "fs")
    A.anotar(c, "disk")
    assert A.reparto(c) == {"disk": 2}


def test_publica_la_metrica():
    A = _A()
    c = _ctx()
    A.anotar(c, "ram")
    A.publicar_y_limpiar(c, 800)
    assert _counters().get("kv_tier_hit_tokens_total|tier=ram") == 800


def test_limpia_la_request_al_publicar():
    A = _A()
    c = _ctx()
    A.anotar(c, "ram")
    A.publicar_y_limpiar(c, 100)
    assert A.stats()["requests_en_vuelo"] == 0
    assert A.publicar_y_limpiar(c, 100) == {}


def test_requests_distintas_no_se_mezclan():
    A = _A()
    a, b = _ctx("ra"), _ctx("rb")
    A.anotar(a, "ram")
    A.anotar(b, "fs")
    assert A.reparto(a) == {"ram": 1}
    assert A.reparto(b) == {"disk": 1}


def test_sin_aciertos_no_publica_nada():
    A = _A()
    c = _ctx()
    assert A.publicar_y_limpiar(c, 0) == {}
    assert A.publicar_y_limpiar(_ctx("otra"), 500) == {}


def test_el_dict_tiene_techo():
    A = _A()
    for i in range(A._MAX_REQUESTS + 50):
        A.anotar(_ctx(f"r{i}"), "ram")
    assert A.stats()["requests_en_vuelo"] <= A._MAX_REQUESTS


def test_contexto_sin_req_id_no_rompe():
    A = _A()
    A.anotar(types.SimpleNamespace(), "ram")
    assert A.publicar_y_limpiar(types.SimpleNamespace(), 100) == {}


def test_anclas_sobre_el_resultado_de_pn88():
    """PN104 ancla sobre lo que deja PN88, asi que va DESPUES."""
    from vllm._genesis.wiring.hybrid import patch_N104_exact_tier_attribution as P

    assert "_g104.anotar(req_context, 'ram')" in P.PRIM_NEW
    assert "tier_type" in P.SEC_NEW
    assert "publicar_y_limpiar" in P.SCHED_NEW
