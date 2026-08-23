# SPDX-License-Identifier: Apache-2.0
"""Tests de PN91 — deferral acotado de promociones SSD → RAM."""

from __future__ import annotations

import pytest


def _g91():
    from vllm._genesis import kv_lazy_streaming as G

    return G


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    G = _g91()
    G.reset_lazy_streaming()
    monkeypatch.delenv("GENESIS_PN91_MAX_DEFER_STEPS", raising=False)
    monkeypatch.delenv("GENESIS_PN91_MAX_DEFER_SECONDS", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_PN91_KV_LAZY_STREAMING", raising=False)
    yield
    G.reset_lazy_streaming()


def test_strict_mode_arms_only_after_the_budget(monkeypatch):
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "9999")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "3")

    for step in range(3):
        G.note_deferral("req-a")
        assert G.is_strict("req-a") is False, f"no debe armarse en el paso {step + 1}"

    G.note_deferral("req-a")
    assert G.is_strict("req-a") is True


def test_zero_budget_arms_immediately(monkeypatch):
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "0")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "0")
    G.note_deferral("req-a")
    assert G.is_strict("req-a") is True


def test_strict_mode_is_per_request(monkeypatch):
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "9999")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "1")
    for _ in range(5):
        G.note_deferral("lento")
    G.note_deferral("rapido")

    assert G.is_strict("lento") is True
    assert G.is_strict("rapido") is False


def test_clear_request_resets_state(monkeypatch):
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "0")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "0")
    G.note_deferral("req-a")
    assert G.is_strict("req-a") is True

    G.clear_request("req-a")
    assert G.is_strict("req-a") is False
    assert G.stats()["tracked_requests"] == 0


def test_disabled_never_arms_strict(monkeypatch):
    G = _g91()
    monkeypatch.setenv("GENESIS_ENABLE_PN91_KV_LAZY_STREAMING", "0")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "0")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "0")
    for _ in range(10):
        G.note_deferral("req-a")
    assert G.is_strict("req-a") is False


def test_tracking_dict_has_a_ceiling(monkeypatch):
    """Un request que nunca llame a clear_request no puede hacer crecer el dict."""
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "9999")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "99999")
    for i in range(G._MAX_TRACKED + 50):
        G.note_deferral(f"req-{i}")
    assert G.stats()["tracked_requests"] <= G._MAX_TRACKED + 1
    # y no vacía el dict entero: los últimos siguen ahí
    assert G.stats()["tracked_requests"] > G._MAX_TRACKED // 2


def test_invalid_budget_falls_back(monkeypatch):
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "ni idea")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "tampoco")
    G = _g91()
    assert G.deferral_budget_seconds() == 2.0
    assert G.deferral_budget_steps() == 32


def test_budget_is_wall_clock_not_steps(monkeypatch):
    """El presupuesto tiene que medirse en TIEMPO.

    Un paso del scheduler va de ~30 ms (motor ocioso) a ~1,6 s (con un prefill
    largo en el batch), así que "N pasos" valía entre 0,12 s y 6,4 s según la
    carga. Muchos pasos rápidos no deben agotar un presupuesto de 2 s.
    """
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "60")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "100000")
    for _ in range(50):
        G.note_deferral("rapido")
    assert G.is_strict("rapido") is False, "50 pasos rápidos no llegan a 60 s"


def test_step_cap_is_the_backstop(monkeypatch):
    """Con pasos instantáneos el reloj nunca llega: manda el tope de pasos."""
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "3600")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "5")
    for _ in range(6):
        G.note_deferral("r")
    assert G.is_strict("r") is True


def test_strict_lookup_yields_only_ready_blocks(monkeypatch):
    """Modela lo que hace el código inyectado en `_maximal_prefix_lookup`.

    Es LA razón por la que el diseño original no podía funcionar: el lookup
    normal cuenta un bloque en vuelo (`None`) COMO HIT para seguir disparando
    promociones, así que `num_hit_tokens` incluye bloques no listos. Devolver
    eso hacía reventar el `assert block.is_ready` de `prepare_load`.
    """
    G = _g91()
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_SECONDS", "0")
    monkeypatch.setenv("GENESIS_PN91_MAX_DEFER_STEPS", "0")

    #      listo  listo  en vuelo  listo  miss
    tiers = [True, True, None, True, False]

    def prefix_lookup(req_id: str):
        hit, defer = 0, False
        for result in tiers:
            if result is None:
                if G.is_strict(req_id):
                    result = False
                else:
                    defer = True
                    result = True
            if not result:
                break
            hit += 1
        return None if defer else hit

    # Sin modo estricto: difiere el request entero.
    assert prefix_lookup("req-a") is None

    G.note_deferral("req-a")  # presupuesto 0 → se arma
    assert G.is_strict("req-a") is True

    # Con modo estricto: devuelve sólo el prefijo LISTO, sin diferir.
    assert prefix_lookup("req-a") == 2
