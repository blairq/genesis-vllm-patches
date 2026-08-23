# SPDX-License-Identifier: Apache-2.0
"""Tests de PN90 — gating de escritura a disco y democión por desalojo."""

from __future__ import annotations

import pytest


def _gate():
    from vllm._genesis import kv_disk_gate as G

    return G


def _ktm():
    from vllm._genesis import kv_tier_metrics as M

    return M


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    G = _gate()
    G.clear_all()
    for var in ("GENESIS_KV_DISK_WRITERS", "GENESIS_PN90_MAX_PENDING_MB"):
        monkeypatch.delenv(var, raising=False)
    yield
    G.clear_all()


# ─────────────────────────── gating ───────────────────────────


def test_explicit_persist_disk_wins_over_agent():
    G = _gate()
    assert G.should_persist_to_secondary_tiers({"persist_disk": True}) is True
    assert (
        G.should_persist_to_secondary_tiers(
            {"persist_disk": True, "genesis_agent": "coder"}
        )
        is True
    )
    assert (
        G.should_persist_to_secondary_tiers(
            {"persist_disk": False, "genesis_agent": "primary"}
        )
        is False
    )


@pytest.mark.parametrize(
    "agent", ["primary", "primary_nothink", "build", "coach", "planner"]
)
def test_default_disk_writers_allowed(agent):
    assert _gate().should_persist_to_secondary_tiers({"genesis_agent": agent}) is True


@pytest.mark.parametrize(
    "agent", ["coder", "verifier", "explorer", "vision", "art", "utility"]
)
def test_ephemeral_agents_denied(agent):
    assert _gate().should_persist_to_secondary_tiers({"genesis_agent": agent}) is False


def test_untagged_requests_denied():
    G = _gate()
    assert G.should_persist_to_secondary_tiers(None) is False
    assert G.should_persist_to_secondary_tiers({}) is False


def test_env_allowlist_overrides_defaults(monkeypatch):
    G = _gate()
    monkeypatch.setenv("GENESIS_KV_DISK_WRITERS", "solo_este")
    assert G.should_persist_to_secondary_tiers({"genesis_agent": "solo_este"}) is True
    assert G.should_persist_to_secondary_tiers({"genesis_agent": "primary"}) is False


# ─────────────────── snapshots de democión ───────────────────


def test_snapshot_only_taken_for_tagged_keys():
    G = _gate()
    G.tag_keys_persistence([b"tagged"])
    assert G.take_demotion_snapshot(b"tagged") is True
    assert G.take_demotion_snapshot(b"untagged") is False


def test_tag_is_consumed_once():
    G = _gate()
    G.tag_keys_persistence([b"k"])
    assert G.take_demotion_snapshot(b"k") is True
    assert G.take_demotion_snapshot(b"k") is False




def test_drain_returns_and_clears():
    G = _gate()
    G.record_demotion(b"a", b"AAAA")
    G.record_demotion(b"b", b"BBBB")
    assert G.pending_demotion_bytes() == 8
    assert G.drain_demotions() == [(b"a", b"AAAA"), (b"b", b"BBBB")]
    assert G.drain_demotions() == []
    assert G.pending_demotion_bytes() == 0


def test_budget_refuses_the_copy_instead_of_dropping_it(monkeypatch):
    """El presupuesto se aplica ANTES de copiar 28 MB, no después.

    La versión anterior copiaba y luego descartaba el snapshot más viejo: pagaba
    la RAM igual y encima perdía un bloque distinto del que causó la presión.
    Ahora se rechaza el snapshot y **se conserva el tag**, así que el bloque
    puede persistirse en un desalojo futuro.
    """
    G = _gate()
    monkeypatch.setenv("GENESIS_PN90_MAX_PENDING_MB", "1")
    G.tag_keys_persistence([b"a", b"b"])

    assert G.take_demotion_snapshot(b"a") is True
    G.record_demotion(b"a", b"x" * (2 * 1024 * 1024))   # supera el presupuesto

    assert G.take_demotion_snapshot(b"b") is False, "tiene que rechazar la copia"
    assert G.is_key_persisted(b"b") is True, "y conservar el tag para reintentar"
    assert G.stats()["snapshots_refused_budget"] == 1


def test_tagging_by_use_not_by_first_writer():
    """El gate mira quién USA el bloque, no quién lo escribió primero.

    Un bloque de KV es compartido: si el tag se decide sólo en prepare_store,
    un bloque que un agente efímero guardó primero no se persiste nunca aunque
    un agente persistente lo reutilice en cada turno.
    """
    G = _gate()
    assert G.is_key_persisted(b"compartido") is False
    assert G.note_persistent_use([b"compartido"]) == 1
    assert G.is_key_persisted(b"compartido") is True
    assert G.note_persistent_use([b"compartido"]) == 0, "no cuenta dos veces"
    assert G.stats()["tagged_by_use"] == 1


def test_tagged_set_is_bounded():
    """Red de seguridad: el set de tags no puede crecer sin techo."""
    G = _gate()
    G.note_persistent_use(i.to_bytes(4, "big") for i in range(G._MAX_TAGGED_KEYS + 500))
    assert G.stats()["tagged_keys"] <= G._MAX_TAGGED_KEYS


def test_failed_demotion_returns_the_tag():
    """Si ningún tier acepta el snapshot, el tag vuelve para reintentar.

    Antes el snapshot ya había salido de la cola y se perdía en silencio.
    """
    G = _gate()

    class _Roto:
        def submit_demote(self, key, snapshot, req_context=None):
            raise RuntimeError("disco lleno")

    G.record_demotion(b"k", b"DATA")
    assert G.submit_pending_demotions([_Roto()], None) == 0
    assert G.is_key_persisted(b"k") is True
    assert G.stats()["demotions_failed"] == 1


def test_snapshot_counted_once_per_block_not_per_tier():
    """Con dos tiers el contador duplicaba."""
    G = _gate()

    class _Tier:
        def submit_demote(self, key, snapshot, req_context=None):
            pass

    G.record_demotion(b"k", b"DATA")
    assert G.submit_pending_demotions([_Tier(), _Tier()], None) == 1
    assert G.stats()["demotions_submitted"] == 1


def test_submit_pending_demotions_reaches_every_capable_tier():
    G = _gate()

    class _Tier:
        def __init__(self):
            self.calls = []

        def submit_demote(self, key, snapshot, req_context=None):
            self.calls.append((key, snapshot))

    class _TierSinDemote:
        pass

    tier, mudo = _Tier(), _TierSinDemote()
    G.record_demotion(b"k", b"DATA")
    assert G.submit_pending_demotions([tier, mudo], None) == 1
    assert tier.calls == [(b"k", b"DATA")]
    assert G.drain_demotions() == [], "la cola queda vacía tras entregar"


def test_submit_survives_a_failing_tier():
    G = _gate()

    class _Roto:
        def submit_demote(self, key, snapshot, req_context=None):
            raise RuntimeError("disco lleno")

    G.record_demotion(b"k", b"DATA")
    assert G.submit_pending_demotions([_Roto()], None) == 0  # no propaga


# ─────────────────── job ids de democión ───────────────────


def test_demote_job_ids_are_negative_and_unique():
    """Los ids negativos no pueden chocar con el contador del tiering manager.

    El esquema viejo arrancaba en 1.000.000 y el contador de vLLM sube desde 0:
    en una sesión larga se cruzaban y un demote terminado popeaba el
    JobMetadata de un store real.
    """
    G = _gate()
    ids = [G.next_demote_job_id() for _ in range(100)]
    assert all(i < 0 for i in ids)
    assert len(set(ids)) == 100
    assert all(G.is_demote_job_id(i) for i in ids)


def test_real_job_ids_are_not_mistaken_for_demotes():
    G = _gate()
    for real_id in (0, 1, 999_999, 1_000_000, 5_000_000):
        assert G.is_demote_job_id(real_id) is False


# ─────────────────── higiene de estado ───────────────────


def test_clear_all_resets_everything():
    """Regresión: `clear_all()` sólo limpiaba `_PERSISTED_KEYS`."""
    G = _gate()
    G.tag_keys_persistence([b"k"])
    G.record_demotion(b"k2", b"DATA")
    G.next_demote_job_id()

    G.clear_all()

    assert G.stats()["tagged_keys"] == 0
    assert G.stats()["pending_snapshots"] == 0
    assert G.pending_demotion_bytes() == 0
    assert G.next_demote_job_id() == -1, "el contador de job id también se resetea"


# ─────────────────── métricas ───────────────────


def test_note_demote_preserves_the_unknown_label(monkeypatch):
    """Regresión: `note_demote` re-normalizaba una etiqueta ya normalizada.

    `agent_label()` devuelve el centinela `unknown`, que no está en la
    allowlist, así que la segunda pasada lo convertía en `other` y las series de
    democión nunca mostraban `agent="unknown"`.
    """
    M = _ktm()
    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)

    mgr = type("Mgr", (), {})()
    M.note_demote(mgr, job_id=-1, nbytes=1024, agent=M.agent_label(None))
    M.finish_job(mgr, type("R", (), {"job_id": -1, "success": True})())

    counters = s.drain()["counters"]
    key = f"kv_tier_bytes_total|agent={M.AGENT_UNKNOWN},direction=write,tier=disk"
    assert counters[key] == 1024


def test_note_demote_records_agent_bytes(monkeypatch):
    M = _ktm()
    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)

    mgr = type("Mgr", (), {})()
    M.note_demote(mgr, job_id=-7, nbytes=2048, agent="primary_nothink")
    M.finish_job(mgr, type("R", (), {"job_id": -7, "success": True})())

    counters = s.drain()["counters"]
    assert (
        counters["kv_tier_bytes_total|agent=primary_nothink,direction=write,tier=disk"]
        == 2048
    )
