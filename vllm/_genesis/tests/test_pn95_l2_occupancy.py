# SPDX-License-Identifier: Apache-2.0
"""Tests de PN95 — ocupación y desalojos del tier L2 (RAM)."""

from __future__ import annotations

import ast
import os
import types

import pytest

_REL = "v1/kv_offload/cpu/manager.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _P():
    from vllm._genesis.wiring.hybrid import patch_N95_l2_occupancy_metrics as P

    return P


def _vllm_tree():
    candidates = []
    try:
        import vllm

        candidates.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    candidates.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    for root in candidates:
        if root and os.path.exists(os.path.join(root, _REL)):
            return root
    return None


VLLM_ROOT = _vllm_tree()
_needs_tree = pytest.mark.skipif(VLLM_ROOT is None, reason="no hay arbol de vLLM")

BLOCK = 28_966_912


@pytest.fixture(autouse=True)
def _sink(monkeypatch):
    from vllm._genesis import kv_sparse_gdn as G
    from vllm._genesis import kv_tier_metrics as M

    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)
    G.reset_state()
    G.register_group_spec(0, types.SimpleNamespace(), BLOCK)
    yield s
    G.reset_state()


def _run(snippet_attr, ns):
    """Ejecuta el bloque inyectado por PN95, desindentado."""
    P = _P()
    body = getattr(P, snippet_attr)
    indent = min(
        (len(l) - len(l.lstrip()) for l in body.splitlines() if l.strip()), default=0
    )
    lines = [l[indent:] if len(l) >= indent else l for l in body.splitlines()]
    # OCCUP_NEW termina con la linea del ancla (`if stored_keys ...`), que en el
    # archivo real lleva su cuerpo detras. Fuera de contexto no compila.
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("if ")):
        lines.pop()
    src = "\n".join(lines)
    exec(compile(ast.parse(src), "<pn95>", "exec"), ns)
    return ns


# ─────────────────── desalojos ───────────────────


def test_eviction_counts_blocks_and_bytes(_sink):
    ns = {
        "evicted": [(f"k{i}", object()) for i in range(3)],
        "to_evict": [],
        "self": types.SimpleNamespace(_free_block=lambda b: None),
        "key": None,
        "block": None,
    }
    _run("EVICT_NEW", ns)
    c = _sink.drain()["counters"]
    assert c["kv_tier_evictions_total|reason=capacity,tier=ram"] == 3
    assert c["kv_tier_evicted_bytes_total|reason=capacity,tier=ram"] == 3 * BLOCK


def test_eviction_is_what_becomes_an_ssd_write(_sink):
    """El punto del parche: 0 desalojos de L2 == 0 escrituras al SSD.

    Si no se desaloja nada, el contador no aparece, y eso es informacion:
    distingue 'no hubo presion' de 'la carga fue chica'.
    """
    ns = {
        "evicted": [],
        "to_evict": [],
        "self": types.SimpleNamespace(_free_block=lambda b: None),
    }
    _run("EVICT_NEW", ns)
    # `drain()` devuelve None cuando no se registro nada.
    assert not (_sink.drain() or {}).get("counters", {})


# ─────────────────── ocupacion ───────────────────


def _mgr(num_blocks, libres):
    return types.SimpleNamespace(
        _num_blocks=num_blocks,
        _get_num_free_blocks=lambda: libres,
    )


def test_occupancy_publishes_used_and_capacity(_sink):
    ns = {"self": _mgr(222, 169), "stored_keys": [], "OffloadingEvent": object}
    ns["self"].events = None
    _run("OCCUP_NEW", ns)
    g = _sink.drain()["gauges"]
    assert g["kv_tier_bytes_used|tier=ram"] == 53 * BLOCK
    assert g["kv_tier_capacity_bytes|tier=ram"] == 222 * BLOCK


def test_occupancy_is_a_gauge_last_write_wins(_sink):
    for libres in (200, 100, 10):
        ns = {"self": _mgr(222, libres), "stored_keys": [], "OffloadingEvent": object}
        ns["self"].events = None
        _run("OCCUP_NEW", ns)
    g = _sink.drain()["gauges"]
    assert g["kv_tier_bytes_used|tier=ram"] == (222 - 10) * BLOCK


def test_occupancy_skipped_when_block_size_unknown(_sink):
    """Sin el tamano de bloque de PN93 se omite, no se inventa un numero."""
    from vllm._genesis import kv_sparse_gdn as G

    G.reset_state()
    ns = {"self": _mgr(222, 169), "stored_keys": [], "OffloadingEvent": object}
    ns["self"].events = None
    _run("OCCUP_NEW", ns)
    assert not (_sink.drain() or {}).get("gauges", {})


def test_occupancy_runs_even_with_events_disabled(_sink):
    """Se engancha ANTES del `if ... self.events is not None` a proposito."""
    ns = {"self": _mgr(222, 200), "stored_keys": ["a"], "OffloadingEvent": object}
    ns["self"].events = None  # eventos apagados
    _run("OCCUP_NEW", ns)
    assert _sink.drain()["gauges"]["kv_tier_bytes_used|tier=ram"] == 22 * BLOCK


def test_metrics_failure_never_propagates(_sink, monkeypatch):
    from vllm._genesis import kv_tier_metrics as M

    def _boom(*a, **k):
        raise RuntimeError("sink roto")

    monkeypatch.setattr(M, "set_occupancy", _boom)
    ns = {"self": _mgr(222, 169), "stored_keys": [], "OffloadingEvent": object}
    ns["self"].events = None
    _run("OCCUP_NEW", ns)  # no debe levantar


# ─────────────────── anclas ───────────────────


@_needs_tree
def test_anchors_appear_exactly_once():
    P = _P()
    with open(os.path.join(VLLM_ROOT, _REL), encoding="utf-8") as fh:
        txt = fh.read()
    assert txt.count(P.EVICT_OLD) == 1
    assert txt.count(P.OCCUP_OLD) == 1


@_needs_tree
def test_patched_source_compiles_with_both_markers():
    P = _P()
    path = os.path.join(VLLM_ROOT, _REL)
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    out = txt.replace(P.EVICT_OLD, P.EVICT_NEW).replace(P.OCCUP_OLD, P.OCCUP_NEW)
    compile(out, path, "exec")
    assert out.count(P.GENESIS_PN95_MARKER) == 2


@_needs_tree
def test_occupancy_hook_lands_before_the_events_block():
    """Si cayera adentro del `if self.events`, no mediria con eventos apagados."""
    P = _P()
    with open(os.path.join(VLLM_ROOT, _REL), encoding="utf-8") as fh:
        out = fh.read().replace(P.EVICT_OLD, P.EVICT_NEW).replace(
            P.OCCUP_OLD, P.OCCUP_NEW
        )
    i_hook = out.index("_g95_usados = self._num_blocks")
    i_ev = out.index("if stored_keys and self.events is not None:")
    assert i_hook < i_ev


# ─────────────── el override de PN90 tapa el hook de la clase base ───────────────


def test_pn90_override_also_counts_evictions():
    """REGRESION: el hook de desalojo en CPUOffloadingManager.prepare_store
    es CODIGO MUERTO con el tiering manager activo.

    PN90 define `@override def prepare_store` en
    CPUPrimaryTierOffloadingManager, que hereda de CPUOffloadingManager. El
    metodo parcheado por PN95 en la clase base nunca se ejecuta. Detectado el
    2026-08-23 leyendo el archivo vivo del contenedor: la ocupacion se
    publicaba y los desalojos no.
    """
    from vllm._genesis.wiring.hybrid import patch_N90_kv_disk_write_gating as P90

    texto = "".join(
        v
        for k, v in vars(P90).items()
        if k.endswith("_NEW") and isinstance(v, str)
    )
    assert "def prepare_store" in texto, "cambio la forma del override de PN90"
    assert "note_eviction" in texto, (
        "el override de PN90 tiene que contar los desalojos: es el unico "
        "prepare_store que corre de verdad"
    )
    assert "_g95_bb = _g95_gdn.block_write_bytes()" in texto


def test_both_prepare_store_paths_count_evictions():
    """Base y override cuentan igual: si se apaga PN90, la metrica sigue."""
    from vllm._genesis.wiring.hybrid import patch_N90_kv_disk_write_gating as P90

    P95 = _P()
    override = "".join(
        v for k, v in vars(P90).items() if k.endswith("_NEW") and isinstance(v, str)
    )
    for frag in ("note_eviction", '"ram"' if '"ram"' in P95.EVICT_NEW else "'ram'"):
        assert frag.strip('"\'') in P95.EVICT_NEW.replace('"', "'").replace("'", "")  \
            or frag in P95.EVICT_NEW
    assert "note_eviction" in P95.EVICT_NEW and "note_eviction" in override
