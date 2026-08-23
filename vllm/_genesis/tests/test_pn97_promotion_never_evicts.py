# SPDX-License-Identifier: Apache-2.0
"""Tests de PN97 — una promoción L3→L2 nunca desaloja."""

from __future__ import annotations

import ast
import os
import types

import pytest

_REL = "v1/kv_offload/tiering/manager.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _P():
    from vllm._genesis.wiring.hybrid import patch_N97_promotion_never_evicts as P

    return P


def _vllm_tree():
    cands = []
    try:
        import vllm

        cands.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    cands.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    for r in cands:
        if r and os.path.exists(os.path.join(r, _REL)):
            return r
    return None


VLLM_ROOT = _vllm_tree()
_needs_tree = pytest.mark.skipif(VLLM_ROOT is None, reason="no hay arbol de vLLM")


def _read():
    with open(os.path.join(VLLM_ROOT, _REL), encoding="utf-8") as fh:
        return fh.read()


# ─────────── el metodo inyectado, ejecutado de verdad ───────────


def _make_tier(libres, ya_en_cache=(), sink=None):
    """Instancia con el cuerpo de prepare_write_no_evict inyectado por PN97."""
    P = _P()
    src = P.METHOD_NEW
    # nos quedamos con la funcion, sin el marker ni el get_kv_memoryview final
    lineas = src.splitlines()
    ini = next(i for i, l in enumerate(lineas) if "def prepare_write_no_evict" in l)
    fin = next(i for i, l in enumerate(lineas) if "def get_kv_memoryview" in l)
    cuerpo = "\n".join(l[4:] for l in lineas[ini:fin])

    llamadas = []

    class _Policy:
        def get(self, k):
            return object() if k in ya_en_cache else None

    ns = {}
    exec(compile(ast.parse(cuerpo), "<pn97>", "exec"), ns)

    tier = types.SimpleNamespace()
    tier._policy = _Policy()
    tier._get_num_free_blocks = lambda: libres
    tier.prepare_write = lambda keys, ctx: llamadas.append(list(keys)) or "OK"
    tier.prepare_write_no_evict = ns["prepare_write_no_evict"].__get__(tier)
    tier.llamadas = llamadas
    return tier


def test_refuses_when_there_is_no_free_slot():
    """EL punto del parche: sin slot libre no se promueve, no se desaloja."""
    t = _make_tier(libres=0)
    assert t.prepare_write_no_evict(["k"], None) is None
    assert t.llamadas == [], "no puede tocar el camino que desaloja"


def test_promotes_when_there_is_room():
    t = _make_tier(libres=5)
    assert t.prepare_write_no_evict(["k"], None) == "OK"
    assert t.llamadas == [["k"]]


def test_exact_fit_is_allowed():
    t = _make_tier(libres=2)
    assert t.prepare_write_no_evict(["a", "b"], None) == "OK"


def test_one_block_over_is_refused():
    t = _make_tier(libres=1)
    assert t.prepare_write_no_evict(["a", "b"], None) is None


def test_keys_already_in_cache_do_not_need_a_slot():
    """Si el bloque ya esta en L2 no hay que allocar nada: no debe rechazar."""
    t = _make_tier(libres=0, ya_en_cache={"a"})
    assert t.prepare_write_no_evict(["a"], None) == "OK"


def test_only_the_missing_keys_count_against_free_space():
    t = _make_tier(libres=1, ya_en_cache={"a", "b"})
    assert t.prepare_write_no_evict(["a", "b", "c"], None) == "OK"
    t2 = _make_tier(libres=1, ya_en_cache={"a"})
    assert t2.prepare_write_no_evict(["a", "b", "c"], None) is None


def test_refusal_is_counted(monkeypatch):
    from vllm._genesis import kv_tier_metrics as M

    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)
    _make_tier(libres=0).prepare_write_no_evict(["k"], None)
    c = (s.drain() or {}).get("counters", {})
    assert c.get("kv_tier_promotion_refused_total|tier=ram") == 1


def test_metrics_failure_never_blocks_the_refusal(monkeypatch):
    from vllm._genesis import kv_tier_metrics as M

    def _boom(*a, **k):
        raise RuntimeError("sink roto")

    monkeypatch.setattr(M, "note_promotion_refused", _boom)
    assert _make_tier(libres=0).prepare_write_no_evict(["k"], None) is None


# ─────────── anclas y convivencia ───────────


@_needs_tree
def test_anchors_appear_exactly_once():
    P = _P()
    txt = _read()
    assert txt.count(P.METHOD_OLD) == 1
    assert txt.count(P.CALL_OLD) == 1


@_needs_tree
def test_patched_source_compiles():
    P = _P()
    out = _read().replace(P.METHOD_OLD, P.METHOD_NEW).replace(P.CALL_OLD, P.CALL_NEW)
    compile(out, _REL, "exec")
    assert out.count(P.GENESIS_PN97_MARKER) == 2


@_needs_tree
def test_promotion_no_longer_calls_the_evicting_path():
    """Regresion: si vuelve a llamar prepare_write directo, el bucle vuelve."""
    P = _P()
    out = _read().replace(P.METHOD_OLD, P.METHOD_NEW).replace(P.CALL_OLD, P.CALL_NEW)
    ini = out.index("def _initiate_promotion")
    fin = out.index("def _flush_pending_promotions")
    cuerpo = out[ini:fin]
    assert "prepare_write_no_evict" in cuerpo
    # Solo la LLAMADA importa; el docstring menciona prepare_write a proposito.
    assert "self.primary_tier.prepare_write(" not in cuerpo


@_needs_tree
def test_coexists_with_pn90_and_pn96_on_the_same_file():
    from vllm._genesis.wiring.hybrid import patch_N90_kv_disk_write_gating as P90
    from vllm._genesis.wiring.hybrid import patch_N96_l2_scan_resistant_admission as P96

    P97 = _P()
    txt = _read()
    for p in P90._patchers() or []:
        if p.target_file.endswith(_REL):
            for sp in p.sub_patches:
                if sp.anchor in txt:
                    txt = txt.replace(sp.anchor, sp.replacement)
    if P96.INIT_OLD in txt:
        txt = txt.replace(P96.INIT_OLD, P96.INIT_NEW)
    assert txt.count(P97.METHOD_OLD) == 1, "PN90/PN96 rompieron el ancla de PN97"
    assert txt.count(P97.CALL_OLD) == 1
    out = txt.replace(P97.METHOD_OLD, P97.METHOD_NEW).replace(
        P97.CALL_OLD, P97.CALL_NEW
    )
    compile(out, _REL, "exec")


@_needs_tree
def test_module_level_apply_actually_applies(tmp_path, monkeypatch):
    import shutil

    tree = tmp_path / "vllm"
    shutil.copytree(VLLM_ROOT, tree, symlinks=True)
    from vllm._genesis import guards

    monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tree))
    P = _P()
    monkeypatch.setattr(P, "vllm_install_root", lambda: str(tree))
    status, reason = P.apply()
    assert status == "applied", reason
    assert P.is_applied() is True
    compile((tree / _REL).read_text(), _REL, "exec")
