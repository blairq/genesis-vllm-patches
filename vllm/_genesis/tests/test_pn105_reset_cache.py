# SPDX-License-Identifier: Apache-2.0
"""Tests de PN105 — reset_cache del tiering manager."""

from __future__ import annotations

import os

import pytest

_REL = "v1/kv_offload/tiering/manager.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _P():
    from vllm._genesis.wiring.hybrid import patch_N105_tiering_reset_cache as P

    return P


def _tree():
    c = []
    try:
        import vllm

        c.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    c.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    for r in c:
        if r and os.path.exists(os.path.join(r, _REL)):
            return r
    return None


ROOT = _tree()
_needs = pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")


@_needs
def test_el_bug_existe_upstream():
    """La razon de ser del parche: la clase NO define reset_cache.

    Si upstream lo agrega, este test falla y hay que revisar si PN105 sigue
    haciendo falta.
    """
    with open(os.path.join(ROOT, _REL), encoding="utf-8") as fh:
        txt = fh.read()
    P = _P()
    if P.GENESIS_PN105_MARKER in txt:
        pytest.skip("PN105 ya aplicado")
    ini = txt.index("class TieringOffloadingManager")
    assert "def reset_cache" not in txt[ini:], (
        "upstream ahora implementa reset_cache: revisar si PN105 sigue siendo necesario"
    )


@_needs
def test_el_no_op_de_la_base_devuelve_None_no_False():
    """Por eso el chequeo `is False` del scheduler nunca detectaba la falla."""
    base = os.path.join(ROOT, "v1/kv_offload/base.py")
    with open(base, encoding="utf-8") as fh:
        txt = fh.read()
    i = txt.index("def reset_cache")
    cuerpo = txt[i : i + 260]
    assert "return" in cuerpo
    assert "False" not in cuerpo.split("\n")[-3:][0]


@_needs
def test_ancla_unica_y_compila():
    P = _P()
    with open(os.path.join(ROOT, _REL), encoding="utf-8") as fh:
        txt = fh.read()
    if P.GENESIS_PN105_MARKER in txt:
        pytest.skip("PN105 ya aplicado")
    assert txt.count(P.ANCHOR_OLD) == 1
    out = txt.replace(P.ANCHOR_OLD, P.ANCHOR_NEW)
    compile(out, _REL, "exec")
    assert "self.primary_tier.reset_cache()" in out


@_needs
def test_delega_en_el_primario_que_si_lo_implementa():
    cpu = os.path.join(ROOT, "v1/kv_offload/cpu/manager.py")
    with open(cpu, encoding="utf-8") as fh:
        txt = fh.read()
    i = txt.index("def reset_cache")
    cuerpo = txt[i : i + 900]  # el comentario de upstream es largo
    assert "self._policy.clear()" in cuerpo
    assert "_num_allocated_blocks = 0" in cuerpo


@_needs
def test_limpia_tambien_el_estado_de_los_parches_genesis():
    """Un reset que deja el score y la atribucion vivos miente igual."""
    P = _P()
    assert "kv_tier_attribution" in P.ANCHOR_NEW
    assert "_g100_reg" in P.ANCHOR_NEW
    assert "_g102" in P.ANCHOR_NEW
