# SPDX-License-Identifier: Apache-2.0
"""Tests de PN99 — L2/L3 comprimidos a 4 bits cuantizando en la GPU."""

from __future__ import annotations

import os

import pytest

_REL_SPEC = "v1/kv_offload/cpu/spec.py"
_REL_W = "v1/kv_offload/cpu/gpu_worker.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
torch = pytest.importorskip("torch", reason="el codec necesita torch")


def _P():
    from vllm._genesis.wiring.hybrid import patch_N99_gpu_compressed_l2 as P

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
        if r and os.path.exists(os.path.join(r, _REL_SPEC)):
            return r
    return None


ROOT = _tree()
_needs = pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")


# ─────────────────── el codec ───────────────────


def test_ratio_es_1_78():
    from vllm._genesis import kv_gpu_codec as C

    assert abs(C.ratio() - 8 / 4.5) < 1e-9


def test_tamano_comprimido_del_bloque_real_cae_en_pagina():
    """8.146.944 = 1989 * 4096. PN94 registra por rank y exige alineacion."""
    from vllm._genesis import kv_gpu_codec as C

    crudo = 14_483_456
    assert C.tamano_comprimido(crudo) == 8_146_944
    assert C.tamano_comprimido(crudo) % 4096 == 0


def test_roundtrip_conserva_la_senal():
    from vllm._genesis import kv_gpu_codec as C

    torch.manual_seed(0)
    n = 32 * 4096
    x = torch.randn(n) * 0.35
    x[::997] *= 12  # outliers, que es lo que rompe una cuantizacion ingenua
    src = x.to(torch.float8_e4m3fn).view(torch.uint8)
    comp = C.comprimir(src)
    assert comp.numel() == C.tamano_comprimido(n)
    rec = C.descomprimir(comp, n)
    assert rec.numel() == n
    assert C.error_relativo(src, rec) < 0.16


def test_los_nan_no_envenenan_la_escala_del_grupo():
    """fp8_e4m3 usa 0x7F/0xFF como NaN; sin nan_to_num se pierde el grupo."""
    from vllm._genesis import kv_gpu_codec as C

    n = 64
    src = torch.full((n,), 0x30, dtype=torch.uint8)
    src[5] = 0x7F  # NaN
    rec = C.descomprimir(C.comprimir(src), n)
    vals = rec.view(torch.float8_e4m3fn).to(torch.float32)
    assert torch.isfinite(vals[:5]).all() and torch.isfinite(vals[6:]).all()
    assert vals[:5].abs().sum() > 0, "el grupo no puede colapsar a cero"


def test_rechaza_tamanos_que_no_son_multiplo_del_grupo():
    from vllm._genesis import kv_gpu_codec as C

    with pytest.raises(ValueError):
        C.tamano_comprimido(33)


# ─────────────────── el gate ───────────────────


def test_gate_apagado_por_defecto(monkeypatch):
    from vllm._genesis import pn99_gate as G

    monkeypatch.delenv("GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2", raising=False)
    G.reset()
    assert G.habilitado() is False
    assert G.activo() is False


def test_habilitado_no_implica_activo(monkeypatch):
    """Pedirlo por env no alcanza: tiene que resultar aplicable."""
    from vllm._genesis import pn99_gate as G

    monkeypatch.setenv("GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2", "1")
    G.reset()
    assert G.habilitado() is True
    assert G.activo() is False
    G.desactivar("block_size_factor != 1")
    assert G.activo() is False and "block_size_factor" in G.motivo()


def test_activar_memoriza_los_tamanos(monkeypatch):
    from vllm._genesis import pn99_gate as G

    G.reset()
    G.activar(14_483_456, 8_146_944)
    assert G.activo() and G.bytes_crudos() == 14_483_456
    assert G.bytes_comprimidos() == 8_146_944
    G.reset()
    assert not G.activo()


# ─────────────────── anclas ───────────────────


def _ya_aplicado(P, rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return P.GENESIS_PN99_MARKER in fh.read()


@_needs
def test_anclas_unicas_y_compilan():
    P = _P()
    for rel, pares in (
        (_REL_SPEC, [("SPEC_OLD", "SPEC_NEW")]),
        (_REL_W, [("ASSERT_OLD", "ASSERT_NEW"), ("COPY_OLD", "COPY_NEW")]),
    ):
        if _ya_aplicado(P, rel):
            pytest.skip(f"PN99 ya aplicado sobre {rel}")
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            out = fh.read()
        for o, n in pares:
            assert out.count(getattr(P, o)) == 1, f"{rel}:{o}"
            out = out.replace(getattr(P, o), getattr(P, n))
        compile(out, rel, "exec")
        assert P.GENESIS_PN99_MARKER in out


@_needs
def test_convive_con_pn82_pn94_en_gpu_worker():
    """PN82 y PN94 tocan el mismo archivo y se aplican antes."""
    from vllm._genesis.wiring.hybrid import patch_N82_host_register_sticky_error as P82
    from vllm._genesis.wiring.hybrid import patch_N94_per_rank_host_register as P94

    P = _P()
    if _ya_aplicado(P, _REL_W):
        pytest.skip("PN99 ya aplicado sobre gpu_worker.py")
    with open(os.path.join(ROOT, _REL_W), encoding="utf-8") as fh:
        txt = fh.read()
    txt = txt.replace(P82.ANCHOR_OLD, P82.ANCHOR_NEW)
    txt = txt.replace(P94.ANCHOR_OLD, P94.ANCHOR_NEW)
    assert txt.count(P.ASSERT_OLD) == 1
    assert txt.count(P.COPY_OLD) == 1
    out = txt.replace(P.ASSERT_OLD, P.ASSERT_NEW).replace(P.COPY_OLD, P.COPY_NEW)
    compile(out, _REL_W, "exec")


@_needs
def test_apply_de_modulo_aplica_de_verdad(tmp_path, monkeypatch):
    import shutil

    if _ya_aplicado(_P(), _REL_SPEC):
        pytest.skip("PN99 ya aplicado sobre el arbol de origen")
    tree = tmp_path / "vllm"
    shutil.copytree(ROOT, tree, symlinks=True)
    from vllm._genesis import guards

    monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tree))
    P = _P()
    monkeypatch.setattr(P, "vllm_install_root", lambda: str(tree))
    status, reason = P.apply()
    assert status == "applied", reason
    assert P.is_applied() is True
    for rel in (_REL_SPEC, _REL_W):
        compile((tree / rel).read_text(), rel, "exec")
