# SPDX-License-Identifier: Apache-2.0
"""Tests de PN96 — admisión a L2 resistente a escaneo."""

from __future__ import annotations

import ast
import os
import types

import pytest

_MANAGER = "v1/kv_offload/tiering/manager.py"
_SPEC = "v1/kv_offload/tiering/spec.py"
_CPU = "v1/kv_offload/cpu/manager.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _P():
    from vllm._genesis.wiring.hybrid import patch_N96_l2_scan_resistant_admission as P

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
        if root and os.path.exists(os.path.join(root, _MANAGER)):
            return root
    return None


VLLM_ROOT = _vllm_tree()
_needs_tree = pytest.mark.skipif(VLLM_ROOT is None, reason="no hay arbol de vLLM")


def _read(rel):
    with open(os.path.join(VLLM_ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ─────────────────── anclas ───────────────────


@_needs_tree
def test_anchors_appear_exactly_once():
    P = _P()
    assert _read(_MANAGER).count(P.INIT_OLD) == 1
    spec = _read(_SPEC)
    assert spec.count(P.BUILD_OLD) == 1
    assert spec.count(P.GUARD_OLD) == 1


@_needs_tree
def test_patched_sources_compile():
    P = _P()
    compile(_read(_MANAGER).replace(P.INIT_OLD, P.INIT_NEW), _MANAGER, "exec")
    spec = _read(_SPEC).replace(P.BUILD_OLD, P.BUILD_NEW).replace(
        P.GUARD_OLD, P.GUARD_NEW
    )
    compile(spec, _SPEC, "exec")


@_needs_tree
def test_guard_uses_a_local_import():
    """`CPUOffloadingManager` NO esta importado en tiering/spec.py.

    Regresion: la primera version del guard lo referenciaba directo y habria
    tirado NameError en el init del engine con store_threshold>=2 — es decir,
    justo en la unica configuracion donde el codigo corre.
    """
    P = _P()
    spec = _read(_SPEC)
    assert "CPUOffloadingManager" not in spec.split("class ")[0], (
        "si upstream lo importa, este guard se puede simplificar"
    )
    assert "from vllm.v1.kv_offload.cpu.manager import" in P.GUARD_NEW
    assert "_g96_base.prepare_store" in P.GUARD_NEW


@_needs_tree
def test_threshold_reaches_the_base_class():
    """El filtro vive en CPUOffloadingManager; sin reenvio no se activa nunca."""
    P = _P()
    assert "store_threshold=store_threshold" in P.INIT_NEW
    assert "store_threshold=_g96_thr" in P.BUILD_NEW
    cpu = _read(_CPU)
    assert "self.counts.get(k, 0) >= self.store_threshold" in cpu
    assert "store_threshold >= 2" in cpu, "asi se decide si se lleva el contador"


@_needs_tree
def test_default_of_one_is_inert():
    """Con el default no cambia nada: counts queda en None."""
    P = _P()
    assert 'self.extra_config.get("store_threshold", 1) or 1' in P.BUILD_NEW
    assert "store_threshold: int = 1," in P.INIT_NEW


# ─────────────────── comportamiento del guard ───────────────────


def _eval_guard(pn90_activo: bool, threshold: int):
    """Ejecuta la logica del guard con la clase base y el override simulados."""
    P = _P()
    cuerpo = P.GUARD_NEW.split("\n")
    # nos quedamos con el codigo, sin los comentarios ni el marker
    codigo = [l for l in cuerpo if l.strip() and not l.strip().startswith("#")]
    indent = min(len(l) - len(l.lstrip()) for l in codigo)
    src = "\n".join(l[indent:] for l in codigo)

    class _Base:
        def prepare_store(self):
            pass

    if pn90_activo:

        class _Prim(_Base):
            def prepare_store(self):  # el override de PN90
                pass

    else:

        class _Prim(_Base):
            pass

    mod = types.ModuleType("vllm.v1.kv_offload.cpu.manager")
    mod.CPUOffloadingManager = _Base
    import sys

    sys.modules["vllm.v1.kv_offload.cpu.manager"] = mod
    ns = {
        "self": types.SimpleNamespace(extra_config={"store_threshold": threshold}),
        "CPUPrimaryTierOffloadingManager": _Prim,
    }
    exec(compile(ast.parse(src), "<guard>", "exec"), ns)


def test_guard_still_raises_without_pn90():
    """Sin la cascada cortada, filtrar la admision dejaria el SSD vacio."""
    with pytest.raises(ValueError, match="PN90"):
        _eval_guard(pn90_activo=False, threshold=2)


def test_guard_allows_with_pn90():
    _eval_guard(pn90_activo=True, threshold=2)  # no debe levantar


@pytest.mark.parametrize("thr", [0, 1])
def test_guard_never_fires_below_two(thr):
    _eval_guard(pn90_activo=False, threshold=thr)


# ─────────────────── convivencia con PN90 ───────────────────


@_needs_tree
def test_pn90_then_pn96_both_apply_in_order():
    """PN90 se aplica ANTES y tambien toca tiering/manager.py y spec.py."""
    from vllm._genesis.wiring.hybrid import patch_N90_kv_disk_write_gating as P90

    P96 = _P()
    txt = _read(_MANAGER)
    for p in P90._patchers() or []:
        if not p.target_file.endswith("tiering/manager.py"):
            continue
        for sp in p.sub_patches:
            if sp.anchor in txt:
                txt = txt.replace(sp.anchor, sp.replacement)
    assert txt.count(P96.INIT_OLD) == 1, "PN90 rompio el ancla de PN96"
    compile(txt.replace(P96.INIT_OLD, P96.INIT_NEW), _MANAGER, "exec")


@_needs_tree
def test_pn96_does_not_disturb_the_pn90_override():
    P = _P()
    out = _read(_MANAGER).replace(P.INIT_OLD, P.INIT_NEW)
    assert out.count("def prepare_store") == _read(_MANAGER).count("def prepare_store")


# ─────────── apply() de verdad, no solo los patchers ───────────
#
# El test de la pila llama `patcher.apply()` directo y se saltea el `apply()`
# del modulo. Por eso no cazo que PN96 usara `MultiFilePatchTransaction.apply()`,
# que no existe — la API es `apply_or_skip()`. Reventaba recien en el arranque
# del contenedor, con el parche silenciosamente no aplicado.


@pytest.mark.parametrize(
    "mod_name",
    [
        "patch_N94_per_rank_host_register",
        "patch_N95_l2_occupancy_metrics",
        "patch_N96_l2_scan_resistant_admission",
    ],
)
@_needs_tree
def test_module_level_apply_actually_applies(mod_name, tmp_path, monkeypatch):
    import importlib
    import shutil

    tree = tmp_path / "vllm"
    shutil.copytree(VLLM_ROOT, tree, symlinks=True)

    from vllm._genesis import guards

    monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tree))
    mod = importlib.import_module(f"vllm._genesis.wiring.hybrid.{mod_name}")
    monkeypatch.setattr(mod, "vllm_install_root", lambda: str(tree))

    status, reason = mod.apply()
    assert status == "applied", f"{mod_name}: {status} — {reason}"
    assert mod.is_applied() is True

    # idempotente: la segunda pasada no puede romper ni duplicar
    status2, _ = mod.apply()
    assert status2 in ("applied", "skipped")

    for path in sorted(tree.rglob("*.py")):
        text = path.read_text(errors="ignore")
        if "_GENESIS_PN9" in text or "[Genesis PN9" in text:
            compile(text, str(path), "exec")
