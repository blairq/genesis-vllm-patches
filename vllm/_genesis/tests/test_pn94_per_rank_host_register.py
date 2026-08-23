# SPDX-License-Identifier: Apache-2.0
"""Tests de PN94 — registro de host pinneado por rank.

El código de PN94 se INYECTA en `pin_mmap_region`, así que no hay función
importable que testear. Estos tests extraen el bloque inyectado del propio
parche y lo ejecutan contra un `cudart` y una `SharedOffloadRegion` falsos.
Así se testea el texto que realmente se instala, no una reimplementación.
"""

from __future__ import annotations

import ast
import os
import types

import pytest

PAGE = 4096


def _anchors():
    from vllm._genesis.wiring.hybrid import patch_N94_per_rank_host_register as P

    return P


def _make_region(rank, num_blocks=4, cpu_page_size=2 * PAGE, world=2, page_size=PAGE):
    r = types.SimpleNamespace()
    r.rank = rank
    r.num_blocks = num_blocks
    r.page_size = page_size
    r._row_stride = world * cpu_page_size
    r.total_size_bytes = num_blocks * r._row_stride
    if rank is not None:
        r._worker_offset = rank * cpu_page_size
        r._worker_area_end = (rank + 1) * cpu_page_size
    r.is_pinned = False
    r._base = types.SimpleNamespace(data_ptr=lambda: _make_region.base_ptr)
    return r


_make_region.base_ptr = 0


class _FakeCudart:
    """Modela cudaHostRegister: falla si el rango pisa páginas ya registradas."""

    def __init__(self, fail_at=None, shared_pages=None):
        self.registered = shared_pages if shared_pages is not None else set()
        self.calls = []
        self.unregistered = []
        self.fail_at = fail_at

    def cudaHostRegister(self, ptr, length, flags):
        self.calls.append((ptr, length))
        if self.fail_at is not None and len(self.calls) > self.fail_at:
            return types.SimpleNamespace(value=1)
        pages = set(range(ptr // PAGE, (ptr + length + PAGE - 1) // PAGE))
        if pages & self.registered:
            return types.SimpleNamespace(value=1)  # cudaErrorInvalidValue
        self.registered |= pages
        return types.SimpleNamespace(value=0)

    def cudaHostUnregister(self, ptr):
        self.unregistered.append(ptr)
        return types.SimpleNamespace(value=0)


def _run(region, cudart, base_ptr=0):
    """Ejecuta el bloque inyectado por PN94 y devuelve (result, namespace)."""
    _make_region.base_ptr = base_ptr
    if not hasattr(region, "_base"):
        region._base = types.SimpleNamespace(data_ptr=lambda: base_ptr)
    P = _anchors()
    body = P.ANCHOR_NEW
    # El bloque vive dentro de una función; lo desindentamos 4 espacios.
    lines = [ln[4:] if ln.startswith("    ") else ln for ln in body.splitlines()]
    src = "\n".join(lines)

    warnings: list = []
    infos: list = []
    logger = types.SimpleNamespace(
        warning=lambda *a: warnings.append(a),
        info=lambda *a: infos.append(a),
        debug=lambda *a: None,
        error=lambda *a: None,
    )
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(cudart=lambda: cudart))
    ns = {
        "region": region,
        "logger": logger,
        "torch": torch,
        "base_ptr": base_ptr,
    }
    exec(compile(ast.parse(src), "<pn94>", "exec"), ns)
    ns["_warnings"], ns["_infos"] = warnings, infos
    return ns["result"], ns


# ─────────────────── el bug que arregla ───────────────────


def test_two_ranks_no_longer_collide():
    """LA regresión: con el registro entero el segundo rank falla; por slots, no.

    Los dos ranks mapean el mismo /dev/shm, así que comparten el set de
    páginas registradas — eso es lo que modela `shared_pages`.
    """
    shared: set[int] = set()
    r0 = _make_region(rank=0)
    r1 = _make_region(rank=1)

    c0 = _FakeCudart(shared_pages=shared)
    res0, ns0 = _run(r0, c0)
    c1 = _FakeCudart(shared_pages=shared)
    res1, ns1 = _run(r1, c1)

    assert res0.value == 0, "rank 0 tiene que pinnear"
    assert res1.value == 0, "rank 1 TAMBIEN, que es el punto del parche"
    assert len(c0.calls) == r0.num_blocks
    assert len(c1.calls) == r1.num_blocks


def test_upstream_behaviour_would_collide():
    """Contraprueba: registrar la region entera desde los dos ranks sí choca."""
    shared: set[int] = set()
    r = _make_region(rank=0)
    c = _FakeCudart(shared_pages=shared)
    assert c.cudaHostRegister(0, r.total_size_bytes, 0).value == 0
    assert c.cudaHostRegister(0, r.total_size_bytes, 0).value == 1


def test_slots_of_the_two_ranks_are_disjoint():
    r0, r1 = _make_region(rank=0), _make_region(rank=1)
    c0, c1 = _FakeCudart(), _FakeCudart()
    _run(r0, c0)
    _run(r1, c1)
    p0 = {p for ptr, ln in c0.calls for p in range(ptr // PAGE, (ptr + ln) // PAGE)}
    p1 = {p for ptr, ln in c1.calls for p in range(ptr // PAGE, (ptr + ln) // PAGE)}
    assert p0 and p1
    assert not (p0 & p1), "los slots de rank 0 y rank 1 no pueden solaparse"


def test_slots_match_the_worker_offset_layout():
    """Las direcciones tienen que ser las mismas que usa create_next_view."""
    cps = 2 * PAGE
    r = _make_region(rank=1, num_blocks=3, cpu_page_size=cps)
    c = _FakeCudart()
    _run(r, c, base_ptr=1 << 20)
    esperado = [(1 << 20) + b * r._row_stride + r._worker_offset for b in range(3)]
    assert [ptr for ptr, _ in c.calls] == esperado
    assert all(ln == cps for _, ln in c.calls)


def test_every_byte_of_the_rank_area_is_covered():
    r = _make_region(rank=0, num_blocks=5)
    c = _FakeCudart()
    _run(r, c)
    cubierto = sum(ln for _, ln in c.calls)
    assert cubierto == r.num_blocks * (r._worker_area_end - 0)


# ─────────────────── guardas de fallback ───────────────────


def test_unaligned_page_size_falls_back_to_whole_region():
    """Sin alineación a página los rangos compartirían la página de frontera."""
    r = _make_region(rank=0, cpu_page_size=PAGE + 7)
    c = _FakeCudart()
    res, _ = _run(r, c)
    assert res.value == 0
    assert c.calls == [(0, r.total_size_bytes)], "tiene que ser UNA llamada entera"


def test_rank_none_falls_back_to_whole_region():
    r = _make_region(rank=None)
    c = _FakeCudart()
    _run(r, c)
    assert c.calls == [(0, r.total_size_bytes)]


def test_slots_that_do_not_fit_the_row_fall_back():
    """Si cpu_page_size * (rank+1) excede la fila, el layout no es el esperado."""
    r = _make_region(rank=1, cpu_page_size=2 * PAGE, world=2)
    r._row_stride = 3 * PAGE  # menos de (1+1) * cpu_page_size
    r.total_size_bytes = r.num_blocks * r._row_stride
    c = _FakeCudart()
    _run(r, c)
    assert c.calls == [(0, r.total_size_bytes)]


def test_broken_region_falls_back_without_raising():
    r = types.SimpleNamespace(rank=0, total_size_bytes=8192)  # sin _worker_area_end
    r._base = types.SimpleNamespace(data_ptr=lambda: 0)
    c = _FakeCudart()
    res, ns = _run(r, c)
    assert res.value == 0
    assert c.calls == [(0, 8192)]
    assert ns["_warnings"], "tiene que avisar por que cayo al fallback"


# ─────────────────── rollback ───────────────────


def test_partial_failure_rolls_back_every_registered_slot():
    """Media región pinneada es peor que ninguna: el driver decide por rango."""
    r = _make_region(rank=0, num_blocks=6)
    c = _FakeCudart(fail_at=3)  # las 3 primeras OK, la 4a falla
    res, _ = _run(r, c)
    assert res.value != 0, "el error se propaga al camino original de vLLM"
    assert len(c.unregistered) == 3, "se revierten exactamente las que entraron"
    assert c.unregistered == [ptr for ptr, _ in c.calls[:3]]


def test_failure_on_the_first_slot_rolls_back_nothing():
    r = _make_region(rank=0, num_blocks=6)
    c = _FakeCudart(fail_at=0)
    res, _ = _run(r, c)
    assert res.value != 0
    assert c.unregistered == []
    assert len(c.calls) == 1, "no sigue intentando tras el primer fallo"


def test_result_keeps_the_cudart_contract():
    """vLLM hace `if result.value != 0`; PN94 no puede romper eso."""
    r = _make_region(rank=0)
    res, _ = _run(r, _FakeCudart())
    assert hasattr(res, "value")


# ─────────────────── ancla y orden de aplicacion ───────────────────

_REL = "v1/kv_offload/cpu/gpu_worker.py"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


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
_needs_tree = pytest.mark.skipif(
    VLLM_ROOT is None, reason="no hay arbol de fuentes de vLLM"
)


@_needs_tree
def test_anchor_appears_exactly_once_in_the_pristine_target():
    P = _anchors()
    with open(os.path.join(VLLM_ROOT, _REL), encoding="utf-8") as fh:
        txt = fh.read()
    assert txt.count(P.ANCHOR_OLD) == 1


@_needs_tree
def test_patched_source_compiles():
    P = _anchors()
    path = os.path.join(VLLM_ROOT, _REL)
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    compile(txt.replace(P.ANCHOR_OLD, P.ANCHOR_NEW), path, "exec")


@_needs_tree
def test_pn82_then_pn94_both_apply_in_registration_order():
    """PN82 se aplica ANTES que PN94 y los dos tocan pin_mmap_region.

    PN82 solo agrega codigo dentro de la rama de fallo, asi que tiene que
    dejar intacto el ancla de PN94. Si algun dia deja de hacerlo, este test
    lo caza antes que el arranque del contenedor.
    """
    from vllm._genesis.wiring.hybrid import patch_N82_host_register_sticky_error as P82

    P94 = _anchors()
    with open(os.path.join(VLLM_ROOT, _REL), encoding="utf-8") as fh:
        txt = fh.read()

    assert txt.count(P82.ANCHOR_OLD) == 1
    tras_82 = txt.replace(P82.ANCHOR_OLD, P82.ANCHOR_NEW)
    assert P82.GENESIS_PN82_MARKER in tras_82

    assert tras_82.count(P94.ANCHOR_OLD) == 1, (
        "PN82 rompio el ancla de PN94"
    )
    tras_94 = tras_82.replace(P94.ANCHOR_OLD, P94.ANCHOR_NEW)
    assert P94.GENESIS_PN94_MARKER in tras_94
    assert P82.GENESIS_PN82_MARKER in tras_94, "PN94 no puede borrar a PN82"
    compile(tras_94, "<pn82+pn94>", "exec")
