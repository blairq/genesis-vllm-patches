# SPDX-License-Identifier: Apache-2.0
"""Verificación de anclas para PN90/PN91/PN92/PN93 contra el fuente real de vLLM.

Por qué existe este archivo
---------------------------
La primera versión de PN91 apuntaba a un ancla que NO EXISTE en `scheduler.py`
(`"        if request.request_id in self._req_status:\\n            del ..."`).
Con `required=True`, `TextPatcher` abortaba y no escribía nada — mientras
`apply()` devolvía "applied". El parche estuvo semanas en el compose sin
aplicarse jamás.

Los tests que había verificaban que el marcador estuviera dentro del string de
reemplazo. Eso es tautológico. Lo único que sirve es contrastar cada ancla
contra el árbol de vLLM, que es exactamente lo que hace esto:

  - cada ancla requerida aparece EXACTAMENTE UNA VEZ (0 → deriva, 2+ → ambiguo,
    y `TextPatcher` reemplazaría sólo la primera dejando estado parcial);
  - aplicadas en secuencia, los archivos resultantes COMPILAN;
  - PN91 y PN93 tocan el mismo archivo y no se pisan.
"""

from __future__ import annotations

import itertools
import os

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _vllm_tree() -> str | None:
    """Localiza un árbol de fuentes de vLLM con los archivos que nos interesan.

    Prueba el vLLM instalado y, si no sirve, el vendorizado en `assets/`.
    """
    probe = os.path.join("v1", "kv_offload", "tiering", "manager.py")
    candidates = []
    try:
        import vllm

        candidates.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    candidates.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    for root in candidates:
        if root and os.path.exists(os.path.join(root, probe)):
            return root
    return None


VLLM_ROOT = _vllm_tree()

pytestmark = pytest.mark.skipif(
    VLLM_ROOT is None,
    reason="no hay árbol de fuentes de vLLM (ni instalado ni en assets/)",
)

_SCHEDULER = "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

# (patch_id, módulo de wiring, [(archivo relativo, [prefijos de constante])])
# El orden refleja el de registro en apply_all: PN91 y PN93 comparten archivo.
PATCH_SPECS = [
    (
        "PN90",
        "patch_N90_kv_disk_write_gating",
        [
            ("v1/kv_offload/tiering/fs/manager.py", ["FS_ANCHOR"]),
            (
                "v1/kv_offload/tiering/manager.py",
                [
                    "PRIMARY_STORE",
                    "TIER_PREPARE",
                    "TIER_SCHEDULE_END",
                    "TIER_COMPLETE",
                    "REQUEST_LEVEL",
                    "FINISHED_JOBS",
                ],
            ),
        ],
    ),
    (
        "PN91",
        "patch_N91_kv_lazy_streaming",
        [(_SCHEDULER, ["PREFIX_LOOKUP", "SLIDING_LOOKUP", "DEFER", "CLEANUP"])],
    ),
    (
        "PN92",
        "patch_N92_kv_fp4_compressed_tiers",
        [("v1/kv_offload/tiering/fs/io.py", ["STORE_BLOCK_HOOK", "LOAD_BLOCK_HOOK"])],
    ),
    (
        "PN93",
        "patch_N93_kv_sparse_gdn",
        [(_SCHEDULER, ["REGISTER", "FILTER", "COST"])],
    ),
]


def _wiring(mod_name: str):
    import importlib

    return importlib.import_module(f"vllm._genesis.wiring.hybrid.{mod_name}")


def _pairs(mod_name: str, prefixes):
    mod = _wiring(mod_name)
    for prefix in prefixes:
        yield prefix, getattr(mod, prefix + "_OLD"), getattr(mod, prefix + "_NEW")


@pytest.mark.parametrize(
    "patch_id,mod_name,rel,prefix",
    [
        (pid, mod, rel, pre)
        for pid, mod, files in PATCH_SPECS
        for rel, prefixes in files
        for pre in prefixes
    ],
)
def test_anchor_exists_exactly_once(patch_id, mod_name, rel, prefix):
    """Cada ancla requerida aparece exactamente una vez en el fuente de vLLM."""
    path = os.path.join(VLLM_ROOT, rel)
    assert os.path.exists(path), f"{patch_id}: no existe el target {rel}"
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    anchor = getattr(_wiring(mod_name), prefix + "_OLD")
    count = content.count(anchor)
    assert count == 1, (
        f"{patch_id}/{prefix}: el ancla aparece {count} veces en {rel} "
        f"(se esperaba 1). 0 = deriva de upstream y el parche se salta entero; "
        f"2+ = ambiguo y TextPatcher deja estado parcial."
    )


def test_marker_present_in_every_replacement():
    """Todo reemplazo tiene que llevar el marcador o se pierde la idempotencia."""
    for patch_id, mod_name, files in PATCH_SPECS:
        marker = getattr(_wiring(mod_name), f"GENESIS_{patch_id}_MARKER")
        for _rel, prefixes in files:
            for prefix, _old, new in _pairs(mod_name, prefixes):
                assert marker in new, f"{patch_id}/{prefix}: falta el marcador"


def test_all_patches_apply_in_sequence_and_compile(tmp_path):
    """Aplicados en orden de registro, los archivos resultantes compilan.

    Cubre la interacción PN91 ↔ PN93 (mismo archivo) y cualquier error de
    indentación en los bloques inyectados, que es el modo de fallo típico de un
    text-patch y que ningún test de strings puede detectar.
    """
    staged: dict[str, str] = {}

    for patch_id, mod_name, files in PATCH_SPECS:
        for rel, prefixes in files:
            if rel not in staged:
                with open(os.path.join(VLLM_ROOT, rel), encoding="utf-8") as fh:
                    staged[rel] = fh.read()
            content = staged[rel]
            for prefix, old, new in _pairs(mod_name, prefixes):
                assert content.count(old) == 1, (
                    f"{patch_id}/{prefix}: ancla no única tras aplicar los "
                    f"parches anteriores sobre {rel}"
                )
                content = content.replace(old, new, 1)
            staged[rel] = content

    for rel, content in staged.items():
        try:
            compile(content, rel, "exec")
        except SyntaxError as exc:
            pytest.fail(f"{rel} no compila tras aplicar los parches: {exc}")


def test_shared_file_patches_do_not_overlap():
    """PN91 y PN93 tocan `scheduler.py`: sus anclas tienen que ser disjuntas."""
    with open(os.path.join(VLLM_ROOT, _SCHEDULER), encoding="utf-8") as fh:
        content = fh.read()
    spans: list[tuple[int, int, str]] = []
    for patch_id, mod_name, files in PATCH_SPECS:
        for rel, prefixes in files:
            if rel != _SCHEDULER:
                continue
            for prefix, old, _new in _pairs(mod_name, prefixes):
                start = content.find(old)
                assert start >= 0, f"{patch_id}/{prefix}: ancla ausente"
                spans.append((start, start + len(old), f"{patch_id}/{prefix}"))

    spans.sort()
    for (_a_start, a_end, a_name), (b_start, _b_end, b_name) in itertools.pairwise(spans):
        assert a_end <= b_start, (
            f"las anclas {a_name} y {b_name} se solapan en scheduler.py"
        )


# Orden REAL de registro en apply_all. Importa: PN88 se aplica antes que PN90 y
# PN92, y los tres comparten archivos (`fs/manager.py`, `fs/io.py`,
# `tiering/manager.py`). Un ancla de PN90 tiene que matchear el archivo YA
# parcheado por PN88, no el original.
_STACK_ORDER = [
    "patch_N81_kv_disk_tier_quota",
    "patch_N88_kv_tier_metrics",
    "patch_N90_kv_disk_write_gating",
    "patch_N91_kv_lazy_streaming",
    "patch_N92_kv_fp4_compressed_tiers",
    "patch_N93_kv_sparse_gdn",
]


def _real_patchers(mod):
    """Patchers reales del wiring, sin depender de nombres de constantes."""
    if hasattr(mod, "_PATCHERS"):
        return [f() for f in mod._PATCHERS]
    out = []
    for name in dir(mod):
        if not (name.endswith("patcher") or name == "_patchers"):
            continue
        fn = getattr(mod, name)
        if not callable(fn) or getattr(fn, "__code__", None) is None:
            continue
        if fn.__code__.co_argcount:
            continue
        got = fn()
        if got is None:
            continue
        out.extend(got if isinstance(got, list) else [got])
    seen, dedup = set(), []
    for p in out:
        k = (p.patch_name, p.target_file)
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    return dedup


def test_whole_offload_stack_applies_in_registration_order(tmp_path, monkeypatch):
    """PN81+PN88+PN90..PN93 aplican en cadena sobre un árbol real y compilan.

    Es el test que cubre el riesgo de ORDEN: los cinco parches se pisan en tres
    archivos, y un ancla escrita contra el fuente virgen puede dejar de
    matchear una vez que el parche anterior lo tocó.
    """
    import importlib
    import shutil

    tree = tmp_path / "vllm"
    shutil.copytree(VLLM_ROOT, tree, symlinks=True)

    from vllm._genesis import guards

    monkeypatch.setattr(guards, "vllm_install_root", lambda: str(tree))

    def _resolve(rel):
        p = tree / rel
        return str(p) if p.exists() else None

    monkeypatch.setattr(guards, "resolve_vllm_file", _resolve)

    applied_any = False
    for mod_name in _STACK_ORDER:
        mod = importlib.import_module(f"vllm._genesis.wiring.hybrid.{mod_name}")
        monkeypatch.setattr(mod, "vllm_install_root", lambda: str(tree))
        if hasattr(mod, "resolve_vllm_file"):
            monkeypatch.setattr(mod, "resolve_vllm_file", _resolve)

        for patcher in _real_patchers(mod):
            result, failure = patcher.apply()
            assert result.name in ("APPLIED", "IDEMPOTENT"), (
                f"{mod_name} → {patcher.target_file}: {result.name} "
                f"({failure.reason if failure else '?'}: "
                f"{failure.detail if failure else ''}). "
                f"Probable conflicto de orden con un parche anterior."
            )
            applied_any = True

    assert applied_any, "ningún patcher se ejecutó; el test no probó nada"

    for path in sorted(tree.rglob("*.py")):
        text = path.read_text(errors="ignore")
        if "[Genesis PN" not in text:
            continue
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            pytest.fail(f"{path.relative_to(tree)} no compila tras la pila: {exc}")


# Nombre de la factory de patchers de cada wiring, para poder sustituirla.
_FACTORY = {
    "patch_N90_kv_disk_write_gating": "_patchers",
    "patch_N91_kv_lazy_streaming": "_scheduler_patcher",
    "patch_N92_kv_fp4_compressed_tiers": "_get_fs_io_patcher",
    "patch_N93_kv_sparse_gdn": "_scheduler_patcher",
}


@pytest.mark.parametrize(
    "patch_id,mod_name", [(pid, mod) for pid, mod, _f in PATCH_SPECS]
)
def test_apply_reports_skipped_when_anchor_drifts(
    patch_id, mod_name, tmp_path, monkeypatch
):
    """Con las anclas ausentes, `apply()` tiene que devolver 'skipped'.

    Éste es EL test que faltaba. El defecto compartido era::

        if status == "failed":
            return "failed", ...
        return "applied", "PNxx ... activo"     # ← un skip salía como éxito

    PN91 vivió así: su segunda ancla no existía, `TextPatcher` abortaba sin
    escribir nada, y el arranque reportaba el parche como aplicado.
    """
    from vllm._genesis import dispatcher
    from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher

    mod = _wiring(mod_name)

    monkeypatch.setattr(dispatcher, "should_apply", lambda pid: (True, "test"))
    monkeypatch.setattr(dispatcher, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(mod, "vllm_install_root", lambda: str(tmp_path))

    victim = tmp_path / "target.py"
    victim.write_text("# un archivo sin ninguna de las anclas\n")

    def _fake_patcher():
        return TextPatcher(
            patch_name=f"{patch_id} (anclas derivadas)",
            target_file=str(victim),
            marker=f"[test {patch_id}]",
            sub_patches=[
                TextPatch(
                    name="ancla_inexistente",
                    anchor="ESTA CADENA NO ESTA EN EL ARCHIVO",
                    replacement="x",
                    required=True,
                )
            ],
        )

    factory_name = _FACTORY[mod_name]
    if factory_name == "_patchers":
        monkeypatch.setattr(mod, factory_name, lambda: [_fake_patcher()])
    else:
        monkeypatch.setattr(mod, factory_name, _fake_patcher)

    status, reason = mod.apply()

    assert status == "skipped", (
        f"{patch_id}: con el ancla ausente apply() devolvió {status!r} "
        f"({reason!r}); tiene que ser 'skipped'."
    )
    assert victim.read_text() == "# un archivo sin ninguna de las anclas\n", (
        f"{patch_id}: no debe escribir nada cuando el ancla no matchea"
    )
