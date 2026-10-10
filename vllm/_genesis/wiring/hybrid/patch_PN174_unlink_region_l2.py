# SPDX-License-Identifier: Apache-2.0
"""PN174 — la region de la L2 (/dev/shm/vllm_offload_*.mmap) se borra del directorio apenas la
mapea el ultimo proceso, como ya hace el spec de CPU.

El bug (vLLM 0.29.0): ``CPUOffloadingSpec`` le pasa ``barrier=`` a ``SharedOffloadRegion`` y el
archivo queda sin nombre cuando todos los workers lo mapearon; ninguna salida, ni un SIGKILL, lo
deja huerfano. ``TieringOffloadingSpec`` no pasa la barrera, y ademas el scheduler (EngineCore)
tambien abre la region. El archivo queda con nombre para siempre: si el motor muere por OOM o por
un error de CUDA, los 12 GB quedan en /dev/shm y el reinicio automatico falla con
"Insufficient space in /dev/shm" en bucle (06-10 dos veces, 10-10 tres).

No alcanza con pasarle la barrera a los workers: el scheduler la abre DESPUES (el EngineCore
recien arma el scheduler cuando terminaron de inicializarse los workers, medido 8 s mas tarde)
y por nombre; si ya no existe, el O_CREAT|O_EXCL le crearia otro archivo vacio y trabajaria
sobre otra memoria. El ultimo en abrir es el scheduler, asi que es el que borra el nombre.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN174: unlink de la region L2]"
_OLD = (
    "                self._scheduler_mmap = scheduler_mmap\n"
)
_NEW = _OLD + (
    "                # " + MARKER + " el scheduler abre la region ultimo (despues de todos los\n"
    "                # workers): sin nombre, ninguna muerte del motor deja los GB en /dev/shm.\n"
    "                if scheduler_mmap._creator:\n"
    "                    raise RuntimeError(\n"
    "                        'PN174: el scheduler creo la region L2; los workers tenian que '\n"
    "                        'haberla creado antes'\n"
    "                    )\n"
    "                import os as _g174_os\n"
    "                try:\n"
    "                    _g174_os.unlink(scheduler_mmap.mmap_path)\n"
    "                    logger.info('PN174: region L2 sin nombre: %s', scheduler_mmap.mmap_path)\n"
    "                except FileNotFoundError:\n"
    "                    pass\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN174")
    log_decision("PN174", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/kv_offload/tiering/spec.py")
    if f is None:
        return "skipped", "falta v1/kv_offload/tiering/spec.py"
    p = TextPatcher(patch_name="PN174 unlink de la region L2", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn174_unlink", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["_g174_os"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="region L2 sin nombre tras mapearla el scheduler",
                                   patch_name=p.patch_name)
