# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN94 — registro de host pinneado POR RANK.

================================================================
QUÉ RESUELVE
================================================================

`pin_mmap_region()` (v1/kv_offload/cpu/gpu_worker.py) registra la región mmap
ENTERA desde todos los ranks:

    result = cudaHostRegister(base_ptr, region.total_size_bytes, 0)

Con TP>1 los ranks mapean el MISMO archivo de /dev/shm, así que el segundo en
llegar registra páginas físicas que ya están registradas y falla con
`cudaErrorInvalidValue`. vLLM loguea el warning y sigue:

    WARNING gpu_worker.py:142 cudaHostRegister failed for rank=1 (code=1)
            — transfers will still work but may be slower (unpinned DMA)

PN82 ya evita que ese fallo tumbe el arranque (consume el error latcheado).
PN94 ataca la otra mitad: que el rank perdedor **no se quede sin pinnear**.

Sin pinneo, `cudaMemcpyAsync` sobre memoria paginable pasa por un buffer de
staging del driver: no hay DMA directo, no se solapa con cómputo, y el ancho
de banda cae a grosso modo a la mitad. Como el tier L2 es hoy el que absorbe
todo el tráfico de KV (PN90 dejó el SSD en cero), ese rank es la mitad lenta
de cada promoción y de cada desalojo.

================================================================
POR QUÉ SE PUEDE ARREGLAR
================================================================

El layout de la región YA es disjunto por rank. De `SharedOffloadRegion`:

    self._row_stride     = kv_bytes_per_block          # fila completa
    self._worker_offset  = rank * cpu_page_size        # slot de este rank
    self._worker_area_end = (rank + 1) * cpu_page_size

o sea que cada fila de bloque contiene un slot por worker, y cada worker sólo
toca el suyo — `create_next_view()` construye los tensores con
`storage_offset=self._worker_offset`. El propio vLLM ya hace esto bien para
`madvise(MADV_POPULATE_WRITE)`: recorre bloque por bloque y popula únicamente
los slots del rank. El registro de host es el único lugar que se olvidó de
hacerlo y agarra la región completa.

PN94 registra sólo los slots propios: `num_blocks` llamadas de `cpu_page_size`
bytes en vez de una de `total_size_bytes`. Los rangos de los dos ranks quedan
disjuntos y los dos pinnean.

Medido en este despliegue (Qwen3.8-27B, TP=2, 2× RTX 3090):

    total_size_bytes = 6.430.654.464
    _row_stride      =    28.966.912   (222 bloques, múltiplo de 4096)
    cpu_page_size    =    14.483.456   (= row_stride / 2, múltiplo de 4096)

Con `vllm:kv_offload_total_{bytes,time}` la corrida de referencia daba
7,32 GB/s CPU→GPU y 6,89 GB/s GPU→CPU, promediando un rank pinneado con uno
que no lo está. Ésas son las series con las que se verifica el efecto.

================================================================
SEGURIDAD
================================================================

- **Guarda de alineación**: `cudaHostRegister` exige punteros alineados a
  página. Sólo entra al camino por slots si `cpu_page_size` y `_row_stride`
  son múltiplos de `region.page_size` y los slots caben en la fila. Si no,
  cae al registro entero de siempre — comportamiento idéntico al de vLLM.
- **Guarda de rank**: si `region.rank is None` (región del scheduler) no hay
  slots que derivar; registro entero.
- **Rollback**: si falla un slot intermedio se des-registran los anteriores
  con `cudaHostUnregister` para no dejar la región a medio pinnear, y se deja
  `result` con el código de error para que siga el camino original de vLLM
  (warning + limpieza del error pegajoso de PN82).
- **No cambia el contrato**: `result` sigue siendo el objeto que devuelve
  cudart, con `.value`, así que el `if result.value != 0:` de vLLM y todo lo
  que PN82 inyecta adentro funcionan sin tocarse.
- Kill switch: `GENESIS_DISABLE_PN94=1`.

Orden: PN94 se aplica DESPUÉS de PN82. El ancla de PN94 son las dos líneas
`base_ptr = ...` / `result = ...`, que PN82 preserva textualmente (PN82 sólo
agrega código dentro de la rama de fallo).
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN94_MARKER = "_GENESIS_PN94_PER_RANK_HOST_REGISTER"

ANCHOR_OLD = (
    "    base_ptr = region._base.data_ptr()\n"
    "    result = torch.cuda.cudart().cudaHostRegister("
    "base_ptr, region.total_size_bytes, 0)\n"
)

ANCHOR_NEW = '''    base_ptr = region._base.data_ptr()
    # ''' + GENESIS_PN94_MARKER + '''
    # vLLM registra la region ENTERA desde todos los ranks. Con TP>1 los dos
    # mapean el mismo /dev/shm, asi que el segundo registra paginas fisicas ya
    # registradas y falla con cudaErrorInvalidValue: ese rank se queda con DMA
    # no-pinneado (staging buffer del driver, sin solape con computo).
    # El layout ya es disjunto por rank -- _worker_offset = rank * cpu_page_size
    # dentro de cada fila, y el madvise(MADV_POPULATE_WRITE) de mas arriba ya
    # recorre solo los slots propios. Hacemos lo mismo con el registro.
    _g94_slots = None
    _g94_rank = getattr(region, "rank", None)
    _g94_cps = 0
    try:
        if _g94_rank is not None:
            _g94_cps = region._worker_area_end // (_g94_rank + 1)
            _g94_stride = region._row_stride
            _g94_pagesz = region.page_size
            # cudaHostRegister exige puntero alineado a pagina. Sin esto los
            # rangos de dos ranks vecinos compartirian la pagina de la frontera
            # y volveriamos al mismo choque que estamos arreglando.
            if (
                _g94_cps > 0
                and _g94_cps % _g94_pagesz == 0
                and _g94_stride % _g94_pagesz == 0
                and _g94_cps * (_g94_rank + 1) <= _g94_stride
            ):
                _g94_base = base_ptr + _g94_rank * _g94_cps
                _g94_slots = [
                    (_g94_base + _b * _g94_stride, _g94_cps)
                    for _b in range(region.num_blocks)
                ]
    except Exception as _g94_exc:  # nunca puede tumbar el arranque
        logger.warning(
            "PN94: no pude derivar los slots del rank (%s); registro la "
            "region entera como upstream",
            _g94_exc,
        )
        _g94_slots = None

    if _g94_slots is None:
        result = torch.cuda.cudart().cudaHostRegister(
            base_ptr, region.total_size_bytes, 0
        )
    else:
        _g94_done = 0
        result = torch.cuda.cudart().cudaHostRegister(*_g94_slots[0], 0)
        if result.value == 0:
            _g94_done = 1
            for _g94_ptr, _g94_len in _g94_slots[1:]:
                result = torch.cuda.cudart().cudaHostRegister(_g94_ptr, _g94_len, 0)
                if result.value != 0:
                    break
                _g94_done += 1
        if result.value == 0:
            logger.info(
                "PN94: rank=%d pinneo %d slots de %.2f MB (%.2f GB, solo sus "
                "paginas). Upstream registraba la region entera y el segundo "
                "rank fallaba, quedandose con DMA no-pinneado.",
                _g94_rank,
                _g94_done,
                _g94_cps / 1e6,
                _g94_done * _g94_cps / 1e9,
            )
        else:
            # Revertir lo parcial: una region a medio pinnear es peor que una
            # sin pinnear, porque el driver decide por rango y el resultado
            # pasa a depender de que bloque toque cada transferencia.
            for _g94_ptr, _g94_len in _g94_slots[:_g94_done]:
                try:
                    torch.cuda.cudart().cudaHostUnregister(_g94_ptr)
                except Exception:
                    pass
            logger.warning(
                "PN94: fallo el registro por slots en %d/%d (code=%d); revertido. "
                "Sigue el camino original de vLLM.",
                _g94_done,
                len(_g94_slots),
                result,
            )
'''


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN94", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(root, "v1", "kv_offload", "cpu", "gpu_worker.py")
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN94 per-rank host register",
        target_file=target,
        marker=GENESIS_PN94_MARKER,
        sub_patches=[
            TextPatch(
                name="pn94_per_rank_slot_registration",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN94")
    log_decision("PN94", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN94 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/cpu/gpu_worker.py not found"

    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN94 aplicado: cada rank registra como pinneados SOLO sus slots de "
            "la region mmap compartida (num_blocks llamadas de cpu_page_size en "
            "vez de una de total_size_bytes). Upstream registraba la region "
            "entera desde todos los ranks; con TP>1 el segundo chocaba con las "
            "mismas paginas fisicas, fallaba con cudaErrorInvalidValue y se "
            "quedaba con DMA no-pinneado. Cae al registro entero si "
            "cpu_page_size no esta alineado a pagina o si region.rank es None. "
            "Kill switch: GENESIS_DISABLE_PN94=1."
        ),
        patch_name="PN94 per-rank host register",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN94_MARKER in f.read()
    except Exception:
        return False
