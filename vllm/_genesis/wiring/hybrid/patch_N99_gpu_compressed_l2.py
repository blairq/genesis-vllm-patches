# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN99 — L2 (y L3) comprimidos a 4 bits, cuantizando EN LA GPU.

================================================================
POR QUE
================================================================

PN98 midio la jerarquia y salio invertida: L2 tenia 155.540 tokens contra
410.176 de L1 (0,40x). Subirla de 6 a 12 GiB llevo la relectura de 0% a 96,7%
cacheado, pero deja el ratio en 0,76x y ya no hay RAM: 30 GB totales, 7
disponibles. Comprar mas no es una opcion en este proyecto.

El codec de 4 bits da 1,78x sobre fp8_e4m3 (4 bits + escala fp16 por grupo de
32 = 4,5 bits/elemento). Aplicado a L2:

    12 GiB reales  ->  21,3 GiB efectivos  ->  553.031 tokens
    L2/L1 = 1,35x   (la jerarquia deja de estar invertida)

O al reves, que es lo que importa siendo pobres: **7 GiB de RAM real rinden
mas que los 12 GiB de hoy**.

================================================================
POR QUE EN LA GPU Y NO REUSANDO PN92
================================================================

`kv_fp4_codec.py` (PN92) es numpy. Sirve para el tier de disco, donde la CPU
ya esta en el camino, pero no para L2: ahi la copia la hace el DMA y meter a
la CPU en el medio mataria el ancho de banda que PN94 acaba de duplicar.

`kv_gpu_codec.py` hace la misma cuantizacion con ops de torch, o sea en el
mismo device donde vive la KV. El DMA sigue siendo una copia de bytes crudos;
lo unico que cambia es que el payload es 1,78x mas chico. Verificado:
ratio 1,778x, error L2 12,13%, coseno 0,992758 — los mismos numeros que el
codec de CPU que ya estaba validado.

**L3 sale gratis**: el tier de disco escribe `primary_kv_view[block]`, o sea
los bytes de L2. Si L2 guarda comprimido, el SSD tambien. Por eso PN92 queda
superseded y tiene que seguir apagado: comprimir dos veces seria un desastre.

================================================================
QUE TOCA
================================================================

1. `cpu/spec.py` — el bloque offloadeado pasa a medir el tamano COMPRIMIDO,
   asi que en los mismos `cpu_bytes_to_use` entran ~1,78x mas bloques.
   La cuenta cierra exacta y alineada a pagina en este despliegue:
       crudo por worker      14.483.456 B
       comprimido            14.483.456 * 0,5625 = 8.146.944 B = 1989 * 4096
   La alineacion a 4096 es un requisito de PN94 (registro de host por rank),
   no un lujo. Si no diera exacta, PN99 se desactiva solo.

2. `cpu/gpu_worker.py` — el assert `cpu_page_size == gpu_page_size *
   block_size_factor` deja de valer cuando el lado CPU esta comprimido.

3. `cpu/gpu_worker.py` — la transferencia. Se reemplaza la copia cruda por
   comprimir-y-copiar (GPU->CPU) o copiar-y-descomprimir (CPU->GPU), dentro
   del MISMO stream y con los mismos eventos, asi que `get_finished()` y toda
   la contabilidad siguen igual.

================================================================
LIMITES Y SEGURIDAD
================================================================

- Solo se activa con `block_size_factor == 1` (mapeo 1:1 de bloques). Con
  sub-bloques la aritmetica de punteros es otra y no vale la pena el riesgo.
- Solo si el tamano comprimido queda alineado a pagina.
- **OFF por defecto**: `GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2=1` para
  activarlo. Cambia el formato de lo que hay en /dev/shm y en /kv-offload,
  asi que al prenderlo o apagarlo hay que borrar el directorio del tier de
  disco: los bloques viejos tienen otro layout.
- Kill switch: `GENESIS_DISABLE_PN99=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN99_MARKER = "_GENESIS_PN99_GPU_COMPRESSED_L2"

# ─────────── 1. el bloque offloadeado mide el tamano comprimido ───────────

SPEC_OLD = (
    "            aligned_kv_bytes_per_offloaded_block = round_up(\n"
    "                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT\n"
    "            )\n"
)

SPEC_NEW = (
    "            # " + GENESIS_PN99_MARKER + "\n"
    "            # El lado CPU guarda la KV comprimida a 4 bits, asi que el\n"
    "            # bloque mide 1,78x menos y en los mismos bytes entran 1,78x\n"
    "            # mas bloques. Se exige alineacion a pagina por worker porque\n"
    "            # PN94 registra los slots de cada rank por separado.\n"
    "            try:\n"
    "                from vllm._genesis import kv_gpu_codec as _g99\n"
    "                from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "                if _g99g.habilitado() and self.block_size_factor == 1:\n"
    "                    _g99_w = kv_bytes_per_offloaded_block // world_size\n"
    "                    _g99_c = _g99.tamano_comprimido(_g99_w)\n"
    "                    if _g99_c % 4096 == 0:\n"
    "                        kv_bytes_per_offloaded_block = _g99_c * world_size\n"
    "                        self.cpu_page_size_per_worker = _g99_c\n"
    "                        _g99g.activar(_g99_w, _g99_c)\n"
    "                    else:\n"
    "                        _g99g.desactivar(\n"
    "                            'el tamano comprimido %d no cae en pagina' % _g99_c\n"
    "                        )\n"
    "                elif _g99g.habilitado():\n"
    "                    _g99g.desactivar('block_size_factor != 1')\n"
    "            except Exception as _g99_exc:\n"
    "                from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "                _g99g.desactivar('error al dimensionar: %s' % _g99_exc)\n"
    "            aligned_kv_bytes_per_offloaded_block = round_up(\n"
    "                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT\n"
    "            )\n"
)

# ─────────── 2. el assert de tamanos deja de valer ───────────

ASSERT_OLD = "            assert cpu_page_size == gpu_page_size * block_size_factor\n"

ASSERT_NEW = (
    "            # " + GENESIS_PN99_MARKER + "\n"
    "            # El assert es POR TENSOR, asi que se compara contra el\n"
    "            # comprimido de ESA pagina, no contra el area del worker.\n"
    "            from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "            if _g99g.activo():\n"
    "                from vllm._genesis import kv_gpu_codec as _g99\n"
    "\n"
    "                _g99_esp = _g99.tamano_comprimido(\n"
    "                    gpu_page_size * block_size_factor\n"
    "                )\n"
    "                assert cpu_page_size == _g99_esp, (\n"
    "                    'PN99: la pagina CPU tiene que medir el comprimido '\n"
    "                    '(%d), mide %d' % (_g99_esp, cpu_page_size)\n"
    "                )\n"
    "            else:\n"
    "                assert cpu_page_size == gpu_page_size * block_size_factor\n"
)

# ─────────── 2b. las vistas del mmap se crean con el tamano comprimido ───────────
#
# ESTE es el que faltaba: create_next_view() avanza _worker_offset con el
# tamano que se le pasa, y el area del worker ahora mide el comprimido. Con el
# tamano crudo revienta con
#     Worker offset 8519680 exceeds worker area end 8146944
# La suma cierra porque tamano_comprimido() es lineal (n * 0,5625), asi que la
# suma de las paginas comprimidas es el area comprimida.

VIEW_OLD = (
    "            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor\n"
)

VIEW_NEW = (
    "            cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor\n"
    "            # " + GENESIS_PN99_MARKER + "\n"
    "            from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "            if _g99g.activo():\n"
    "                from vllm._genesis import kv_gpu_codec as _g99\n"
    "\n"
    "                cpu_page_size_bytes = _g99.tamano_comprimido(\n"
    "                    cpu_page_size_bytes\n"
    "                )\n"
)

# ─────────── 3. la transferencia comprime / descomprime ───────────

COPY_OLD = (
    "            start_event.record(stream)\n"
    "            if num_copy_ops > 0:\n"
    "                self._swap_blocks_batch(\n"
    "                    src,\n"
    "                    dst,\n"
    "                    sizes,\n"
    "                    is_src_access_order_any=is_src_access_order_any,\n"
    "                )\n"
    "            end_event.record(stream)\n"
)

COPY_NEW = (
    "            start_event.record(stream)\n"
    "            # " + GENESIS_PN99_MARKER + "\n"
    "            # Mismo stream y mismos eventos que el camino crudo, asi que\n"
    "            # get_finished() y toda la contabilidad siguen igual. Lo unico\n"
    "            # que cambia es que el payload va comprimido.\n"
    "            from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "            if _g99g.activo() and num_copy_ops > 0:\n"
    "                from vllm._genesis import kv_gpu_codec as _g99\n"
    "\n"
    "                # OJO: el tamano crudo es el de ESTE tensor, no el area\n"
    "                # del worker. Con bytes_crudos() reventaba con\n"
    "                #   expanded size (7241728) must match existing (479232)\n"
    "                # porque 7241728 = 14483456/2, o sea n//2 del area entera.\n"
    "                for _g99_t_src, _g99_t_dst in zip(\n"
    "                    self.src_tensors, self.dst_tensors\n"
    "                ):\n"
    "                    for _g99_i in range(len(src_blocks)):\n"
    "                        _g99_s = _g99_t_src[int(src_blocks[_g99_i])]\n"
    "                        _g99_d = _g99_t_dst[int(dst_blocks[_g99_i])]\n"
    "                        if self.gpu_to_cpu:\n"
    "                            _g99_d.copy_(\n"
    "                                _g99.comprimir(_g99_s).view(torch.int8),\n"
    "                                non_blocking=True,\n"
    "                            )\n"
    "                        else:\n"
    "                            _g99_stage = _g99_s.to(\n"
    "                                _g99_d.device, non_blocking=True\n"
    "                            )\n"
    "                            _g99_d.copy_(\n"
    "                                _g99.descomprimir(\n"
    "                                    _g99_stage.view(torch.uint8),\n"
    "                                    _g99_d.numel(),\n"
    "                                ).view(torch.int8),\n"
    "                                non_blocking=True,\n"
    "                            )\n"
    "            elif num_copy_ops > 0:\n"
    "                self._swap_blocks_batch(\n"
    "                    src,\n"
    "                    dst,\n"
    "                    sizes,\n"
    "                    is_src_access_order_any=is_src_access_order_any,\n"
    "                )\n"
    "            end_event.record(stream)\n"
)


# ─────────── 4. borrar el tier de disco al arrancar ───────────
#
# El layout comprimido es incompatible con el crudo: un bloque viejo leido
# como comprimido da basura. Como no interesa conservar KV entre boots, con
# PN99 activo se limpia el directorio del run al crear el tier. Corre solo en
# el proceso del scheduler (el fs manager se instancia ahi), asi que no hay
# carrera entre workers.

WIPE_OLD = (
    "        # Write config file\n"
    "        config_path = self.file_mapper.get_config_file_path()\n"
    "        os.makedirs(os.path.dirname(config_path), exist_ok=True)\n"
)

WIPE_NEW = (
    "        # " + GENESIS_PN99_MARKER + "\n"
    "        # Con PN99 los bloques van comprimidos: los de un boot anterior\n"
    "        # tienen otro layout y se leerian como basura. No interesa\n"
    "        # conservar KV entre boots, asi que se limpia y listo.\n"
    "        config_path = self.file_mapper.get_config_file_path()\n"
    "        try:\n"
    "            from vllm._genesis import pn99_gate as _g99g\n"
    "\n"
    "            if _g99g.activo():\n"
    "                import shutil as _g99_sh\n"
    "\n"
    "                _g99_dir = os.path.dirname(config_path)\n"
    "                if os.path.isdir(_g99_dir):\n"
    "                    _g99_sh.rmtree(_g99_dir, ignore_errors=True)\n"
    "                    logger.info(\n"
    "                        'PN99: borrado el tier de disco de boots anteriores "
    "(%s): '\n"
    "                        'el layout comprimido es incompatible con el crudo.',\n"
    "                        _g99_dir,\n"
    "                    )\n"
    "        except Exception as _g99_exc:\n"
    "            logger.warning('PN99: no pude limpiar el tier: %s', _g99_exc)\n"
    "        os.makedirs(os.path.dirname(config_path), exist_ok=True)\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN99", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patchers() -> list[TextPatcher] | None:
    root = vllm_install_root()
    if root is None:
        return None
    spec = os.path.join(root, "v1", "kv_offload", "cpu", "spec.py")
    worker = os.path.join(root, "v1", "kv_offload", "cpu", "gpu_worker.py")
    if not (os.path.exists(spec) and os.path.exists(worker)):
        return None
    if not os.path.exists(
        os.path.join(root, "v1", "kv_offload", "tiering", "fs", "manager.py")
    ):
        return None
    fsm = os.path.join(root, "v1", "kv_offload", "tiering", "fs", "manager.py")
    return [
        TextPatcher(
            patch_name="PN99 GPU-compressed L2 (fs tier wipe)",
            target_file=fsm,
            marker=GENESIS_PN99_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn99_wipe_stale_tier_on_boot",
                    anchor=WIPE_OLD,
                    replacement=WIPE_NEW,
                    required=True,
                ),
            ],
        ),
        TextPatcher(
            patch_name="PN99 GPU-compressed L2 (cpu spec)",
            target_file=spec,
            marker=GENESIS_PN99_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn99_compressed_block_sizing",
                    anchor=SPEC_OLD,
                    replacement=SPEC_NEW,
                    required=True,
                ),
            ],
        ),
        TextPatcher(
            patch_name="PN99 GPU-compressed L2 (gpu worker)",
            target_file=worker,
            marker=GENESIS_PN99_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn99_compressed_cpu_page_view",
                    anchor=VIEW_OLD,
                    replacement=VIEW_NEW,
                    required=True,
                ),
                TextPatch(
                    name="pn99_relax_page_size_assert",
                    anchor=ASSERT_OLD,
                    replacement=ASSERT_NEW,
                    required=True,
                ),
                TextPatch(
                    name="pn99_compress_on_transfer",
                    anchor=COPY_OLD,
                    replacement=COPY_NEW,
                    required=True,
                ),
            ],
        ),
    ]


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN99")
    log_decision("PN99", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN99 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patchers = _patchers()
    if patchers is None:
        return "skipped", "targets de cpu/{spec,gpu_worker}.py no encontrados"

    txn = MultiFilePatchTransaction(patchers, name="PN99")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
        "PN99 aplicado: L2 (y por arrastre L3) guardan la KV cuantizada a 4 "
        "bits, con la cuantizacion corriendo EN LA GPU. 1,78x mas bloques en "
        "los mismos bytes de RAM. Se activa con "
        "GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2=1; al prender o apagar hay que "
        "borrar /kv-offload porque cambia el layout. PN92 queda superseded y "
        "tiene que seguir apagado. Kill switch: GENESIS_DISABLE_PN99=1."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN99_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
