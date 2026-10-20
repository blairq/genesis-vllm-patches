# SPDX-License-Identifier: Apache-2.0
"""PN177 — memoria que los parches piden despues del perfil de vLLM, creada DENTRO del perfil.

vLLM dimensiona la KV con lo que queda despues de ``profile_run``, que corre sin atencion
(``skip_attn``): los transitorios de la atencion y del GDN los cubre solo la holgura de
``--gpu-memory-utilization``. Lo que se crea recien en la primera inferencia (buffers de SK-18, el
workspace de FlashInfer, el pool de salida del GDN de PN59) o al enlazar la KV (la cinta de PN122)
quedaba fuera de la cuenta y se comia esa holgura: el 20-10, un paso con 8 prefills sobre contexto
cacheado mas 3 decodes dio OOM por 42 MiB y colgo el motor (con caches de compilacion nuevas; con
las de produccion pasaba por poco). Aca se crea todo eso antes de medir; lo que necesita la KV (la
cinta) se aparta con un tensor del mismo tamano que se suelta justo antes de crearla.

Lo engancha el text-patch de ``wiring/hybrid/patch_PN177_reserva.py`` al principio de
``GPUModelRunner.profile_run`` (model runner v2).
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.pn177")
_apartado: dict[str, torch.Tensor] = {}


def antes_del_perfil(runner) -> None:
    """Se llama al empezar ``profile_run``, en cada worker. Nunca tumba el arranque."""
    try:
        _reservar(runner)
    except Exception as e:
        log.warning("[PN177] no se pudo reservar antes del perfil (%s: %s): la KV no cuenta los "
                    "buffers de SK-18/GDN y un paso con varios prefills largos puede dar OOM",
                    type(e).__name__, e)


def _reservar(runner) -> None:
    from vllm._genesis import gdn_cinta, sk18_attn
    from vllm._genesis.kernels.gdn_scratch_pool import GdnScratchPool

    dev = runner.device
    cfg = runner.vllm_config
    mc, pc, sc = cfg.model_config, cfg.parallel_config, cfg.scheduler_config
    partes = {}
    antes = torch.cuda.memory_allocated(dev)

    if sk18_attn.dtype_activo(cfg.cache_config.cache_dtype):
        partes["sk18"] = sk18_attn.reservar(
            dev, mc.get_num_kv_heads(pc), cfg.cache_config.block_size,
            mc.get_num_attention_heads(pc), mc.max_model_len, sc.max_num_seqs,
            sc.max_num_batched_tokens)

    capas = [m for m in runner.model.modules()
             if hasattr(m, "num_v_heads") and getattr(m, "num_spec", 0) > 0]
    if capas and GdnScratchPool.is_production_eligible():
        H, HV, K, V = gdn_cinta.dims(capas[0])
        x = torch.cuda.memory_allocated(dev)
        GdnScratchPool.reservar(sc.max_num_batched_tokens, HV, V, mc.dtype, dev)
        partes["gdn_salida"] = torch.cuda.memory_allocated(dev) - x
    if capas and gdn_cinta.activo():
        n = sum(gdn_cinta.bytes_cinta(c, sc.max_num_seqs) for c in capas)
        _apartado["cinta"] = torch.empty(n, dtype=torch.uint8, device=dev)
        partes["cinta"] = n

    log.info("[PN177] %.0f MiB creados antes del perfil de memoria: %s",
             (torch.cuda.memory_allocated(dev) - antes) / 2**20,
             ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in partes.items()) or "nada")


def soltar(nombre: str) -> None:
    """Suelta lo apartado para ``nombre`` justo antes de crear lo de verdad."""
    _apartado.pop(nombre, None)
