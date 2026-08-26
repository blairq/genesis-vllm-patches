# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN106 — Marlin FP8 repack disk-persistent cache.

Genesis-original 2026-08-24 (checkpoint CK-0.3 del plan de optimización).

El problema
-----------
En vLLM v0.23.0 sobre Ampere (sin tensor cores FP8), cada Linear del
checkpoint cuantizado corre por Marlin W8A16. Para eso,
`prepare_fp8_layer_for_marlin` repackea los pesos EN CADA ARRANQUE:
pack_fp8_to_int32 + gptq_marlin_repack + permute/repeat de escalas +
pliegue del bias de exponente. Para un 27B son segundos-decenas de
segundos por boot, y este stack reinicia seguido (restart policy=no a
propósito desde 2026-08-14: un crash debe quedar caído y visible).

La solución
-----------
Cache en disco del RESULTADO del repack, keyed por sha256 de:
  FORMAT_VERSION + fuente de la función original (drift automático:
  si upstream cambia el repack, el hash cambia y el cache se invalida)
  + formas/dtypes + bytes de pesos + bytes de escalas + bias + block_size.

Hit  → se cargan los tensores y se asignan vía replace_parameter
       (workspace se regenera fresco: es scratch de dispositivo).
Miss → corre el original y persiste (atomic rename, último escritor
       gana con contenido idéntico entre ranks TP).

Mismo patrón que PN57 (centroids TQ disk cache). Diferencia clave:
PN57 es text-patch sobre una función chiquita; acá usamos REBIND de
módulo porque envolver la función entera es robusto a cualquier cambio
del cuerpo (el hash de la fuente detecta el drift sin anclas frágiles).

Seguridad
---------
- Nunca lanza: cualquier falla de cache cae al repack original.
- Env-gated: GENESIS_ENABLE_PN106_MARLIN_REPACK_CACHE=1 (default OFF).
- Cache dir: GENESIS_MARLIN_CACHE_DIR o ~/.cache/genesis/marlin_repack.
- Los tensores se persisten en CPU y se mueven al device del layer al
  cargar (portable entre índices de GPU).

Models affected
---------------
Todo checkpoint FP8-bloque servido en Ampere (27B/35B Qwen3.x en este
stack). No-op en Hopper/Blackwell (FP8 nativo no pasa por Marlin) y en
checkpoints no-FP8 (la función ni se llama).

Author: ox-alpha 2026-08-24 (checkpoint CK-0.3, plan KERNELS-OPTIMIZACION).
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os

import torch

log = logging.getLogger("genesis.wiring.pn106_marlin_repack_cache")

ENV_FLAG = "GENESIS_ENABLE_PN106_MARLIN_REPACK_CACHE"
CACHE_DIR_ENV = "GENESIS_MARLIN_CACHE_DIR"
GENESIS_PN106_MARKER = "Genesis PN106 Marlin repack disk cache"
FORMAT_VERSION = "pn106-v1"

_TRUTHY = ("1", "true", "yes", "on")

_ORIGINAL = None
_INSTALLED_MODULE = None


def _cache_dir() -> str:
    root = os.environ.get(CACHE_DIR_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "genesis", "marlin_repack")
    os.makedirs(root, exist_ok=True)
    return root


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Bytes crudos del tensor, portable a dtypes sin equivalente numpy."""
    t = t.detach().cpu().contiguous()
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.view(torch.uint8).numpy().tobytes()
    if t.element_size() == 2:
        return t.view(torch.uint16).numpy().tobytes()
    if t.element_size() == 4:
        return t.view(torch.uint32).numpy().tobytes()
    return t.numpy().tobytes()


def _scale_attr_of(layer) -> str | None:
    for attr in ("weight_scale", "weight_scale_inv"):
        if hasattr(layer, attr):
            return attr
    return None


def _cache_key(layer, size_k_first: bool, input_dtype, src_hash: str) -> str:
    h = hashlib.sha256()
    h.update(FORMAT_VERSION.encode())
    h.update(src_hash.encode())
    h.update(repr((
        tuple(layer.weight.shape), str(layer.weight.dtype),
        getattr(layer, "output_size_per_partition", None),
        getattr(layer, "input_size_per_partition", None),
        bool(size_k_first), str(input_dtype),
        repr(getattr(layer, "weight_block_size", None)),
    )).encode())
    h.update(_raw_bytes(layer.weight))
    scale_attr = _scale_attr_of(layer)
    if scale_attr is not None:
        h.update(scale_attr.encode())
        h.update(_raw_bytes(getattr(layer, scale_attr)))
    bias = getattr(layer, "bias", None)
    if bias is not None:
        h.update(b"bias")
        h.update(_raw_bytes(bias))
    return h.hexdigest()


def install(module) -> bool:
    """Instala el wrapper sobre module.prepare_fp8_layer_for_marlin."""
    global _ORIGINAL, _INSTALLED_MODULE
    if getattr(module, "_genesis_pn106_installed", False):
        return True
    original = module.prepare_fp8_layer_for_marlin
    try:
        src_hash = hashlib.sha256(
            inspect.getsource(original).encode()).hexdigest()[:16]
    except Exception:
        src_hash = "nosrc"

    def cached_prepare(layer, size_k_first=True, input_dtype=None):
        key = None
        path = None
        try:
            key = _cache_key(layer, size_k_first, input_dtype, src_hash)
            path = os.path.join(_cache_dir(), key + ".pt")
            if os.path.isfile(path):
                payload = torch.load(path, map_location="cpu")
                device = layer.weight.device
                module.replace_parameter(
                    layer, "weight", payload["weight"].to(device))
                module.replace_parameter(
                    layer, payload["scale_attr"],
                    payload["scale"].to(device))
                layer.workspace = module.marlin_make_workspace_new(device)
                if payload.get("bias") is not None and hasattr(layer, "bias"):
                    module.replace_parameter(
                        layer, "bias", payload["bias"].to(device))
                log.info("PN106 cache HIT %s", key[:12])
                return
        except Exception as e:  # cache hit falló → repack normal
            log.warning("PN106 cache hit falló (%s); repack original", e)
        original(layer, size_k_first=size_k_first, input_dtype=input_dtype)
        if key is None or path is None:
            return
        try:
            scale_attr = _scale_attr_of(layer)
            if scale_attr is None:
                return
            payload = {
                "weight": layer.weight.detach().cpu().clone(),
                "scale_attr": scale_attr,
                "scale": getattr(layer, scale_attr).detach().cpu().clone(),
            }
            bias = getattr(layer, "bias", None)
            if bias is not None:
                payload["bias"] = bias.detach().cpu().clone()
            tmp = f"{path}.tmp{os.getpid()}"
            torch.save(payload, tmp)
            os.replace(tmp, path)
            log.info("PN106 cache SAVE %s", key[:12])
        except Exception as e:  # cache write failure non-fatal
            log.warning("PN106 cache save falló (%s)", e)

    module.prepare_fp8_layer_for_marlin = cached_prepare
    module._genesis_pn106_installed = True
    _ORIGINAL = original
    _INSTALLED_MODULE = module
    log.info("PN106 instalado sobre %s", getattr(module, "__name__", module))
    return True


def revert() -> bool:
    """Restaura la función original (para tests y apagado limpio)."""
    global _ORIGINAL, _INSTALLED_MODULE
    if _INSTALLED_MODULE is None or _ORIGINAL is None:
        return False
    _INSTALLED_MODULE.prepare_fp8_layer_for_marlin = _ORIGINAL
    _INSTALLED_MODULE._genesis_pn106_installed = False
    _INSTALLED_MODULE = None
    _ORIGINAL = None
    return True


def apply():
    """Punto de entrada del orquestador. Nunca lanza."""
    if os.environ.get(ENV_FLAG, "").lower() not in _TRUTHY:
        return "skipped", (
            "opt-in only — set GENESIS_ENABLE_PN106_MARLIN_REPACK_CACHE=1")
    try:
        from vllm.model_executor.layers.quantization.utils import (
            marlin_utils_fp8 as M,)
    except Exception as e:
        return "failed", f"import marlin_utils_fp8: {e}"
    try:
        if install(M):
            return "applied", (
                "rebind de prepare_fp8_layer_for_marlin con cache en disco "
                f"({CACHE_DIR_ENV} o ~/.cache/genesis/marlin_repack)")
        return "skipped", "ya estaba instalado"
    except Exception as e:
        return "failed", f"install: {e}"
