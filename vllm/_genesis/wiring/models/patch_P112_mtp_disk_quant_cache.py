# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch P112 — MTP Quantization Disk Cache and Fast Loading.

Allows automatic offline/online disk caching of quantized MTP draft weights
with cryptographic fingerprint verification (SHA256).

- Default OFF unless `GENESIS_ENABLE_MTP_QUANT_CACHE=1`.
- On Cache HIT: Loads in <0.05s with 0% CPU overhead via memory-mapped safetensors.
- On Cache MISS: Vectorized quantization in GPU/PyTorch, atomic disk save + manifest.
"""
# ═══════════════════════════════════════════════════════════════════════════
# ⚠️  AUDITORÍA 2026-08-19 — MANTENER DESACTIVADO (GENESIS_ENABLE_MTP_QUANT_CACHE=0)
#
# La auditoría completa (7 bugs, con evidencia medida) está en el encabezado de
# vllm/_genesis/mtp_cache.py. Resumen: sobre un checkpoint FP8 este parche
# aplica un round-trip FP8 → INT8 → FP16 que pierde precisión y no entrega ni
# IMMA ni VRAM, porque los pesos se dequantizan a FP16 antes de llegar a la GPU.
#
# Dos de esos bugs nacen EN ESTE ARCHIVO, en el replacement de más abajo:
#
#   [B4] `source_file=str(_genesis_p112_model_name)` — se pasa el NOMBRE del
#        repo HF donde get_cache_key() espera un PATH de archivo. Como el path
#        no existe, compute_file_hash() devuelve la string literal "missing" y
#        termina dentro de la clave del cache. Verificado:
#          sha256("orcarouter/Qwen3.8-27B-Uncensored-FP8|missing|int8|tp2|v1.0")
#            = afe2f5a591956d2072608ffa == fingerprint del manifest real.
#        O sea que la verificación SHA256 no hashea ni un byte de los pesos y el
#        cache no se invalida nunca al cambiar el checkpoint.
#        → FIX: resolver los safetensors reales del modelo y pasar sus paths.
#
#   [B5] Nunca se pasa `target_dtype` a process_mtp_weights(), así que gana el
#        default `torch.float16` de la firma. En un engine bfloat16 los pesos
#        del drafter saldrían fp16 en silencio.
#        → FIX: pasar vllm_config.model_config.dtype, y que además entre en la
#          clave del cache.
#
# NOTA AL EDITAR: los bugs de arriba viven DENTRO de P112_LOAD_WEIGHTS_REPLACEMENT,
# que es texto fuente que se inyecta en vLLM. No agregar comentarios adentro de
# ese string sin querer: cambia el código parcheado y el marker de idempotencia
# (GENESIS_P112_MARKER). Los comentarios de auditoría van acá afuera.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.p112_mtp_disk_quant_cache")
GENESIS_P112_MARKER = "Genesis P112 MTP Quant Disk Cache v1.1"

P112_LOAD_WEIGHTS_ANCHOR = (
    "        loader = AutoWeightsLoader(self)\n"
    "        return loader.load_weights(remap_weight_names(weights))\n"
)

P112_LOAD_WEIGHTS_REPLACEMENT = (
    "        # [Genesis P112 MTP Quant Disk Cache v1.1]\n"
    "        import os as _genesis_os\n"
    "        _genesis_p112_cache_enabled = (\n"
    "            _genesis_os.getenv(\"GENESIS_ENABLE_MTP_QUANT_CACHE\", \"0\") == \"1\"\n"
    "        )\n"
    "        _genesis_p112_remapped = remap_weight_names(weights)\n"
    "        if _genesis_p112_cache_enabled:\n"
    "            try:\n"
    "                from vllm._genesis.mtp_cache import MTPQuantDiskCacheManager\n"
    "                _genesis_p112_model_name = \"qwen3_mtp\"\n"
    "                if hasattr(self, \"vllm_config\") and hasattr(self.vllm_config, \"model_config\"):\n"
    "                    _genesis_p112_model_name = getattr(self.vllm_config.model_config, \"model\", \"qwen3_mtp\")\n"
    "                _genesis_p112_tp_size = 1\n"
    "                if hasattr(self, \"vllm_config\") and hasattr(self.vllm_config, \"parallel_config\"):\n"
    "                    _genesis_p112_tp_size = getattr(self.vllm_config.parallel_config, \"tensor_parallel_size\", 1)\n"
    "                _genesis_p112_quant_fmt = _genesis_os.getenv(\"GENESIS_MTP_QUANT_FORMAT\", \"int8\")\n"
    "                _genesis_p112_cache_mgr = MTPQuantDiskCacheManager(\n"
    "                    enabled=True, quant_format=_genesis_p112_quant_fmt\n"
    "                )\n"
    "                _genesis_p112_remapped = _genesis_p112_cache_mgr.process_mtp_weights(\n"
    "                    model_name=str(_genesis_p112_model_name),\n"
    "                    source_file=str(_genesis_p112_model_name),\n"
    "                    tp_size=_genesis_p112_tp_size,\n"
    "                    weights=_genesis_p112_remapped,\n"
    "                )\n"
    "            except Exception as _p112_exc:\n"
    "                import logging as _genesis_logging\n"
    "                _genesis_logging.getLogger(\"genesis.p112\").warning(\"[Genesis P112] MTP cache hook fallback: %s\", _p112_exc)\n"
    "                _genesis_p112_remapped = remap_weight_names(weights)\n"
    "\n"
    "        loader = AutoWeightsLoader(self)\n"
    "        return loader.load_weights(_genesis_p112_remapped)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/qwen3_5_mtp.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P112 Qwen3 MTP Quant Disk Cache",
        target_file=str(target),
        marker=GENESIS_P112_MARKER,
        sub_patches=[
            TextPatch(
                name="p112_mtp_load_weights_cache",
                anchor=P112_LOAD_WEIGHTS_ANCHOR,
                replacement=P112_LOAD_WEIGHTS_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision

    decision, reason = True, "MTP disk quant cache"
    log_decision("P112", decision, reason)

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file qwen3_5_mtp.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message="P112 applied: MTP quantization disk cache active with SHA256 verification.",
        patch_name=patcher.patch_name,
    )
