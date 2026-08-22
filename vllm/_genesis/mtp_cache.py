# SPDX-License-Identifier: Apache-2.0
"""Genesis MTP Quantization Disk Cache Manager.

Provides cryptographic fingerprinting, atomic disk caching, and fast loading
for Qwen3.8 / Qwen3.5 MTP (Multi-Token Prediction) layers.
"""
# ═══════════════════════════════════════════════════════════════════════════
# ⚠️  AUDITORÍA 2026-08-19 — DESACTIVAR ANTES DE USAR (GENESIS_ENABLE_MTP_QUANT_CACHE=0)
#
# Auditado sobre genesis-27b-qwen38-fp8 (orcarouter/Qwen3.8-27B-Uncensored-FP8,
# TP=2, --dtype float16, MTP K=3). En ese engine el parche es NETO NEGATIVO:
# aplica toda la pérdida de precisión de un round-trip FP8 → INT8 → FP16 y no
# entrega NADA a cambio — ni IMMA, ni VRAM, ni velocidad. Ver B2.
#
# Contexto de la medición: aceptación MTP 58,6% (acceptance length 2,76) sobre
# 3,2M de drafts / 8,8M de tokens generados. NO se aisló cuánto de esa caída es
# atribuible a este parche: ese es el experimento pendiente (ver "PARA
# REACTIVARLO", paso 0).
#
# ── LOS SIETE PROBLEMAS ────────────────────────────────────────────────────
#
# B1. PREMISA FALSA, SIN GUARD DE APLICABILIDAD.
#     El log de la línea ~138 declara "Backbone model is quantized (W4A16/W8A16)
#     but MTP layers are unquantized (FP16/BF16)". Eso es cierto para los engines
#     AutoRound/TurboQuant, para los que se escribió esto. NO es cierto para un
#     checkpoint FP8: en Qwen3.8-27B-Uncensored-FP8, 7 de las 8 proyecciones del
#     head MTP ya vienen en F8_E4M3 con escalas de bloque 128x128, y sólo
#     `mtp.fc` viene en BF16. El parche no inspecciona nada: asume y actúa.
#     → FIX: detectar dtype de entrada. Si ya viene cuantizado (F8_E4M3 / I8, o
#       existen tensores `*_scale_inv` acompañantes) → no-op + log explícito.
#
# B2. EL INT8 NO EXISTE EN RUNTIME — SÓLO EN DISCO. (el problema central)
#     Tanto el camino de cache HIT (línea ~126) como el de MISS (línea ~161)
#     llaman a dequantize_linear_weight_int8() y entregan FP16 al loader de vLLM.
#     Los pesos viven en VRAM en FP16 denso y el matmul corre en HMMA.
#     Consecuencias medidas:
#       · IMMA / tensor cores INT8 de Ampere: NUNCA se usan.
#       · Ahorro de VRAM: CERO (el peso final es FP16 igual).
#       · Ahorro en disco: 444 MiB → 405 MiB = 39 MiB. No los "~400MB" del log.
#       · Pérdida de precisión: COMPLETA.
#     → FIX: decidir qué feature es esto y separarlo en dos flags (ver abajo).
#
# B3. CUANTIZA LAS ESCALAS DE BLOQUE DEL FP8.
#     El filtro de la línea ~156 es `weight.ndim == 2 and weight.numel() > 1024`:
#     por TAMAÑO, no por ROL. Los `weight_scale_inv` del checkpoint FP8 son 2D y
#     grandes, así que entran. En el cache de este engine quedaron así:
#         I8   (40, 136)  mlp.down_proj.weight_scale_inv     ← escala en INT8
#         I8   (136, 40)  mlp.gate_proj.weight_scale_inv
#         I8   (96, 40)   self_attn.q_proj.weight_scale_inv
#         BF16 (8, 40)    self_attn.k_proj.weight_scale_inv  ← se salvó: 320 elem
#     k_proj/v_proj sobrevivieron sólo por ser chicos (8*40 = 320 <= 1024).
#     Una escala cuantizada mete error que MULTIPLICA al error del peso.
#     → FIX: filtrar por rol. Whitelist de proyecciones lineales reales
#       (q/k/v/o_proj, gate/up/down_proj, fc). Blacklist explícita de
#       *scale*, *scale_inv*, *zero_point*, *norm*, embed*, lm_head.
#
# B4. EL FINGERPRINT NO HASHEA NI UN BYTE DE LOS PESOS.
#     El wiring (patch_P112_mtp_disk_quant_cache.py) pasa el NOMBRE del modelo
#     donde get_cache_key() espera un PATH. compute_file_hash() no lo encuentra
#     y devuelve la string literal "missing". Reproducido exactamente:
#         sha256("orcarouter/Qwen3.8-27B-Uncensored-FP8|missing|int8|tp2|v1.0")
#           = afe2f5a591956d2072608ffa  ==  el fingerprint del manifest real
#     O sea que la "verificación criptográfica SHA256" que anuncia el parche es
#     hueca: cambiás los pesos manteniendo el nombre del repo y el cache viejo
#     se reusa en silencio.
#     → FIX: el wiring debe pasar paths reales de safetensors; y hacer que
#       compute_file_hash() devolviendo "missing" sea un ERROR duro, no un valor
#       aceptable para construir una clave.
#
# B5. target_dtype HARDCODEADO A float16 Y AUSENTE DE LA CLAVE.
#     El wiring nunca pasa target_dtype, así que queda el default float16 de la
#     firma (línea ~96). Un engine en bfloat16 recibiría pesos fp16 en silencio.
#     Y como target_dtype tampoco entra en get_cache_key(), cambiar --dtype no
#     invalida el cache.
#     → FIX: pasarlo desde el wiring (vllm_config.model_config.dtype) y meterlo
#       en la clave junto con el esquema de cuantización de origen.
#
# B6. LOS LOGS AFIRMAN COSAS FALSAS.
#     · "~400MB saved"            → son 39 MiB, y en disco, no en VRAM (B2).
#     · "MTP layers are unquantized" → falso en checkpoints FP8 (B1).
#     · "cryptographic fingerprint"  → no hashea los pesos (B4).
#     Un parche que miente en los logs es peor que uno que no loguea: hace que
#     el diagnóstico apunte al lado equivocado durante meses.
#     → FIX: reportar SIEMPRE lo medido (bytes en disco antes/después, bytes en
#       VRAM antes/después, error del canario), nunca lo esperado.
#
# B7. NO HAY VALIDACIÓN NUMÉRICA. (la causa raíz de que B1..B3 pasaran)
#     El parche cambia los pesos de un drafter y nunca comprueba cuánto los
#     cambió. Un canario de 6 líneas — error relativo máximo + similitud coseno
#     entre original y reconstruido, por tensor, log del peor caso, abort si
#     supera umbral — habría atrapado B1, B2, B3 y el punto de granularidad de
#     abajo el primer día.
#     → FIX: canario obligatorio, con fallback al camino sin cache si falla.
#
# ── ADEMÁS: LA GRANULARIDAD ES UN DOWNGRADE ────────────────────────────────
#     quantize_linear_weight_int8() hace RTN absmax per-row (una escala por
#     canal de salida), sin calibración. El origen FP8 tiene escalas por bloque
#     128x128. O sea que el "cache" reemplaza un esquema fino por uno grueso.
#     La granularidad debería DERIVARSE del quantization_config del checkpoint,
#     no salir de la env var fija GENESIS_MTP_QUANT_FORMAT.
#
# ── PARA REACTIVARLO, EN ORDEN ─────────────────────────────────────────────
#   0. Medir el baseline SIN el parche. Deltas de los contadores de /metrics:
#      spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total y los
#      accepted_tokens_per_pos. La métrica es acceptance length, NO el %:
#          AL = (accepted + drafts) / drafts        (baseline 2026-08: 2,76)
#      Sin este número no hay forma de saber si el parche ayuda o estorba.
#   1. Arreglar B1 (guard) y B3 (filtro por rol). Con eso el parche queda
#      correcto e inerte acá, y sano en los engines para los que se escribió.
#   2. Arreglar B4, B5 (fingerprint honesto) y bumpear GENESIS_CACHE_VERSION
#      para invalidar todos los caches escritos con los bugs de arriba.
#   3. Agregar B7 (canario) y arreglar B6 (logs medidos).
#   4. Recién ahí decidir qué feature se quiere, porque hoy hay DOS confundidas
#      bajo un solo nombre y se paga el costo de una sin el beneficio de ninguna:
#        (a) CACHE DE CARGA — ahorrar el tiempo de cuantización en CPU al
#            arrancar. Legítimo, pero debe guardar el dtype FINAL, sin
#            round-trip lossy. Sugerido: GENESIS_MTP_LOAD_CACHE.
#        (b) CUANTIZACIÓN REAL — bajar VRAM / usar IMMA. Exige que el peso
#            QUEDE en INT8 en memoria y que la capa se registre como lineal
#            cuantizada. Sugerido: GENESIS_MTP_QUANT, off por defecto.
#
# ── ANTES DE INVERTIR EN (b): HACER LA CUENTA ──────────────────────────────
#     Para IMMA hace falta W8A8 (peso Y activación en INT8) + SmoothQuant para
#     los outliers de activación + kernels CUTLASS de compressed-tensors.
#     Weight-only INT8 (W8A16) NO toca los tensor cores INT8: dequantiza a FP16
#     y corre HMMA. Sólo compra ancho de banda.
#     Y el premio, medido en este rig:
#         drafter: 444 MiB x 3 forwards / TP2 = 0,65 GB/GPU ~ 0,7  ms
#         target : 28,7 GB x 1 forward   / TP2 = 14,3 GB/GPU ~ 15,3 ms
#         → el drafter es 4,5% del paso; pasarlo a INT8 gana ~2%
#     ~2% a cambio de arriesgar aceptación, sobre un head de UNA capa. El costo
#     del drafter no está en su tamaño sino en que corre K veces en serie, y eso
#     ninguna cuantización lo arregla. Recomendación de la auditoría: hacer
#     1..3, dejar (a), y NO construir (b) para este head.
#     (El INT8 de Ampere sí valdría sobre el TARGET de 28,7 GB — SM 8.6 no tiene
#      FP8 nativo y hoy lo emula — pero eso es otro proyecto, no este parche.)
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Iterable, Iterator, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from vllm.logger import init_logger
    log = init_logger("vllm.genesis.mtp_cache")
except Exception:
    log = logging.getLogger("genesis.mtp_cache")

CACHE_DIR_DEFAULT = "/root/.cache/vllm/mtp_quant_cache"
GENESIS_CACHE_VERSION = "v1.0"


def compute_file_hash(filepath: str) -> str:
    """Compute a fast SHA256 fingerprint from file metadata and boundary chunks."""
    # [B4] Devolver "missing" como si fuera un hash válido es lo que vuelve hueco
    # todo el fingerprint: el llamador la concatena en la clave sin darse cuenta
    # de que no se leyó ningún peso. Debería ser una excepción, o al menos forzar
    # al caller a desactivar el cache. Ver el bloque de auditoría del encabezado.
    if not filepath or not os.path.exists(filepath):
        return "missing"
    st = os.stat(filepath)
    h = hashlib.sha256()
    h.update(f"{filepath}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    try:
        with open(filepath, "rb") as f:
            chunk_head = f.read(65536)
            h.update(chunk_head)
            if st.st_size > 131072:
                f.seek(-65536, os.SEEK_END)
                chunk_tail = f.read(65536)
                h.update(chunk_tail)
    except Exception as e:
        log.warning("Could not read full chunks for hash: %s", e)
    return h.hexdigest()[:16]


def quantize_linear_weight_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D linear weight matrix to symmetric INT8 with per-channel scales."""
    # [GRANULARIDAD] RTN absmax per-row, sin calibración. Si el checkpoint de
    # origen es Block-FP8 con escalas 128x128, esto es un DOWNGRADE de esquema:
    # cambia una escala por bloque por una escala por canal de salida. La
    # granularidad debería derivarse del quantization_config del modelo.
    # Para W8A8 de verdad hace falta además SmoothQuant (outliers de activación)
    # y compensación de error tipo GPTQ. Ver auditoría del encabezado.
    # weight: [out_features, in_features] in float16/bfloat16
    orig_device = weight.device
    w_float = weight.to(torch.float32)
    # per-channel max absolute value along input features (dim 1)
    max_val = torch.amax(torch.abs(w_float), dim=1, keepdim=True).clamp(min=1e-8)
    scale = (max_val / 127.0).to(torch.float16)
    qweight = torch.clamp(torch.round(w_float / scale.to(torch.float32)), -128, 127).to(torch.int8)
    return qweight.to(orig_device), scale.to(orig_device)


def dequantize_linear_weight_int8(qweight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize INT8 weight back to target float precision (FP16/BF16) on GPU."""
    # [B2] ⚠️ ESTA FUNCIÓN ES EL PROBLEMA CENTRAL DEL PARCHE.
    # Se la llama en los DOS caminos (cache HIT y cache MISS) antes de entregar
    # los pesos al loader de vLLM. Resultado: el INT8 sólo existe en el archivo
    # de disco; en VRAM el head MTP queda FP16 denso y el matmul corre en HMMA.
    # No se usa IMMA, no se ahorra VRAM, y la pérdida de precisión se paga entera.
    # Si el objetivo es cuantización real, el peso tiene que QUEDAR en INT8 y la
    # capa registrarse como lineal cuantizada; si el objetivo es sólo cachear la
    # carga, entonces no hay que cuantizar nada y se guarda el dtype final.
    return (qweight.to(torch.float32) * scale.to(torch.float32)).to(dtype)


class MTPQuantDiskCacheManager:
    """Manages disk caching and loading of quantized MTP draft weights."""

    def __init__(
        self,
        cache_dir: str = CACHE_DIR_DEFAULT,
        enabled: bool = True,
        quant_format: str = "int8",
    ):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.quant_format = quant_format

    def get_cache_key(
        self,
        model_name_or_path: str,
        source_file: str,
        tp_size: int,
    ) -> str:
        # [B4][B5] La clave NO incluye: target_dtype, el esquema de cuantización
        # del checkpoint de origen, ni un hash real de los pesos (source_file
        # llega como nombre de repo, no como path → source_hash == "missing").
        # Verificado en genesis-27b-qwen38-fp8:
        #   sha256("orcarouter/Qwen3.8-27B-Uncensored-FP8|missing|int8|tp2|v1.0")
        #     = afe2f5a591956d2072608ffa == fingerprint del manifest en disco.
        # Al arreglarlo, bumpear GENESIS_CACHE_VERSION para invalidar los caches
        # escritos con los bugs viejos.
        source_hash = compute_file_hash(source_file)
        raw_key = f"{model_name_or_path}|{source_hash}|{self.quant_format}|tp{tp_size}|{GENESIS_CACHE_VERSION}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def process_mtp_weights(
        self,
        model_name: str,
        source_file: str,
        tp_size: int,
        weights: Iterable[Tuple[str, torch.Tensor]],
        target_dtype: torch.dtype = torch.float16,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Processes weights with Cache Hit / Cache Miss logic."""
        # [B5] El wiring nunca pasa target_dtype → siempre gana este default.
        # Un engine en bfloat16 recibiría pesos fp16 sin ningún aviso.
        # [B1] Acá falta el guard de aplicabilidad: antes de tocar nada habría
        # que inspeccionar el stream y hacer no-op si los pesos YA vienen
        # cuantizados (F8_E4M3 / I8, o con `*_scale_inv` acompañantes).
        if not self.enabled:
            for name, weight in weights:
                yield name, weight
            return

        cache_key = self.get_cache_key(model_name, source_file, tp_size)
        entry_dir = os.path.join(self.cache_dir, cache_key)
        cache_file = os.path.join(entry_dir, "mtp_quant.safetensors")
        manifest_file = os.path.join(entry_dir, "manifest.json")

        # ── CACHE HIT ────────────────────────────────────────────────────────
        if os.path.exists(cache_file) and os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                if manifest.get("fingerprint") == cache_key and manifest.get("valid", False):
                    t0 = time.perf_counter()
                    log.info("[Genesis MTP Cache] ⚡ Cache HIT for MTP key %s. Loading from %s", cache_key, cache_file)
                    with safe_open(cache_file, framework="pt", device="cpu") as sf:
                        keys = list(sf.keys())
                        # Load and yield dequantized or quantized tensors
                        for k in keys:
                            if k.endswith(".scale"):
                                continue
                            if f"{k}.scale" in keys:
                                qweight = sf.get_tensor(k)
                                scale = sf.get_tensor(f"{k}.scale")
                                # [B2] Camino de cache HIT: dequantiza a FP16 y
                                # entrega FP16. Confirmado en los logs del engine
                                # ("Cache HIT load completed in 2.2 seconds").
                                # El INT8 nunca llega a la GPU como INT8.
                                weight = dequantize_linear_weight_int8(qweight, scale, target_dtype)
                                yield k, weight
                            else:
                                yield k, sf.get_tensor(k).to(target_dtype)
                    dt = time.perf_counter() - t0
                    log.info("[Genesis MTP Cache] ⚡ Cache HIT load completed in %.3f seconds (0%% CPU quantization overhead).", dt)
                    return
            except Exception as e:
                log.warning("[Genesis MTP Cache] Cache read error (%s). Regenerating cache...", e)

        # ── CACHE MISS ───────────────────────────────────────────────────────
        # [B6] Los tres logs de abajo afirman cosas que no se verificaron:
        #   · "Backbone quantized but MTP unquantized" → falso en checkpoints FP8:
        #     en Qwen3.8-27B-Uncensored-FP8 el head MTP ya viene F8_E4M3 (7 de 8
        #     proyecciones). Es la premisa B1 escrita como si fuera un hecho.
        #   · "~400MB saved" / "reduce VRAM footprint" → el ahorro real medido es
        #     39 MiB (444 → 405) y es EN DISCO. En VRAM el ahorro es 0 (ver B2).
        # Al arreglar el parche, estos logs deben reportar bytes medidos
        # antes/después (disco y VRAM) y el error del canario numérico (B7).
        log.info("[Genesis MTP Quant] 🧠 Starting INT8 Quantization for MTP Draft Model (Key: %s)", cache_key)
        log.info("[Genesis MTP Quant] ℹ️ Context: Backbone model is quantized (W4A16/W8A16) but MTP layers are unquantized (FP16/BF16).")
        log.info("[Genesis MTP Quant] ℹ️ Action: Compressing draft weights to INT8 to reduce VRAM footprint (~400MB saved) and caching to disk.")
        t0 = time.perf_counter()
        os.makedirs(entry_dir, exist_ok=True)
        tmp_cache_file = cache_file + ".tmp"
        tmp_manifest_file = manifest_file + ".tmp"

        collected_tensors: dict[str, torch.Tensor] = {}
        yield_tensors: list[Tuple[str, torch.Tensor]] = []

        weights_list = list(weights)
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(weights_list, desc="Quantizing MTP tensors (INT8)", unit="tensor", leave=True)
        except Exception:
            pbar = weights_list

        for name, weight in pbar:
            # [B3] ⚠️ FILTRO POR TAMAÑO, NO POR ROL. Este `numel() > 1024` es lo
            # que hace que entren los `weight_scale_inv` del checkpoint FP8 —
            # que son las escalas de bloque 128x128, no pesos. Cuantizarlas mete
            # error que MULTIPLICA al del peso. En el cache de este engine:
            #     I8   (40, 136)  mlp.down_proj.weight_scale_inv      ← corrupto
            #     I8   (136, 40)  mlp.gate_proj.weight_scale_inv      ← corrupto
            #     I8   (96, 40)   self_attn.q_proj.weight_scale_inv   ← corrupto
            #     BF16 (8, 40)    self_attn.k_proj.weight_scale_inv   ← 320 <= 1024
            # k_proj/v_proj se salvaron por casualidad, por ser chicos.
            # Reemplazar por whitelist de proyecciones (q/k/v/o_proj,
            # gate/up/down_proj, fc) + blacklist de *scale*, *scale_inv*,
            # *zero_point*, *norm*, embed*, lm_head.
            if weight.ndim == 2 and weight.numel() > 1024 and self.quant_format == "int8":
                qweight, scale = quantize_linear_weight_int8(weight)
                collected_tensors[name] = qweight.contiguous().cpu()
                collected_tensors[f"{name}.scale"] = scale.contiguous().cpu()
                # [B7] Acá va el canario: comparar `weight` vs `rec_weight`
                # (error relativo máximo + similitud coseno), loguear el peor
                # caso y abortar al camino sin cache si supera el umbral.
                # [B2] Yield reconstructed weight → sale FP16, no INT8.
                rec_weight = dequantize_linear_weight_int8(qweight, scale, target_dtype)
                yield_tensors.append((name, rec_weight))
            else:
                collected_tensors[name] = weight.contiguous().cpu()
                yield_tensors.append((name, weight))

        try:
            log.info("[Genesis MTP Quant] 💾 Persisting %d tensors to disk cache at %s...", len(collected_tensors), cache_file)
            save_file(collected_tensors, tmp_cache_file)
            manifest_data = {
                "fingerprint": cache_key,
                "model_name": model_name,
                "source_file": source_file,
                "quant_format": self.quant_format,
                "tp_size": tp_size,
                "version": GENESIS_CACHE_VERSION,
                "tensors_count": len(collected_tensors),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "valid": True,
            }
            with open(tmp_manifest_file, "w", encoding="utf-8") as mf:
                json.dump(manifest_data, mf, indent=2)

            # Atomic swap
            os.replace(tmp_cache_file, cache_file)
            os.replace(tmp_manifest_file, manifest_file)
            dt = time.perf_counter() - t0
            log.info("[Genesis MTP Quant] ✅ MTP Quant Cache written successfully in %.3f seconds (0%% CPU overhead on future boots).", dt)
        except Exception as e:
            log.error("[Genesis MTP Quant] Failed to persist MTP cache: %s", e)
            if os.path.exists(tmp_cache_file):
                try:
                    os.remove(tmp_cache_file)
                except Exception:
                    pass

        for name, weight in yield_tensors:
            yield name, weight
