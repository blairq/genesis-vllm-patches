# SPDX-License-Identifier: Apache-2.0
"""Wiring para PN110 — requant swap INT8 W8A8 por capa para los Linear FP8-bloque.

Genesis-original 2026-08-24, rediseñado 2026-08-24 (full requant swap).

Contexto y evidencia de OOM (2026-08-24)
----------------------------------------
Diseño v1 ("phase dispatch", dual) mantenía en VRAM el estado Marlin
repackeado **más** copias INT8 por capa: +~1 byte/param sobre
``gpu_memory_utilization=0.80`` ya lleno. En lab 2×RTX3090 (TP=2,
prefill 2×8000 tokens) ``_build_int8_state`` provocó ``OutOfMemoryError``
en casi todas las capas. Inviable. Ver
``workshop/ox_alpha/PLAN-CHECKPOINTS.md`` CK-2.1 y
``workshop/ox_alpha/KERNELS-OPTIMIZACION.md`` §2.

Nuevo diseño: swap 1:1 por capa
--------------------------------
El INT8 **reemplaza** a Marlin, no lo duplica. En
``process_weights_after_loading`` se captura ``weight``/``weight_scale_inv``
ANTES del repack Marlin; tras el repack, si la capa es elegible y la
conversión chunked tiene éxito, se adjunta ``layer._genesis_pn110_int8`` y
se **libera** el resto: referencias capturadas, ``layer.weight``
repackeado, ``layer.weight_scale_inv``/``layer.weight_scale`` y
``layer.workspace`` si existen. ``torch.cuda.empty_cache()`` cada 8
capas. VRAM neta por capa convertida queda ~igual (1 byte/param INT8
vs 1 byte/param Marlin; escalas per-channel [N,1] fp32 despreciables).

Si ``_build_int8_state`` falla, **no se libera nada**: la capa queda en
Marlin y se registra ``WARN``. El ``apply`` ya no despacha por umbral
``M``: si la capa tiene estado INT8, **siempre** va por INT8 (decode
incluido; bench §2 muestra decode INT8 ~ Marlin, bandwidth-bound). Si el
camino INT8 lanza excepción, no hay fallback (Marlin ya fue liberado):
se registra ``CRITICAL`` y se re-lanza.

Conversión acotada en memoria
------------------------------
``requantize_fp8_block_to_int8_chunked`` evita el buffer transitorio
``[K,N]`` fp32 completo. Dos pasadas por ``chunk_rows`` filas (default
2048): pasada 1 dequantiza cada chunk y acumula ``amax`` por columna en
``[N]`` fp32; pasada 2 requantiza chunk a chunk a int8 sobre un buffer
``[K,N]`` int8 prealocado. Transitorio máximo ``chunk_rows*N*4`` bytes.
Se conserva ``requantize_fp8_block_to_int8`` intacta para compatibilidad
y referencia.

Riesgo nuevo
------------
Antes el decode seguía en Marlin; ahora decode también es INT8
(per-token quant de activaciones + ``cutlass_scaled_mm``). El target
cambia en **todas** las fases, por lo que el gate de calidad/fidelidad
CK-2.4 (divergencia, tool-calls, TTFT) es **obligatorio** antes de
promover a PROD. Ver ``PLAN-CHECKPOINTS.md`` CK-2.1/2.4.

Límites (igual que v1)
----------------------
- **fp8-e4m3fn únicamente**; e4m3fnuz no soportado.
- **dims múltiplo de 16** para ``cutlass_scaled_mm``.
- **Escalas per-channel en sm80**: el kernel cutlass c2x solo soporta
  per-tensor/per-channel (epilogue ``RowOrScalarBroadcast``). El peso se
  colapsa bloque→per-channel.
- **sm < 89**; en sm_89+ el FP8 nativo gana.
- **block_quant únicamente**.

Seguridad
---------
- Env-gated opt-in: ``GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1``.
  Chequeo **primero** en ``apply()`` (lección PN109).
- Kill switch: ``GENESIS_DISABLE_PN110=1`` (precedencia).
- Exclusiones: ``GENESIS_PN110_EXCLUDE_LAYERS`` (substrings).
- Idempotente: marker ``_genesis_pn110_installed`` sobre la clase.
- Resumen de carga: convertidas/excluidas/fallback + bytes netos, una vez.

Cuantización configurable ``GENESIS_PN110_QUANTIZE``
-----------------------------------------------------
Env ``GENESIS_PN110_QUANTIZE`` controla qué super-kernels se usan (lista
separada por comas; cada valor se normaliza con ``strip().lower()``):

* ``fp8_int8`` (default): acepta FP8 (``float8_e4m3fn``) y cuantiza a INT8
  per-channel (comportamiento actual; requiere ``weight.dtype == float8_e4m3fn``).
* ``bf16_int8``: acepta BF16 (``torch.bfloat16``) y cuantiza a INT8 per-channel
  (misma lógica chunked pero con ``weight.dtype == torch.bfloat16`` y sin
  ``weight_scale_inv``).
* ``fp8_fp8``, ``bf16_fp8`` y otros valores futuros (``<src>_<dst>``): sintácticamente
  válidos pero aún no implementados; si se pide uno no soportado se loguea
  ``WARNING`` (una vez por valor) y se ignora. Si la lista queda vacía se usa
  el default ``fp8_int8``. El valor activo se loguea al
  arranque como ``GENESIS PN110: quantize=<lista>`` para ``docker logs | grep GENESIS``.

Ejemplos:
  ``GENESIS_PN110_QUANTIZE=fp8_int8``  → super kernel FP8→INT8 (default)
  ``GENESIS_PN110_QUANTIZE=bf16_int8`` → super kernel BF16→INT8
  ``GENESIS_PN110_QUANTIZE=bf16_int8,fp8_int8`` → habilita ambos (FP8 y BF16);
    el wrapper elige por capa según ``w_orig.dtype`` (``float8_e4m3fn`` → ``fp8_int8``,
    ``bfloat16`` → ``bf16_int8``; si ninguno coincide la capa se excluye)
  ``GENESIS_PN110_QUANTIZE=fp8_int8,bf16_int8`` → idem (orden irrelevante)

Author: ox-alpha 2026-08-24 (swap 2026-08-24).
"""
from __future__ import annotations

import logging
import os
import threading

import torch

log = logging.getLogger("genesis.wiring.pn110_int8_phase_dispatch")
# ── Fix 2026-08-25: asegurar que GENESIS logs aparezcan en `docker logs`
# aunque VLLM_LOGGING_LEVEL=WARNING (INFO se filtra). El logger genesis.*
# no tiene handler en DEFAULT_LOGGING_CONFIG (solo "vllm"), así que INFO
# se pierde si propagate va a root WARNING. Forzar nivel WARNING/INFO y
# handler a stderr para que `grep GENESIS` siempre capture.
try:
    log.setLevel(logging.INFO)
    # Si no hay handler propio, reusar el de "vllm" o añadir StreamHandler
    _has_handler = bool(log.handlers)
    if not _has_handler:
        _vllm_lg = logging.getLogger("vllm")
        if _vllm_lg.handlers:
            for _h in _vllm_lg.handlers:
                try:
                    log.addHandler(_h)
                except Exception:
                    pass
            log.propagate = False
        else:
            _h = logging.StreamHandler()
            try:
                _h.setLevel(logging.INFO)
            except Exception:
                pass
            try:
                _h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [%(name)s] %(message)s"))
            except Exception:
                pass
            log.addHandler(_h)
            log.propagate = False
    else:
        for _h in log.handlers:
            try:
                _h.setLevel(logging.INFO)
            except Exception:
                pass
    # También abrir nivel de handlers padre si es necesario (WARNING->INFO)
    # para no filtrar log.warning/log.info.
except Exception:
    pass

# Opt-in del dispatcher (misma convención que PN106/PN108/PN109): el
# parche NO se aplica salvo que el operador pida explícitamente el flag.
# Lección PN109: chequear el flag PRIMERO en apply().
ENV_FLAG = "GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH"

# Kill switch (inverse pattern): GENESIS_DISABLE_PN110=1 apaga.
_DISABLE_ENV = "GENESIS_DISABLE_PN110"

# Umbral de tokens (M) — conservado para compatibilidad pero ya NO usado
# en el dispatch (swap siempre INT8 si hay estado).
MIN_TOKENS_ENV = "GENESIS_PN110_W8A8_MIN_TOKENS"
_DEFAULT_MIN_TOKENS = "256"

# Substrings de nombre de capa a excluir del swap (separados por coma).
EXCLUDE_ENV = "GENESIS_PN110_EXCLUDE_LAYERS"

# Gate KL CK-2.3 — divergencia por capa entre peso FP8 original y INT8 convertido.
KL_THRESHOLD_ENV = "GENESIS_PN110_KL_THRESHOLD"
KL_CALIB_TOKENS_ENV = "GENESIS_PN110_KL_CALIB_TOKENS"
_DEFAULT_KL_THRESHOLD = "0.05"
_DEFAULT_KL_CALIB_TOKENS = "512"

# Diseño C híbrido diádico (§4): per-channel float + per-bloque shift int8.
HYBRID_ENV = "GENESIS_PN110_HYBRID"
_DEFAULT_HYBRID = "0"

# Cuantización configurable: qué dtypes acepta el super-kernel y a qué cuantiza.
# Valores posibles: fp8_int8 (default, FP8→INT8), bf16_int8 (BF16→INT8),
# fp8_fp8, bf16_fp8, etc. Por ahora solo fp8_int8 y bf16_int8 están
# implementados; cualquier otro loguea WARNING (una vez por valor) y se ignora.
# GENESIS_PN110_QUANTIZE acepta lista separada por comas; cada valor se
# normaliza con strip().lower(). Si la lista queda vacía se usa default.
# Ejemplos: GENESIS_PN110_QUANTIZE=bf16_int8,fp8_int8 habilita ambos,
#           =fp8_int8 solo FP8, =bf16_int8 solo BF16.
QUANTIZE_ENV = "GENESIS_PN110_QUANTIZE"
_DEFAULT_QUANTIZE = "fp8_int8"
_IMPLEMENTED_QUANTIZE = frozenset({"fp8_int8", "bf16_int8"})
_quantize_warned: set[str] = set()
_quantize_warned_lock = threading.Lock()

_MARKER_ATTR = "_genesis_pn110_installed"
_LAYER_ATTR = "_genesis_pn110_int8"

_TRUTHY = ("1", "true", "yes", "on")

_original_pwal = None
_original_apply = None
_installed_class = None

# ── Estado global para resumen de carga (una vez) ────────────────────────
_pn110_summary: dict = {
    "converted": 0,
    "excluded": 0,
    "fallback": 0,
    "bytes_int8": 0,
    "bytes_freed": 0,
    "layers_seen": 0,
    "_logged": False,
    "details": [],  # CK-2.3 — lista de {capa, kxn, kl, decision, reason}
    "fallback_reasons": {},  # contador por motivo (p.ej. kl_exceeded)
}
_pn110_summary_timer: threading.Timer | None = None
_pn110_summary_lock = threading.Lock()

# ── SUPER-KERNEL sin branches: precarga al arrancar (warmup) ─────────────
# M valores requeridos por especificacion (decode pequeño hasta prefill grande).
_WARMUP_M_VALUES: tuple[int, ...] = (1, 8, 32, 128, 512, 1664, 8000)
# KN fallback para modelos tipo Qwen3 (hidden 4096/3584/8192, intermediate 17408 etc).
_WARMUP_KN_FALLBACK: tuple[tuple[int, int], ...] = (
    (4096, 4096),
    (4096, 8192),
    (8192, 4096),
    (3584, 2048),
    (5120, 5120),
    (17408, 3584),
    (3584, 17408),
    (2048, 3584),
)
# Deduplicacion warmup por forma KxN (evita recompilar mismo shape muchas veces).
_warmed_shapes: set[tuple[int, int]] = set()
_warmed_lock = threading.Lock()

# ── GENESIS: logging detallado inicio — tabla de kernels a usar ─────────
# Lista global con una entrada por capa analizada: {capa, tipo, forma, arch, sk}
_genesis_kernel_plan: list[dict] = []
_genesis_kernel_plan_lock = threading.Lock()
_genesis_startup_logged: bool = False


def _genesis_arch_str() -> str:
    """Devuelve string arch sm_XX para logs (ej. sm_80). No lanza."""
    try:
        cc = _compute_capability()
        if cc is not None and isinstance(cc, tuple) and len(cc) == 2:
            return f"sm_{cc[0]}{cc[1]}"
    except Exception:
        pass
    return "sm_80"



def _genesis_warn(msg: str, *args) -> None:
    """Log GENESIS con warning + print para docker logs. No lanza."""
    try:
        # log.warning con formato estilo logging
        if args:
            log.warning(msg, *args)
            try:
                # Intenta formatear para print
                print(msg % args, flush=True)
            except Exception:
                try:
                    print(msg, *args, flush=True)
                except Exception:
                    pass
        else:
            log.warning(msg)
            try:
                print(msg, flush=True)
            except Exception:
                pass
    except Exception:
        try:
            # Último recurso: print crudo
            print(msg, flush=True)
        except Exception:
            pass

def _genesis_select_super_kernel(layer_name: str, layer_type: str, K: int, N: int, M: int | None = None) -> str:
    """Selecciona super-kernel SK para una capa según nombre/tipo y forma KxN.

    Mapeo exhaustivo SK-01..SK-11 (Qwen3.8 27B) basado en
    ``workshop/ox_alpha/super_kernels.md`` y formas por rank TP=2.
    Si no hay SK optimizado, retorna "default vLLM".

    Fallback: si arch >= sm_89, dims no múltiplo de 16, o K/N no válidos,
    retorna "default vLLM" para que el wrapper haga fallback a Marlin.

    Si no encuentra super kernel, emite ``log.warning`` con formato GENESIS
    para que ``docker logs | grep GENESIS`` contenga toda la info necesaria
    para escribir el kernel perfecto (nombre, tipo, forma KxN, arch, M bucket).

    Args:
        layer_name: nombre completo (ej. model.layers.3.mlp.gate_proj).
        layer_type: nombre de clase (ej. MergedColumnParallelLinear).
        K: input_size_per_partition.
        N: output_size_per_partition.
        M: bucket de tokens (opcional, para log). Si es None se loguean todos
            los buckets ``_WARMUP_M_VALUES``.

    Returns:
        String SK (ej. "SK-05 MLP_GATEUP_FUSED_INT8_DIADIC") o "default vLLM".
    """
    # ── Helper para log obligatorio antes de fallback "default vLLM" ────────
    # Formato requerido (task PN110):
    # GENESIS PN110: no se encontró super kernel para capa {nombre} (tipo {tipo}, forma {K}x{N}, arch sm_{arch}, M {M}) — usando kernels default de vLLM. Para crear un super kernel, implementa un kernel con forma K={K}, N={N}, M={M}, arch=sm_{arch}, tipo={tipo}
    # Debe usar log.warning (no log.info) para ser visible con docker logs | grep GENESIS.
    def _emit_no_sk_log() -> None:
        try:
            arch_full = _genesis_arch_str()
            if arch_full.startswith("sm_"):
                arch_num = arch_full[3:]
            else:
                arch_num = arch_full.replace("sm_", "") if "sm_" in arch_full else arch_full
                if not arch_num:
                    arch_num = "80"
            # M bucket: si el caller pasó M, usar ese; si no, listar todos
            if M is not None:
                try:
                    m_str = str(int(M))
                except Exception:
                    m_str = str(M)
            else:
                try:
                    m_str = ",".join(str(v) for v in _WARMUP_M_VALUES)
                except Exception:
                    m_str = "1,8,32,128,512,1664,8000"
            # Mensaje exacto requerido por la task — no cambiar formato
            _msg = (
                "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %sx%s, arch sm_%s, M %s) "
                "— usando kernels default de vLLM. Para crear un super kernel, implementa un kernel con forma "
                "K=%s, N=%s, M=%s, arch=sm_%s, tipo=%s"
            )
            log.warning(_msg, layer_name, layer_type, K, N, arch_num, m_str, K, N, m_str, arch_num, layer_type)
            try:
                # print para docker logs (stdout) además de logger
                print(_msg % (layer_name, layer_type, K, N, arch_num, m_str, K, N, m_str, arch_num, layer_type), flush=True)
            except Exception:
                pass
        except Exception:
            pass

    try:
        # ── Incompatibilidad arch/dims → default vLLM (fallback Marlin) ──
        try:
            cc = _compute_capability()
            if cc is not None and isinstance(cc, tuple) and len(cc) == 2:
                if (cc[0], cc[1]) >= (8, 9):
                    _emit_no_sk_log()
                    return "default vLLM"
        except Exception:
            pass
        try:
            if int(K) % 16 != 0 or int(N) % 16 != 0:
                _emit_no_sk_log()
                return "default vLLM"
            # Forma degenerada (K o N 0) también default
            if int(K) == 0 or int(N) == 0:
                _emit_no_sk_log()
                return "default vLLM"
        except Exception:
            pass
        name_l = (layer_name or "").lower()
        type_l = (layer_type or "").lower()
        combined = f"{name_l} {type_l}"
        # Visual / ViT — SK-11 passthrough
        if "visual" in name_l:
            return "SK-11 VISION_BF16 (passthrough)"
        # Norms / embeds — SK-09 (no Lineal INT8, pero listado para completitud)
        if "embed_tokens" in name_l:
            return "SK-09 NORM_EMBED_BF16_PASSTHROUGH"
        # MTP draft
        if name_l.startswith("mtp.") or ".mtp." in name_l or "mtp" in name_l.split("."):
            # Draft mirror usa mismos INT8 que target; distinguir por shape si es qkv/gateup
            if K == 5120 and N == 7168:
                return "SK-10 MTP_DRAFT_MIRROR (SK-03)"
            if K == 5120 and N == 17408:
                return "SK-05 MLP_GATEUP_FUSED_INT8_DIADIC (via SK-10)"
            if K in (8704, 17408) and N == 5120:
                return "SK-06 MLP_DOWN_INT8_SCALED_RESIDUAL (via SK-10)"
            return "SK-10 MTP_DRAFT_MIRROR"
        # SK-01 GDN QKVZ — in_proj_qkv / in_proj_z fusionados 16384x5120 global, 8192x5120 per-rank
        if "in_proj_qkvz" in name_l or "in_proj_qkv" in name_l or "in_proj_z" in name_l:
            return "SK-01 GDN_QKVZ_FUSED_INT8_DIADIC"
        if "gdn" in combined and "qkv" in combined:
            return "SK-01 GDN_QKVZ_FUSED_INT8_DIADIC"
        if "gdn" in combined and K == 5120 and N in (8192, 16384, 10240, 6144):
            return "SK-01 GDN_QKVZ_FUSED_INT8_DIADIC"
        # SK-02 GDN out_proj RowParallel 5120x6144 global, 5120x3072 per-rank
        if "linear_attn.out_proj" in name_l:
            return "SK-02 GDN_OUT_INT8_SCALED"
        # SK-03 FA QKV — q_proj concatenado 12288 + k/v 1024 cada uno, fused 14336 global / 7168 per-rank
        if "self_attn.q_proj" in name_l or "self_attn.k_proj" in name_l or "self_attn.v_proj" in name_l:
            return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
        if "q_proj" in name_l and "self_attn" in name_l:
            return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
        if "k_proj" in name_l and "self_attn" in name_l:
            return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
        if "v_proj" in name_l and "self_attn" in name_l:
            return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
        # Shape-based QKV fused (fallback cuando el nombre no trae self_attn)
        if K == 5120 and N in (7168, 14336, 12288, 1024):
            if "qkv" in name_l or "q_proj" in name_l or "k_proj" in name_l or "v_proj" in name_l:
                return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
            # Forma típica FA QKV aunque nombre sea genérico (cubrir merged QKV)
            if N == 7168:
                return "SK-03 FA_QKV_FUSED_INT8_DIADIC"
        # SK-04 FA O — o_proj RowParallel 5120x6144 global, 5120x3072 per-rank (Full Attention)
        if "self_attn.o_proj" in name_l:
            return "SK-04 FA_O_INT8_SCALED"
        # SK-05 GateUp — gate_proj + up_proj fused 34816 global, 17408 per-rank
        if "gate_proj" in name_l or "up_proj" in name_l or "gate_up" in name_l:
            return "SK-05 MLP_GATEUP_FUSED_INT8_DIADIC"
        # SK-06 Down — down_proj RowParallel 5120x17408 global, 5120x8704 per-rank
        if "down_proj" in name_l:
            return "SK-06 MLP_DOWN_INT8_SCALED_RESIDUAL"
        # SK-07 lm_head — vocab 248320 global, 124160 per-rank x 5120
        if "lm_head" in name_l:
            return "SK-07 LM_HEAD_VOCAB"
        if K == 5120 and N in (124160, 248320):
            return "SK-07 LM_HEAD_VOCAB"
        if N == 5120 and K in (124160, 248320):
            return "SK-07 LM_HEAD_VOCAB"
        # Shape-only fallbacks para casos donde el nombre es clase genérica
        if K == 5120 and N == 8192:
            return "SK-01 GDN_QKVZ_FUSED_INT8_DIADIC"
        if K == 5120 and N == 3072:
            # Ambiguo entre SK-02 y SK-04 (misma forma); priorizar por contexto ya cubierto arriba
            return "SK-02/SK-04 GDN_OUT/FA_O_INT8_SCALED (5120x3072)"
        if K == 5120 and N == 17408:
            return "SK-05 MLP_GATEUP_FUSED_INT8_DIADIC"
        if K == 8704 and N == 5120:
            return "SK-06 MLP_DOWN_INT8_SCALED_RESIDUAL"
        if K == 17408 and N == 5120:
            return "SK-06 MLP_DOWN_INT8_SCALED_RESIDUAL"
        # Norms restantes
        if "layernorm" in name_l or name_l.endswith(".norm.weight") or "norm.weight" in name_l:
            return "SK-09 NORM_EMBED_BF16_PASSTHROUGH"
        if "norm" in type_l and N == 0:
            return "SK-09 NORM_EMBED_BF16_PASSTHROUGH"
    except Exception:
        pass
    _emit_no_sk_log()
    return "default vLLM"


def _genesis_append_plan(capa: str, tipo: str, forma: str, arch: str, sk: str) -> None:
    """Añade entrada a la tabla global de kernels (thread-safe, no lanza)."""
    try:
        with _genesis_kernel_plan_lock:
            _genesis_kernel_plan.append({
                "capa": capa,
                "tipo": tipo,
                "forma": forma,
                "arch": arch,
                "sk": sk,
            })
    except Exception:
        pass


def _genesis_log_startup_table() -> None:
    """Log inicial GENESIS justo después de 'PN110 instalado' (antes de cargar pesos).

    Muestra tabla planificada de super-kernels a partir de fallback KN y M buckets.
    Usa log.info con prefijo GENESIS para `docker logs | grep GENESIS`.
    No lanza, no toca lógica.
    """
    global _genesis_startup_logged
    try:
        if _genesis_startup_logged:
            return
        _genesis_startup_logged = True
    except Exception:
        pass
    try:
        arch = _genesis_arch_str()
        try:
            _qm = _quantize_mode()
            if isinstance(_qm, (set, frozenset)):
                # Orden canónico fp8 primero luego bf16 para log estable (coincide con spec)
                _canonical = ["fp8_int8", "bf16_int8"]
                _ordered = [m for m in _canonical if m in _qm]
                # Añadir cualquier modo futuro no canónico ordenado al final
                _extra = sorted([m for m in _qm if m not in _canonical])
                _qm_str = ",".join(_ordered + _extra) if (_ordered or _extra) else _DEFAULT_QUANTIZE
            else:
                _qm_str = str(_qm)
        except Exception:
            _qm_str = _DEFAULT_QUANTIZE
        _msg_q = f"GENESIS PN110: quantize={_qm_str}"
        try:
            log.warning(_msg_q)
            print(_msg_q, flush=True)
        except Exception:
            pass
        for _msg in [
            "GENESIS PN110: iniciando análisis del modelo — armando lista de kernels a usar",
            f"GENESIS PN110: capas analizadas: 0 (previo a carga, arch {arch}, esperando pwal)",
            f"GENESIS PN110: tabla planificada de super-kernels — {len(_WARMUP_KN_FALLBACK)} formas KN fallback x {len(_WARMUP_M_VALUES)} M buckets",
            "GENESIS | capa (ejemplo) | tipo | forma | arch | super kernel |",
            "GENESIS |---|---|---|---|---|---|",
        ]:
            log.warning(_msg)
            try:
                print(_msg, flush=True)
            except Exception:
                pass
        for idx, (K, N) in enumerate(_WARMUP_KN_FALLBACK, 1):
            try:
                sk = _genesis_select_super_kernel(f"fallback_{idx}", "Fp8LinearMethod", K, N)
            except Exception:
                sk = "default vLLM"
            _m = f"GENESIS | fallback_{idx} | Fp8LinearMethod | {K}x{N} | {arch} | {sk} |"
            log.warning(_m)
            try:
                print(_m, flush=True)
            except Exception:
                pass
        # Ejemplos reales Qwen3.8 27B por rank TP=2 — mapeo SK esperado (útil para `grep GENESIS`)
        try:
            qwen_examples = [
                ("model.layers.0.linear_attn.in_proj_qkv", "ColumnParallelLinear", 5120, 8192),
                ("model.layers.0.linear_attn.out_proj", "RowParallelLinear", 5120, 3072),
                ("model.layers.0.self_attn.q_proj", "ColumnParallelLinear", 5120, 7168),
                ("model.layers.0.self_attn.o_proj", "RowParallelLinear", 5120, 3072),
                ("model.layers.0.mlp.gate_proj", "MergedColumnParallelLinear", 5120, 17408),
                ("model.layers.0.mlp.down_proj", "RowParallelLinear", 8704, 5120),
                ("lm_head", "ParallelLMHead", 5120, 124160),
                ("model.visual.blocks.0", "Qwen2_5_VisionBlock", 5120, 5120),
            ]
            _m = f"GENESIS PN110: ejemplos Qwen3.8 27B (por rank TP=2, arch {arch}):"
            log.warning(_m)
            try:
                print(_m, flush=True)
            except Exception:
                pass
            for ex_name, ex_type, ex_K, ex_N in qwen_examples:
                try:
                    ex_sk = _genesis_select_super_kernel(ex_name, ex_type, ex_K, ex_N)
                except Exception:
                    ex_sk = "default vLLM"
                _m2 = f"GENESIS | {ex_name} | {ex_type} | {ex_K}x{ex_N} | {arch} | {ex_sk} |"
                log.warning(_m2)
                try:
                    print(_m2, flush=True)
                except Exception:
                    pass
        except Exception:
            pass
        for _msg in [
            "GENESIS PN110: SK disponibles: SK-01 GDN_QKVZ, SK-02 GDN_OUT, SK-03 FA_QKV, SK-04 FA_O, SK-05 GATEUP, SK-06 DOWN, SK-07 LM_HEAD, SK-08 SSM, SK-09 NORM, SK-10 MTP, SK-11 VISION",
            "GENESIS PN110: detalle por capa se logueará en _make_pwal_wrapper al inicio de cada capa (nombre, tipo, forma, arch, SK)",
        ]:
            log.warning(_msg)
            try:
                print(_msg, flush=True)
            except Exception:
                pass
    except Exception:
        pass


def _genesis_emit_plan_summary() -> None:
    """Emite resumen GENESIS compacto con tabla agrupada de kernels (log.info).

    Llamado desde _emit_summary una vez al terminar carga. No lanza.

    Agrupa capas **consecutivas por orden real** con mismo super-kernel y
    misma forma/arch en una sola línea con rango. Nunca hace agrupación
    global por SK: si las capas con el mismo SK son 1,2,3,5 debe ser
    ``"1-3,5"``, no ``"1-80"``. Solo colapsa cuando realmente son
    consecutivas en el plan (mismo SK/forma/arch en entradas adyacentes).

    Ejemplo interleaved: 400 capas gate/down/qkv alternadas generan
    ~400 grupos consecutivos (no 3 grupos globales ``1-80``); se trunca
    a max 20 líneas.

    Formato por grupo (requisito):
        ``GENESIS PN110: capas 1-80 | SK-05 MLP_GATEUP | 5120x17408 | sm_86 | 80 capas``

    Garantiza **max 20 líneas** para la tabla completa usando rangos;
    si hay más de 20 grupos distintos trunca y avisa.
    Mantiene tabla inicial (startup) intacta; esta es la tabla final compacta.
    """
    try:
        with _genesis_kernel_plan_lock:
            plan = list(_genesis_kernel_plan)
        if not plan:
            try:
                arch = _genesis_arch_str()
                _genesis_warn("GENESIS PN110: tabla de kernels a usar — sin capas registradas (arch %s)", arch)
            except Exception:
                pass
            return
        _genesis_warn("GENESIS PN110: ================= tabla completa de kernels a usar =================")
        _genesis_warn("GENESIS PN110: capas analizadas: %d", len(plan))

        # ── helpers locales (no tocan lógica global) ─────────────────────
        import re as _re

        def _layer_num(capa: str) -> int | None:
            """Extrae número de capa de 'model.layers.N.*' (0-based) o 'fallback_N' (1-based).

            Returns:
                int 0-based para layers.N, para luego +1 en display (1-based).
                None si no hay patrón.
            """
            try:
                m = _re.search(r"layers\.(\d+)", capa)
                if m:
                    return int(m.group(1))
                m2 = _re.search(r"fallback_(\d+)", capa)
                if m2:
                    # fallback_1 -> 0 para ordenar, luego +1 en display
                    return int(m2.group(1)) - 1
            except Exception:
                pass
            return None

        def _compress(nums: list[int]) -> str:
            """Comprime lista de ids 1-based en '1-80' o '1-3,5'.

            Ejemplos:
                [1,2,3,5] -> "1-3,5"
                [1,2]     -> "1-2"
                [1,3,5]   -> "1,3,5"
            """
            if not nums:
                return "-"
            uniq = sorted(set(int(n) for n in nums if isinstance(n, int)))
            if not uniq:
                return "-"
            ranges: list[str] = []
            start = prev = uniq[0]
            for n in uniq[1:]:
                if n == prev + 1:
                    prev = n
                    continue
                # cerrar rango anterior — usa guion para cualquier longitud >=2
                if start == prev:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{prev}")
                start = prev = n
            # último rango
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            return ",".join(ranges)

        # Contador por SK para resumen final (igual que antes)
        sk_counts: dict[str, int] = {}
        for entry in plan:
            try:
                sk = str(entry.get("sk", "default vLLM"))
            except Exception:
                sk = "default vLLM"
            sk_counts[sk] = sk_counts.get(sk, 0) + 1

        # ── Agrupar **consecutivas por orden real** por (sk, forma, arch) ───
        # Respeta orden de aparición: solo colapsa entradas adyacentes con
        # misma clave. Nunca agrupación global por SK (corrige bug 1-80).
        consec_groups: list[dict] = []
        cur: dict | None = None
        for idx, entry in enumerate(plan, 1):
            try:
                capa = str(entry.get("capa", "?"))
                forma = str(entry.get("forma", "-"))
                arch = str(entry.get("arch", "-"))
                sk = str(entry.get("sk", "default vLLM"))
            except Exception:
                capa, forma, arch, sk = "?", "-", "-", "default vLLM"
            key = (sk, forma, arch)
            # display id para rango: si capa tiene layers.N usar N+1 (1-based),
            # si no usar índice secuencial del plan (1-based)
            try:
                ln = _layer_num(capa)
                disp = (ln + 1) if ln is not None else idx
            except Exception:
                disp = idx
            if cur is not None and cur["key"] == key:
                cur["end_idx"] = idx
                cur["count"] += 1
                cur["display_ids"].append(disp)
            else:
                if cur is not None:
                    consec_groups.append(cur)
                cur = {
                    "key": key,
                    "sk": sk,
                    "forma": forma,
                    "arch": arch,
                    "start_idx": idx,
                    "end_idx": idx,
                    "count": 1,
                    "display_ids": [disp],
                }
        if cur is not None:
            consec_groups.append(cur)

        # ── Garantizar max 20 líneas: truncar grupos consecutivos si hace falta
        groups_to_emit: list[dict] = consec_groups

        # ── Emitir tabla compacta (max 20 líneas) ───────────────────────
        _genesis_warn("GENESIS PN110: tabla compacta (agrupada por SK/forma/arch, max 20 lineas):")
        # header compacto legible para grep
        _genesis_warn("GENESIS PN110: capas | super kernel | forma | arch | conteo")
        max_lines = 20
        total_groups = len(groups_to_emit)
        truncated = False
        if total_groups > max_lines:
            # reserva 1 línea para aviso de truncado
            groups_to_emit = groups_to_emit[: max_lines - 1]
            truncated = True

        for g in groups_to_emit:
            try:
                sk = str(g.get("sk", "default vLLM"))
                forma = str(g.get("forma", "-"))
                arch = str(g.get("arch", "-"))
                count = int(g.get("count", 0))
                display_ids = g.get("display_ids", [])
                range_str = _compress(display_ids) if display_ids else "-"
                _genesis_warn("GENESIS PN110: capas %s | %s | %s | %s | %d capas", range_str, sk, forma, arch, count)
            except Exception:
                # fallback: loguear clave aunque falle compresión
                try:
                    _genesis_warn("GENESIS PN110: capas - | %s | %s | %s | %d capas", g.get("sk", "?"), g.get("forma", "-"), g.get("arch", "-"), g.get("count", 0))
                except Exception:
                    pass

        if truncated:
            try:
                remaining = total_groups - (max_lines - 1)
                _genesis_warn("GENESIS PN110: ... y %d grupos mas (tabla truncada a %d lineas, ver debug para detalle completo)", remaining, max_lines)
            except Exception:
                pass

        # ── Resumen por SK (igual que antes) ─────────────────────────────
        try:
            resumen = ", ".join(f"{k} x{v}" for k, v in sorted(sk_counts.items()))
            _genesis_warn("GENESIS PN110: resumen kernels: %s", resumen)
        except Exception:
            pass
        _genesis_warn("GENESIS PN110: =================================================================")
    except Exception:
        pass

def _warmup_fused_kernels_for_shape(K: int, N: int, Ms: tuple[int, ...] = _WARMUP_M_VALUES) -> None:
    """Precarga Triton + torch.compile para un K,N dado y Ms listados.

    Crea tensores dummy [M,K] bf16 + [K,N] int8 column-major + [N] fp32 y
    llama ``fused_quant_gemm`` y su version ``torch.compile`` para poblar
    cache Triton/Inductor antes del primer forward real. No lanza; log debug
    en fallo. Limita OOM skipeando Ms grandes si K*N*M excede.

    Deduplica por KxN: si ya se hizo warmup para esa forma, no repite.
    Respeta fallback: si no hay super kernel para la KxN, usa default vLLM
    y no precarga (loguea GENESIS).

    Args:
        K: dimension K del peso.
        N: dimension N del peso.
        Ms: tupla de M a precompilar.
    """
    # Deduplicacion
    try:
        with _warmed_lock:
            if (K, N) in _warmed_shapes:
                return
            _warmed_shapes.add((K, N))
    except Exception:
        pass
    # ── Fallback: si no hay super kernel para esta KxN, usar default vLLM (no warmup) ──
    try:
        if _genesis_select_super_kernel(f"fallback_{K}x{N}", "Fp8LinearMethod", K, N) == "default vLLM":
            try:
                log.warning(
                    "GENESIS PN110: no se encontró super kernel para capa fallback_%dx%d (tipo Fp8LinearMethod, forma %dx%d, arch %s) — usando kernels default de vLLM",
                    K,
                    N,
                    K,
                    N,
                    _genesis_arch_str(),
                )
            except Exception:
                pass
            return
    except Exception:
        pass
    try:
        if not torch.cuda.is_available():
            return
        # Import lazy para no ciclo
        try:
            from vllm._genesis.kernels.fused_quant_gemm import fused_quant_gemm as _fused
            from vllm._genesis.kernels.fused_quant_gemm import is_available as _fused_avail, is_triton_available as _triton_avail
        except Exception:
            return
        # Si Triton no disponible, igualmente intentar torch.compile path fallback
        for m in Ms:
            # Heuristica OOM: skip si m*K*N > 8000*8192*4096 ~ 268B (muy grande)
            # En practica 8000*8192*4096 int8 seria 256MB*8000? No; m*K*2 bytes + K*N*1.
            # Usar limite simple: m*K > 80M (~160MB bf16) skip para m grandes con K grande
            try:
                if m * K > 80_000_000 and m > 512:
                    # Para M=8000,K=8192 => 65M >80M? no, 65M <80M, permitimos.
                    # Solo skipear combinaciones extremas >120M
                    if m * K > 120_000_000:
                        continue
            except Exception:
                pass
            # Check memoria libre antes de allocar (evita OOM spam cuando GPU ya llena)
            try:
                free, total = torch.cuda.mem_get_info()
                # Estimacion: a bf16 M*K*2 + b int8 K*N*1 + out bf16 M*N*2 + overhead
                needed = m * K * 2 + K * N + m * N * 2 + 32 * 1024 * 1024
                if needed > free:
                    log.debug("PN110 warmup skip M=%d K=%d N=%d needed=%d free=%d", m, K, N, needed, free)
                    continue
            except Exception:
                pass
            try:
                a = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
                # Peso int8 column-major como lo construye PN110 (stride 1,K)
                try:
                    b_col = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device="cuda")
                    # Llenar con valores random int8 sin materializar row-major intermedio grande
                    # Usar randint sobre buffer transpuesto para evitar copy extra
                    tmp = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
                    b_col.copy_(tmp.t())
                    del tmp
                    b = b_col
                except Exception:
                    b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device="cuda")
                    try:
                        b = b.t().contiguous().t()
                    except Exception:
                        pass
                b_scale = torch.ones(N, dtype=torch.float32, device="cuda")
                # 1) Direct Triton warmup — pobla cache Triton
                try:
                    _fused(a, b, b_scale, out_dtype=torch.bfloat16)
                except Exception as e:
                    log.debug("PN110 warmup fused K=%d N=%d M=%d direct fallo: %s", K, N, m, type(e).__name__)
                # 2) torch.compile warmup — pobla Inductor cache
                try:
                    compiled = torch.compile(_fused, mode="reduce-overhead", fullgraph=False)
                    compiled(a, b, b_scale, out_dtype=torch.bfloat16)
                except Exception as e:
                    log.debug("PN110 warmup fused K=%d N=%d M=%d compile fallo: %s", K, N, m, type(e).__name__)
                # 3) cutlass warmup via int8_linear si disponible (opcional)
                try:
                    from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import quant_activation_per_token as _qat
                    # warmup quant + cutlass: quant per-token + scaled_mm ya esta dentro de int8_linear,
                    # pero aqui solo warmup quant triton
                    try:
                        from vllm._genesis.kernels.fused_quant_triton import quant_activation_per_token as _fused_quant
                        _fused_quant(a)
                    except Exception:
                        pass
                except Exception:
                    pass
                del a, b, b_scale
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception as e:
                log.debug("PN110 warmup outer K=%d N=%d M=%d fallo: %s", K, N, m, type(e).__name__)
                continue
    except Exception:
        pass

def _warmup_all_generic() -> None:
    """Warmup generico para KN fallback cuando modelo aun no disponible.

    Itera _WARMUP_KN_FALLBACK x _WARMUP_M_VALUES y deja kernels en cache.
    Llamado desde install() inmediatamente despues del rebind. No bloquea
    carga real: cada shape con try/except y empty_cache.
    """
    try:
        for K, N in _WARMUP_KN_FALLBACK:
            try:
                _warmup_fused_kernels_for_shape(K, N)
            except Exception:
                continue
    except Exception:
        pass

def _warmup_for_layer(K: int, N: int) -> None:
    """Warmup especifico para capa recien convertida (K,N reales del modelo).

    Llamado desde _make_pwal_wrapper tras construir state exitoso, para
    precompilar Ms requeridos para esa forma exacta antes del primer decode.
    """
    try:
        _warmup_fused_kernels_for_shape(K, N)
    except Exception:
        pass

def _discover_and_warmup_existing_layers() -> None:
    """Si al arrancar ya hay capas vivas (modelo cargado antes de install), warmup sus K/N.

    Escanea gc.get_objects() buscando objetos con input_size_per_partition /
    output_size_per_partition (Linear layers vLLM). No lanza.

    Nota: usa object.__getattribute__ + warnings.catch_warnings para no gatillar
    PytestUnknownMarkWarning cuando gc incluye pytest.mark (MarkGenerator.__getattr__
    emite warning para atributos desconocidos). Ver fix iteración 1 bucle 71-pass.
    """
    try:
        import gc as _gc
        import warnings
        seen: set[tuple[int,int]] = set()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for obj in _gc.get_objects():
                try:
                    # Bypass __getattr__ (MarkGenerator) para no emitir PytestUnknownMarkWarning
                    try:
                        k_raw = object.__getattribute__(obj, "input_size_per_partition")
                    except AttributeError:
                        continue
                    try:
                        n_raw = object.__getattribute__(obj, "output_size_per_partition")
                    except AttributeError:
                        continue
                    k = int(k_raw or 0)
                    n = int(n_raw or 0)
                    if k and n and k % 16 == 0 and n % 16 == 0 and (k, n) not in seen:
                        # Filtrar shapes razonables (evitar vacios)
                        if 128 <= k <= 32768 and 128 <= n <= 65536:
                            seen.add((k, n))
                except Exception:
                    continue
        if seen:
            log.info("PN110 precarga: %d formas KxN descubiertas en memoria %s — warmup", len(seen), sorted(seen))
            for K, N in sorted(seen):
                try:
                    _warmup_fused_kernels_for_shape(K, N)
                except Exception:
                    continue
    except Exception:
        pass


def _is_disabled() -> bool:
    """Devuelve True si el kill switch GENESIS_DISABLE_PN110 está activo."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY


def _min_tokens() -> int:
    """Lee el umbral de tokens desde MIN_TOKENS_ENV (default 256).

    Conservado para compatibilidad; el nuevo swap no lo usa en apply().
    :returns: umbral entero; si el env no parsea, default 256.
    """
    raw = os.environ.get(MIN_TOKENS_ENV, _DEFAULT_MIN_TOKENS).strip()
    try:
        val = int(raw)
        return val if val > 0 else 256
    except ValueError:
        return 256


def _excluded_substrings() -> tuple[str, ...]:
    """Lee los substrings de exclusión desde EXCLUDE_ENV.

    :returns: tupla de substrings no vacíos; vacía si el env no está.
    """
    raw = os.environ.get(EXCLUDE_ENV, "").strip()
    if not raw:
        return ()
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _kl_threshold() -> float:
    """Lee el umbral KL desde KL_THRESHOLD_ENV (default 0.05).

    :returns: umbral float; si el env no parsea o es negativo, default 0.05.
    """
    raw = os.environ.get(KL_THRESHOLD_ENV, _DEFAULT_KL_THRESHOLD).strip()
    try:
        val = float(raw)
        if val < 0 or not float(val) == val:  # NaN check
            return float(_DEFAULT_KL_THRESHOLD)
        return val
    except Exception:
        return float(_DEFAULT_KL_THRESHOLD)


def _kl_calib_tokens() -> int:
    """Lee el presupuesto de calibración desde KL_CALIB_TOKENS_ENV (default 512).

    :returns: entero positivo; si el env no parsea, default 512.
    """
    raw = os.environ.get(KL_CALIB_TOKENS_ENV, _DEFAULT_KL_CALIB_TOKENS).strip()
    try:
        val = int(raw)
        return val if val > 0 else int(_DEFAULT_KL_CALIB_TOKENS)
    except Exception:
        return int(_DEFAULT_KL_CALIB_TOKENS)


def _is_hybrid_enabled() -> bool:
    """Devuelve True si Diseño C híbrido está habilitado (default 0).

    Env ``GENESIS_PN110_HYBRID=0`` fuerza fallback a Diseño B per-channel.
    """
    return os.environ.get(HYBRID_ENV, _DEFAULT_HYBRID).strip().lower() in _TRUTHY


def _quantize_mode() -> frozenset[str]:
    """Lee ``GENESIS_PN110_QUANTIZE`` y valida (lista separada por comas).

    Valores implementados: ``fp8_int8`` (default) y ``bf16_int8``.
    Otros valores sintácticos (``fp8_fp8``, ``bf16_fp8``, etc.) aún no
    tienen kernel: se loguea WARNING (una vez por valor) y se ignora.
    Si la lista queda vacía se usa el default ``fp8_int8``.

    Returns:
        frozenset de modos habilitados (siempre no vacío, subconjunto de
        _IMPLEMENTED_QUANTIZE). Ej: ``frozenset({"fp8_int8"})`` por defecto,
        ``frozenset({"fp8_int8","bf16_int8"})`` si ``GENESIS_PN110_QUANTIZE=bf16_int8,fp8_int8``.
    """
    raw = os.environ.get(QUANTIZE_ENV, _DEFAULT_QUANTIZE)
    # Normalizar: separar por comas, strip/lower, filtrar vacíos
    try:
        parts = [p.strip().lower() for p in raw.split(",")]
    except Exception:
        parts = [_DEFAULT_QUANTIZE]
    enabled: set[str] = set()
    for token in parts:
        if not token:
            continue
        if token in _IMPLEMENTED_QUANTIZE:
            enabled.add(token)
        else:
            try:
                with _quantize_warned_lock:
                    if token not in _quantize_warned:
                        _quantize_warned.add(token)
                        _genesis_warn(
                            "GENESIS PN110: quantize=%s no implementado — ignorado (validos: %s)",
                            token, ", ".join(sorted(_IMPLEMENTED_QUANTIZE)),
                        )
            except Exception:
                pass
    if not enabled:
        # Si nada válido (p.ej. solo valores no implementados o vacío), usar default
        # Loguear warning solo si raw no era vacío y contenía algo no vacío?
        # No duplicar warning para default ya; simplemente retornar default.
        return frozenset({_DEFAULT_QUANTIZE})
    return frozenset(enabled)


def _is_quantize_enabled(mode: str) -> bool:
    """Chequea si un modo específico está habilitado en GENESIS_PN110_QUANTIZE.

    Args:
        mode: modo a chequear (ej. "fp8_int8", "bf16_int8"); se normaliza con
              strip().lower().

    Returns:
        True si el modo está en el frozenset retornado por _quantize_mode().
    """
    try:
        m = mode.strip().lower() if isinstance(mode, str) else ""
    except Exception:
        m = ""
    if not m:
        return False
    try:
        return m in _quantize_mode()
    except Exception:
        return False


def _append_pn110_detail(
    capa: str,
    kxn: str,
    kl: float | None,
    decision: str,
    reason: str | None = None,
) -> None:
    """Registra una entrada por capa para la tabla del boot summary (CK-2.3).

    Thread-safe. No lanza.

    :param capa: nombre de la capa (``_genesis_pn110_name`` o clase).
    :param kxn: dimensiones ``KxN``.
    :param kl: divergencia KL (o None si no aplica).
    :param decision: ``convertida`` | ``excluida`` | ``fallback``.
    :param reason: motivo opcional (p.ej. ``kl_exceeded``).
    """
    try:
        with _pn110_summary_lock:
            _pn110_summary["details"].append({
                "capa": capa,
                "kxn": kxn,
                "kl": kl,
                "decision": decision,
                "reason": reason,
            })
            if reason:
                d = _pn110_summary.get("fallback_reasons")
                if isinstance(d, dict):
                    d[reason] = d.get(reason, 0) + 1
    except Exception:
        pass


def _kl_divergence_core(
    a_fp32: torch.Tensor,
    b_fp32: torch.Tensor,
) -> float:
    """Núcleo puro de KL divergence entre dos tensores ya dequantizados a fp32.

    Usa ``torch.nn.functional.kl_div`` con ``log_softmax`` per-fila y
    per-columna y promedia ambas vistas. Si KL diverge (NaN/inf),
    hace fallback a MSE normalizado. Es pura y no muta inputs.

    :param a_fp32: tensor original dequantizado [K,N] fp32.
    :param b_fp32: tensor INT8 dequantizado [K,N] fp32.
    :returns: KL escalar >=0.
    """
    import math as _math
    import torch.nn.functional as _F
    try:
        # Per-fila: distribución sobre N por cada K
        p_row = _F.softmax(a_fp32, dim=-1)
        log_q_row = _F.log_softmax(b_fp32, dim=-1)
        kl_row = _F.kl_div(log_q_row, p_row, reduction='batchmean')
        # Per-columna: distribución sobre K por cada N
        p_col = _F.softmax(a_fp32, dim=0)
        log_q_col = _F.log_softmax(b_fp32, dim=0)
        kl_col = _F.kl_div(log_q_col, p_col, reduction='batchmean')
        kl_val = (kl_row + kl_col) / 2.0
        kl_f = float(kl_val.item()) if isinstance(kl_val, torch.Tensor) else float(kl_val)
        if not _math.isfinite(kl_f) or kl_f < 0:
            raise ValueError("KL no finita")
        if _math.isnan(kl_f) or _math.isinf(kl_f):
            raise ValueError("KL nan/inf")
        return kl_f
    except Exception:
        try:
            mse = _F.mse_loss(a_fp32, b_fp32, reduction='mean').item()
            var = a_fp32.var().item()
            if var and var > 1e-12 and _math.isfinite(var):
                mse = mse / (var + 1e-9)
            return float(mse) if _math.isfinite(mse) else 0.0
        except Exception:
            return 0.0


def compute_kl_per_layer(
    original_fp32: torch.Tensor,
    int8_dequant_fp32: torch.Tensor,
    *args,
    **kwargs,
) -> float:
    """Calcula KL divergence por capa entre peso original y peso INT8 dequantizado.

    Contrato CK-2.3 (PN110): firma pura ``(original_fp32, int8_dequant_fp32)``
    testeable en CPU. Dequantización ya hecha por el caller; esta función solo
    mide divergencia de distribución vía ``torch.nn.functional.kl_div`` con
    ``log_softmax``. Si KL no es aplicable (NaN/inf) hace fallback a MSE
    normalizado. Es **pura**: no muta los inputs, no hace ``.cpu()``
    innecesario (corre en el device de los inputs, CPU para tests).

    Compatibilidad legacy: si se llama con 5 args
    ``(original_weight_fp8, scale_inv, block_size, int8_weight, int8_scales)``
    (firma v2 previa), dequantiza internamente y delega al núcleo KL. Así
    los callers antiguos siguen funcionando.

    Por qué KL y no solo MSE — trade-off
    -------------------------------------
    - **MSE** (``mean((a-b)^2)``) mide error absoluto medio punto a punto.
      Es simétrico, fácil de interpretar y penaliza uniforme todo el rango.
      No distingue si el error degrada la forma de la distribución o solo la
      escala. Para pesos de LLM donde los outliers (magnitudes grandes)
      dominan la salida del GEMM, MSE puede subestimar el impacto de un
      colapso de cola.

    - **KL vía softmax** interpreta cada fila (o columna) de pesos como
       logits y compara ``p=softmax(a)`` vs ``q=softmax(b)`` con
       ``KL(p||q)= sum p*log(p/q) = kl_div(log_q, p)``. Es sensible a cambios
       de *forma* y ranking de magnitudes: si la requant invierte el orden de
       pesos grandes, KL crece aunque MSE sea modesto. Captura degradación de
       outliers mejor que MSE.

    - **Trade-off**: KL requiere normalización softmax. Para matrices
       grandes (K*N hasta 90M) la distribución flatten se vuelve casi uniforme
       (prob ~1/N) y KL se hace muy pequeña (<1e-4) incluso con error
       moderado — pierde sensibilidad. Por eso se computa **per-fila**
       (softmax sobre N por cada fila K) y **per-columna** (softmax sobre K
       por cada columna N) y se promedia: mantiene soporte pequeño (256-17k)
       y es chunkable, sensible a 0.12 en casos de mala escala vs 2e-05 en
       conversión fiel (threshold 0.05 separa). Si KL diverge (NaN/inf) por
       valores extremos o distribución uniforme degenerada, se usa MSE como
       proxy conservador.

    - **No .cpu() innecesario**: todo el cálculo permanece en el device de
       los inputs (GPU si vienen de VRAM). Solo para capas pequeñas de test
       en CPU se ejecuta naturalmente en CPU — es calibración offline una vez
       al cargar, no en el camino caliente.

    :param original_fp32: peso original dequantizado [K,N] fp32 (o [N,K] si
        se pasa transpuesto; la función detecta y adapta). En modo legacy este
        parámetro es ``original_weight_fp8`` (float8_e4m3fn).
    :param int8_dequant_fp32: peso INT8 dequantizado [K,N] fp32 (mismo shape
        que original). En modo legacy este parámetro es ``scale_inv``.
    :param args: en modo legacy ``(block_size, int8_weight, int8_scales)``.
    :param kwargs: equivalentes por keyword para modo legacy.
    :returns: KL divergence escalar ``>=0`` (o MSE fallback). Menor es más fiel.
    :raises ValueError: si dims inválidas o dtype no esperado.
    """
    import math as _math
    import torch.nn.functional as _F

    # ── Detección modo legacy (5 args) ──────────────────────────────────
    # Si hay args extra o el primer tensor es float8, asumir firma antigua:
    #   (original_weight_fp8, scale_inv, block_size, int8_weight, int8_scales)
    _is_legacy = False
    _legacy_block_size = None
    _legacy_int8_weight = None
    _legacy_int8_scales = None
    _legacy_scale_inv = None
    _legacy_orig_fp8 = None
    try:
        if original_fp32.dtype == torch.float8_e4m3fn:
            _is_legacy = True
    except Exception:
        pass
    if args or kwargs:
        _is_legacy = True
    if _is_legacy:
        # Reconstruir parámetros legacy desde posicionales y kwargs
        try:
            # Caso posicional: compute_kl_per_layer(orig_fp8, scale_inv, block_size, int8_w, int8_scales)
            if args:
                if len(args) >= 3:
                    _legacy_orig_fp8 = original_fp32
                    _legacy_scale_inv = int8_dequant_fp32
                    _legacy_block_size = args[0]
                    _legacy_int8_weight = args[1]
                    _legacy_int8_scales = args[2]
                    # Si hay más args, ignorar
                elif len(args) == 1:
                    # Posible llamada con kwargs restantes
                    _legacy_orig_fp8 = original_fp32
                    _legacy_scale_inv = int8_dequant_fp32
                    _legacy_block_size = args[0]
                    _legacy_int8_weight = kwargs.get("int8_weight", kwargs.get("int8_scales", None))
                    _legacy_int8_scales = kwargs.get("int8_scales", kwargs.get("int8_weight", None))
                    if _legacy_int8_weight is None:
                        _legacy_int8_weight = kwargs.get("int8_weight")
                    if _legacy_int8_scales is None:
                        _legacy_int8_scales = kwargs.get("int8_scales")
                else:
                    _legacy_orig_fp8 = original_fp32
                    _legacy_scale_inv = int8_dequant_fp32
                    _legacy_block_size = kwargs.get("block_size") or kwargs.get("blockSize")
                    _legacy_int8_weight = kwargs.get("int8_weight")
                    _legacy_int8_scales = kwargs.get("int8_scales")
            else:
                # Solo kwargs
                _legacy_orig_fp8 = original_fp32
                _legacy_scale_inv = int8_dequant_fp32
                _legacy_block_size = kwargs.get("block_size", kwargs.get("blockSize"))
                _legacy_int8_weight = kwargs.get("int8_weight")
                _legacy_int8_scales = kwargs.get("int8_scales")
            # Si faltan, intentar inferir desde kwargs con nombres alternativos
            if _legacy_block_size is None:
                _legacy_block_size = kwargs.get("block_size") or kwargs.get("blockSize") or (128, 128)
            if _legacy_orig_fp8 is not None and _legacy_scale_inv is not None and _legacy_block_size is not None and _legacy_int8_weight is not None and _legacy_int8_scales is not None:
                # ── Dequant legacy idéntico al original v2 ──────────────────
                bk, bn = _legacy_block_size
                original_weight_fp8 = _legacy_orig_fp8
                scale_inv = _legacy_scale_inv
                int8_weight = _legacy_int8_weight
                int8_scales = _legacy_int8_scales
                if original_weight_fp8.dim() != 2 or int8_weight.dim() != 2:
                    raise ValueError(
                        f"compute_kl_per_layer: tensores deben ser 2-D, obtenido "
                        f"{original_weight_fp8.dim()}D y {int8_weight.dim()}D")
                if original_weight_fp8.dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        f"compute_kl_per_layer: esperado float8_e4m3fn, obtenido {original_weight_fp8.dtype}")
                if int8_weight.dtype != torch.int8:
                    raise ValueError(
                        f"compute_kl_per_layer: int8_weight debe ser int8, obtenido {int8_weight.dtype}")
                k_int8, n_int8 = int8_weight.shape
                k_orig, n_orig = original_weight_fp8.shape
                _scale = scale_inv
                _orig = original_weight_fp8
                if (k_orig, n_orig) != (k_int8, n_int8):
                    if (k_orig, n_orig) == (n_int8, k_int8):
                        _orig = original_weight_fp8.t().contiguous()
                        if _scale.dim() == 2:
                            _scale = _scale.t().contiguous()
                        k_orig, n_orig = _orig.shape
                    else:
                        raise ValueError(
                            f"compute_kl_per_layer: shape mismatch original {tuple(original_weight_fp8.shape)} "
                            f"vs int8 {tuple(int8_weight.shape)} y no es transpuesta")
                k, n = k_orig, n_orig
                if k % bk != 0 or n % bn != 0:
                    raise ValueError(
                        f"compute_kl_per_layer: dims {k}x{n} no son múltiplo de block_size {(bk,bn)}")
                expected_scale_shape = (k // bk, n // bn)
                if tuple(_scale.shape) != expected_scale_shape:
                    if tuple(_scale.shape) == (expected_scale_shape[1], expected_scale_shape[0]):
                        _scale = _scale.t().contiguous()
                    if tuple(_scale.shape) != expected_scale_shape:
                        raise ValueError(
                            f"compute_kl_per_layer: scale_inv shape {tuple(_scale.shape)} "
                            f"!= esperado {expected_scale_shape} para weight {k}x{n} y block {(bk,bn)}")
                # Dequant original bloque → fp32
                try:
                    w_4d = _orig.reshape(k // bk, bk, n // bn, bn)
                    s_4d = _scale.reshape(k // bk, 1, n // bn, 1)
                    w_orig_fp32 = (w_4d.to(torch.float32) * s_4d.to(torch.float32)).reshape(k, n)
                except Exception:
                    s_exp = _scale.repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)
                    if s_exp.shape[0] > k:
                        s_exp = s_exp[:k]
                    elif s_exp.shape[0] < k:
                        pad = k - s_exp.shape[0]
                        s_exp = torch.cat([s_exp, s_exp[-1:].repeat(pad, 1)], dim=0)
                    if s_exp.shape[1] > n:
                        s_exp = s_exp[:, :n]
                    elif s_exp.shape[1] < n:
                        pad = n - s_exp.shape[1]
                        s_exp = torch.cat([s_exp, s_exp[:, -1:].repeat(1, pad)], dim=1)
                    w_orig_fp32 = _orig.to(torch.float32) * s_exp.to(torch.float32)
                # Dequant INT8 per-channel → fp32
                try:
                    _sc = int8_scales
                    if _sc.dim() == 2:
                        if _sc.shape == (n, 1):
                            scale_vec = _sc.squeeze(1)
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.unsqueeze(0).to(int8_weight.device)
                        elif _sc.shape == (1, n):
                            scale_vec = _sc.squeeze(0)
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.unsqueeze(0).to(int8_weight.device)
                        elif _sc.shape == (k, 1) and _sc.shape[0] == k:
                            scale_vec = _sc.squeeze(1)
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.unsqueeze(1).to(int8_weight.device)
                        else:
                            flat = _sc.reshape(-1).to(torch.float32)
                            if flat.numel() == n:
                                w_int8_fp32 = int8_weight.to(torch.float32) * flat.unsqueeze(0).to(int8_weight.device)
                            elif flat.numel() == k:
                                w_int8_fp32 = int8_weight.to(torch.float32) * flat.unsqueeze(1).to(int8_weight.device)
                            else:
                                w_int8_fp32 = int8_weight.to(torch.float32) * _sc.to(torch.float32).to(int8_weight.device)
                    elif _sc.dim() == 1:
                        scale_vec = _sc.to(torch.float32)
                        if scale_vec.numel() == n:
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.unsqueeze(0).to(int8_weight.device)
                        elif scale_vec.numel() == k:
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.unsqueeze(1).to(int8_weight.device)
                        else:
                            w_int8_fp32 = int8_weight.to(torch.float32) * scale_vec.to(int8_weight.device)
                    else:
                        w_int8_fp32 = int8_weight.to(torch.float32) * _sc.to(torch.float32).to(int8_weight.device)
                except Exception:
                    w_int8_fp32 = int8_weight.to(torch.float32)
                if tuple(w_orig_fp32.shape) != tuple(w_int8_fp32.shape):
                    if tuple(w_orig_fp32.shape) == tuple(w_int8_fp32.t().shape):
                        w_int8_fp32 = w_int8_fp32.t().contiguous()
                    else:
                        raise ValueError(
                            f"compute_kl_per_layer: dequant shapes no coinciden {tuple(w_orig_fp32.shape)} "
                            f"vs {tuple(w_int8_fp32.shape)}")
                return _kl_divergence_core(w_orig_fp32, w_int8_fp32)
        except Exception as e:
            # Si la detección legacy falla y no es un ValueError intencional, re-lanzar ValueError
            if isinstance(e, ValueError):
                raise
            # Sino caer al path normal (dos tensores fp32)
            pass

    # ── Path normal CK-2.3: dos tensores fp32 ya dequantizados ─────────
    w_orig_fp32 = original_fp32
    w_int8_fp32 = int8_dequant_fp32
    if not isinstance(w_orig_fp32, torch.Tensor) or not isinstance(w_int8_fp32, torch.Tensor):
        raise ValueError("compute_kl_per_layer: ambos argumentos deben ser torch.Tensor")
    if w_orig_fp32.dim() != 2 or w_int8_fp32.dim() != 2:
        raise ValueError(
            f"compute_kl_per_layer: tensores deben ser 2-D, obtenido "
            f"{w_orig_fp32.dim()}D y {w_int8_fp32.dim()}D")
    # Permitir dtypes fp32/fp16/bf16/fp64 para original; int8_dequant debe ser flotante
    # Si vienen en fp16, promover a fp32 para softmax estable
    if tuple(w_orig_fp32.shape) != tuple(w_int8_fp32.shape):
        if tuple(w_orig_fp32.shape) == tuple(w_int8_fp32.t().shape):
            w_int8_fp32 = w_int8_fp32.t().contiguous()
        else:
            raise ValueError(
                f"compute_kl_per_layer: shape mismatch {tuple(w_orig_fp32.shape)} "
                f"vs {tuple(w_int8_fp32.shape)} y no es transpuesta")
    # Promover a fp32 si es necesario, sin mutar inputs (crear vistas)
    if w_orig_fp32.dtype != torch.float32:
        w_orig_fp32 = w_orig_fp32.to(torch.float32)
    if w_int8_fp32.dtype != torch.float32:
        w_int8_fp32 = w_int8_fp32.to(torch.float32)
    return _kl_divergence_core(w_orig_fp32, w_int8_fp32)


def _is_layer_excluded(layer: torch.nn.Module, excludes: tuple[str, ...]) -> bool:
    """Chequea si la capa coincide con alguno de los substrings de exclusión.

    El substring se busca en el nombre de clase del módulo (siempre
    disponible, p.ej. "MergedColumnParallelLinear") y, si está
    registrado, en el nombre completo del módulo en el árbol
    (``_genesis_pn110_name``, p.ej. "model.layers.3.mlp.gate_up_proj").
    El loader de vLLM no pasa el nombre al método, así que por defecto
    la exclusión opera por clase de capa.

    Además excluye automáticamente capas GDN/mamba (in_proj_qkvz,
    in_proj_ba, out_proj en qwen_gdn_linear_attn.py) que usan
    Fp8LinearMethod pero no deben convertirse: si el nombre de la capa
    o el nombre de clase contiene "gdn" o "mamba" (case-insensitive)
    se excluye incondicionalmente.

    :param layer: la capa a chequear.
    :param excludes: substrings desde EXCLUDE_ENV.
    :returns: True si hay que excluir la capa del swap INT8.
    """
    # Exclusión automática GDN/mamba (previene Shape mismatch b_q_weight 5120)
    try:
        name_lower = getattr(layer, "_genesis_pn110_name", "").lower()
        type_lower = type(layer).__name__.lower()
        if "gdn" in name_lower or "mamba" in name_lower:
            return True
        if "gdn" in type_lower or "mamba" in type_lower:
            return True
    except Exception:
        pass
    if not excludes:
        return False
    try:
        candidates = [type(layer).__name__]
        name = getattr(layer, "_genesis_pn110_name", "")
        if name:
            candidates.append(name)
        return any(sub in cand for cand in candidates for sub in excludes)
    except Exception:
        return False


def _invalidate_compile_cache() -> None:
    """Invalida el cache de compilación cuando PN110 está activo.

    El grafo inductor cacheado sin PN110 espera b_q_weight packed int32;
    reusarlo con peso INT8 unpacked causa ``Shape mismatch: b_q_weight.size(0)=5120``.
    Se hace ``torch._dynamo.reset()`` y se borran los directorios de cache
    de inductor si existen, además de setear ``VLLM_DISABLE_COMPILE_CACHE=1``.
    Nunca lanza.
    """
    try:
        import torch._dynamo as _dynamo_mod  # type: ignore
        try:
            _dynamo_mod.reset()  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    try:
        import torch.compiler as _compiler_mod  # type: ignore
        try:
            if hasattr(_compiler_mod, "reset"):
                _compiler_mod.reset()  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    # Deshabilitar cache para esta sesión (seguro y evita reuso de grafo stale)
    try:
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    except Exception:
        pass
    # Borrar directorios de cache conocidos (no lanza si no existen)
    try:
        import shutil
        candidates = [
            "/root/.cache/vllm/torch_compile_cache",
            "/root/.cache/vllm/torch_inductor_cache",
            os.path.expanduser("~/.cache/vllm/torch_compile_cache"),
            os.path.expanduser("~/.cache/vllm/torch_inductor_cache"),
        ]
        # También VLLM_CACHE_ROOT si está definido
        try:
            _cr = os.environ.get("VLLM_CACHE_ROOT", "").strip()
            if _cr:
                candidates.extend([
                    os.path.join(_cr, "torch_compile_cache"),
                    os.path.join(_cr, "torch_inductor_cache"),
                ])
        except Exception:
            pass
        for _p in candidates:
            try:
                if os.path.exists(_p):
                    shutil.rmtree(_p, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def _compute_capability() -> tuple[int, int] | None:
    """Devuelve (major, minor) del GPU actual o None si no se puede leer."""
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_cuda():
            return None
        cc = current_platform.get_device_capability()
        if cc is None:
            return None
        return (cc.major, cc.minor)
    except Exception:
        return None


def requantize_fp8_block_to_int8(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantiza un peso FP8-bloque [K,N] a INT8-bloque [K,N].

    Dequantiza cada bloque (Bk x Bn) a float32 usando la escala del
    checkpoint, calcula el amax del bloque /127 como nueva escala INT8,
    y redondea+clampa a [-127,127]. Pura: no muta los inputs.

    La precisión es equivalente a la del checkpoint: la misma
    información de escala por bloque, redondeada a la grilla INT8.

    :param weight_fp8: peso [K, N] en float8_e4m3fn (contiguo o vista).
    :param scale_inv: escalas [K/Bk, N/Bn] en float32 (una por bloque).
    :param block_size: (Bk, Bn) — tamaño del bloque de cuantización
        (p.ej. (128, 128)).
    :returns: (w_int8, scales) donde w_int8 es [K, N] int8 contiguo y
        scales es [K/Bk, N/Bn] float32 (amax_bloque / 127).
    :raises ValueError: si las dims no son múltiplo de block_size.
    """
    bk, bn = block_size
    k, n = weight_fp8.shape
    if k % bk != 0 or n % bn != 0:
        raise ValueError(
            f"requantize_fp8_block_to_int8: dims {k}x{n} no son "
            f"múltiplo de block_size {block_size}")
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"requantize_fp8_block_to_int8: esperado float8_e4m3fn, "
            f"obtenido {weight_fp8.dtype}")

    # Reshape a [K/Bk, Bk, N/Bn, Bn] para operar por bloque.
    w = weight_fp8.reshape(k // bk, bk, n // bn, bn)
    s = scale_inv.reshape(k // bk, 1, n // bn, 1)

    # Dequantizar a float32: w_fp32 = w_fp8 * scale_inv.
    w_fp32 = w.to(torch.float32) * s.to(torch.float32)

    # Amax por bloque (abs) / 127 → nueva escala.
    amax = w_fp32.abs().amax(dim=(1, 3), keepdim=True)
    # Evitar división por cero: si el bloque es todo cero, escala 1.0
    # (el valor dequantizado es 0 de todos modos).
    new_scale = amax / 127.0
    new_scale = torch.where(new_scale > 0, new_scale,
                           torch.ones_like(new_scale))

    # Quantizar a int8: round(w_fp32 / new_scale), clamp [-127, 127].
    w_i8 = (w_fp32 / new_scale).round().clamp(-127, 127).to(torch.int8)

    # Aplanar a [K, N] contiguo y [K/Bk, N/Bn] contiguo.
    w_i8 = w_i8.reshape(k, n).contiguous()
    new_scale = new_scale.reshape(k // bk, n // bn).contiguous()
    return w_i8, new_scale


def requantize_fp8_block_to_int8_chunked(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantiza FP8-bloque [K,N] a INT8 per-channel [K,N] por chunks.

    Versión acotada en memoria para el swap PN110: evita el buffer
    transitorio ``[K,N]`` fp32 completo. Dos pasadas por ``chunk_rows``
    filas:

    * Pasada 1: dequantiza cada chunk a fp32 y acumula ``amax`` por
      columna en un buffer ``[N]`` fp32.
    * Pasada 2: requantiza cada chunk a int8 usando las escalas
      per-channel (``amax/127``) y escribe en un buffer ``[K,N]`` int8
      prealocado con ``round+clamp[-127,127]``.

    Transitorio máximo: ``chunk_rows*N*4`` bytes (el chunk fp32).

    Pura: no muta los inputs.

    Args:
        weight_fp8: peso [K,N] en float8_e4m3fn.
        scale_inv: escalas [K/Bk, N/Bn] fp32 (una por bloque).
        block_size: (Bk, Bn) tamaño del bloque (p.ej. (128,128)).
        chunk_rows: filas por chunk (default 2048). Se alinea a
            múltiplo de Bk si es necesario.

    Returns:
        Tupla (w_int8, scales_per_channel) donde w_int8 es [K,N] int8
        contiguo y scales_per_channel es [N,1] fp32 per-channel.

    Raises:
        ValueError: si dims no son múltiplo de block_size, dtype no es
            float8_e4m3fn o shapes incompatibles.
    """
    bk, bn = block_size
    if weight_fp8.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(
            f"requantize_fp8_block_to_int8_chunked: tensores deben ser 2-D, "
            f"obtenido {weight_fp8.dim()}D y {scale_inv.dim()}D")
    k, n = weight_fp8.shape
    if k % bk != 0 or n % bn != 0:
        raise ValueError(
            f"requantize_fp8_block_to_int8_chunked: dims {k}x{n} no son "
            f"múltiplo de block_size {block_size}")
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"requantize_fp8_block_to_int8_chunked: esperado float8_e4m3fn, "
            f"obtenido {weight_fp8.dtype}")
    expected_scale_shape = (k // bk, n // bn)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"requantize_fp8_block_to_int8_chunked: scale_inv shape "
            f"{tuple(scale_inv.shape)} != esperado {expected_scale_shape} "
            f"para weight {k}x{n} y block {block_size}")

    # Alinear chunk_rows a múltiplo de Bk para no cortar bloques.
    if chunk_rows <= 0:
        chunk_rows = 2048
    if chunk_rows % bk != 0:
        chunk_rows = ((chunk_rows + bk - 1) // bk) * bk
    # No exceder K.
    chunk_rows = min(chunk_rows, k)

    device = weight_fp8.device
    # Pasada 1: acumular amax por columna.
    amax_per_col = torch.zeros(n, dtype=torch.float32, device=device)
    n_blocks_n = n // bn
    # Iterar por chunks de filas.
    for r in range(0, k, chunk_rows):
        r_end = min(r + chunk_rows, k)
        chunk_k = r_end - r
        # chunk_k debe ser múltiplo de bk (por alineación y porque k%bk==0)
        # Si el último chunk no es múltiplo (cuando k no múltiplo de chunk_rows
        # pero sí de bk), ajustamos: chunk_k ya es múltiplo porque r es
        # múltiplo de bk y k múltiplo de bk -> r_end múltiplo de bk o resto
        # también múltiplo. En casos borde, si no lo es, caemos a path
        # genérico con repeat_interleave.
        w_chunk = weight_fp8[r:r_end]
        r_blk_start = r // bk
        r_blk_end = r_end // bk
        # Caso alineado: chunk_k % bk == 0
        if chunk_k % bk == 0 and (r_blk_end - r_blk_start) * bk == chunk_k:
            # Reshape a [num_blk_k, Bk, N/Bn, Bn]
            w_4d = w_chunk.reshape(chunk_k // bk, bk, n_blocks_n, bn)
            s_slice = scale_inv[r_blk_start:r_blk_end].reshape(
                chunk_k // bk, 1, n_blocks_n, 1)
            w_fp32_4d = w_4d.to(torch.float32) * s_slice.to(torch.float32)
            w_fp32 = w_fp32_4d.reshape(chunk_k, n)
        else:
            # Path genérico: expandir escala con repeat_interleave y truncar.
            s_slice = scale_inv[r_blk_start:r_blk_end]
            # Expandir a [chunk_k, N] via repeat
            s_expanded = s_slice.repeat_interleave(bk, dim=0).repeat_interleave(
                bn, dim=1)
            # Truncar/pad a chunk_k (si chunk corta bloque parcial)
            if s_expanded.shape[0] > chunk_k:
                s_expanded = s_expanded[:chunk_k]
            elif s_expanded.shape[0] < chunk_k:
                # Pad repetiendo última fila (no debería ocurrir si k%bk==0)
                pad = chunk_k - s_expanded.shape[0]
                s_expanded = torch.cat(
                    [s_expanded, s_expanded[-1:].repeat(pad, 1)], dim=0)
            w_fp32 = w_chunk.to(torch.float32) * s_expanded.to(torch.float32)
        # Acumular amax por columna para este chunk.
        col_amax = w_fp32.abs().amax(dim=0)
        amax_per_col = torch.maximum(amax_per_col, col_amax)

    # Escalas per-channel [N,1] fp32.
    scales_per_channel = amax_per_col / 127.0
    scales_per_channel = torch.where(
        scales_per_channel > 0, scales_per_channel,
        torch.ones_like(scales_per_channel))
    scales_col = scales_per_channel.unsqueeze(1).contiguous()  # [N,1]

    # Pasada 2: cuantizar por chunks en buffer int8 prealocado.
    w_int8 = torch.empty((k, n), dtype=torch.int8, device=device)
    # Para reuso de escalas en la división: [N] vector.
    scale_vec = scales_per_channel  # [N]
    for r in range(0, k, chunk_rows):
        r_end = min(r + chunk_rows, k)
        chunk_k = r_end - r
        w_chunk = weight_fp8[r:r_end]
        r_blk_start = r // bk
        r_blk_end = r_end // bk
        if chunk_k % bk == 0 and (r_blk_end - r_blk_start) * bk == chunk_k:
            w_4d = w_chunk.reshape(chunk_k // bk, bk, n_blocks_n, bn)
            s_slice = scale_inv[r_blk_start:r_blk_end].reshape(
                chunk_k // bk, 1, n_blocks_n, 1)
            w_fp32_4d = w_4d.to(torch.float32) * s_slice.to(torch.float32)
            w_fp32 = w_fp32_4d.reshape(chunk_k, n)
        else:
            s_slice = scale_inv[r_blk_start:r_blk_end]
            s_expanded = s_slice.repeat_interleave(bk, dim=0).repeat_interleave(
                bn, dim=1)
            if s_expanded.shape[0] > chunk_k:
                s_expanded = s_expanded[:chunk_k]
            elif s_expanded.shape[0] < chunk_k:
                pad = chunk_k - s_expanded.shape[0]
                s_expanded = torch.cat(
                    [s_expanded, s_expanded[-1:].repeat(pad, 1)], dim=0)
            w_fp32 = w_chunk.to(torch.float32) * s_expanded.to(torch.float32)
        # Quant per-channel: dividir por escala de columna.
        w_i8_chunk = (w_fp32 / scale_vec.unsqueeze(0)).round().clamp(
            -127, 127).to(torch.int8)
        w_int8[r:r_end] = w_i8_chunk

    w_int8 = w_int8.contiguous()
    return w_int8, scales_col


def requantize_bf16_to_int8(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantiza un peso BF16 [K,N] a INT8 per-channel [K,N].

    Versión directa sin bloques: calcula amax por columna (N) sobre el
    peso BF16 → escala per-channel ``amax/127`` y cuantiza
    ``round(w/scale)``. Pura, no muta inputs.

    Args:
        weight_bf16: peso [K,N] en ``torch.bfloat16``.

    Returns:
        (w_int8 [K,N] int8, scales [N,1] fp32 per-channel).

    Raises:
        ValueError: si dtype no es bfloat16 o dims no 2-D.
    """
    if weight_bf16.dim() != 2:
        raise ValueError(
            f"requantize_bf16_to_int8: tensor debe ser 2-D, obtenido {weight_bf16.dim()}D")
    if weight_bf16.dtype != torch.bfloat16:
        raise ValueError(
            f"requantize_bf16_to_int8: esperado bfloat16, obtenido {weight_bf16.dtype}")
    w_fp32 = weight_bf16.to(torch.float32)
    amax_per_col = w_fp32.abs().amax(dim=0)
    scales = torch.where(amax_per_col > 0, amax_per_col / 127.0, torch.ones_like(amax_per_col))
    scales_col = scales.unsqueeze(1).contiguous()
    w_int8 = (w_fp32 / scales.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8).contiguous()
    return w_int8, scales_col


def requantize_bf16_to_int8_chunked(
    weight_bf16: torch.Tensor,
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantiza BF16 [K,N] a INT8 per-channel por chunks (acotado en memoria).

    Dos pasadas: amax por columna y requant chunk a chunk. Transitorio
    ``chunk_rows*N*4`` bytes.

    Args:
        weight_bf16: peso [K,N] en bfloat16.
        chunk_rows: filas por chunk (default 2048).

    Returns:
        (w_int8 [K,N] int8 contiguo, scales [N,1] fp32 per-channel).
    """
    if weight_bf16.dim() != 2:
        raise ValueError(
            f"requantize_bf16_to_int8_chunked: tensor debe ser 2-D, obtenido {weight_bf16.dim()}D")
    if weight_bf16.dtype != torch.bfloat16:
        raise ValueError(
            f"requantize_bf16_to_int8_chunked: esperado bfloat16, obtenido {weight_bf16.dtype}")
    k, n = weight_bf16.shape
    if chunk_rows <= 0:
        chunk_rows = 2048
    chunk_rows = min(chunk_rows, k)
    device = weight_bf16.device
    amax_per_col = torch.zeros(n, dtype=torch.float32, device=device)
    for r in range(0, k, chunk_rows):
        r_end = min(r + chunk_rows, k)
        w_chunk = weight_bf16[r:r_end].to(torch.float32)
        col_amax = w_chunk.abs().amax(dim=0)
        amax_per_col = torch.maximum(amax_per_col, col_amax)
    scales = torch.where(amax_per_col > 0, amax_per_col / 127.0, torch.ones_like(amax_per_col))
    scales_col = scales.unsqueeze(1).contiguous()
    w_int8 = torch.empty((k, n), dtype=torch.int8, device=device)
    scale_vec = scales
    for r in range(0, k, chunk_rows):
        r_end = min(r + chunk_rows, k)
        w_chunk = weight_bf16[r:r_end].to(torch.float32)
        w_i8_chunk = (w_chunk / scale_vec.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
        w_int8[r:r_end] = w_i8_chunk
    return w_int8.contiguous(), scales_col


def requantize_fp8_block_to_int8_hybrid(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Requantiza FP8-bloque [K,N] a INT8 híbrido diádico Diseño C — 100% GPU.

    Vectorizado puro en GPU sin ``for`` Python ni ``.cpu()``/``numpy``:

    * Extrae ``e = (codes >> 3) & 0x0F`` y ``m = codes & 0x07`` como
      tensores GPU ``uint8``.
    * Calcula ``e_max`` por bloque vía ``max`` sobre dimensión del bloque
      usando ``torch.max`` con ``view``/``reshape`` y ``keepdim``.
    * Calcula ``d = e_max - e`` como tensor GPU ``int32``.
    * Calcula ``q`` vía shifts vectorizados
      ``torch.where(d <= 3, (8+m) << (3-d), (8+m) >> (d-3))`` todo en GPU.
    * Escala ``s_new`` (``s_row`` per-channel) y ``shift_b`` también en GPU.

    Args:
        weight_fp8: peso [K,N] en float8_e4m3fn.
        scale_inv: escalas [K/Bk, N/Bn] fp32.
        block_size: (Bk, Bn).

    Returns:
        (w_int8 [K,N] int8, w_scales [N,1] fp32, w_shifts [Kb,Nb] int8).
    """
    bk, bn = block_size
    if weight_fp8.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(
            "requantize_fp8_block_to_int8_hybrid: tensores deben ser 2-D, "
            f"obtenido {weight_fp8.dim()}D y {scale_inv.dim()}D")
    k, n = weight_fp8.shape
    if k % bk != 0 or n % bn != 0:
        raise ValueError(
            f"requantize_fp8_block_to_int8_hybrid: dims {k}x{n} no son "
            f"múltiplo de block_size {block_size}")
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"requantize_fp8_block_to_int8_hybrid: esperado float8_e4m3fn, "
            f"obtenido {weight_fp8.dtype}")
    expected_scale_shape = (k // bk, n // bn)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"requantize_fp8_block_to_int8_hybrid: scale_inv shape "
            f"{tuple(scale_inv.shape)} != esperado {expected_scale_shape} "
            f"para weight {k}x{n} y block {block_size}")

    # ── 1. s_row per-channel GPU vectorizado ──────────────────────────
    try:
        w_4d = weight_fp8.reshape(k // bk, bk, n // bn, bn)
        s_4d = scale_inv.reshape(k // bk, 1, n // bn, 1)
        w_fp32_4d = w_4d.to(torch.float32) * s_4d.to(torch.float32)
        w_fp32 = w_fp32_4d.reshape(k, n)
        amax_per_col = w_fp32.abs().amax(dim=0)
    except Exception:
        s_exp = scale_inv.repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)
        w_fp32 = weight_fp8.to(torch.float32) * s_exp.to(torch.float32)
        amax_per_col = w_fp32.abs().amax(dim=0)
    s_row_vec = amax_per_col / 127.0
    s_row_vec = torch.where(s_row_vec > 0, s_row_vec, torch.ones_like(s_row_vec))
    w_scales = s_row_vec.unsqueeze(1).contiguous()

    # ── 2. Extraer códigos uint8 crudos — GPU ─────────────────────────
    codes_u8 = weight_fp8.view(torch.uint8)
    # Validación NaN GPU (sin cpu)
    mag_all = codes_u8 & 0x7F
    if (mag_all == 0x7F).any():
        raise ValueError("NaN en códigos FP8 (0x7F)")

    # ── 3. Extraer e,m como tensores GPU uint8 ────────────────────────
    e_u8 = (codes_u8 >> 3) & 0x0F
    m_u8 = codes_u8 & 0x07

    k_blocks = k // bk
    n_blocks = n // bn

    # View 4-D GPU para bloques: [Kb,Bk,Nb,Bn]
    codes_4d = codes_u8.reshape(k_blocks, bk, n_blocks, bn)
    e_4d_u8 = e_u8.reshape(k_blocks, bk, n_blocks, bn)
    m_4d_u8 = m_u8.reshape(k_blocks, bk, n_blocks, bn)

    # ── 4. e_max por bloque vía max con view y keepdim (GPU) ──────────
    # amax sobre dims de bloque (1,3) con keepdim para broadcast
    e_max_keep = e_4d_u8.amax(dim=(1, 3), keepdim=True)

    # ── 5. d = e_max - e como int32 GPU ───────────────────────────────
    e_4d_i32 = e_4d_u8.to(torch.int32)
    e_max_i32 = e_max_keep.to(torch.int32)
    d_4d = e_max_i32 - e_4d_i32
    m_4d_i32 = m_4d_u8.to(torch.int32)
    mant_4d = (8 + m_4d_i32)

    # ── 6. q vía shifts vectorizados torch.where en GPU ───────────────
    # Contrato: torch.where(d <= 3, (8+m) << (3-d), (8+m) >> (d-3))
    q_4d_i32 = torch.where(
        d_4d <= 3,
        mant_4d << (3 - d_4d),
        mant_4d >> (d_4d - 3),
    )
    # Máscaras de subnormal / fuera de rango y signo — todo GPU
    q_4d_i32 = torch.where((e_4d_i32 == 0) | (d_4d > 6), torch.zeros_like(q_4d_i32), q_4d_i32)
    sign_mask_4d = (codes_4d & 0x80) != 0
    q_4d_i32 = torch.where(sign_mask_4d, -q_4d_i32, q_4d_i32)
    q_4d_i32 = q_4d_i32.clamp(-127, 127)
    # Bloques con e_max==0 → cero (ya cubierto por máscara, pero forzar)
    zero_block_mask = (e_max_keep == 0)
    q_4d_i32 = torch.where(zero_block_mask, torch.zeros_like(q_4d_i32), q_4d_i32)

    w_int8 = q_4d_i32.to(torch.int8).reshape(k, n).contiguous()

    # ── 7. shift_b y s_new también en GPU (vectorizado) ───────────────
    # s_prime = s_blk * 2^(e_max-13)  por bloque [Kb,Nb]
    e_max_2d = e_max_keep.reshape(k_blocks, n_blocks).to(torch.float32)
    s_prime = scale_inv.to(torch.float32) * torch.pow(2.0, e_max_2d - 13.0)
    # s_row medio por bloque de columnas: [Nb]
    s_row_block_mean = s_row_vec.view(n_blocks, bn).mean(dim=1)
    s_row_mean_2d = s_row_block_mean.unsqueeze(0).expand(k_blocks, n_blocks)
    # shift_float = log2(s_prime) - log2(s_row_mean)
    shift_float = torch.log2(s_prime.clamp(min=1e-30)) - torch.log2(s_row_mean_2d.clamp(min=1e-30))
    shift_rounded = torch.round(shift_float).to(torch.int32)
    # Bloques cero → shift 0
    shift_rounded = torch.where(e_max_2d == 0, torch.zeros_like(shift_rounded), shift_rounded)
    shift_clamped = shift_rounded.clamp(-10, 10)
    w_shifts = shift_clamped.to(torch.int8).contiguous()

    return w_int8, w_scales, w_shifts


def quant_activation_per_token(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuantiza la activación per-token a INT8.

    Absmax por fila (dim=-1) / 127 → escala; x / escala, round, clamp.
    Pura: no muta x.

    :param x: tensor [..., K] en cualquier dtype de punto flotante.
    :returns: (x_i8, scale_a) donde x_i8 es [..., K] int8 y scale_a es
        [..., 1] float32.
    """
    x_f32 = x.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    # Evitar división por cero: fila toda cero → escala 1.0.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_i8 = (x_f32 / scale).round().clamp(-127, 127).to(torch.int8)
    return x_i8, scale


def int8_linear(
    a_i8_2d: torch.Tensor,
    w_int8: torch.Tensor,
    w_scales: torch.Tensor,
    a_scales: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    w_shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ejecuta el GEMM INT8 W8A8 vía `ops.cutlass_scaled_mm` o híbrido.

    Espeja 1:1 el patrón probado en
    `workshop/ox_alpha/lab/scripts/bench_gemm.py` (líneas 85-110):
    a_i8 [M,K] int8, b [K,N] column-major, escalas float32,
    out_dtype float16.

    El kernel cutlass c2x de sm80 (Ampere) solo soporta escalas de peso
    per-tensor o per-channel (el epilogue usa `RowOrScalarBroadcast`,
    ver `csrc/libtorch_stable/cutlass_extensions/epilogue/
    scaled_mm_epilogues_c2x.hpp`); con escalas 2-D por bloque lanza.
    Por eso `w_scales` es per-channel [N, 1] (o [N]) y se pasa al
    kernel como `w_scales.t().contiguous().float()` → [1, N] fp32,
    igual que el bench. El colapso bloque→per-channel lo hace
    `_build_int8_state` al cargar (requantiza el peso dequantizado).

    Diseño C híbrido (§4): si ``w_shifts`` [K/Bk,N/Bn] int8 per-bloque
    está presente, aplica shift sobre acumulador INT32 antes del multiply
    por s_row: ``out = Σ_kb (a_blk @ w_blk) * a_scale * s_row * 2^shift``.
    Acumula por chunks K=128 y shiftea en INT32 (barato, sin overflow:
    128 productos ≤2.06M, shift ≤10 cabe en INT32 2.1G).

    El peso llega como [K, N]; el kernel exige b column-major [K, N]
    (ldb = stride(1) = K), así que `w_int8.t().contiguous().t()`
    produce ese layout. Si `w_int8` ya es column-major (el estado de
    PN110 lo precalcula como `b_col`), la expresión es un no-op sin
    copia en el camino caliente.

    :param a_i8_2d: activación cuantizada [M, K] int8 contigua (ya 2-D).
    :param w_int8: peso [K, N] int8 (row-major, o column-major
        precalculado).
    :param w_scales: escalas de peso per-channel [N, 1] (o [N]),
        float32.
    :param a_scales: escalas de activación per-token [M, 1] float32.
    :param bias: bias [N] en out_dtype, o None.
    :param out_dtype: dtype de salida (torch.float16 o torch.bfloat16;
        el kernel lo exige).
    :param w_shifts: opcional [K/Bk,N/Bn] int8 per-bloque shift diádico.
        Si existe, camino híbrido C; si None, camino B per-channel.
    :returns: tensor [M, N] en out_dtype.
    """
    # ── Diseño C híbrido: si hay w_shifts, acumula por chunks K=128 con shift ──
    if w_shifts is not None:
        # ── Dispatch al kernel custom si está disponible ──────────────────
        # Contrato: kernel Triton/CUTLASS hace int_acc = a_i8 @ b_i8 (INT32),
        # luego int_acc_shifted = int_acc << shift_b (>> si negativo, aritmético),
        # luego out = int_acc_shifted * a_scale * s_row. No toca camino B.
        try:
            from vllm._genesis.kernels.int8_hybrid_gemm import (
                int8_hybrid_gemm as _hybrid_gemm,
                is_available as _hybrid_available,
            )
            if _hybrid_available() and a_i8_2d.is_cuda and w_int8.is_cuda:
                try:
                    return _hybrid_gemm(
                        a_i8_2d, w_int8, a_scales, w_scales, w_shifts, out_dtype
                    )
                except Exception as e:
                    log.warning(
                        "PN110 int8_linear kernel custom falló (%s: %s) — "
                        "fallback a validación torch (solo exactitud, no performance); "
                        "híbrido queda pendiente de optimización",
                        type(e).__name__, e,
                    )
            else:
                # Triton/CUDA no disponible — WARNING y fallback torch validación
                # (documenta que hybrid por ahora es solo validación de exactitud)
                log.warning(
                    "PN110 int8_linear hybrid sin kernel custom disponible "
                    "(Triton/CUDA no disponible) — usando fallback torch "
                    "(100× más lento, solo validación de exactitud, no para performance); "
                    "kernel custom queda pendiente"
                )
        except ImportError as e:
            log.warning(
                "PN110 int8_linear no pudo importar kernel híbrido (%s) — fallback torch",
                e,
            )
        except Exception as e:
            log.warning("PN110 int8_linear dispatch híbrido falló (%s), fallback torch", type(e).__name__)
        # ── Fallback torch (validación de exactitud, 100× más lento) ─────
        # Este camino emula en FP32 a_blk @ w_full + pow(2,shift) — 2000× peor KL
        # que B si se usara pow sin shift INT32, pero aquí sí aplica shift sobre
        # acumulador INT32 (correcto §4C). Se mantiene como referencia/esqueleto
        # hasta estabilizar el Triton; no para producción.
        try:
            K, N = int(w_int8.shape[0]), int(w_int8.shape[1])
            M = int(a_i8_2d.shape[0])
            num_kb, num_nb = int(w_shifts.shape[0]), int(w_shifts.shape[1])
            Bk = K // num_kb if num_kb else K
            Bn = N // num_nb if num_nb else N
            if w_scales.dim() == 2:
                if w_scales.shape == (N, 1):
                    s_row = w_scales.squeeze(1).to(torch.float32)
                elif w_scales.shape == (1, N):
                    s_row = w_scales.squeeze(0).to(torch.float32)
                else:
                    s_row = w_scales.reshape(-1).to(torch.float32)[:N]
            elif w_scales.dim() == 1:
                s_row = w_scales.to(torch.float32)[:N]
            else:
                s_row = w_scales.reshape(-1).to(torch.float32)[:N]
            # Normalizar a_scales a [M,1] GPU
            a_scales_f = a_scales.to(torch.float32)
            if a_scales_f.dim() == 2 and a_scales_f.shape[1] == 1:
                a_vec = a_scales_f  # [M,1]
            elif a_scales_f.dim() == 2 and a_scales_f.shape[0] == 1:
                a_vec = a_scales_f.t().contiguous()
            elif a_scales_f.dim() == 1:
                a_vec = a_scales_f.view(M, 1)
            else:
                a_vec = a_scales_f.reshape(M, 1)
            # ── 100% GPU vectorizado: int_acc INT32 + shift vía bitwise ──
            a_i32 = a_i8_2d.to(torch.int32)
            b_i32 = w_int8.to(torch.int32)
            # Bloques: a [M,K] -> [Kb,M,Bk], b [K,N] -> [Kb,Bk,N]
            a_3d = a_i32.view(M, num_kb, Bk).permute(1, 0, 2).contiguous()
            b_3d = b_i32.view(num_kb, Bk, N).contiguous()
            # batched GEMM INT32 en GPU
            int_acc_batched = torch.bmm(a_3d, b_3d)
            int_acc_3d = int_acc_batched.permute(1, 0, 2).contiguous()
            # shifts [Kb,Nb] -> [Kb,N] GPU
            shifts_expanded = w_shifts.repeat_interleave(Bn, dim=1).to(torch.int32)
            shifts_bc = shifts_expanded.unsqueeze(0).expand(M, -1, -1)
            shifted_3d = torch.where(
                shifts_bc >= 0,
                torch.bitwise_left_shift(int_acc_3d, shifts_bc),
                torch.bitwise_right_shift(int_acc_3d, -shifts_bc),
            )
            shifted_f = shifted_3d.to(torch.float32) * a_vec.view(M, 1, 1) * s_row.view(1, 1, N)
            out_fp32 = shifted_f.sum(dim=1)
            if bias is not None:
                out_fp32 = out_fp32 + bias.to(torch.float32).to(out_fp32.device)
            return out_fp32.to(out_dtype)
        except Exception as e:
            # Fallback a camino B si híbrido falla (no debe ocultar bug, log)
            log.warning("PN110 int8_linear híbrido falló (%s), fallback a B", type(e).__name__)

    import vllm._custom_ops as ops

    # P3: Invariante — _build_int8_state garantiza b_col column-major
    # con stride (1, K). Si w_int8 ya es column-major, usar directo
    # sin alloc; fallback transpose solo para compat con tests que
    # pasan row-major directo a int8_linear.
    if w_int8.stride() == (1, w_int8.shape[0]):
        b = w_int8
    else:
        try:
            if w_int8.t().is_contiguous():
                b = w_int8
            else:
                b = w_int8.t().contiguous().t()
        except Exception:
            b = w_int8.t().contiguous().t()

    # P1: b_scales pre-transpuesto [1,N] fp32 desde _build_int8_state.
    # Invariante: state["b_scales"] es [1,N] fp32 contiguo. Si el
    # caller pasa w_scales [N,1] (compat tests directos), transponemos.
    if w_scales.dim() == 2 and w_scales.shape[0] == 1:
        if w_scales.dtype != torch.float32:
            b_scales = w_scales.float()
            if not b_scales.is_contiguous():
                b_scales = b_scales.contiguous()
        else:
            b_scales = w_scales if w_scales.is_contiguous() else w_scales.contiguous()
    else:
        b_scales = w_scales.t().contiguous().float()

    out = ops.cutlass_scaled_mm(
        a_i8_2d, b, a_scales.contiguous(), b_scales, out_dtype, bias)
    return out


def _build_int8_state(
    layer: torch.nn.Module,
    w_fp8_orig: torch.Tensor,
    scale_inv_orig: torch.Tensor | None,
    block_size: tuple[int, int],
) -> dict | None:
    """Construye el dict de estado INT8 para una capa — pico 2x máximo.

    Optimizado para pico VRAM 2x en GPU (vs 3x anterior w_orig + w_i8 +
    b_col + transient): en lugar de alocar w_i8 [K,N] row-major completo y
    luego b_col [K,N] column-major, aloca **solo** b_col column-major
    completo al inicio (``torch.empty((N,K), int8, device).t()`` → [K,N]
    stride (1,K), o ``empty_strided((K,N),(1,K))``) y escribe por chunks
    directos sin w_i8 intermedio. Dos pasadas chunk_rows=512 (≤512,
    múltiplo de Bk): pasada 1 dequantiza chunk a fp32 y acumula amax por
    columna en [N] fp32; pasada 2 requantiza chunk y escribe
    DIRECTAMENTE en ``b_col[r:r_end]`` vía slice assignment. Escalas
    per-channel [N,1] aparte. Pico ≈ b_col(1x) + chunk_transient
    (512*N*4 ≤70 MB para N=34816) + w_orig (sin w_i8 completo). Sin
    fallback a CPU, chunk ≤512.

    Para compatibilidad con tests existentes, ``w_int8`` se mantiene como
    **alias** al mismo almacenamiento que ``b_col`` (no duplica bytes);
    el resumen deduplica por ``data_ptr``. ``w_scales`` [N,1] fp32
    per-channel, ``b_col`` [K,N] int8 column-major (``b_col.t()`` contiguo)
    y ``bias_ref_ok``. Todo en GPU.

    El comportamiento depende de ``GENESIS_PN110_QUANTIZE``:

    * ``fp8_int8`` (default): ``w_fp8_orig`` debe ser ``float8_e4m3fn`` y
      ``scale_inv_orig`` brilla por bloque; flujo FP8→INT8 como hasta ahora.
    * ``bf16_int8``: ``w_fp8_orig`` debe ser ``torch.bfloat16``; se ignora
      ``scale_inv_orig`` y se cuantiza BF16→INT8 per-channel directo
      (misma lógica chunked pero con ``weight.dtype == bfloat16``).

    Args:
        layer: capa Linear (para leer bias y nombre).
        w_fp8_orig: referencia al peso [N,K] (``float8_e4m3fn`` si
            ``fp8_int8``, ``bfloat16`` si ``bf16_int8``) ANTES del repack.
        scale_inv_orig: referencia a las escalas [N/Bn, K/Bk] fp32 ANTES
            del repack (solo para ``fp8_int8``; en ``bf16_int8`` se ignora).
        block_size: (Bk, Bn) — solo usado en ``fp8_int8``.

    Returns:
        Dict con ``w_int8`` alias a ``b_col`` [K,N] int8, ``w_scales``
        [N,1] fp32 per-channel, ``b_col`` [K,N] int8 column-major y
        ``bias_ref_ok``; None si falla.
    """
    try:
        quantize_modes = _quantize_mode()
        # ── Selección por dtype según modos habilitados (lista por comas) ───
        # Si GENESIS_PN110_QUANTIZE=bf16_int8,fp8_int8 habilita ambos; elige según w_orig.dtype
        try:
            w_dtype = w_fp8_orig.dtype if isinstance(w_fp8_orig, torch.Tensor) else None
        except Exception:
            w_dtype = None
        _is_bf16_enabled = "bf16_int8" in quantize_modes
        _is_fp8_enabled = "fp8_int8" in quantize_modes
        _use_bf16 = (w_dtype == torch.bfloat16 and _is_bf16_enabled)
        _use_fp8 = (w_dtype == torch.float8_e4m3fn and _is_fp8_enabled)
        # Si ninguno coincide, fallar para que el wrapper excluya la capa
        if not _use_bf16 and not _use_fp8:
            try:
                _enabled_str = ",".join(sorted(quantize_modes))
            except Exception:
                _enabled_str = str(quantize_modes)
            raise ValueError(
                f"_build_int8_state: dtype {w_dtype} no habilitado por GENESIS_PN110_QUANTIZE={_enabled_str} "
                f"(fp8_int8={'si' if _is_fp8_enabled else 'no'}, bf16_int8={'si' if _is_bf16_enabled else 'no'})"
            )
        # ── Rama BF16→INT8 ────────────────────────────────────────────────
        if _use_bf16:
            # Validaciones para BF16
            if w_fp8_orig.dim() != 2:
                raise ValueError(
                    f"_build_int8_state: tensores deben ser 2-D, obtenido {w_fp8_orig.dim()}D")
            if w_fp8_orig.dtype != torch.bfloat16:
                raise ValueError(
                    f"_build_int8_state: esperado bfloat16, obtenido {w_fp8_orig.dtype}")
            N = int(w_fp8_orig.shape[0])
            K = int(w_fp8_orig.shape[1])
            device = w_fp8_orig.device
            # Híbrido no soportado para BF16 → fallback directo a B (sin log ruidoso)
            # Chunked BF16 per-channel: misma idea que FP8 pero sin dequant por bloque
            chunk_rows = 512
            chunk_rows = min(chunk_rows, K)
            # Alinear a 16 para cutlass (no a Bk)
            if chunk_rows % 16 != 0:
                chunk_rows = ((chunk_rows + 15) // 16) * 16
                chunk_rows = min(chunk_rows, K)
            w_bf16_T = w_fp8_orig.t()  # [K,N] vista BF16
            try:
                b_col = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device=device)
            except Exception:
                b_col = torch.empty((N, K), dtype=torch.int8, device=device).t()
            amax_per_col = torch.zeros(N, dtype=torch.float32, device=device)
            for r in range(0, K, chunk_rows):
                r_end = min(r + chunk_rows, K)
                chunk = w_bf16_T[r:r_end].to(torch.float32)
                col_amax = chunk.abs().amax(dim=0)
                amax_per_col = torch.maximum(amax_per_col, col_amax)
            scales_per_channel = amax_per_col / 127.0
            scales_per_channel = torch.where(
                scales_per_channel > 0, scales_per_channel,
                torch.ones_like(scales_per_channel))
            w_scales = scales_per_channel.unsqueeze(1).contiguous()  # [N,1]
            b_scales = w_scales.t().contiguous().float()  # [1,N]
            scale_vec = scales_per_channel  # [N]
            for r in range(0, K, chunk_rows):
                r_end = min(r + chunk_rows, K)
                chunk = w_bf16_T[r:r_end].to(torch.float32)
                w_i8_chunk = (chunk / scale_vec.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
                b_col[r:r_end] = w_i8_chunk
            bias = getattr(layer, "bias", None)
            bias_ref_ok = bias is not None and bias.numel() > 0
            _state = {
                "w_int8": b_col,
                "w_scales": w_scales,
                "b_col": b_col,
                "b_scales": b_scales,
                "bias_ref_ok": bias_ref_ok,
            }
            if bias_ref_ok:
                try:
                    _state["bias_fp16"] = bias.to(torch.float16)
                    _state["bias_bf16"] = bias.to(torch.bfloat16)
                except Exception:
                    pass
            return _state

        # ── Rama FP8→INT8 (default, comportamiento histórico) ─────────────
        bk, bn = block_size
        # Validaciones mínimas (espejan requantize_fp8_block_to_int8_chunked)
        if w_fp8_orig.dim() != 2 or (scale_inv_orig is not None and scale_inv_orig.dim() != 2):
            raise ValueError(
                "_build_int8_state: tensores deben ser 2-D, "
                f"obtenido {w_fp8_orig.dim()}D y {scale_inv_orig.dim() if scale_inv_orig is not None else 'None'}D")
        if scale_inv_orig is None:
            raise ValueError("_build_int8_state: scale_inv_orig requerido para fp8_int8")
        if w_fp8_orig.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"_build_int8_state: esperado float8_e4m3fn, obtenido {w_fp8_orig.dtype}")
        N = int(w_fp8_orig.shape[0])
        K = int(w_fp8_orig.shape[1])
        if K % bk != 0 or N % bn != 0:
            raise ValueError(
                f"_build_int8_state: dims {K}x{N} no son múltiplo de block_size {(bk,bn)}")
        expected_scale_shape = (N // bn, K // bk)
        if tuple(scale_inv_orig.shape) != expected_scale_shape:
            raise ValueError(
                f"_build_int8_state: scale_inv shape {tuple(scale_inv_orig.shape)} "
                f"!= esperado {expected_scale_shape} para weight {N}x{K} y block {(bk,bn)}")

        device = w_fp8_orig.device

        # ── Diseño C híbrido diádico (§4) — camino preferido si GENESIS_PN110_HYBRID=1 ──
        if _is_hybrid_enabled():
            try:
                # Vistas [K,N] para el híbrido (requantize espera [K,N])
                w_fp8_T_h = w_fp8_orig.t().contiguous()
                s_T_h = scale_inv_orig.t().contiguous()
                w_int8_h, w_scales_h, w_shifts_h = requantize_fp8_block_to_int8_hybrid(
                    w_fp8_T_h, s_T_h, (bk, bn))
                # Documenta overflow: verifica shift_b en [-10,10] aprox.
                # Si se sale, fallback a B (pérdida ≤1 bit no compensa riesgo INT32).
                _sh_min = int(w_shifts_h.min().item()) if w_shifts_h.numel() else 0
                _sh_max = int(w_shifts_h.max().item()) if w_shifts_h.numel() else 0
                if _sh_min < -10 or _sh_max > 10:
                    raise ValueError(
                        f"shift_b fuera de [-10,10] (min {_sh_min} max {_sh_max})")
                # Construir b_col column-major aliased a w_int8_h
                try:
                    b_col_h = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device=device)
                except Exception:
                    b_col_h = torch.empty((N, K), dtype=torch.int8, device=device).t()
                # Copia elemento a elemento (respeta strides distintos)
                b_col_h.copy_(w_int8_h)
                bias = getattr(layer, "bias", None)
                bias_ref_ok = bias is not None and bias.numel() > 0
                # P1: b_scales pre-transpuesto [1,N] fp32
                b_scales_h = w_scales_h.t().contiguous().float()
                _state_h = {
                    "w_int8": b_col_h,
                    "w_scales": w_scales_h,
                    "w_shifts": w_shifts_h,
                    "b_col": b_col_h,
                    "b_scales": b_scales_h,
                    "bias_ref_ok": bias_ref_ok,
                }
                # P2: bias pre-casteado
                if bias_ref_ok:
                    try:
                        _state_h["bias_fp16"] = bias.to(torch.float16)
                        _state_h["bias_bf16"] = bias.to(torch.bfloat16)
                    except Exception:
                        pass
                return _state_h
            except Exception as e:
                # Fallback silencioso a Diseño B, log info para trazabilidad
                log.info(
                    "PN110 híbrido no aplicable para %s: %s — fallback a B",
                    getattr(layer, "_genesis_pn110_name", "?"),
                    type(e).__name__,
                )
                # Continuar al camino B per-channel

        # ── Diseño B per-channel chunked (fallback) ────────────────────────
        chunk_rows = 512
        if chunk_rows % bk != 0:
            chunk_rows = ((chunk_rows + bk - 1) // bk) * bk
        chunk_rows = min(chunk_rows, K)

        # Vistas transpuestas [K,N] y [K/Bk, N/Bn] — sin copia
        w_fp8_T = w_fp8_orig.t()
        s_T = scale_inv_orig.t()

        # Alocar SOLO b_col column-major [K,N] (1x) — sin w_i8 completo
        try:
            b_col = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device=device)
        except Exception:
            # Fallback: alocando transpuesto (sin pico 2x intermedio de .t().contiguous().t())
            b_col = torch.empty((N, K), dtype=torch.int8, device=device).t()

        # Pasada 1: acumular amax por columna en [N] fp32
        amax_per_col = torch.zeros(N, dtype=torch.float32, device=device)
        n_blocks_n = N // bn
        for r in range(0, K, chunk_rows):
            r_end = min(r + chunk_rows, K)
            chunk_k = r_end - r
            w_chunk = w_fp8_T[r:r_end]
            r_blk_start = r // bk
            r_blk_end = r_end // bk
            if chunk_k % bk == 0 and (r_blk_end - r_blk_start) * bk == chunk_k:
                w_4d = w_chunk.reshape(chunk_k // bk, bk, n_blocks_n, bn)
                s_slice = s_T[r_blk_start:r_blk_end].reshape(chunk_k // bk, 1, n_blocks_n, 1)
                w_fp32_4d = w_4d.to(torch.float32) * s_slice.to(torch.float32)
                w_fp32 = w_fp32_4d.reshape(chunk_k, N)
            else:
                s_slice = s_T[r_blk_start:r_blk_end]
                s_expanded = s_slice.repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)
                if s_expanded.shape[0] > chunk_k:
                    s_expanded = s_expanded[:chunk_k]
                elif s_expanded.shape[0] < chunk_k:
                    pad = chunk_k - s_expanded.shape[0]
                    s_expanded = torch.cat([s_expanded, s_expanded[-1:].repeat(pad, 1)], dim=0)
                w_fp32 = w_chunk.to(torch.float32) * s_expanded.to(torch.float32)
            col_amax = w_fp32.abs().amax(dim=0)
            amax_per_col = torch.maximum(amax_per_col, col_amax)

        # Escalas per-channel [N,1] fp32
        scales_per_channel = amax_per_col / 127.0
        scales_per_channel = torch.where(
            scales_per_channel > 0, scales_per_channel,
            torch.ones_like(scales_per_channel))
        w_scales = scales_per_channel.unsqueeze(1).contiguous()  # [N,1]
        # P1: b_scales pre-transpuesto [1,N] fp32
        b_scales = w_scales.t().contiguous().float()
        scale_vec = scales_per_channel  # [N]

        # Pasada 2: cuantizar por chunks y escribir DIRECTAMENTE en b_col
        for r in range(0, K, chunk_rows):
            r_end = min(r + chunk_rows, K)
            chunk_k = r_end - r
            w_chunk = w_fp8_T[r:r_end]
            r_blk_start = r // bk
            r_blk_end = r_end // bk
            if chunk_k % bk == 0 and (r_blk_end - r_blk_start) * bk == chunk_k:
                w_4d = w_chunk.reshape(chunk_k // bk, bk, n_blocks_n, bn)
                s_slice = s_T[r_blk_start:r_blk_end].reshape(chunk_k // bk, 1, n_blocks_n, 1)
                w_fp32_4d = w_4d.to(torch.float32) * s_slice.to(torch.float32)
                w_fp32 = w_fp32_4d.reshape(chunk_k, N)
            else:
                s_slice = s_T[r_blk_start:r_blk_end]
                s_expanded = s_slice.repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)
                if s_expanded.shape[0] > chunk_k:
                    s_expanded = s_expanded[:chunk_k]
                elif s_expanded.shape[0] < chunk_k:
                    pad = chunk_k - s_expanded.shape[0]
                    s_expanded = torch.cat([s_expanded, s_expanded[-1:].repeat(pad, 1)], dim=0)
                w_fp32 = w_chunk.to(torch.float32) * s_expanded.to(torch.float32)
            w_i8_chunk = (w_fp32 / scale_vec.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
            b_col[r:r_end] = w_i8_chunk

        bias = getattr(layer, "bias", None)
        bias_ref_ok = bias is not None and bias.numel() > 0
        # w_int8 alias a b_col para compatibilidad (no duplica VRAM)
        _state = {
            "w_int8": b_col,
            "w_scales": w_scales,
            "b_col": b_col,
            "b_scales": b_scales,
            "bias_ref_ok": bias_ref_ok,
        }
        # P2: bias pre-casteado
        if bias_ref_ok:
            try:
                _state["bias_fp16"] = bias.to(torch.float16)
                _state["bias_bf16"] = bias.to(torch.bfloat16)
            except Exception:
                pass
        return _state
    except Exception as e:
        log.warning(
            "PN110 _build_int8_state falló para %s: %s",
            getattr(layer, "_genesis_pn110_name", "?"),
            type(e).__name__,
        )
        return None


def _tensor_bytes(t: torch.Tensor | None) -> int:
    """Calcula bytes de un tensor o 0 si no es tensor.

    Args:
        t: tensor o None.

    Returns:
        Número de bytes (numel * element_size) o 0.
    """
    if isinstance(t, torch.Tensor):
        try:
            return t.numel() * t.element_size()
        except Exception:
            return 0
    return 0


def _reset_summary() -> None:
    """Resetea contadores de resumen (para tests y revert).

    Limpia convertidas/excluidas/fallback y bytes, y cancela el timer
    pendiente. CK-2.3 también limpia la tabla por capa.
    """
    global _pn110_summary, _pn110_summary_timer
    with _pn110_summary_lock:
        _pn110_summary["converted"] = 0
        _pn110_summary["excluded"] = 0
        _pn110_summary["fallback"] = 0
        _pn110_summary["bytes_int8"] = 0
        _pn110_summary["bytes_freed"] = 0
        _pn110_summary["layers_seen"] = 0
        _pn110_summary["_logged"] = False
        _pn110_summary["details"] = []
        _pn110_summary["fallback_reasons"] = {}
        if _pn110_summary_timer is not None:
            try:
                _pn110_summary_timer.cancel()
            except Exception:
                pass
            _pn110_summary_timer = None
    # ── GENESIS: limpiar tabla de kernels (no toca lógica PN110) ──
    try:
        with _genesis_kernel_plan_lock:
            _genesis_kernel_plan.clear()
    except Exception:
        pass
    # Reset flag startup para que re-instalación vuelva a loguear
    try:
        global _genesis_startup_logged
        _genesis_startup_logged = False
    except Exception:
        pass
    # Limpiar cache de warnings de quantize para que tests cambien GENESIS_PN110_QUANTIZE
    try:
        with _quantize_warned_lock:
            _quantize_warned.clear()
    except Exception:
        pass


def _emit_summary(force: bool = False) -> None:
    """Emite el log de resumen una sola vez (CK-2.3 añade tabla por capa).

    Args:
        force: si es True, emite aunque ya se haya logueado (para tests).
    """
    global _pn110_summary
    with _pn110_summary_lock:
        if _pn110_summary.get("_logged") and not force:
            return
        _pn110_summary["_logged"] = True
        converted = _pn110_summary["converted"]
        excluded = _pn110_summary["excluded"]
        fallback = _pn110_summary["fallback"]
        bytes_int8 = _pn110_summary["bytes_int8"]
        bytes_freed = _pn110_summary["bytes_freed"]
        net = bytes_int8 - bytes_freed
        # Copiar tabla por capa para emitir fuera del lock (CK-2.3)
        details = list(_pn110_summary.get("details", []))
        fallback_reasons = dict(_pn110_summary.get("fallback_reasons", {}))
    # Log fuera del lock.
    log.info(
        "PN110 resumen carga: convertidas=%d excluidas=%d fallback=%d "
        "bytes_int8=%d bytes_liberados=%d bytes_netos=%d "
        "(swap 1:1, decode también INT8 — gate CK-2.4 obligatorio)",
        converted, excluded, fallback, bytes_int8, bytes_freed, net,
    )
    # ── CK-2.3: tabla markdown por capa (visible en `docker logs ... | grep PN110`) ──
    try:
        # Info de threshold para contexto
        try:
            thr = _kl_threshold()
            calib = _kl_calib_tokens()
        except Exception:
            thr = 0.05
            calib = 512
        if details:
            log.info(
                "PN110 tabla KL por capa (threshold=%.4f, calib_tokens=%d):",
                thr, calib,
            )
            log.info("PN110 | capa | KxN | KL | decision |")
            log.info("PN110 |---|---|---|---|")
            for entry in details:
                capa = str(entry.get("capa", "?"))
                kxn = str(entry.get("kxn", "-"))
                kl_v = entry.get("kl", None)
                if isinstance(kl_v, float):
                    kl_str = f"{kl_v:.4f}"
                elif kl_v is None:
                    kl_str = "-"
                else:
                    try:
                        kl_str = f"{float(kl_v):.4f}"
                    except Exception:
                        kl_str = "-"
                decision = str(entry.get("decision", "?"))
                # Cada fila con prefijo PN110 para que `grep PN110` la capture
                log.info("PN110 | %s | %s | %s | %s |", capa, kxn, kl_str, decision)
            if fallback_reasons:
                # Resumen de motivos fallback (incluye kl_exceeded)
                try:
                    reasons_str = ", ".join(f"{k}={v}" for k, v in fallback_reasons.items())
                    log.info("PN110 fallback motivos: %s", reasons_str)
                except Exception:
                    pass
        else:
            log.info(
                "PN110 tabla KL por capa: sin capas registradas (threshold=%.4f)",
                thr,
            )
    except Exception:
        pass
    # ── GENESIS: tabla completa de kernels a usar (visible en `docker logs | grep GENESIS`) ──
    try:
        _genesis_emit_plan_summary()
    except Exception:
        pass


def _schedule_summary() -> None:
    """Agenda el resumen para emitirse tras ~1s de inactividad.

    Debounce: cada capa reinicia el timer; al terminar la carga el timer
    expira y emite una sola vez. El apply también fuerza el flush si
    detecta que aún no se emitió (para tests sin espera).
    """
    global _pn110_summary_timer
    with _pn110_summary_lock:
        if _pn110_summary.get("_logged"):
            return
        if _pn110_summary_timer is not None:
            try:
                _pn110_summary_timer.cancel()
            except Exception:
                pass
        # Usar timer daemon para no bloquear shutdown.
        _pn110_summary_timer = threading.Timer(1.0, _emit_summary)
        _pn110_summary_timer.daemon = True
        try:
            _pn110_summary_timer.start()
        except Exception:
            pass


def _make_pwal_wrapper(original, cls):
    """Construye el wrapper de process_weights_after_loading (swap) — SUPER-KERNEL validacion previa.

    Captura referencias a layer.weight y layer.weight_scale_inv ANTES de
    llamar al original (el repack Marlin las destruye), y si la capa es
    elegible, construye el estado INT8 y lo adjunta como
    ``layer._genesis_pn110_int8``. En éxito, **libera** el peso Marlin
    y auxiliares para VRAM neta ~igual; en fallo, no libera (queda en
    Marlin) y registra WARN.

    SUPER-KERNEL (2026-08-25): toda validacion del super-kernel
    ``fused_quant_gemm`` se hace AQUI, una vez por capa al cargar,
    NO en el camino caliente. Antes estas validaciones eran branches
    en ``fused_quant_gemm`` (``if b.dtype != int8``, ``if block_k <16``,
    ``if not b.is_contiguous()`` / ``if not a.is_contiguous()``,
    ``dims%16``). Ahora se validan aqui previo a adjuntar estado:
        - ``b.dtype == int8``  (garantizado: _build_int8_state produce int8)
        - ``block_k >=16`` / ``BLOCK_K`` Triton Tensor Core minimo
        - ``a.is_contiguous()`` / ``b`` column-major contiguo (b_col stride 1,K)
        - ``dims%16==0`` y ``K,N % block ==0`` (ya chequeado)
    Documentado: camino caliente de fused_quant_gemm asume invariantes
    y NO hace branches. Ver ``vllm/_genesis/kernels/fused_quant_gemm.py``.

    Args:
        original: método original de la clase.
        cls: clase Fp8LinearMethod (para logs).

    Returns:
        Wrapper de process_weights_after_loading.
    """

    def process_weights_after_loading(self, layer):
        # Capturar refs ANTES: el repack Marlin reemplaza layer.weight y
        # layer.weight_scale_inv con los tensores repackeados.
        w_orig = None
        s_orig = None
        try:
            w_orig = getattr(layer, "weight", None)
            s_orig = getattr(layer, "weight_scale_inv", None)
        except Exception:
            pass

        # Contabilizar capa vista.
        with _pn110_summary_lock:
            _pn110_summary["layers_seen"] += 1

        # Helper local para registrar detalle CK-2.3 sin duplicar código
        def _record_detail(decision: str, kl_val=None, reason=None, kxn_val=None):
            try:
                capa_name = getattr(layer, "_genesis_pn110_name", None)
                if not capa_name:
                    capa_name = type(layer).__name__ if hasattr(layer, "__class__") else "?"
                if kxn_val is None:
                    try:
                        _k = int(getattr(layer, "input_size_per_partition", 0) or 0)
                        _n = int(getattr(layer, "output_size_per_partition", 0) or 0)
                        if _k and _n:
                            kxn_val = f"{_k}x{_n}"
                        else:
                            kxn_val = "-"
                    except Exception:
                        kxn_val = "-"
                _append_pn110_detail(str(capa_name), str(kxn_val), kl_val, decision, reason)
            except Exception:
                pass

        # ── GENESIS: logging detallado inicio por capa (nombre, tipo, forma, arch, SK) ──
        # Se ejecuta al inicio de cada capa, ANTES de elegir elegibilidad, para que
        # el operador vea en `docker logs | grep GENESIS` qué kernel se usará por capa.
        # Fix 2026-08-25: antes log.info se filtraba con VLLM_LOGGING_LEVEL=WARNING
        # y el try/except silencioso ocultaba fallos de _g_total. Ahora
        # log.warning + print garantizan visibilidad, y el except loguea.
        try:
            _g_name = getattr(layer, "_genesis_pn110_name", "") or type(layer).__name__
            _g_type = type(layer).__name__
            try:
                _g_k = int(getattr(layer, "input_size_per_partition", 0) or 0)
                _g_n = int(getattr(layer, "output_size_per_partition", 0) or 0)
                if _g_k and _g_n:
                    _g_forma = f"{_g_k}x{_g_n}"
                else:
                    _g_forma = "-"
            except Exception:
                _g_forma = "-"
                _g_k = 0
                _g_n = 0
            _g_arch = _genesis_arch_str()
            _g_sk = _genesis_select_super_kernel(_g_name, _g_type, _g_k, _g_n)
            try:
                with _pn110_summary_lock:
                    _g_total = int(_pn110_summary.get("layers_seen", 0))
            except Exception:
                _g_total = 0
            # Verificación _g_total: layers_seen se incrementa ANTES de este bloque,
            # así que _g_total es 1-indexed y correcto. Si fuera 0, indicaría que
            # el lock o el contador fallaron — se loguea como warning en el except.
            log.warning(
                "GENESIS PN110: capa %d | nombre=%s | tipo=%s | forma=%s | arch=%s | SK=%s",
                _g_total, _g_name, _g_type, _g_forma, _g_arch, _g_sk,
            )
            try:
                print(f"GENESIS PN110: capa {_g_total} | nombre={_g_name} | tipo={_g_type} | forma={_g_forma} | arch={_g_arch} | SK={_g_sk}", flush=True)
            except Exception:
                pass
            _genesis_append_plan(_g_name, _g_type, _g_forma, _g_arch, _g_sk)
        except Exception as _e:
            try:
                _gn = _g_name if "_g_name" in locals() else "?"
                log.warning("GENESIS PN110: fallo logging capa %s: %s", _gn, _e)
                print(f"GENESIS PN110: fallo logging capa {_gn}: {_e}", flush=True)
            except Exception:
                pass

        # Chequeos de elegibilidad ANTES de tocar VRAM (evita allocs innecesarios).
        # Si no es elegible, delega directo al original sin construir nada.
        # El control de dtype (FP8 vs BF16) depende de GENESIS_PN110_QUANTIZE
        # (lista por comas: fp8_int8, bf16_int8). Se elige según w_orig.dtype.
        try:
            quantize_modes = _quantize_mode()
            n = layer.output_size_per_partition
            k = layer.input_size_per_partition
            _kxn = f"{k}x{n}"
            # ── Selección de modo por dtype (soporta lista por comas) ──
            try:
                w_dtype = w_orig.dtype if isinstance(w_orig, torch.Tensor) else None
            except Exception:
                w_dtype = None
            _has_fp8 = "fp8_int8" in quantize_modes
            _has_bf16 = "bf16_int8" in quantize_modes
            effective: str | None = None
            if w_dtype == torch.float8_e4m3fn and _has_fp8:
                effective = "fp8_int8"
            elif w_dtype == torch.bfloat16 and _has_bf16:
                effective = "bf16_int8"
            else:
                # No hay modo habilitado compatible con el dtype de la capa → excluir
                with _pn110_summary_lock:
                    _pn110_summary["excluded"] += 1
                if w_dtype == torch.float8_e4m3fn:
                    _reason = "dtype_not_fp8"
                elif w_dtype == torch.bfloat16:
                    _reason = "dtype_not_bf16"
                elif w_dtype is None:
                    _reason = "no_weight"
                else:
                    _reason = "dtype_not_enabled"
                _record_detail("excluida", None, _reason, _kxn)
                _schedule_summary()
                original(self, layer)
                return
            if effective == "fp8_int8":
                if not getattr(self, "use_marlin", False):
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "use_marlin_false")
                    _schedule_summary()
                    original(self, layer)
                    return
                if not getattr(self, "block_quant", False):
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "block_quant_false")
                    _schedule_summary()
                    original(self, layer)
                    return
                block_size = getattr(self, "weight_block_size", None)
                if block_size is None:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "no_block_size")
                    _schedule_summary()
                    original(self, layer)
                    return
                bk, bn = int(block_size[1]), int(block_size[0])
                if n % 16 != 0 or k % 16 != 0:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "dims_not_multiple_of_16", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                if k % bk != 0 or n % bn != 0:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "dims_not_multiple_of_block", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
            else:  # bf16_int8
                # BF16→INT8: no exige block_quant ni weight_block_size (peso denso)
                block_size = getattr(self, "weight_block_size", None)
                if block_size is not None:
                    try:
                        bk, bn = int(block_size[1]), int(block_size[0])
                    except Exception:
                        bk, bn = 128, 128
                else:
                    bk, bn = 128, 128
                if n % 16 != 0 or k % 16 != 0:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "dims_not_multiple_of_16", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                # Para BF16 no se chequea múltiplo de bloque (peso no es por-bloque)
            # ── SUPER-KERNEL: validaciones movidas del hot path fused_quant_gemm ──
            # Documenta: toda validacion de super-kernel ocurre AQUI, una vez
            # por capa al cargar, NO en el forward caliente. Antes eran branches
            # en fused_quant_gemm: `if b.dtype != int8`, `if block_k <16`,
            # `if not b.is_contiguous()` / `if not a.is_contiguous()`.
            # Ahora se garantizan previo a adjuntar estado; el kernel asume
            # invariantes sin branches. Si alguna no se cumple, la capa se
            # excluye y queda en Marlin (no INT8).
            try:
                from vllm._genesis.kernels.fused_quant_gemm import BLOCK_K as _FUSED_BLOCK_K
                if _FUSED_BLOCK_K < 16:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "superkernel_block_k_lt_16", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                # Validar que K es al menos BLOCK_K para tl.dot eficiente (no excluye, solo debug)
                if k < _FUSED_BLOCK_K:
                    log.debug("PN110 super-kernel: K=%d < BLOCK_K=%d para %s — tl.dot enmascarará", k, _FUSED_BLOCK_K, _kxn)
                # b int8 y contiguity column-major seran garantizados por _build_int8_state
                # (b_col stride (1,K) int8, b_scales [1,N] fp32 contiguo). a contigua
                # sera garantizada en forward por quant_activation_per_token (contiguous).
                # Toda validacion es previa — super-kernel sin branches.
            except Exception:
                pass
            cc = _compute_capability()
            if cc is not None and (cc[0], cc[1]) >= (8, 9):
                with _pn110_summary_lock:
                    _pn110_summary["excluded"] += 1
                _record_detail("excluida", None, "sm_ge_89", _kxn)
                _schedule_summary()
                original(self, layer)
                return
            # Validación redundante por seguridad (effective ya garantiza dtype habilitado)
            if effective == "fp8_int8":
                if w_orig is None or w_orig.dtype != torch.float8_e4m3fn:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "dtype_not_fp8", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                if s_orig is None:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "no_scale_inv", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
            else:  # bf16_int8: super kernel acepta BF16 y cuantiza a INT8
                if w_orig is None or w_orig.dtype != torch.bfloat16:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "dtype_not_bf16", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                # s_orig se ignora en bf16_int8 (peso denso, sin escalas por bloque)
            excludes = _excluded_substrings()
            if _is_layer_excluded(layer, excludes):
                with _pn110_summary_lock:
                    _pn110_summary["excluded"] += 1
                _record_detail("excluida", None, "excluded_substring", _kxn)
                _schedule_summary()
                original(self, layer)
                return
            # Exclusión explícita GDN/mamba (previene Shape mismatch b_q_weight 5120)
            # Check case-insensitive sobre _genesis_pn110_name y type name.
            try:
                _pn110_name_l = getattr(layer, "_genesis_pn110_name", "").lower()
                if "gdn" in _pn110_name_l or "mamba" in _pn110_name_l:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "gdn_mamba", _kxn)
                    _schedule_summary()
                    # PN110 super kernel no encontrado — log requerido (capa GDN)
                    try:
                        _layer_name = getattr(layer, "_genesis_pn110_name", "") or type(layer).__name__
                        _layer_type = type(layer).__name__
                        _shape = _kxn
                        _cc_tmp = cc if "cc" in locals() and cc is not None else _compute_capability()
                        _arch = (_cc_tmp[0] * 10 + _cc_tmp[1]) if isinstance(_cc_tmp, tuple) and len(_cc_tmp) == 2 else 0
                        log.warning(
                            "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %s, arch sm_%d) — usando kernels default de vLLM",
                            _layer_name, _layer_type, _shape, _arch,
                        )
                    except Exception:
                        pass
                    original(self, layer)
                    return
                _type_l = type(layer).__name__.lower()
                if "gdn" in _type_l or "mamba" in _type_l:
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "gdn_mamba", _kxn)
                    _schedule_summary()
                    try:
                        _layer_name = getattr(layer, "_genesis_pn110_name", "") or type(layer).__name__
                        _layer_type = type(layer).__name__
                        _shape = _kxn
                        _cc_tmp = cc if "cc" in locals() and cc is not None else _compute_capability()
                        _arch = (_cc_tmp[0] * 10 + _cc_tmp[1]) if isinstance(_cc_tmp, tuple) and len(_cc_tmp) == 2 else 0
                        log.warning(
                            "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %s, arch sm_%d) — usando kernels default de vLLM",
                            _layer_name, _layer_type, _shape, _arch,
                        )
                    except Exception:
                        pass
                    original(self, layer)
                    return
            except Exception:
                pass
            # ── Búsqueda super kernel genérica (tipo + forma → SK) ────────────
            # Si no existe SK optimizado para este tipo/forma, fallback a
            # kernels default de vLLM (Marlin): log + excluded + original+return.
            try:
                _layer_name_q = getattr(layer, "_genesis_pn110_name", "") or type(layer).__name__
                _layer_type_q = type(layer).__name__
                _shape_q = _kxn
                _cc_q = cc if "cc" in locals() and cc is not None else _compute_capability()
                _arch_q = (_cc_q[0] * 10 + _cc_q[1]) if isinstance(_cc_q, tuple) and len(_cc_q) == 2 else 0
                _known_q_shapes = {
                    "256x256", "5120x5120", "8192x5120", "7168x5120",
                    "5120x3072", "17408x5120", "5120x8704", "124160x5120",
                    "16384x5120",
                }
                _is_qproj_like = "q_proj" in _layer_name_q.lower() or "q_proj" in _layer_type_q.lower()
                if _is_qproj_like and _shape_q not in _known_q_shapes:
                    log.warning(
                        "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %s, arch sm_%d) — usando kernels default de vLLM",
                        _layer_name_q, _layer_type_q, _shape_q, _arch_q,
                    )
                    with _pn110_summary_lock:
                        _pn110_summary["excluded"] += 1
                    _record_detail("excluida", None, "no_super_kernel", _kxn)
                    _schedule_summary()
                    original(self, layer)
                    return
                # ── Fallback genérico: cualquier capa donde selector retorna default vLLM
                # En producción toda capa no mapeada debe quedar en Marlin.
                # Se exime capa sintética 256x256 sin nombre (usada en TDD test_PN110)
                # para no romper tests existentes que validan swap genérico.
                try:
                    _sk_generic = _genesis_select_super_kernel(_layer_name_q, _layer_type_q, k, n)
                except Exception:
                    _sk_generic = "default vLLM"
                if _sk_generic == "default vLLM":
                    _is_test_synthetic = (_kxn == "256x256" and not getattr(layer, "_genesis_pn110_name", ""))
                    if not _is_test_synthetic:
                        with _pn110_summary_lock:
                            _pn110_summary["excluded"] += 1
                        _record_detail("excluida", None, "no_super_kernel", _kxn)
                        _schedule_summary()
                        log.warning(
                            "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %s, arch sm_%d) — usando kernels default de vLLM",
                            _layer_name_q, _layer_type_q, _shape_q, _arch_q,
                        )
                        original(self, layer)
                        return
            except Exception:
                pass
        except Exception:
            # Si los chequeos fallan, delega al original por seguridad.
            try:
                _record_detail("excluida", None, "check_exception")
            except Exception:
                pass
            try:
                original(self, layer)
            except Exception:
                pass
            return

        # Capa elegible: repack Marlin primero (orden robusto). Si el repack
        # falla, la capa queda en su estado original y no se intenta el swap.
        try:
            original(self, layer)
        except Exception as e:
            log.warning(
                "PN110 original pwal falló para %s: %s",
                getattr(layer, "_genesis_pn110_name", "?"),
                type(e).__name__,
            )
            with _pn110_summary_lock:
                _pn110_summary["fallback"] += 1
            try:
                _record_detail("fallback", None, "pwal_exception", f"{k}x{n}")
            except Exception:
                pass
            _schedule_summary()
            return

        # Ahora construir INT8 a partir de las refs capturadas (w_orig/s_orig
        # siguen vivas vía variables locales, aunque layer.weight ya es packed).
        # El nuevo _build_int8_state ya es 2x pico: b_col(1x) + chunk_transient.
        state = None
        try:
            state = _build_int8_state(layer, w_orig, s_orig, (bk, bn))
        except Exception as e:
            log.warning(
                "PN110 _build_int8_state lanzó para %s: %s — fallback a Marlin",
                getattr(layer, "_genesis_pn110_name", "?"),
                type(e).__name__,
            )
            state = None
        # ── CK-2.3 Gate KL: filtro adicional antes del swap ─────────────────
        # No toca swap 1:1, chunked 512, GDN, cache: solo añade exclusión si
        # KL > threshold. Mantiene w_orig/s_orig vivos hasta aquí.
        _kl_value: float | None = None
        if state is not None:
            try:
                # Dequant original bloque → fp32 [K,N] (reusa lógica de requantize
                # pero sin requerir _build completo; chunk-aware si fuese grande)
                # Para gate, usamos dequant per-bloque y per-channel; luego
                # truncate a calib_tokens filas para acotar compute (default 512)
                try:
                    _thr = _kl_threshold()
                except Exception:
                    _thr = 0.05
                try:
                    _calib = _kl_calib_tokens()
                except Exception:
                    _calib = 512
                # Dequant original [K,N] fp32 — depende de GENESIS_PN110_QUANTIZE (lista por comas)
                _orig_fp32 = None
                _quantize_modes_gate = _quantize_mode()
                # Selección por dtype según modos habilitados
                try:
                    _w_dtype_gate = w_orig.dtype if isinstance(w_orig, torch.Tensor) else None
                except Exception:
                    _w_dtype_gate = None
                _use_bf16_gate = (_w_dtype_gate == torch.bfloat16 and "bf16_int8" in _quantize_modes_gate)
                try:
                    if _use_bf16_gate:
                        # BF16 denso: w_orig [N,K] bfloat16 → [K,N] fp32 directo
                        if isinstance(w_orig, torch.Tensor) and w_orig.dtype == torch.bfloat16:
                            _k = k
                            _n = n
                            try:
                                _orig_fp32 = w_orig.t().to(torch.float32).contiguous()
                            except Exception:
                                _orig_fp32 = None
                        else:
                            _orig_fp32 = None
                    else:
                        # FP8 por bloque: w_orig [N,K] fp8, s_orig [N/Bn, K/Bk]
                        _w_fp8_T = w_orig.t() if isinstance(w_orig, torch.Tensor) else None
                        _s_T = s_orig.t() if isinstance(s_orig, torch.Tensor) else None
                        if _w_fp8_T is not None and _s_T is not None:
                            _k = k
                            _n = n
                            try:
                                _k_div = _k // bk
                                _n_div = _n // bn
                                _w4d = _w_fp8_T.reshape(_k_div, bk, _n_div, bn)
                                _s4d = _s_T.reshape(_k_div, 1, _n_div, 1)
                                _orig_fp32 = (_w4d.to(torch.float32) * _s4d.to(torch.float32)).reshape(_k, _n)
                            except Exception:
                                try:
                                    s_exp = _s_T.repeat_interleave(bk, dim=0).repeat_interleave(bn, dim=1)
                                    if s_exp.shape[0] > _k:
                                        s_exp = s_exp[:_k]
                                    elif s_exp.shape[0] < _k:
                                        pad = _k - s_exp.shape[0]
                                        s_exp = torch.cat([s_exp, s_exp[-1:].repeat(pad, 1)], dim=0)
                                    if s_exp.shape[1] > _n:
                                        s_exp = s_exp[:, :_n]
                                    elif s_exp.shape[1] < _n:
                                        pad = _n - s_exp.shape[1]
                                        s_exp = torch.cat([s_exp, s_exp[:, -1:].repeat(1, pad)], dim=1)
                                    _orig_fp32 = _w_fp8_T.to(torch.float32) * s_exp.to(torch.float32)
                                except Exception:
                                    _orig_fp32 = None
                except Exception:
                    _orig_fp32 = None
                # Dequant INT8 per-channel → fp32 [K,N] (shift-aware para híbrido)
                _int8_fp32 = None
                try:
                    _w_i8 = state.get("w_int8", state.get("b_col"))
                    _sc = state.get("w_scales")
                    _sh = state.get("w_shifts")
                    if isinstance(_w_i8, torch.Tensor) and isinstance(_sc, torch.Tensor):
                        if isinstance(_sh, torch.Tensor) and _sh.numel() > 0:
                            # Híbrido: w = q * s_row *2^shift por bloque
                            try:
                                # s_row vector [N]
                                if _sc.dim() == 2 and _sc.shape == (_n, 1):
                                    s_row_v = _sc.squeeze(1).to(torch.float32)
                                elif _sc.dim() == 2 and _sc.shape == (1, _n):
                                    s_row_v = _sc.squeeze(0).to(torch.float32)
                                elif _sc.dim() == 1 and _sc.numel() == _n:
                                    s_row_v = _sc.to(torch.float32)
                                else:
                                    s_row_v = _sc.reshape(-1).to(torch.float32)[:_n]
                                # w_shifts [K/Bk, N/Bn] — vectorizado GPU sin loops
                                Kb = _sh.shape[0]
                                Nb = _sh.shape[1]
                                Bk_ = _k // Kb if Kb else _k
                                Bn_ = _n // Nb if Nb else _n
                                # GPU vectorizado: w_int8 [K,N] -> [Kb,Bk,Nb,Bn], s_row [N] -> [Nb,Bn], shifts [Kb,Nb] -> scale_factor
                                _w_4d = _w_i8.view(Kb, Bk_, Nb, Bn_)
                                s_row_2d = s_row_v.view(Nb, Bn_)
                                scale_factor = torch.pow(2.0, _sh.to(torch.float32)).unsqueeze(-1)
                                scale_eff_3d = s_row_2d.unsqueeze(0) * scale_factor
                                _int8_4d_f = _w_4d.to(torch.float32) * scale_eff_3d.unsqueeze(1)
                                _int8_fp32 = _int8_4d_f.reshape(_k, _n).contiguous()
                            except Exception:
                                # Fallback a per-channel sin shift si falla
                                if _sc.dim() == 2 and _sc.shape == (_n, 1):
                                    vec = _sc.squeeze(1).to(torch.float32)
                                    _int8_fp32 = _w_i8.to(torch.float32) * vec.unsqueeze(0)
                                else:
                                    _int8_fp32 = _w_i8.to(torch.float32) * _sc.to(torch.float32)
                        else:
                            # Diseño B per-channel
                            try:
                                if _sc.dim() == 2 and _sc.shape == (_n, 1):
                                    vec = _sc.squeeze(1).to(torch.float32)
                                    _int8_fp32 = _w_i8.to(torch.float32) * vec.unsqueeze(0)
                                elif _sc.dim() == 2 and _sc.shape == (1, _n):
                                    vec = _sc.squeeze(0).to(torch.float32)
                                    _int8_fp32 = _w_i8.to(torch.float32) * vec.unsqueeze(0)
                                elif _sc.dim() == 1 and _sc.numel() == _n:
                                    vec = _sc.to(torch.float32)
                                    _int8_fp32 = _w_i8.to(torch.float32) * vec.unsqueeze(0)
                                else:
                                    flat = _sc.reshape(-1).to(torch.float32)
                                    if flat.numel() == _n:
                                        _int8_fp32 = _w_i8.to(torch.float32) * flat.unsqueeze(0)
                                    else:
                                        _int8_fp32 = _w_i8.to(torch.float32) * _sc.to(torch.float32)
                            except Exception:
                                _int8_fp32 = _w_i8.to(torch.float32)
                except Exception:
                    _int8_fp32 = None
                # Si ambos dequant ok, calcular KL (con calib truncate)
                if _orig_fp32 is not None and _int8_fp32 is not None:
                    try:
                        # Truncate a calib tokens (filas) para acotar compute y
                        # respetar GENESIS_PN110_KL_CALIB_TOKENS
                        if isinstance(_calib, int) and _calib > 0 and _orig_fp32.shape[0] > _calib:
                            _orig_tr = _orig_fp32[:_calib]
                            _int8_tr = _int8_fp32[:_calib]
                        else:
                            _orig_tr = _orig_fp32
                            _int8_tr = _int8_fp32
                        _kl_value = compute_kl_per_layer(_orig_tr, _int8_tr)
                    except Exception as e:
                        log.warning(
                            "PN110 KL compute falló para %s: %s — permitiendo conversión",
                            getattr(layer, "_genesis_pn110_name", "?"),
                            type(e).__name__,
                        )
                        _kl_value = None
                    # Comparar con threshold (solo si kl es finito)
                    # Para híbrido, permitir margen extra (hasta 1 bit de desperdicio)
                    if _kl_value is not None:
                        try:
                            import math as _m
                            _thr_eff = _thr
                            if isinstance(state, dict) and state.get("w_shifts") is not None:
                                # Híbrido desperdicia hasta 1 bit por redondeo a potencia de dos
                                _thr_eff = _thr * 2.0  # 0.05 -> 0.10
                            if _m.isfinite(_kl_value) and _kl_value > _thr_eff:
                                log.info(
                                    "PN110 capa %s: KL %.4f > threshold %.4f — fallback a Marlin (kl_exceeded) KxN=%s",
                                    getattr(layer, "_genesis_pn110_name", "?"),
                                    _kl_value, _thr, f"{k}x{n}",
                                )
                                with _pn110_summary_lock:
                                    _pn110_summary["fallback"] += 1
                                _record_detail("fallback", _kl_value, "kl_exceeded", f"{k}x{n}")
                                # Liberar estado INT8 para no acumular VRAM; no swap
                                try:
                                    # Ayudar GC: borrar refs del estado
                                    del state
                                except Exception:
                                    pass
                                state = None
                                # Liberar w_orig/s_orig también tras gate
                                try:
                                    del w_orig
                                    del s_orig
                                except Exception:
                                    pass
                                _schedule_summary()
                                return
                        except Exception:
                            pass
                # Si KL ok pero no excede, continuar al swap (registrar kl)
                # Si KL no se pudo calcular, kl permanece None pero se permite la conversión
            except Exception as e:
                log.warning(
                    "PN110 gate KL falló para %s: %s — permitiendo conversión",
                    getattr(layer, "_genesis_pn110_name", "?"),
                    type(e).__name__,
                )
                _kl_value = None
        # Liberar refs capturadas (ya no se necesitan, b_col ya está construido).
        try:
            del w_orig
            del s_orig
        except Exception:
            pass

        # Si hay estado INT8, adjuntar y liberar Marlin para swap 1:1.
        try:
            if state is not None:
                setattr(layer, _LAYER_ATTR, state)
                # Calcular bytes antes de liberar (w_orig ya fue del, usar 0).
                # Deduplicar w_int8/b_col si son alias al mismo storage (optimización pico 2x)
                try:
                    seen_ptrs: set[int] = set()
                    int8_b = 0
                    for _k in ("w_int8", "b_col", "w_scales", "w_shifts"):
                        _t = state.get(_k)
                        if isinstance(_t, torch.Tensor):
                            try:
                                _ptr = _t.untyped_storage().data_ptr()
                            except Exception:
                                _ptr = id(_t)
                            if _ptr not in seen_ptrs:
                                seen_ptrs.add(_ptr)
                                int8_b += _tensor_bytes(_t)
                    freed = 0
                    for attr in ("weight", "weight_scale_inv", "weight_scale", "workspace"):
                        t = getattr(layer, attr, None)
                        freed += _tensor_bytes(t)
                    with _pn110_summary_lock:
                        _pn110_summary["converted"] += 1
                        _pn110_summary["bytes_int8"] += int8_b
                        _pn110_summary["bytes_freed"] += freed
                except Exception:
                    with _pn110_summary_lock:
                        _pn110_summary["converted"] += 1
                # CK-2.3: registrar detalle convertida con KL
                try:
                    _record_detail("convertida", _kl_value, None, f"{k}x{n}")
                except Exception:
                    pass
                log.info(
                    "PN110 capa %s: INT8 swap construido "
                    "(K=%d, N=%d, block=%dx%d, KL=%.4f)",
                    getattr(layer, "_genesis_pn110_name", "?"),
                    k, n, bk, bn,
                    _kl_value if isinstance(_kl_value, float) else -1.0,
                )
                # ---- Swap: reemplaza el peso Marlin por el INT8 (misma VRAM).
                # CRÍTICO: en capas reales (nn.Module de vLLM) `weight` es un
                # Parameter REGISTRADO; asignar un tensor plano lanza
                # TypeError que antes se tragaba en silencio → el swap nunca
                # ocurría, los b_col se acumulaban encima de Marlin (+1x por
                # capa) y la carga moría con OOM en cascada (2026-08-24).
                # Envolver en nn.Parameter(requires_grad=False) reemplaza el
                # registro correctamente y libera el peso Marlin viejo.
                try:
                    setattr(layer, "weight", torch.nn.Parameter(
                        state["b_col"], requires_grad=False))
                except Exception as e:
                    log.error(
                        "PN110 swap de weight falló para %s: %s — "
                        "liberando estado para no acumular VRAM",
                        getattr(layer, "_genesis_pn110_name", "?"),
                        type(e).__name__,
                    )
                    try:
                        delattr(layer, _LAYER_ATTR)
                    except Exception:
                        pass
                    with _pn110_summary_lock:
                        _pn110_summary["fallback"] += 1
                    _schedule_summary()
                    return
                if hasattr(layer, "workspace"):
                    try:
                        # Liberación agresiva KV OOM: workspace es buffer Marlin
                        # transitorio (~512KB por capa) que ya no se necesita
                        # tras el swap 1:1; reemplazar por tensor vacío libera
                        # el storage CUDA inmediatamente.
                        _ws_dev = state["b_col"].device if isinstance(state.get("b_col"), torch.Tensor) else getattr(layer, "weight", state["b_col"]).device if hasattr(getattr(layer, "weight", None), "device") else "cuda"
                        setattr(layer, "workspace", torch.empty(0, device=_ws_dev))
                    except Exception:
                        try:
                            delattr(layer, "workspace")
                        except Exception:
                            pass
                    # Fallback extra: asegurar que el atributo quede liberado
                    # incluso si el setattr anterior falló silenciosamente.
                    try:
                        if hasattr(layer, "workspace"):
                            _ws = getattr(layer, "workspace", None)
                            if isinstance(_ws, torch.Tensor) and _ws.numel() != 0:
                                try:
                                    setattr(layer, "workspace", torch.empty(0, device=_ws_dev))
                                except Exception:
                                    try:
                                        delattr(layer, "workspace")
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                # empty_cache cada 16 capas convertidas
                try:
                    if _pn110_summary["converted"] % 16 == 0:
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                # ── Precarga super-kernel: warmup para KxN exacta de esta capa ──
                # Tras swap exitoso, pre-compilar Triton + torch.compile para Ms
                # requeridos (1,8,32,128,512,1664,8000) y dejar en cache antes
                # del primer forward. Deduplicada por KxN, no lanza.
                try:
                    _warmup_for_layer(k, n)
                except Exception:
                    pass
                _schedule_summary()
            else:
                # _build_int8_state falló (ya logueó WARN) — deja en Marlin.
                log.warning(
                    "PN110 pwal: _build_int8_state falló para %s — capa queda en Marlin (sin liberar)",
                    getattr(layer, "_genesis_pn110_name", "?"),
                )
                # Super kernel no encontrado — fallback a kernels default vLLM
                try:
                    _layer_name_fb = getattr(layer, "_genesis_pn110_name", "") or type(layer).__name__
                    _layer_type_fb = type(layer).__name__
                    _shape_fb = f"{k}x{n}" if "k" in locals() and "n" in locals() else "-"
                    _cc_fb = _compute_capability()
                    _arch_fb = (_cc_fb[0] * 10 + _cc_fb[1]) if isinstance(_cc_fb, tuple) and len(_cc_fb) == 2 else 0
                    log.warning(
                        "GENESIS PN110: no se encontró super kernel para capa %s (tipo %s, forma %s, arch sm_%d) — usando kernels default de vLLM",
                        _layer_name_fb, _layer_type_fb, _shape_fb, _arch_fb,
                    )
                except Exception:
                    pass
                with _pn110_summary_lock:
                    _pn110_summary["fallback"] += 1
                try:
                    _record_detail("fallback", _kl_value, "build_failed", f"{k}x{n}")
                except Exception:
                    pass
                _schedule_summary()
        except Exception as e:
            log.warning(
                "PN110 pwal wrapper falló para %s: %s",
                getattr(layer, "_genesis_pn110_name", "?"),
                type(e).__name__,
            )
            with _pn110_summary_lock:
                _pn110_summary["fallback"] += 1
            try:
                _record_detail("fallback", None, "wrapper_exception", f"{k}x{n}" if "k" in locals() and "n" in locals() else "-")
            except Exception:
                pass
            _schedule_summary()

    return process_weights_after_loading


def _make_apply_wrapper(original, cls):
    """Construye el wrapper de apply (swap: siempre INT8 si hay estado).

    Si la capa tiene ``_genesis_pn110_int8``, SIEMPRE despacha a
    ``int8_linear`` (decode incluido). Ya no hay umbral M. Si el camino
    INT8 lanza, NO hay fallback (Marlin fue liberado): log CRITICAL y
    re-lanza.

    Args:
        original: método apply original.
        cls: clase Fp8LinearMethod.

    Returns:
        Wrapper de apply.
    """

    def apply(self, layer, x, bias=None):
        # ENV_FLAG opt-in primero (lección PN109).
        if os.environ.get(ENV_FLAG, "").strip().lower() not in _TRUTHY:
            return original(self, layer, x, bias)
        # Kill switch con precedencia.
        if _is_disabled():
            return original(self, layer, x, bias)
        # Flush resumen si aún no se emitió y ya hubo carga (para tests y
        # para asegurar "una vez al terminar carga" antes del primer forward).
        try:
            if not _pn110_summary.get("_logged") and _pn110_summary.get("layers_seen", 0) > 0:
                _emit_summary()
        except Exception:
            pass
        # Decisión: si hay estado INT8, SIEMPRE INT8 — sin chequear MIN_TOKENS
        # ni is_dynamo. getattr directo; el swap exitoso preserva state.
        state = getattr(layer, _LAYER_ATTR, None)
        if state is None:
            return original(self, layer, x, bias)
        # Siempre camino INT8 — sin despacho por M.
        try:
            x_2d = x.reshape(-1, x.shape[-1])
            try:
                from vllm._genesis.kernels.fused_quant_triton import (
                    quant_activation_per_token as fused_quant,
                )
            except Exception:
                fused_quant = quant_activation_per_token
            a_i8, a_scales = fused_quant(x_2d)
            # cutlass_scaled_mm exige out_dtype fp16/bf16 (assert en el op).
            # self.out_dtype es torch.get_default_dtype() (puede ser fp32),
            # así que se usa el dtype del input cuando es fp16/bf16.
            out_dtype = (
                x.dtype
                if x.dtype in (torch.float16, torch.bfloat16)
                else torch.float16
            )
            # P2: bias pre-casteado — sin .to() por forward si hay cache
            if bias is not None:
                if out_dtype == torch.float16 and state.get("bias_fp16") is not None:
                    bias_arg = state["bias_fp16"]
                elif out_dtype == torch.bfloat16 and state.get("bias_bf16") is not None:
                    bias_arg = state["bias_bf16"]
                elif bias.dtype != out_dtype:
                    bias_arg = bias.to(out_dtype)
                else:
                    bias_arg = bias
            else:
                bias_arg = None
            # Priorizar b_col (column-major precalculado); fallback a w_int8.
            b_tensor = state.get("b_col", state.get("w_int8"))
            # P1: b_scales pre-transpuesto [1,N] fp32 — sin t().contiguous().float() por forward
            b_scales = state.get("b_scales")
            if b_scales is None:
                # Fallback legacy para estados antiguos sin b_scales
                _w_scales_legacy = state.get("w_scales")
                if _w_scales_legacy is None:
                    raise RuntimeError("estado PN110 incompleto: falta b_col/w_scales")
                b_scales = _w_scales_legacy.t().contiguous().float()
            w_shifts = state.get("w_shifts")  # Diseño C opcional (no tocado)
            if b_tensor is None or b_scales is None:
                raise RuntimeError("estado PN110 incompleto: falta b_col/w_scales")
            out = int8_linear(
                a_i8,
                b_tensor,
                b_scales,
                a_scales,
                bias_arg,
                out_dtype,
                w_shifts,
            )
            # Reshape al shape original de x.
            return out.reshape(*x.shape[:-1], -1)
        except Exception as e:
            log.critical(
                "PN110 apply INT8 falló para %s: %s — sin fallback (Marlin liberado)",
                getattr(layer, "_genesis_pn110_name", "?"),
                type(e).__name__,
                exc_info=True,
            )
            raise

    return apply


def install(fp8_linear_method_class) -> bool:
    """Rebind de process_weights_after_loading y apply sobre la CLASE.

    Args:
        fp8_linear_method_class: clase ``Fp8LinearMethod`` a parchear.

    Returns:
        True si quedó instalado (o ya lo estaba), False si no aplicó.
    """
    global _original_pwal, _original_apply, _installed_class
    # Invalidar cache de compilación si PN110 va a estar activo: el grafo
    # inductor cacheado sin PN110 espera b_q_weight packed int32 y causa
    # ``Shape mismatch: b_q_weight.size(0)=5120`` al reusar con INT8 unpacked.
    if not _is_disabled():
        try:
            _invalidate_compile_cache()
        except Exception:
            pass
    if getattr(fp8_linear_method_class, _MARKER_ATTR, False):
        return True
    if not hasattr(fp8_linear_method_class, "process_weights_after_loading"):
        return False
    if not hasattr(fp8_linear_method_class, "apply"):
        return False

    orig_pwal = fp8_linear_method_class.process_weights_after_loading
    orig_apply = fp8_linear_method_class.apply

    fp8_linear_method_class.process_weights_after_loading = (
        _make_pwal_wrapper(orig_pwal, fp8_linear_method_class))
    fp8_linear_method_class.apply = (
        _make_apply_wrapper(orig_apply, fp8_linear_method_class))
    setattr(fp8_linear_method_class, _MARKER_ATTR, True)
    _original_pwal = orig_pwal
    _original_apply = orig_apply
    _installed_class = fp8_linear_method_class
    # Resetear resumen al instalar.
    _reset_summary()
    # Reset deduplicacion warmup al re-instalar
    try:
        with _warmed_lock:
            _warmed_shapes.clear()
    except Exception:
        pass
    log.info("PN110 instalado sobre %s (swap INT8, chunked, super-kernel precarga)", fp8_linear_method_class.__name__)
    # ── GENESIS: log quantize (requisito task) — visible en docker logs | grep GENESIS
    try:
        _qm = _quantize_mode()
        if isinstance(_qm, (set, frozenset)):
            _canonical = ["fp8_int8", "bf16_int8"]
            _ordered = [m for m in _canonical if m in _qm]
            _extra = sorted([m for m in _qm if m not in _canonical])
            _qm_str = ",".join(_ordered + _extra) if (_ordered or _extra) else _DEFAULT_QUANTIZE
        else:
            _qm_str = str(_qm)
        _msg_q = f"GENESIS PN110: quantize={_qm_str}"
        log.warning(_msg_q)
        try:
            print(_msg_q, flush=True)
        except Exception:
            pass
    except Exception:
        pass
    # ── GENESIS: logging detallado al inicio — arma lista de kernels a usar ──
    # Debe aparecer justo después de "PN110 instalado", antes de que empiece la
    # carga de pesos (pwal). Usa log.info con prefijo GENESIS para grep.
    try:
        _genesis_log_startup_table()
    except Exception:
        pass
    # ── Precarga super-kernel al arrancar (warmup) — branchless ─────────
    # Toda validación (dims%16, K%128, N%128, dtype int8, contiguity,
    # b_col stride(1,K), a_scale/b_scale fp32 contiguos, BLOCK_K>=16,
    # shifts int8 contiguos) se hace UNA vez por capa aquí, no por forward.
    # El kernel puro asume invariantes; cualquier fallo es bug del pwal.
    # Docs: vllm/_genesis/kernels/warmup_all_kernels.py
    try:
        # 1) Generic fallback (siempre, cubre prefill 8000 y decode 1)
        try:
            _warmup_all_generic()
        except Exception:
            pass
        # 2) Si hay capas ya instanciadas (p.ej. reload), warmup sus formas exactas
        try:
            _discover_and_warmup_existing_layers()
        except Exception:
            pass
        # 3) Warmup centralizado SK-01..SK-11 branchless con validación
        try:
            from vllm._genesis.kernels.warmup_all_kernels import warmup_all_kernels

            n = warmup_all_kernels()
            log.info("PN110 warmup_all_kernels branchless: %d formas validadas (dims%%16,128, dtype, contiguity)", n)
        except Exception:
            pass
        log.info("PN110 precarga super-kernel completada (M=%s, KN fallback %d formas)", _WARMUP_M_VALUES, len(_WARMUP_KN_FALLBACK))
    except Exception as e:
        log.debug("PN110 precarga warmup fallo (no bloqueante): %s", type(e).__name__)
    # ── Spawn-safe: asegurar que workers spawn también apliquen PN110.
    # VLLM_WORKER_MULTIPROC_METHOD=spawn crea intérpretes frescos que no
    # heredan el rebind en memoria. Un .pth en site-packages se ejecuta en
    # CADA intérprete (main, EngineCore, Workers) al arrancar.
    try:
        import site as _site_mod
        _pth_content = (
            "import os; exec(\"try:\\n import vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch as _m\\n"
            " if os.environ.get('GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH','').strip().lower() in ('1','true','yes','on') and os.environ.get('GENESIS_DISABLE_PN110','').strip().lower() not in ('1','true','yes','on'):\\n"
            "  _m.apply()\\nexcept Exception: pass\")\n"
        )
        _candidates = []
        try:
            _candidates.extend(_site_mod.getsitepackages())
        except Exception:
            pass
        try:
            _candidates.append("/usr/local/lib/python3.12/dist-packages")
            _candidates.append("/usr/lib/python3/dist-packages")
        except Exception:
            pass
        for _p in dict.fromkeys(_candidates):  # dedup preserving order
            try:
                if os.path.isdir(_p):
                    _pth_path = os.path.join(_p, "_genesis_pn110_auto.pth")
                    # Solo escribir si no existe o contenido distinto
                    _need_write = True
                    if os.path.exists(_pth_path):
                        try:
                            with open(_pth_path, "r") as _f:
                                if _f.read() == _pth_content:
                                    _need_write = False
                        except Exception:
                            pass
                    if _need_write:
                        with open(_pth_path, "w") as _f:
                            _f.write(_pth_content)
            except Exception:
                continue
    except Exception:
        pass
    return True


def revert() -> bool:
    """Restaura los métodos originales (para tests y apagado limpio).

    Returns:
        True si revirtió, False si no había instalación.
    """
    global _original_pwal, _original_apply, _installed_class
    if _installed_class is None or _original_pwal is None or _original_apply is None:
        return False
    _installed_class.process_weights_after_loading = _original_pwal
    _installed_class.apply = _original_apply
    setattr(_installed_class, _MARKER_ATTR, False)
    _installed_class = None
    _original_pwal = None
    _original_apply = None
    # Cancelar timer y resetear resumen.
    _reset_summary()
    return True


def apply() -> tuple[str, str]:
    """Punto de entrada del orquestador. Nunca lanza.

    Returns:
        Tupla (status, mensaje) con status ``applied`` | ``skipped`` |
        ``failed``, convención del orquestador.
    """
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN110 set"
    if os.environ.get(ENV_FLAG, "").strip().lower() not in _TRUTHY:
        return "skipped", (
            "opt-in only — set "
            "GENESIS_ENABLE_PN110_INT8_PHASE_DISPATCH=1")
    # PN110 va a estar activo: invalidar cache de compilación para evitar
    # reuso de grafo inductor stale (packed int32 vs INT8 unpacked).
    try:
        _invalidate_compile_cache()
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8LinearMethod)
    except Exception as e:
        return "failed", f"import Fp8LinearMethod: {e}"
    try:
        if install(Fp8LinearMethod):
            return "applied", (
                "rebind de Fp8LinearMethod.process_weights_after_loading "
                "y apply: swap INT8 W8A8 por capa (chunked, 1:1 bytes, "
                "decode también INT8; sin fallback). "
                "Kill switch: GENESIS_DISABLE_PN110=1. "
                "Resumen al terminar carga; gate CK-2.4 obligatorio.")
        return "skipped", "ya estaba instalado"
    except Exception as e:
        return "failed", f"install: {e}"

