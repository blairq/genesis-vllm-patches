# SPDX-License-Identifier: Apache-2.0
"""Analisis de arranque del engine, en castellano y para humanos.

Responde de una sola lectura las preguntas que uno se hace cada vez que
levanta el engine y que hoy hay que reconstruir a mano leyendo mil lineas de
log:

  - ¿A donde se fue la VRAM?
  - ¿Cuanto contexto entra realmente? ¿Con MI patron de uso?
  - ¿Que riesgos conocidos tiene esta configuracion?
  - ¿Que flag me daria mas KV, y cuanto, y a costa de que?

Se invoca desde PN83 en dos puntos del arranque:

  1. `gpu_worker.py`, cuando el worker ya perfilo la memoria -> `guardar_memoria()`
     deja el desglose en un JSON (solo rank 0).
  2. `kv_cache_utils.py`, cuando ya se sabe el tamaño del KV en tokens ->
     `emitir_informe()` junta las dos mitades y escribe el informe.

Hace falta partirlo en dos porque los numeros viven en procesos distintos: el
desglose de memoria lo calcula el WORKER y el total de tokens de KV lo calcula
el ENGINE CORE.

Tambien corre solo, contra el ultimo arranque:

    docker exec <contenedor> python3 -m vllm._genesis.analisis_arranque

Nunca puede tumbar el arranque: todo va en try/except y ante cualquier duda
se calla.
"""

from __future__ import annotations

import json
import os
import sys

ESTADO = "/tmp/genesis_analisis_arranque.json"

ANCHO = 78

# Mediciones propias, en este rig (2x RTX 3090, TP=2, Qwen3.8-27B W8A16).
# Ver docs/KV-OFFLOADING.md §9.
GANANCIA_MM_KWARGS = 0.054  # +5,4% de KV
GANANCIA_BATCHED_4K = 0.135  # +13,5% de KV
# FlashInfer aloca esto para el wrapper de spec-decode prefill, y lo hace
# LAZY: en el primer request, cuando el profiler ya repartio todo. Ver §3.4.
FLASHINFER_WORKSPACE_MIB = 394

_YA_EMITIDO = False


# ─────────────────────────────────────────────────────────────────────────
# 1. recoleccion
# ─────────────────────────────────────────────────────────────────────────


def guardar_memoria(**datos) -> None:
    """Llamado desde el worker (rank 0) con el desglose de memoria."""
    try:
        if int(datos.get("rank", 0)) != 0:
            return
        with open(ESTADO, "w") as fh:
            json.dump(datos, fh)
    except Exception:
        pass


def _leer_memoria() -> dict:
    try:
        with open(ESTADO) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _g(obj, *nombres, default=None):
    """getattr encadenado y tolerante: el primero que exista."""
    for n in nombres:
        cur = obj
        ok = True
        for parte in n.split("."):
            if cur is None or not hasattr(cur, parte):
                ok = False
                break
            cur = getattr(cur, parte)
        if ok and cur is not None:
            return cur
    return default


# ─────────────────────────────────────────────────────────────────────────
# 2. formato
# ─────────────────────────────────────────────────────────────────────────


def _gib(b) -> float:
    try:
        return float(b) / (1 << 30)
    except Exception:
        return 0.0


def _miles(n) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)


def _titulo(t: str) -> str:
    return f"\n{t}\n{'─' * min(len(t), ANCHO)}"


def _barra(frac: float, ancho: int = 22) -> str:
    frac = max(0.0, min(1.0, frac))
    lleno = int(round(frac * ancho))
    return "█" * lleno + "·" * (ancho - lleno)


def _fila_mem(etiqueta: str, gib: float, total: float) -> str:
    frac = gib / total if total else 0.0
    g = f"{gib:6.2f}".replace(".", ",")
    f = f"{frac * 100:4.1f}".replace(".", ",")
    return f"  {etiqueta:<24} {g} GiB  {_barra(frac)} {f}%"


# ─────────────────────────────────────────────────────────────────────────
# 3. secciones del informe
# ─────────────────────────────────────────────────────────────────────────


def _sec_modelo(vc) -> list[str]:
    mc = _g(vc, "model_config")
    pc = _g(vc, "parallel_config")
    sc = _g(vc, "scheduler_config")
    modelo = str(_g(mc, "model", default="?")).split("/")[-1]
    cuant = _g(mc, "quantization", default=None) or "sin cuantizar (bf16/fp16)"
    tp = _g(pc, "tensor_parallel_size", default=1)
    ml = _g(mc, "max_model_len", default=0)
    seqs = _g(sc, "max_num_seqs", default=0)

    out = [_titulo("MODELO")]
    out.append(f"  {modelo}")
    out.append(f"  cuantizacion: {cuant}   ·   tensor parallel: {tp} GPU(s)")
    out.append(
        f"  contexto maximo por request: {_miles(ml)} tokens"
        f"   ·   hasta {seqs} secuencias a la vez"
    )
    return out


def _sec_vram(mem: dict) -> list[str]:
    total = _gib(mem.get("total_memory"))
    if not total:
        return []
    pesos = _gib(mem.get("weights_memory"))
    kv = _gib(mem.get("available_kv_cache_memory_bytes"))
    graphs = _gib(mem.get("cudagraph_memory_estimate"))
    no_kv = _gib(mem.get("non_kv_cache_memory"))
    # lo que el profiler reservo para activaciones y overhead del runtime,
    # descontando los pesos (que ya se muestran aparte)
    # OJO con la aritmetica: vLLM reporta `non_kv_cache_memory` medido DENTRO
    # del presupuesto pedido (total x util), no sobre la VRAM entera, y sale
    # MENOR que los pesos. Su cuenta es:
    #     requested - non_kv_cache - cudagraphs = KV
    # Para el desglose sobre la VRAM total la unica cuenta sostenible es
    # restar lo que si esta medido en terminos absolutos.
    libre = max(0.0, total - pesos - kv - graphs)

    out = [_titulo("A DONDE SE FUE LA VRAM  (por GPU, de "
                    + f"{total:.1f}".replace(".", ",") + " GiB)")]
    out.append(_fila_mem("pesos del modelo", pesos, total))
    out.append(_fila_mem("cache de KV", kv, total))
    out.append(_fila_mem("CUDA graphs", graphs, total))
    out.append(_fila_mem("libre / activaciones", libre, total))
    out.append("")
    out.append(
        "  Ese resto NO esta desperdiciado: ahi entran las activaciones y los picos"
    )
    out.append(
        "  transitorios de las capas GDN (~12 KiB por token del batch) y las"
    )
    out.append("  alocaciones lazy que el profiler no ve (ver RIESGOS).")
    return out


def _sec_capacidad(vc, num_tokens: int) -> list[str]:
    sc = _g(vc, "scheduler_config")
    mc = _g(vc, "model_config")
    seqs = int(_g(sc, "max_num_seqs", default=0) or 0)
    ml = int(_g(mc, "max_model_len", default=0) or 0)

    principal = int(os.environ.get("GENESIS_ANALISIS_HILO_PRINCIPAL", 220_000))
    agente = int(os.environ.get("GENESIS_ANALISIS_AGENTE", 40_000))

    out = [_titulo("QUE ENTRA EN ESA CACHE")]
    out.append(f"  {_miles(num_tokens)} tokens de contexto en total.")
    out.append("")

    if agente > 0:
        caben = max(0, (num_tokens - principal) // agente)
        sobra = num_tokens - principal - caben * agente
        if num_tokens >= principal:
            out.append(
                f"  Con tu patron (1 hilo de {_miles(principal)} + agentes de "
                f"{_miles(agente)}):"
            )
            out.append(
                f"    -> el hilo principal + {caben} agentes en paralelo"
                f"   (sobran {_miles(sobra)} tokens)"
            )
        else:
            out.append(
                f"  ⚠ NO entra un hilo de {_miles(principal)} tokens: la cache"
                f" tiene {_miles(num_tokens)}."
            )
        out.append(f"    -> o {num_tokens // agente} agentes de {_miles(agente)} sin hilo principal")
        out.append("")

    if seqs:
        out.append(
            f"  Limite aparte: --max-num-seqs {seqs}, o sea {seqs} secuencias"
            " ejecutandose a la vez."
        )
        out.append(
            "  Los requests de mas se encolan (no fallan), pero si la suma de"
        )
        out.append(
            "  contextos activos supera la cache, los bloques viejos se desalojan."
        )
    if ml and num_tokens:
        out.append(
            f"  Concurrencia a contexto lleno ({_miles(ml)} tok/request):"
            + f" {num_tokens / ml:.2f}x".replace(".", ",")
        )
    out.append("")
    out.append("  Ajustable: GENESIS_ANALISIS_HILO_PRINCIPAL / _AGENTE.")
    return out


def _sec_offloading(vc, num_tokens: int) -> list[str]:
    kt = _g(vc, "kv_transfer_config")
    if kt is None:
        return [
            _titulo("CACHE EN DOS CAPAS (RAM + DISCO)"),
            "  APAGADO. Cuando la cache de VRAM se llena, los bloques viejos se",
            "  DESCARTAN y hay que recalcularlos: un hilo de 220k son minutos de",
            "  prefill cada vez que unos agentes te lo pisan.",
            "  Ver docs/KV-OFFLOADING.md para activarlo.",
        ]

    extra = _g(kt, "kv_connector_extra_config", default={}) or {}
    spec = extra.get("spec_name", "?")
    cpu_b = extra.get("cpu_bytes_to_use", 0)
    pol = extra.get("eviction_policy", "lru")
    tiers = extra.get("secondary_tiers", []) or []

    out = [_titulo("CACHE EN DOS CAPAS (RAM + DISCO)")]
    out.append(f"  spec: {spec}   ·   politica de desalojo: {pol}")

    # tokens que caben en RAM, a la misma densidad que la VRAM
    mem = _leer_memoria()
    kv_bytes = mem.get("available_kv_cache_memory_bytes") or 0
    if kv_bytes and num_tokens and cpu_b:
        por_token = kv_bytes / float(num_tokens)
        tok_ram = int(cpu_b / por_token)
        out.append(
            "  tier 1 · RAM:   " + f"{_gib(cpu_b):.1f}".replace(".", ",")
            + f" GiB  ≈ {_miles(tok_ram)} tokens mas de contexto guardado"
        )
    elif cpu_b:
        out.append("  tier 1 · RAM:   " + f"{_gib(cpu_b):.1f}".replace(".", ",") + " GiB")

    for t in tiers:
        raiz = t.get("root_dir", "?")
        out.append(f"  tier 2 · disco: {raiz}  (tipo {t.get('type', '?')})")

    if pol == "arc":
        out.append("")
        out.append(
            "  ARC separa lo accedido UNA vez de lo accedido VARIAS: los agentes"
        )
        out.append(
            "  efimeros se desalojan primero y tu hilo largo queda protegido solo."
        )
    else:
        out.append("")
        out.append(
            f"  ⚠ Con '{pol}' cuatro coders recientes valen mas que tu prefijo de"
        )
        out.append("    hace diez minutos y te lo empujan. Considera 'arc'.")

    if tiers and os.environ.get("GENESIS_ENABLE_PN81_KV_DISK_QUOTA", "") not in (
        "1",
        "true",
        "yes",
    ):
        out.append("")
        out.append(
            "  ⚠ PN81 APAGADO y hay tier de disco: vLLM no le pone cuota NI"
        )
        out.append(
            "    limpieza a ese directorio. Crece sin techo (medido: 38 GB en"
        )
        out.append("    una sesion). Activar GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1.")
    return out


def _sec_mtp(vc) -> list[str]:
    sp = _g(vc, "speculative_config")
    if sp is None:
        return []
    met = _g(sp, "method", default="?")
    k = _g(sp, "num_speculative_tokens", default=0)
    out = [_titulo("SPECULATIVE DECODING (MTP)")]
    out.append(f"  metodo: {met}   ·   {k} tokens de draft por paso")
    out.append("")
    out.append("  Para ver la aceptacion real, con el engine sirviendo trafico:")
    out.append("    curl -s :PUERTO/metrics -H 'Authorization: Bearer $VLLM_API_KEY' \\")
    out.append("      | grep spec_decode_num_.*_total")
    out.append("  aceptacion = accepted_tokens_total / draft_tokens_total")
    out.append("  Referencia en este rig (W8A16 + MTP): 73%.  Por debajo de ~30%")
    out.append("  el MTP esta costando mas de lo que ahorra.")
    return out


def _sec_riesgos(vc, mem: dict) -> list[str]:
    out = [_titulo("RIESGOS DE ESTA CONFIGURACION")]
    hay = False

    usa_offload = _g(vc, "kv_transfer_config") is not None
    tiene_mtp = _g(vc, "speculative_config") is not None

    # ── PN82: el crash intermitente de arranque ──────────────────────────
    if usa_offload:
        tp = int(_g(vc, "parallel_config.tensor_parallel_size", default=1) or 1)
        if tp > 1:
            activo = _pn82_activo()
            if activo:
                out.append(
                    "  ✔ PN82 activo. Sin el, con offloading + TP>1 el engine muere"
                )
                out.append(
                    "    de forma INTERMITENTE en la captura de CUDA graphs"
                )
                out.append(
                    "    (sm.fill_(-1) -> 'CUDA error: invalid argument'). No es la"
                )
                out.append(
                    "    VRAM: es un cudaHostRegister fallido que deja el error"
                )
                out.append("    latcheado en el contexto. Ver §3.3.")
            else:
                out.append(
                    "  ✖ PN82 NO detectado, y estas con offloading + TP>1."
                )
                out.append(
                    "    Este arranque puede morir en la captura de CUDA graphs,"
                )
                out.append(
                    "    y el mismo compose a veces arranca y a veces no. Bajar"
                )
                out.append(
                    "    --gpu-memory-utilization NO lo arregla. Ver §3.3."
                )
            hay = True

    # ── PN84: hit de prefijo inconsistente (hibrido + MTP) ───────────────
    # Necesita las dos cosas: capas GDN/mamba Y speculative decoding. Es la
    # config de los cuatro engines qwen38 de este rig.
    if tiene_mtp and _hay_capas_mamba(vc):
        if _pn84_activo():
            out.append(
                "  ✔ PN84 activo. Sin el, en hibrido + MTP el lookup del cache"
            )
            out.append(
                "    de prefijos puede devolver 'N tokens computados' con CERO"
            )
            out.append(
                "    bloques en el grupo de atencion. Con offloading eso mata el"
            )
            out.append(
                "    EngineCore entero; sin offloading contesta con KV basura"
            )
            out.append("    en silencio. Ver §10.")
        else:
            out.append(
                "  ✖ PN84 NO detectado, y estas en hibrido + MTP."
            )
            out.append(
                "    find_longest_cache_hit puede reportar un hit de prefijo que"
            )
            out.append(
                "    el grupo de atencion NO tiene. Sintoma: AssertionError sin"
            )
            out.append(
                "    mensaje en offloading/scheduler.py:612 -> EngineDeadError,"
            )
            out.append(
                "    tras horas de uso normal. Sin offloading no crashea: da"
            )
            out.append("    respuestas incorrectas sin avisar. Ver §10.")
        hay = True

    # ── FlashInfer: los 394 MiB que se alocan en el PRIMER request ───────
    if tiene_mtp:
        total = _gib(mem.get("total_memory"))
        pesos = _gib(mem.get("weights_memory"))
        kv = _gib(mem.get("available_kv_cache_memory_bytes"))
        graphs = _gib(mem.get("cudagraph_memory_estimate"))
        libre_mib = max(0.0, total - pesos - kv - graphs) * 1024
        if total:
            hay = True
            if libre_mib < FLASHINFER_WORKSPACE_MIB * 1.25:
                out.append("")
                out.append(
                    f"  ✖ Margen libre {libre_mib:.0f} MiB y FlashInfer necesita"
                )
                out.append(
                    f"    {FLASHINFER_WORKSPACE_MIB} MiB para el workspace de"
                    " spec-decode, que aloca"
                )
                out.append(
                    "    en el PRIMER REQUEST (no ahora). Riesgo alto de que el"
                )
                out.append(
                    "    engine levante bien y muera con el primer prompt:"
                )
                out.append(
                    "      OutOfMemoryError en flashinfer._get_workspace_buffer"
                )
                out.append(
                    "    Recomendacion: bajar --gpu-memory-utilization ~0,02."
                )
            else:
                out.append("")
                out.append(
                    f"  ✔ Margen libre {libre_mib:.0f} MiB, alcanza para los"
                    f" {FLASHINFER_WORKSPACE_MIB} MiB que"
                )
                out.append(
                    "    FlashInfer aloca lazy en el primer request. Ver §3.4."
                )

    # ── cumem + expandable_segments ──────────────────────────────────────
    if usa_offload:
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" in conf and "max_split_size_mb" not in conf:
            hay = True
            out.append("")
            out.append(
                "  ✖ PYTORCH_CUDA_ALLOC_CONF tiene expandable_segments pero NO"
            )
            out.append(
                "    max_split_size_mb. Con cumem activo la captura de CUDA"
            )
            out.append("    graphs falla. Agregar max_split_size_mb:512. Ver §3.2.")

    if not hay:
        out.append("  Nada que reportar.")
    return out


def _pn82_activo() -> bool:
    try:
        from vllm._genesis.guards import resolve_vllm_file

        f = resolve_vllm_file("v1/kv_offload/cpu/gpu_worker.py")
        if f is None:
            return False
        with open(f) as fh:
            return "_GENESIS_PN82_HOST_REGISTER_STICKY" in fh.read()
    except Exception:
        return False


def _pn84_activo() -> bool:
    try:
        from vllm._genesis.guards import resolve_vllm_file

        f = resolve_vllm_file("v1/core/kv_cache_coordinator.py")
        if f is None:
            return False
        with open(f) as fh:
            return "Genesis PN84" in fh.read()
    except Exception:
        return False


def _hay_capas_mamba(vc) -> bool:
    """Detecta el hibrido por la config del modelo, no por el nombre."""
    try:
        hf = _g(vc, "model_config.hf_config")
        for attr in ("linear_attn_config", "layer_types", "layers_block_type"):
            val = getattr(hf, attr, None)
            if isinstance(val, (list, tuple)):
                return any("linear" in str(x) or "mamba" in str(x) for x in val)
            if val:
                return True
        return "qwen3_5" in str(getattr(hf, "model_type", ""))
    except Exception:
        return False


def _sec_recomendaciones(vc, num_tokens: int) -> list[str]:
    mc = _g(vc, "model_config")
    sc = _g(vc, "scheduler_config")
    out = []

    # ── mm_processor_kwargs ──────────────────────────────────────────────
    es_mm = bool(_g(mc, "limit_mm_per_prompt", default=None)) or bool(
        _g(mc, "multimodal_config", default=None)
    )
    if es_mm and not _g(mc, "mm_processor_kwargs", default=None):
        gan = int(num_tokens * GANANCIA_MM_KWARGS)
        out.append("")
        out.append(f"  + ~{_miles(gan)} tokens  (" + f"{GANANCIA_MM_KWARGS * 100:.1f}".replace(".", ",") + "%)")
        out.append(
            "      --mm-processor-kwargs '{\"max_pixels\":2000000,\"min_pixels\":65536}'"
        )
        out.append(
            "    El profiler corre un dummy multimodal con el item MAS GRANDE"
        )
        out.append(
            "    permitido. Sin esto usa el max_pixels del modelo y reserva"
        )
        out.append(
            "    activaciones para una imagen enorme que nunca vas a mandar."
        )
        out.append("    Costo: ninguno si no mandas imagenes gigantes.")

    # ── max_num_batched_tokens ───────────────────────────────────────────
    mnbt = int(_g(sc, "max_num_batched_tokens", default=0) or 0)
    if mnbt > 4096:
        gan = int(num_tokens * GANANCIA_BATCHED_4K)
        out.append("")
        out.append(f"  + ~{_miles(gan)} tokens  (" + f"{GANANCIA_BATCHED_4K * 100:.1f}".replace(".", ",") + "%)")
        out.append(f"      --max-num-batched-tokens 4096   (ahora: {mnbt})")
        out.append(
            "    Achica el pico transitorio de las capas GDN, que escala lineal"
        )
        out.append("    con los tokens del batch (~12 KiB por token, por GPU).")
        out.append(
            "    Costo REAL: parte el prefill en chunks la mitad de grandes, o"
        )
        out.append("    sea que un prompt de 200k tarda mas en procesarse.")

    if not out:
        return []
    return [
        _titulo("COMO CONSEGUIR MAS KV"),
        "  Medido en este rig, ver docs/KV-OFFLOADING.md §9:",
    ] + out + [
        "",
        "  ⚠ Al subir el KV, revisa el margen de FlashInfer (RIESGOS): esos",
        "    394 MiB se alocan en el primer request, no en el arranque.",
    ]


# ─────────────────────────────────────────────────────────────────────────
# 4. armado
# ─────────────────────────────────────────────────────────────────────────


def construir_informe(vc, num_tokens: int) -> str:
    mem = _leer_memoria()
    lineas: list[str] = []
    lineas.append("")
    lineas.append("╔" + "═" * ANCHO + "╗")
    lineas.append("║" + "  GENESIS · ANALISIS DE ARRANQUE".ljust(ANCHO) + "║")
    lineas.append("╚" + "═" * ANCHO + "╝")

    for sec in (
        _sec_modelo(vc),
        _sec_vram(mem),
        _sec_capacidad(vc, num_tokens),
        _sec_offloading(vc, num_tokens),
        _sec_mtp(vc),
        _sec_riesgos(vc, mem),
        _sec_recomendaciones(vc, num_tokens),
    ):
        lineas.extend(sec)

    lineas.append("")
    lineas.append("─" * ANCHO)
    lineas.append(
        "  Re-ejecutar:  python3 -m vllm._genesis.analisis_arranque"
        "   ·   apagar: GENESIS_DISABLE_PN83=1"
    )
    lineas.append("")
    return "\n".join(lineas)


def emitir_informe(vllm_config, num_tokens: int) -> None:
    """Llamado desde kv_cache_utils una vez que se sabe el tamaño del KV."""
    try:
        if os.environ.get("GENESIS_DISABLE_PN83", "").lower() in ("1", "true", "yes"):
            return
        # el anchor puede ejecutarse mas de una vez por proceso (vLLM llama al
        # calculo del tamaño de KV mas de una vez en algunos arranques); el
        # informe tiene que salir UNA sola vez
        global _YA_EMITIDO
        if _YA_EMITIDO:
            return
        _YA_EMITIDO = True
        texto = construir_informe(vllm_config, int(num_tokens))
        # stderr y no logger: el logger prefija cada linea y rompe el formato
        print(texto, file=sys.stderr, flush=True)
        try:
            with open(ESTADO + ".txt", "w") as fh:
                fh.write(texto)
        except Exception:
            pass
    except Exception as e:  # nunca puede tumbar el arranque
        print(f"[Genesis PN83] no pude armar el analisis: {e}", file=sys.stderr)


def main() -> int:
    """Modo standalone: reimprime el informe del ultimo arranque."""
    try:
        with open(ESTADO + ".txt") as fh:
            sys.stdout.write(fh.read())
        return 0
    except FileNotFoundError:
        print(
            "No hay informe guardado. Se genera solo al arrancar el engine con "
            "PN83 activo (default ON, kill switch GENESIS_DISABLE_PN83=1).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
