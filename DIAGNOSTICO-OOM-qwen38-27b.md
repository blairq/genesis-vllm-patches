# Diagnóstico OOM — genesis-27b-qwen38-mtp (Qwen3.8-27B W8A16 MTP)

Fechas: 2026-08-14 / 2026-08-15
Rig: 2× RTX 3090 (24 GB, SM 8.6), TP=2, vLLM 0.23.0, `gpu-memory-utilization 0.975`

---

## 1. Resumen ejecutivo

El engine OOMea bajo carga concurrente con prompts grandes. Se investigó a fondo y
**hay dos sitios de OOM distintos**, no uno:

| # | Disparador | Sitio del fallo | Naturaleza |
|---|---|---|---|
| (a) | `max-num-seqs 12` | `chunk_gated_delta_rule_fwd_h` → `h = k.new_empty(B, NT, H, V, K)` | GDN / atención lineal — "Cliff 2" |
| (b) | `max-num-seqs 5` + prompts ~12k | grafo compilado por inductor (`torch/_inductor/output_code.py`) | FFN / SiluAndMul — "Cliff 1" |

**El primer diagnóstico culpaba a GDN para los dos casos y estaba equivocado para (b).**
La corrección salió de instrumentar el código y leer el traceback real.

En ninguno de los dos casos el KV cache es el límite: hay 345-357k tokens de KV libres
y los contextos reales son de 4-16k. Lo que falta es margen para **buffers transitorios**,
que viven fuera del pool que vLLM perfila con `--gpu-memory-utilization` y por eso no
aparecen en "Available KV cache memory".

---

## 2. Presupuesto de memoria medido (por GPU, de 23,56 GiB usables)

```
pesos del modelo        14,26 GiB   61%
KV cache                 6,34 GiB   27%   (357.335 tokens)
CUDA graphs              0,79 GiB    3%
contexto CUDA / NCCL    ~1,00 GiB    4%
─────────────────────────────────────
libre para transitorios ~1,16 GiB    5%
```

Ese ~1,16 GiB es todo el margen disponible. Los OOM ocurren pidiendo 40-272 MiB.

---

## 3. Fórmula de `h` (GDN) — verificada con instrumentación

Se insertó una sonda temporal en `chunk_delta_h.py:350` que registró 200 llamadas reales.

```
h_bytes_por_GPU = tokens_en_el_forward × 12,05 KiB

  = NT × H × V × K × 2 bytes
  con NT = T/64 (FLA_CHUNK_SIZE), H = 24, V = K = 128, fp16
```

**Datos crudos:**

```
T=1650  nseq=2  h=19,5 MiB   → 12,1 KiB/token
T=3120  nseq=1  h=36,8 MiB   → 12,1 KiB/token
T=3200  nseq=1  h=37,5 MiB   → 12,0 KiB/token
T=3760  nseq=2  h=44,2 MiB   → 12,0 KiB/token
```

**Dos conclusiones firmes:**

1. **`h` se asigna por BATCH COMPLETO, no por secuencia.** `T=1650` con `nseq=2`
   produce un solo `h` de 19,5 MiB. Esto refuta la hipótesis de "una asignación por
   secuencia".
2. **Los 48 heads GDN SÍ se reparten entre las 2 GPUs** (`H=24` observado, no 48).
   Duda que quedaba de la estimación previa: resuelta.

Con ~1,16 GiB de margen, el techo teórico solo para `h` es ~99.000 tokens por forward.
Hay más buffers en el mismo camino (`v_new` suma otros ~6 KiB/token).

---

## 4. Identificación del sitio (b) — el que rompe con `max-num-seqs 5`

Traceback real del OOM (no inferido):

```
File "torch/_inductor/output_code.py", line 656, in __call__
File "torch/_inductor/utils.py", line 3401, in run
File ".../inductor_cache/uz/cuzdm5b....py", line 1236, in call
OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB
```

**No es `chunk_delta_h.py`.** Es el grafo compilado por inductor.

Aritmética que lo identifica:

```
108 MiB / (intermediate_size 17408 × 2 bytes) = 3253 tokens
```

Y la sonda registró forwards de T=3120, 3200 y 3760 tokens. Coincide: el buffer es de
shape `(T, 17408)` en fp16 — un **intermedio del FFN (gate-up / SiluAndMul)**.

Esto es **Cliff 1**, no Cliff 2.

---

## 5. Estado real de los parches Genesis

### 5.1 El pin-gate está en UNKNOWN

```
[Genesis pin-gate] running vllm pin = 0.23.0
[Genesis pin-gate] allowlist (2 entries): ['0.20.1rc1.dev16+g7a1eb8ac2',
                                            '0.20.2rc1.dev9+g01d4d1ad3']
[WARNING] UNKNOWN — vllm pin '0.23.0' is NOT on the Genesis known-good list.
          To accept this pin, add it to KNOWN_GOOD_VLLM_PINS in guards.py
```

**Toda la suite corre contra una versión de vLLM que nunca se validó.** Ésta es la causa
común de toda la deriva de anclas de abajo.

### 5.2 Parches encendidos en el compose que NO se aplican

De 15 encendidos, **4 fallan en silencio** (el dispatcher loguea `APPLY` y después el
parcheo real falla):

| Parche | Motivo | Tipo de deriva |
|---|---|---|
| `PN50` GDN fused proj | `gdn_linear_attn.py not found` | ruta movida |
| `PN54` GDN contiguous dedup | `gdn_linear_attn.py not found` | ruta movida |
| `PN59` streaming GDN orchestrator | `required_anchor_missing` | ancla de texto |
| `P107` MTP truncation detector | `required_anchor_missing` | ancla de texto |

También `P28` (nunca pedido) falla por la misma ruta vieja.

**Sí se aplican (11):** P5b, P62, P61b, P68/P69, P100, PN8, PN51, PN56, PN57, PN66, PN77.

### 5.3 Causa de la deriva de ruta

En vLLM 0.23 el archivo se movió y se especializó por modelo:

```
buscan:  vllm/model_executor/layers/fla/ops/gdn_linear_attn.py       (no existe)
está en: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
```

### 5.4 Causa de la deriva de ancla en PN59 — localizada a UNA línea

```python
# ANCHOR_OLD que espera PN59 (patch_N59_streaming_gdn.py):
    chunk_offsets: torch.Tensor | None = None,
):

# lo que tiene vLLM 0.23 (fla/ops/chunk.py):
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,   # ← agregado en 0.23
):
```

Arreglarlo requiere además decidir qué hacer con `core_attn_out` en el camino de
streaming (lo seguro: si viene distinto de `None`, caer a vanilla).

---

## 6. Por qué la familia "Cliff 2" NO resuelve este caso

Los tres parches anti-OOM de GDN comparten una suposición de diseño: **solo actúan en
prefill de secuencia única**. Verificado leyendo el código de cada uno:

```
PN32  →  "single-sequence prefill ... multi-sequence bypasses to original"
PN59  →  is_single_seq = (cu_seqlens is None or cu_seqlens.shape == (2,))  →  bypass
P103  →  "cu_seqlens is not None (variable-length batches don't trigger Cliff 2)"
```

**El OOM medido acá es con batch multi-secuencia**, así que los tres se saltean.

> ⚠️ Hallazgo reportable aguas arriba: la premisa *"variable-length batches don't trigger
> Cliff 2"* es falsa en este rig. El OOM (a) con `max-num-seqs 12` ocurrió justamente en
> un batch multi-secuencia, dentro de `chunk_gated_delta_rule_fwd_h`.

---

## 7. Candidatos correctos, sin aplicar todavía

Del barrido completo del registry (127 entradas), filtrando por VRAM y descartando lo
gateado a TurboQuant (`PN31`, `P98`, `P101`) y a MoE (`P37` — este modelo **no** es MoE):

| Parche | Categoría | Por qué aplica |
|---|---|---|
| **PN25** SiluAndMul opaque-op pool | `memory_savings`, sin gate | *"137,6 MiB transitorios en una 3090 con ~131 MiB libres"*; reduce *"~4,7-18 GiB a un buffer pooleado (~73-285 MiB)"*. Ataca el sitio (b). |
| **PN12** FFN intermediate scratch pool | `memory_savings` | *"73-285 MiB por capa × 64 capas = 4,7-18 GiB"*. Su docstring abre con un OOM idéntico: *"Tried to allocate 138.00 MiB. GPU 0 has 122.56 MiB free"*. Ataca el sitio (b). |
| **PN19** Scoped max_split_size_mb en carga | `memory_savings`, siempre aplicable CUDA | *"deja 200-500 MiB inutilizables"* por fragmentación al cargar. |
| **P95** Marlin TP cudagraph cap en Ampere | `stability` | Rig es Ampere + TP=2 + Marlin. Sin ahorro documentado; falta leerlo. |
| **P84 / P85** prefix cache híbrido | `kv_cache` | No ahorran VRAM. P85: `MambaManager.cache_blocks` retorna temprano para prompts < block_size ⇒ el prefix cache puede estar rindiendo menos de lo observado (70% hit). |

**PN25 y PN12 son los que corresponden al sitio (b) confirmado por traceback.**

---

## 8. Sobre `PYTORCH_CUDA_ALLOC_CONF`

Valor actual: `expandable_segments:True,max_split_size_mb:512`

Verificado dentro del contenedor con `torch.cuda.memory_snapshot()`:
`is_expandable: True`, 1/1 segmentos expandibles. **`expandable_segments` está activo y
funcionando.**

`max_split_size_mb:512` es **redundante**: gobierna el troceo del allocator legacy, un
mecanismo que `expandable_segments` reemplaza. No hace daño (comprobado que no lo
desactiva) pero es ruido. PN19 lo maneja con alcance acotado a la carga del modelo.

---

## 9. Cambios aplicados durante la investigación

- `restart: unless-stopped` **removido** del compose. Con esa política un crash por OOM
  se relevantaba solo y el contenedor aparecía `Up` a los segundos — indistinguible de no
  haber muerto, lo que hizo imposible saber si un barrido había roto algo.
- Detector de reinicio confiable documentado en el compose:
  ```
  curl -s :8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" | grep process_start_time_seconds
  ```
  Si ese valor cambia, el proceso se reinició y los acumulativos `vllm:*` volvieron a cero.
- Comentario de `--max-num-seqs` corregido con el diagnóstico real y la fórmula.
- `--cudagraph-capture-sizes` revertido de 48 a 40 (las 2 tallas extra costaban +0,26 GiB).
- Sonda de instrumentación **removida**; `chunk_delta_h.py` restaurado desde
  `chunk_delta_h.py.genesis-bak` (el backup queda dentro del contenedor).

---

## 10. Throughput medido (para dimensionar los lotes de fan-out)

```
1 secuencia    ->  88 tok/s   TTFT 1,46s   waiting 0
5 secuencias   -> 308 tok/s   (3,5x)       waiting 0, 0 fallos
12 secuencias  -> el engine muere
```

Con 5 secuencias y prompts de ~1,4k anda bien. Con ~12k OOMea.

---

## 11. RESOLUCIÓN (2026-08-15)

### 11.1 La causa real: `max-num-seqs 5` era exactamente uno de más

Escalado con el **payload real** de un subagente de opencode (5.390 tokens, 8 tools):

```
1 concurrente  -> OK
2 concurrentes -> OK
3 concurrentes -> OK
4 concurrentes -> OK   (4/4, 8s)
5 concurrentes -> 0/5, HTTP 500   <-- frontera exacta
```

La frontera coincide con el `--max-num-seqs 5` configurado: cuando los 5 slots se
llenan a la vez, el prefill simultáneo agota el margen. Por eso el operador "nunca
tuvo problemas": el fan-out real rara vez dispara exactamente 5 prefills grandes en
el mismo instante.

### 11.2 El fix: `--max-num-seqs 4`

Con 4, vLLM **nunca puede agendar** el 5º prefill simultáneo que revienta. El cliente
puede mandar los que quiera: se encolan.

Verificado con el payload real:

```
5 concurrentes -> 5/5 OK   (16s)
6 concurrentes -> 6/6 OK   (12s)
8 concurrentes -> 8/8 OK   (19s)
```

Cero fallos. El encolado absorbe el exceso.

### 11.3 Ganancias acumuladas de headroom (medidas con PN80)

| Configuración | VRAM libre en el pico | Proyección |
|---|---|---|
| baseline (0.975, sin pools) | 27 MiB | 2.299 tok/forward |
| + PN25 + PN12 (pools de FFN) | 47 MiB | 3.974 tok/forward |
| + `gpu-memory-utilization 0.97` | 67 MiB | 5.693 tok/forward |

Cada palanca sumó ~20 MiB. Ninguna alcanzaba sola; el fix estructural fue `max-num-seqs 4`.

### 11.4 Barrido completo de parches — qué NO dio más VRAM

Revisadas las 127 entradas del registry buscando ahorro de VRAM:

| Parche | Ahorro que reclama | Veredicto |
|---|---|---|
| `PN19` scoped max_split | 200-500 MiB | **Ya está en vLLM 0.23** (`upstream_merged`, marker `_scoped_allocator_max_split` presente). Sin ganancia disponible. |
| `P38` TQ continuation workspace | 13-25 GB | TurboQuant. No aplica (usamos `fp8_e4m3`). |
| `P51` dequant buffer | 516 MiB | Ya automático al no ser TurboQuant. |
| `P37` MoE intermediate cache | 553 MiB | **El modelo no es MoE.** No aplica. |
| `PN35` inputs_embeds text-only | 64 MiB | **NO-OP en modelos multimodales** — el nuestro tiene torre de visión. |
| `PN62` text-only ViT scratch | 3-5 GiB (predicho) | `MARKER-ONLY — real hook pending`. No implementado. |
| `PN32`/`PN59`/`P103` (Cliff 2) | — | Solo prefill de **secuencia única**; hacen bypass en batch multi-secuencia. |
| `PN8` MTP draft quant (600 MiB), `PN77` FP8 LM head (1,2 GiB) | — | **Ya aplicados** desde antes. |

**Torre de visión:** ~420M params ≈ 0,78 GiB (0,39 GiB/GPU con TP=2). Es la única
reserva grande que se podría liberar, pero requiere renunciar a `agi_vision_inspector`.
`--limit-mm-per-prompt '{"image":1,"video":0}'` se aplicó y dio poco: el budget del
encoder cache sigue fijo en 16.384 tokens (no lo domina el video). KV: 350.013 →
351.477 tokens.

### 11.5 PN80 — parche nuevo de observabilidad

Se creó `PN80 GDN h budget probe` (`wiring/hybrid/patch_N80_gdn_h_budget_probe.py`,
registrado en `dispatcher.py` y `apply_all.py`). Emite desde el punto exacto de la
asignación:

```
[PN80] h alloc: T=6820 tok (nseq=3, B=1, NT=107) | shape=(1,107,24,128,128) float16
       | h=80.2 MiB/GPU = 12.05 KiB/tok | VRAM libre=67.0 MiB
       -> proyeccion: ~5693 tok/forward antes de agotar POR h (margen 0.8x)
       <-- MARGEN BAJO (<1.5x)
```

Fue la herramienta que permitió medir cada palanca en la misma unidad y ver la
condición de OOM **antes** del crash. Tunables: `GENESIS_PN80_EVERY`,
`GENESIS_PN80_MIN_T`, `GENESIS_PN80_WARN_RATIO`.

---

## 12. Próximos pasos sugeridos

1. **Probar PN25 + PN12** (opt-in, sin gate, atacan el sitio (b) confirmado). Medir con
   la misma carga que hoy rompe: 5 concurrentes × ~12k tokens.
2. **Arreglar la deriva de PN50/PN54** (reapuntar ruta a `mamba/gdn/qwen_gdn_linear_attn.py`).
3. **Arreglar la deriva de PN59/P107** (actualizar anclas a las firmas de 0.23).
4. **Evaluar agregar `0.23.0` a `KNOWN_GOOD_VLLM_PINS`** en `guards.py` con la validación
   documentada, que es lo que el propio guard pide.
5. **Reportar aguas arriba** que la premisa "variable-length batches don't trigger Cliff 2"
   no se sostiene en este rig.
