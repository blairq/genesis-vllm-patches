# Traspaso: perfil fino del paso de idiotSavant y dónde queda margen (25-09-2026)

Documento para arrancar una sesión nueva. Resume las mediciones del 25-09 kernel por kernel y capa por capa
sobre el stack que sirve hoy (idiotSavant + DFlash2 + árbol), el margen de cada palanca y cómo reproducir
todo para medir si un cambio mejora.

- Informe visual (tablas, barras, tipos de dato): https://claude.ai/artifact/3SYMzZ8Pr233XH1VU9biAT (secciones
  "El paso medido…" y "Dónde queda margen"). Fuente: `docs/informe-tipos-sm86-2026-09-25.html`.
- Commit del perfil: `fdddf65`.
- **Este es un servidor de desarrollo, no hay producción:** se pueden bajar y recrear contenedores para medir.
- **Regla de esta etapa: todo se vuelve a medir sobre idiotSavant.** Los resultados obtenidos con noon (PN135,
  PN136, el umbral de PN120, etc.) son hipótesis, no conclusiones. idiotSavant tiene el residuo rotado (cresta
  ~2,7 contra ~56 de noon) y la norma plegada, así que cambia lo que el int8 aguanta y lo que cuesta cada fusión.

---

## 1. El stack medido

| | |
|---|---|
| Modelo | `qwen3.8_27b_idiotSavant_sm_86`: Qwen3.5/3.8-27B híbrido, 48 capas GDN + 16 de atención (en la posición 3 de cada 4). Residuo rotado (Hadamard 1024), GPTQ W4 g128 |
| Borrador | `qwen3.8_27b_idiotSavant_sm_86_dflash2`: 5 capas, W4A16, K=8, árbol de 8 nodos (9 filas por pedido y paso) |
| Servir | vLLM 0.29.0, TP=2 en 2× RTX 3090 capadas a 220 W (el SM baja a ~810 MHz sostenido) |
| Lineales | Marlin W4A8 (`VLLM_MARLIN_INPUT_DTYPE=int8`) + PN130; down_proj vía SK-23 (PN148 fusionado) |
| Atención | PN131 (SK-18h) sobre TRITON_ATTN, KV int8 por token-cabeza, PN126 (Hadamard q/k) |
| GDN | PN122 (cinta de rollback), kernels del árbol `_k_spec_arbol` / `_k_escribir` / `_k_salidas` |
| All-reduce | decode: NCCL LL fp16; prefill: PN120 int8 (AllGather de int8, grupo 64, M≥512) |
| Compose | `compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml` (contenedor `genesis-27b-idiotsavant`, puerto 8360) |

Configuración: Qwen hidden 5120, 24 cabezas q / 4 kv, head_dim 256, inter 17408, vocab 248320;
GDN con 16 cabezas k / 48 v de 128. Borrador: 32 q / 8 kv × 128, 5 capas, target_layer_ids [5,19,33,47,61].
Por rango (TP=2): 2 cabezas kv, 12 q, 24 cabezas v de GDN y medio vocabulario (124k).

---

## 2. Herramientas (para reproducir y comparar)

### 2.1 Captura: `tests/bench/medicion/perfil_idiotsavant.sh <etiqueta>`

- Usa el compose de idiotSavant **más** `compose/ov-aislado.yml`: contenedor `genesis-27b-pruebas`, puerto
  8361, red propia, sin tráfico ajeno. Arranca con `--profiler-config` a partir de las variables
  PROF_KIND, PROF_DIR, PROF_DELAY y PROF_MAX.
- **Baja `genesis-27b-idiotsavant`** y, como el override tiene el mismo servicio, al terminar **lo borra**. No lo
  vuelve a levantar. Después de medir hay que correr:
  `cd compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d`
- Tres fases, cada una con un arranque limpio:

| Fase | Carga | Profiler |
|---|---|---|
| `decode1` | 1 pedido: ~62k de prompt (fuentes de `_genesis`) ya en la caché de prefijos, 400 tokens, temp 0, sin thinking | PROF_DELAY=8, PROF_MAX=60 pasos |
| `decode4` | 4 pedidos concurrentes de ~14–17k, 400 tokens cada uno | ídem |
| `prefill` | prefill frío de 15,4k (chunks 1760), 1 token | PROF_DELAY=0, PROF_MAX=12 |

- Correr desacoplado, porque tarda ~10 min:
  `setsid nohup bash tests/bench/medicion/perfil_idiotsavant.sh is26 > tests/bench/medicion/perfil_is26.log 2>&1 &`
  Termina cuando el log dice `LISTO`.
- Las trazas quedan en `tests/bench/medicion/trazas/<etiqueta>_<fase>/rank{0,1}.*.pt.trace.json.gz`, de ~9 MB
  cada una. El dueño es root, así que no se puede escribir en esa carpeta.
- Para probar un cambio con flags, agregá las variables al override o a un `ov-*.yml` extra.
  - Las variables del compose **no pasan si no están declaradas en el `environment:`**.
  - Para parches de nivel modelo, usá `--force-recreate` y borrá la caché de torch.compile: si no, el parche
    queda inerte (memoria: parches-inertes-por-cache-de-torch-compile).

### 2.2 Análisis: `tests/bench/medicion/analizar_perfil.py`

```bash
cd tests/bench/medicion
python3 analizar_perfil.py trazas/is25_decode1 0            # decode, rango 0 (o 1)
python3 analizar_perfil.py trazas/is25_decode4 0
python3 analizar_perfil.py --prefill trazas/is25_prefill 0
python3 analizar_perfil.py --comparar analisis_perfil/is25_decode1_rank0.json analisis_perfil/is26_decode1_rank0.json
```

- **Cómo corta el paso de decode:**
  - Ancla: SK-23 (`sk23_silu_had_q8`) corre una vez por capa del target y el borrador no la usa. Cada 64 SK-23
    forman un forward.
  - El forward arranca en el último `vocab_parallel_embedding` antes de la SK-23 de la capa 0.
  - Cada capa termina en el primer all-reduce después de su SK-23.
  - Fuera de las capas:
    - `lm_head + gather`: hasta el primer AllGather después del primer Marlin fp16;
    - `verificación`: hasta el primer `_per_token_quant_int8`, que es la fc del borrador;
    - `borrador: capas`: hasta el segundo Marlin fp16;
    - `borrador: lm_head + top-k`;
    - `árbol + preparar paso`: hasta el embedding siguiente.
  - Una capa es de atención si contiene `sk18*`.
  - Se descartan el primer paso, los pasos con otra cantidad de filas (la moda de la grilla de SK-23) y los
    que no tienen esa forma (pasos mixtos con prefill).
- **Qué reporta:**
  - por región: pared, GPU (unión de intervalos) y GPU ociosa;
  - por capa: pared;
  - kernel por kernel por región: µs por paso o por capa, lanzamientos y tipo de dato;
  - la secuencia completa de una capa de cada tipo.
  - Lo guarda en `analisis_perfil/<traza>_rank<r>.json`.
- **Prefill:** categorías sobre toda la traza (GEMM, all-reduce int8 con su (des)cuantización, atención, GDN
  chunk, resto, ociosa) y el tiempo de pared por chunk.
- **Limitaciones:**
  - la clasificación va por nombre de kernel, así que un kernel nuevo cae en "otros": agregalo a `corto()` y `TIPO`;
  - la categoría del all-reduce int8 en prefill reconoce los triton de (des)cuantizar por su grilla 4400/8800,
    que solo vale con chunks de 1760;
  - la duración de un kernel de NCCL **incluye la espera al otro rango**. Para separar espera de
    transferencia, compará los dos rangos: hoy dan igual (0,1%).
- Para inspeccionar una frontera de pasos a mano, sirve el snippet `corto.py` de la sesión (nombres cortos
  más secuencia con ts, dur y stream). Está resumido en la función `corto()` del analizador.

### 2.3 Medición punta a punta (la que decide)

El perfil dice dónde está el tiempo; la ganancia real se mide con carga. Hay que usar las herramientas de
siempre y las trampas de memoria:
- varios arranques por brazo (medir-arranques-repetidos-no-uno);
- el cap hace derivar el reloj un 5% (cap-de-potencia-costo-y-deriva);
- `tok_s_mediana` engaña entre modelos: usar totales.

Para calidad, el oráculo es la generación larga (fidelidad sobre las respuestas, no sobre el prompt), y
cualquier cambio de kernel se verifica **bit a bit** contra el camino viejo antes de medir velocidad.

---

## 3. Línea base (25-09, trazas `is25_*`)

### 3.1 Decode

| | 1 pedido · 62k · 9 filas | 4 pedidos · ~15k · 36 filas |
|---|---|---|
| Paso (pared) | **29,7 ms** | **41,2 ms** |
| GPU ocupada | 92% | 95% |
| GEMM del target (Marlin W4A8) | 8,97 ms · 30,2% | 10,83 ms · 26,3% |
| Atención SK-18h | 5,44 ms · 18,3% | 7,11 ms · 17,3% |
| All-reduce del target | 3,46 ms · 11,7% | 7,91 ms · 19,2% |
| GDN: recurrencia del árbol | 3,01 ms · 10,1% | 4,34 ms · 10,5% |
| GDN: in_proj_a/b y copias | 0,61 ms · 2,0% | 0,87 ms · 2,1% |
| Normas, cuantización, fusiones | 1,74 ms · 5,9% | 1,95 ms · 4,7% |
| lm_head + gather de logits | 0,64 ms · 2,1% | 1,83 ms · 4,4% |
| Borrador DFlash2 (GPU) | 3,04 ms · 10,2% | 4,06 ms · 9,9% |
| Verificación + árbol (GPU) | 0,32 ms · 1,1% | 0,41 ms · 1,0% |
| GPU ociosa | 2,48 ms · 8,4% | 2,03 ms · 4,9% |

Las capas son planas: GDN 297–303 µs y atención 550–559 µs con 1 pedido; la capa 0 es más larga (466 µs).
Los rangos dan lo mismo (paso 29.670 contra 29.682 µs).

**Una capa GDN, 1 pedido, 300 µs:**

| # | Kernel | µs | Nota |
|---|---|---|---|
| 1 | norma | 3,5 | |
| 2 | int8 | 2,1 | |
| 3 | in_proj_a/b (cutlass fp16) | 3,8 + 2,4 | + split-K |
| 4 | in_proj_qkvz (Marlin, 21 MB) | 33,4 | techo |
| 5 | `_k_salidas` (grilla 1×5) | 9,0 | |
| 6 | 3 copias de torch | 6,6 | |
| 7 | `_k_spec_arbol` (grilla 4×24, 128 registros) | 18,7 | |
| 8 | `_k_escribir` (**grilla 1×1**) | 34,7 | |
| 9 | norma con compuerta + int8 | 5,2 | |
| 10 | out_proj | 15,6 | |
| 11 | all-reduce | 46,0 | |
| 12 | norma + int8 | 6,5 | |
| 13 | gate_up (44,6 MB) | 60,5 | 737 GB/s, techo |
| 14 | SK-23 | 5,0 | |
| 15 | down | 32,4 | |
| 16 | all-reduce | 27,1 | |

**Una capa de atención, 1 pedido, 554 µs:**

| # | Kernel | µs | Nota |
|---|---|---|---|
| 1 | norma + int8 | 7 | |
| 2 | qkv | 28,0 | |
| 3 | `_fused_qk_rmsnorm_rope_gate` | 2,9 | |
| 4 | `sk18h_escribir2` + `prep2` | 7,6 | |
| 5 | **`sk18h_batch2`** | **284** | grilla 298×4, 128 hilos, 254 registros, 77 KB de memoria compartida: 1 bloque de 4 warps por SM |
| 6 | `sk18h_union4` | 30 | |
| 7 | `sk18h_salida` | 16,5 | grilla 9×12 de **1 warp** |
| 8 | o_proj | 16 | |
| 9 | all-reduce | 47,9 | |
| 10 | MLP | 104,6 | |
| 11 | all-reduce | 27,2 | |

Con 4 pedidos, `sk18h_salida` sube a **84,5 µs** (grilla 36×12 de 1 warp), `union4` a 49 y `batch2` a 299
(grilla 298×16).

**Fuera de las capas, 1 pedido / 4 pedidos (µs de pared):**

| Región | 1 pedido | 4 pedidos | Qué hay |
|---|---|---|---|
| lm_head + gather | 644 | 1.841 | Marlin W4A16: 402 µs en el techo con 9 filas, 972 con 36 (HMMA fp16 limita por cómputo). AllGather de los logits completos: 220 y 844 µs |
| Verificación | 453 | 556 | 82–156 µs de GPU; el resto es Python |
| Borrador: capas | 3.306 | 3.992 | detalle abajo |
| Borrador: lm_head + top-k | 468 | 796 | lm_head entero 386–674 µs para 16 candidatos por posición |
| Árbol + preparar el paso | 964 | 489 | ~110 kernels de torch de 1–3 µs, 235 µs de GPU |

Detalle de `borrador: capas`:
- `kernel_unified_attention` Triton: 5 × 203 µs = 1.014 µs, **grilla 3×4 = 12 bloques** en 82 SM, sin partir la KV;
- Marlin: 688;
- all-reduce: 320 / 695;
- GEMM **fp16 sin cuantizar**: 298, entre ellos uno de 70 µs después de `_prepare_dflash_inputs`, que es la
  proyección K/V del contexto;
- ociosa: 734.

### 3.2 Prefill (15,4k frío, chunks 1760; con profiler: 2.730 tok/s)

| Categoría | ms | % |
|---|---|---|
| GEMM Marlin W4A8 | 2.740 | 48,6 |
| All-reduce int8 (PN120: AllGather + (des)cuantizar y sumar) | 1.255 | 22,3 |
| Atención (FlashInfer BatchPrefill fp16 + `sk18h_decuant` + qk_norm + fwht) | 656 | 11,6 |
| Normas, cuantización, fusiones, resto | 369 | 6,5 |
| GPU ociosa | 362 | 6,4 |
| GDN chunk (fla) | 251 | 4,5 |

- **Por chunk de 1760:** 575–650 ms.
- **GEMM:** gate_up son 3 lanzamientos de Marlin, 1464 + 1013 + 70 µs, 314 GOP en 2,55 ms, es decir 123 TOPS:
  **~90% del pico int8 a 810 MHz**.
- **All-reduce int8:** dos por capa de ~0,88–0,95 ms en los dos rangos. Son 9 MB a ~10,3 GB/s: es
  transferencia, no espera.
- **Atención:** FlashInfer, 3,0 ms por capa a ~8k de posición media.
- **Preparación de la atención:** `torch.unique` / sort de páginas en `sk18_attn.py` (líneas 723 y 759),
  ~0,5 ms por capa de atención con la GPU parada.
- **Pasos mixtos** (prefill de un pedido con otros decodificando, visto en decode4): `sk18h_decuant`, con
  grilla (70, 880), tarda **0,6–1,3 ms por capa** de atención, **10–20 ms por paso**. En un prefill solo son 0,14 ms.

---

## 3b. Hecho el 27-09 (después del P1): Python a la GPU, pasos mixtos, re-mediciones

- **PN150:** fc y contexto del borrador en grafos CUDA. **PN151:** metadata del GDN en un kernel, verificada en 3000
  llamadas. **Sincronización oculta** en `compactar_kernel`: `torch.tensor(list, device=cuda)` es sincrónico y dejaba
  la CPU bloqueada 19,8 de 26 ms por paso.
  - GPU ociosa entre fases con 1 pedido: 1.719 → 3 µs por paso.
  - Paso acumulado del día: 29,36 → 25,67 ms (−12,6%) con 1 pedido; 40,94 → 38,31 ms (−6,4%) con 4.
  - Herramientas: `huecos_cpu.py`, `huecos_py.py` (con `PROF_STACK=true`).
- **Pasos mixtos** (`GENESIS_PN131_MIXTO=1`, prendido): los decodes van por SK-18h y solo el prefill por descuantizar +
  FlashInfer. Descuantizar 147 → 55 ms en la fase decode4; prefill idéntico; decode 0,4–1,8% (la diferencia normal de
  SK-18h). Herramienta: `pasos_mixtos.py`.
- **PN135 NUNCA estuvo cableado.** Su "+0,4% / +0,7%" era del kernel aislado. Cablearlo es el patrón de SK-23: buffers
  int8 fijos en la capa. En las capas GDN la norma también alimenta `in_proj_a/b` en fp16. Pendiente.
- **PN136 inerte con idiotSavant:** se aplica el parche de texto pero el MLP lo reemplaza PN148/SK-23, y nunca
  entra ("MLP partido" no aparece). TTFT igual en 2 × 2 corridas (fase `ttft`, prompts fríos con nonce). Para
  medirlo hay que integrar el solape en `rot_down.py`.
- **Literatura para el all-reduce, traducida a SM86 (2 × 3090, PCIe 4.0 x8 ~12 GB/s, con P2P, sin NVLink/NVSwitch):**
  - **SiFAR** (arXiv 2607.08973), medido en H200/NVSwitch: `multimem.ld_reduce` NO aplica. El doble buffer (sin
    barrera final) y las banderas especulativas SÍ, por P2P con escrituras remotas.
  - **TokenWeave** (2505.11329, Hopper): partir el lote en dos y fusionar AR + RMSNorm. En Ampere, sin multimem.
  - **ISO** (2409.11155): solapar dentro de la secuencia.
  - **NanoFlow**: nano-lotes por stream.
- **Disco lleno el 27-09:** el tier de KV (40 GB) más 49 GB de capturas del borrador y 6 variantes de 3,6 GB en
  `trazas/`. Se vació el tier de KV. Las capturas son datos de investigación: preguntar antes de borrar.

## 4. Palancas, en el orden sugerido

Ahorro = medido menos el techo del kernel: **es un máximo**. Los porcentajes son del paso (1 pedido / 4 pedidos).
Todas deben dar la **misma salida** (bit a bit o dentro del ruido fp16): verificarlo antes de medir velocidad.

### ✅ P1 HECHO (27-09): cinta y conv del árbol en PTX — paso −5,2% (1 pedido), −3,0% (4 pedidos)

**Resultado**

| | 1 pedido | 4 pedidos |
|---|---|---|
| `_k_escribir` → `pn122_cinta` | 33,6 → 2,2 µs | 35,5 → 2,7 µs |
| `_k_salidas` → `arbol_conv` | 9,0 → 3,9 µs | 9,3 → 5,1 µs |
| Capas GDN | −11% | −7% |
| **Paso** | **−5,2%** (perfil `is27_ptx2` contra `is27_base`) | **−3,0%** |

- Ruido entre arranques (`is25` contra `is27_base`): capas ±0,4%, paso ±1%.
- **Calidad:** bit a bit en 28 + 13 casos offline, y el oráculo greedy (`tests/bench/medicion/oraculo_greedy.sh`) dio el
  **mismo texto** en 3 respuestas (una de 7063 caracteres) y la **misma aceptación** (832 borradores, 2406 aceptados).
- **Prendido por omisión** en el compose: `GENESIS_PN122_ESCRIBIR_PAR=ptx`, `GENESIS_ARBOL_SALIDAS_PAR=ptx`.

**Código**
- `kernels/cuda/pn122_cinta.cu` y `kernels/cuda/arbol_conv.cu`, lanzados por `ptx_lab` (que ahora acepta grillas 3D).
- Respaldo Triton, con el motivo en el log si el PTX no aplica: `_k_escribir_par` y `_k_salidas_par`.
- Pruebas: `tests/proto/pn122_escribir_par.py`, `arbol_salidas_par.py`, `arbol_conv_ptx_barrido.py`.

**Método aplicado** (pedido del usuario: Triton para prototipar, después PTX revisado a mano)
1. Leer el SASS de lo que genera Triton. En la cinta: 128 hilos con cargas de 2 bytes, dos `bar.sync` por la
   reducción, los escalares calculados por los 128 hilos y `slots` cargado después del chequeo de salida.
2. Reescribir en CUDA con **toda la aritmética en PTX explícito**. `ptx_lab` compila con `--use_fast_math`, y eso
   cambiaría los bits. Mismo orden de reducción que Triton (el árbol del butterfly), el `log` de libdevice copiado del
   PTX de Triton, `div.full` y `ex2.approx`.
3. Mirar el SASS propio y corregir:
   - `shfl.sync` dentro de un `if` generó 14 `CALL` de respaldo: se sacó la rama;
   - la conv con `if (i >= 0)` por fila tenía 76 `BRA` y era más lenta que Triton: cargas sin ramas;
   - **la palanca grande: dos latencias de memoria en vez de tres.** Las filas x de los 9 tokens se cargan junto con
     `anc3` y los ancestros se eligen de registros. Triton no puede hacerlo con su estructura.

**Trampas que aparecieron**
- `anc3` es un buffer FIJO de `n_slots·(K+1)` filas (grafos CUDA), no del largo del lote. `T = anc3.shape[0] // N`
  dio cientos de tokens y 122 µs por capa. Ahora sale de `gdn_cinta.tokens_arbol()`.
  **Las pruebas offline tienen que usar los buffers con la forma del servidor.**
- En el servidor los PTX caían al respaldo: `A_log`/`dt_bias` no venían en fp32 y el estado conv tenía otro layout.
  Solución: copia fp32 exacta y creada fuera de la captura del grafo, strides genéricos, y el motivo al log.
  `perfil_idiotsavant.sh` ahora guarda esos avisos en `analisis_perfil/<etiqueta>_<fase>.log`.

### ✅ P1-bis HECHO (27-09): recurrencia del árbol en PTX y sin copias — paso acumulado −7,0% (1 pedido), −4,7% (4)

Perfil `is27_arbol22` contra `is27_base`. Las tres piezas juntas: cinta, conv y recurrencia en PTX, más sin las copias.

| | 1 pedido | 4 pedidos |
|---|---|---|
| `_k_spec_arbol` → `gdn_arbol.cu` | 19,7 → 11,7 µs | 44,6 → 27,4 µs |
| Copias de q/k/v | 6,4 → 0 µs | 9,9 → 3,1 µs |
| Capas GDN | −15,6% | −11,8% |
| **Paso** | **−7,0%** | **−4,7%** |

**Qué hacía Triton** (TTGIR): el estado [32, 128] repartido por COLUMNAS entre los 128 hilos. Cada h·k y h·q era una
reducción entre 4 warps por shared con barrera, dos por token.

**Qué hace el PTX**
- **Layout:** cada warp es dueño de 4 filas enteras (`RPW=4`, barrido contra 8 y 2) y reduce dentro del warp.
- **Una reducción por token:** h·k y h·q juntas, con `o = e·(h·q) + d·(k·q)`. Es reduce-scatter; el decaimiento
  queda fuera del camino crítico.
- **Prólogo:** normas, g, beta y filas de cinta, con todas las cargas primero.
- **Máscaras del árbol** en shared, leídas en dirección uniforme. Sin eso, 71 CALL de respaldo de `shfl.sync`.
- **ncu:** issue 36% contra 20% de Triton. Lo que queda son 3 latencias de memoria encadenadas en el prólogo y los 14
  pasos secuenciales de la regla delta.

**Validación**
- **NO es bit a bit** con Triton, porque cambia el orden de las sumas. Error contra fp64 igual al de Triton en 5
  casos (`tests/proto/gdn_arbol_ptx.py`).
- **Oráculo:** el texto diverge, la aceptación queda igual o mejor (2,96 contra 2,89 aceptados por borrador).

**Copias:** eran de PN54 (`rearrange_mixed_qkv`), no nuestras. Ahora, con el PTX del árbol activo, devuelve vistas con
stride (`gdn_cinta.qkv_con_stride()`), y la cinta PTX también lee con stride. Control: quitar las copias dio el
MISMO texto del oráculo.

**Flags** (prendidos por omisión): `GENESIS_PN122_ARBOL_PTX=1`. Si el PTX no aplica, cae a Triton con el motivo en el log.

**Pendiente**
- Con 4 pedidos quedan 3,1 µs de copias por capa: identificar de dónde salen.
- `perfil_idiotsavant.sh` se reescribió: un solo arranque para las fases de decode y el oráculo (`ORACULO=1`),
  `/health` cada 5 s. Un brazo del A/B pasa de ~12 a ~5 min.

### P1 · Recurrencia del árbol GDN: `_k_escribir`, `_k_salidas`, `_k_spec_arbol` — **−7% / −7%, costo bajo**

- **Dónde:** `vllm/_genesis/gdn_cinta.py`
  - `_k_escribir`: kernel en la l. ~795, lanzamiento en la l. ~948 con grilla `(N,)`;
  - `_k_spec_arbol`: l. 415, lanzamiento en la l. 935 con grilla `(cdiv(V,BV), N*HV)`.
- `_k_salidas`: `vllm/_genesis/arbol_conv.py`, l. 31 y lanzamiento en la l. 107 con grilla `(N, cdiv(dim,BN))`.
- **Qué pasa:** `_k_escribir` es **un solo programa por pedido** que recorre en serie `for t in 1..T-1` y
  `for hh in 0..H` y escribe ~33 KB en 34 µs. Es pura latencia. Cada fila `(slot, t)` de la cinta es independiente.
- **Hacer:**
  - `_k_escribir`: grilla `(N, T-1)`, o `(N, T-1, H+1)` con las cabezas k y el bloque v/a/b separados;
  - `_k_salidas`: más bloques;
  - `_k_spec_arbol`: 45 µs con 4 pedidos, ocupación 10%, 128 registros. Revisar BV y cuántos programas.
- **Esperado:** 34 → ~3 µs, 9 → ~3 y 19–45 → ~10–25, es decir −2,2 ms / −2,8 ms por paso.
- **Verificar:** la cinta y las salidas bit a bit contra el kernel actual con entradas reales. Después, el
  oráculo de generación larga (PN122 + DFlash2 degeneraba al cruzar el bloque de 880; ver memoria
  pn122-incompatible-con-dflash2-runner-v2).
- **Ojo:** el docstring explica que `_k_escribir` va en un lanzamiento aparte porque dentro de `_k_spec`
  había una carrera. Paralelizarlo no reintroduce eso mientras siga separado.

### P2 · Atención de decode SK-18h — **−9% / −10%, costo medio-alto**

- **Dónde:**
  - `vllm/_genesis/kernels/cuda/sk18h_batch2.cu`, `sk18h_union4` (en el mismo directorio) y `sk18h_salida.cu`;
  - el lanzamiento y la selección de kernels están en `vllm/_genesis/sk18_attn.py` (`_kernels`, `Kernel(...)`).
- **Qué pasa:**
  - `batch2` lee 63 MB de KV int8 por capa y rango (62k × 2 cabezas × 256 × K y V) en 284 µs: **224 GB/s, 35%**
    del techo sostenido (~640 GB/s con el cap);
  - 254 registros y 77 KB de memoria compartida dejan **1 bloque de 4 warps por SM**. Memoria
    ancho-de-banda-es-warps-por-sm: el ancho de banda lo fijan los warps en vuelo por SM;
  - `salida` usa bloques de 1 warp, con 84 µs y 4 pedidos, para datos chicos.
- **Hacer:**
  - bajar registros y memoria compartida para tener 2–3 bloques por SM, o 8 warps por bloque;
  - revisar el tamaño de la partición (298 particiones);
  - `salida` con un warp por cabeza y varias cabezas por bloque, o fusionada en `union4`.
- **Esperado:** batch2 ~140 µs, union + salida ~15 µs, es decir −2,8 / −4,1 ms.
- **Verificar:**
  - bit a bit contra el kernel actual; hay un comparador en `sk18_attn.py` (l. ~562, SK-18h contra
    decuantizado + Triton);
  - medir con el reloj fijo (medir-kernels-fijar-el-reloj) y con ncu dentro del contenedor (perfilar-con-ncu-como).
- **Historia:** memorias sk18f-streaming-entero (perfil ncu: barreras), ptx-acumulador-mma-no-leer-escribir
  (una carrera silenciosa con el reescalado), pn131-sk18-integrado-vllm (trampas de grafos).

### P3 · All-reduce de decode — **−6% / −11%, costo medio** · HECHO: −6,1% / −9,7% (PN152 + i8)

- **Resultado (27/28-09):** PN152 (SK-24, `ar_p2p.py`) escribe el parcial en la memoria de la otra placa,
  bandera de época, doble buffer; bit a bit con NCCL: paso −4,9% / −6,4%. Encima, `sk24_ar_i8`
  (`GENESIS_PN152_MIN_I8=1`, int8 por grupo de 64, las dos placas suman las dos partes cuantizadas en orden de
  rango): −1,3% / −3,5% más. Calidad: el KL por prompt_logprobs no pasa por el decode, así que se midió la misma
  cuenta en el prefill (PN120 prendido vs apagado, `tests/bench/medicion/kl_ar_int8.sh`): 0,0193 vs 0,0192 en
  respuestas, determinista entre arranques. Pendiente: latencia base (~13 µs para 9 filas contra ~8 teóricos) y
  el all-reduce del embedding. Lo que sigue es el diagnóstico original.

- **Qué pasa:** NCCL `AllReduce_Sum_f16_RING_LL`, grilla 2, **128 por paso** más 11 del borrador:
  - 26 µs para 92 KB (9 filas) y 61 µs para 369 KB (36 filas);
  - el primero de cada capa tarda 46–48 µs contra 27 del segundo, y los dos rangos dan lo mismo.
    Revisar si es espera por el stream (va por otro stream, 3041/3101) o protocolo.
  - vLLM `cross_device_reduce_1stage` aparece solo en el embedding, con 17–41 µs: no es más rápido.
- **Hay material:**
  - `vllm/_genesis/kernels/cuda/sk21_p2p.cu`, `vllm/_genesis/p2p_buzon.py`;
  - pruebas: `tests/proto/sk21_p2p_test.py`, `sk21_ipc_crudo.py`, `p2p_micro.py`;
  - memoria p2p-anda-bien-con-memoria-cruda: 11,4 GB/s con memoria cruda, 10% de SM, solape 93%;
  - memoria all-reduce-int4-dominado: int4 no.
- **Hacer:** all-reduce de una pasada por P2P crudo (cada rango lee el parcial del otro y suma) para los tamaños
  de decode, capturable en grafo CUDA. Opcional: int8 como PN120, que en decode hoy está apagado por
  `GENESIS_PN120_M_MIN=512`.
- **Esperado:** 26 → ~12 µs y 61 → ~25 µs, es decir −1,8 / −4,6 ms.
- **Verificar:** suma exacta en fp16 (mismo orden), grafos CUDA, sin carreras entre pasos (buzón con época).

### P4 · GPU ociosa entre fases — **−5% / −2,5%, costo medio**

- **Qué pasa:** entre verificación, borrador y armado del árbol la GPU espera a Python unos 2,0–2,5 ms por paso.
  El armado del árbol lanza ~110 kernels de torch (index, add, copy, scatter, fill, cub scan) en un patrón que
  se repite ~6 veces, uno por nivel o nodo del árbol.
- **Dónde:** `vllm/_genesis/arbol_borrador.py` (constructor, `_k`), `arbol_conv.py` (`_k_compactar_v2`,
  `_k_materializar`) y el runner de spec decode.
- **Hacer:** fusionar el armado del árbol en 1–2 kernels de Triton; ver si la verificación y el borrador pueden ir
  en un grafo CUDA; evitar sincronizaciones con el host.
- **Historia:** memoria arbol-ab-resultado: la primera versión del árbol perdía por ~250 kernels chicos por
  paso, y ya se bajaron. Es el mismo frente.

### P5 · Borrador — **−3% / −3%, costo medio**

- **Atención:**
  - hoy va por `kernel_unified_attention` de Triton (PN124, `triton_attn_ampere.py`) con **12 bloques** (grilla 3×4),
    sin partir la KV: 1 ms por paso;
  - opciones: 3D o split-KV, SK-18h para el borrador (KV int8 de head 128, así que hay que ver si el kernel
    soporta 128), o limitar la ventana de contexto del borrador (cambia la aceptación: medir).
- **GEMM fp16 sin cuantizar:** 298 µs. Cuantizarlas a W4A8 o W4A16.
- **lm_head completo del borrador:** 386–674 µs para top-16. Vocabulario podado (los 32k más frecuentes)
  → ~1/8 de los bytes. Cambia los candidatos: medir la aceptación.

### P6 · Logits sin AllGather — **−0,7% / −2%, costo bajo-medio**

- **Qué pasa:** después del lm_head se juntan los logits enteros (NCCL AllGather: 220 / 844 µs) y después
  `_gumbel_sample_kernel`.
- **Hacer:**
  - gumbel-max por rango sobre su mitad del vocabulario, con el **mismo ruido por índice global**, y juntar solo
    (valor, índice). Da el mismo token;
  - revisar que el rechazo del árbol no necesite las probabilidades completas (con temperatura > 0).
- **Extra:** con 36 filas, el lm_head W4A16 (972 µs) pasa a limitar por cómputo HMMA. Probar W4A8 en el lm_head
  (≈ −0,4 ms), midiendo la calidad.

### P7 · Norma que escribe int8 (PN135 / SK-20) — **re-medir sobre idiotSavant**

- **Ya existe:** `vllm/_genesis/norm_quant.py`, `kernels/cuda/sk20_norm_quant.cu`. Con noon dio +0,4% en decode y
  +0,7% en prefill. **En idiotSavant cambia:** el peso de la norma está plegado (g = 1), así que el kernel se
  simplifica y el patrón de SK-23 (la lineal llama a Marlin directo) permite cablearlo sin pasar el int8 por
  Python.
- En el perfil hay 1,7–1,9 ms por paso de normas, cuantización y a_scales (5,9% con 1 pedido). **Techo: −2–4%.**

### P8 · Prefill: solapar el all-reduce — **0 a −15%: medir PN136 sobre idiotSavant**

- **El perfil:** el all-reduce int8 es 22,3% del prefill, sin solapar (transferencia real, igual en los dos rangos).
- **Ya existe PN136** (`vllm/_genesis/mlp_solapado.py`, transporte `p2p_buzon.py` por DMA a 13 GB/s y 105% de
  solape). Con noon no ganó (2.383 contra 2.390 tok/s a 42k), pero **eso no vale para idiotSavant**: hay que medirlo de nuevo.
- **Primera tarea:** perfilar PN136 prendido con idiotSavant (misma fase `prefill`, agregando la variable) y
  ver en la traza si la transferencia realmente queda escondida y dónde reaparece el tiempo. Posibles
  explicaciones a comprobar:
  - solo solapa el AR del MLP y no el de la atención/GDN;
  - la copia DMA compite por DRAM con Marlin;
  - el tiempo se mueve a otra espera.
  Sin eso no hay estimación seria.

### P9 · Prefill: atención sin descuantizar, estilo SageAttention — **−10–15% del prefill a 60k; y los pasos mixtos**

**Qué pasa hoy**
- FlashInfer fp16 con acumulador fp32 (media tasa en GA102) sobre KV descuantizada por `sk18h_decuant`
  (`sk18_attn.py`: `_prefill_flashinfer`, `decuantizar_kv`; alternativa `GENESIS_PN131_PREFILL=triton`, que también
  descuantiza).
- En el perfil FlashInfer va a ~64 TFLOPS (3,0 ms por capa, 1760 queries contra ~9k): ya está en el techo de ese
  tipo de dato. Un FlashInfer mejor no existe; hay que cambiar el tipo.
- **Pasos mixtos (verificado en `is25_decode4`):** en esos pasos SK-18h **no corre**. Todo el lote, incluidos los que
  solo decodifican, va por descuantizar + FlashInfer, y `sk18h_decuant` recorre las páginas de **todos** los pedidos
  (16 → 72 páginas de 880): hasta ~20 ms por paso.

**Qué hay hecho (ninguno es un kernel de prefill)**
- **Numérica:** SK-18 A0 / A0-bis (`tests/proto/sk18_a0*.py`). Q8K8V8 + Hadamard, 0,5–0,65% de error; prefill 0,56–0,58%.
  Trampa: pesos del softmax de 8 bits rompen a 22k (6,7%).
- **Kernels `kernels/cuda/sk18{a..i}`:** todos de decode. Reparten el trabajo **por páginas de keys** y juntan parciales
  con una unión. SK-18g acepta cualquier cantidad de filas con causal (`lim[]`), pero para un prefill el buffer parcial
  (páginas × filas × 256 × int32 × 2) da ~3 GB a 60k: no sirve tal cual.
- **Reutilizable:**
  - el layout de página (K por token, V por dimensión);
  - `prep2` (q a int8 con la Hadamard en PTX);
  - la causal por fila;
  - los emuladores y tests (`sk18e_test.py`, `sk18g_paginado.py`, `pn131_offline.py`);
  - `tests/proto/attn_int8_qk_eval.py` (numérica estilo Sage, sin kernel).

**Por qué NO el softmax entero de SK-18 (corrección del usuario, confirmada)**

Por cada par (query, key) el camino entero hace:
- la exp por desplazamiento + cuadrática Q15;
- el peso partido en dos bytes (hi/lo);
- **dos pasadas** de mma para P·V.

Son ~15 operaciones enteras por par en el pipe INT32 (64 por ciclo y SM en GA102, la mitad que FP32), más el doble
de mma en P·V y más registros. Con int4 eso ya hizo que el kernel no fuera más rápido. El pipe de float tiene el
doble de ancho y la exp tiene su propia unidad (MUFU.EX2, 16 por ciclo y SM, que corre en paralelo con los tensor
cores y el FMA).

**Cuenta por par con head_dim 256 en GA102** (ciclos por SM, órdenes de magnitud):

| Qué | Hoy (FlashInfer) | Propuesta |
|---|---|---|
| QKᵀ | fp16/acc32: 1 ciclo | int8 IMMA: 0,25 |
| P·V | fp16/acc32: 1 ciclo | fp16/acc16: 0,5 |
| exp | MUFU, 0,06 | igual |
| FMA del softmax | ~0,04 | igual |

Con head_dim 256 la exp **no** es el cuello; a diferencia de H100 con hdim 128, que es el caso de FA3/FA4. El cuello es
el tipo de los mma. Techo: ~2 → ~0,85 ciclos por par contando el softmax (~2,3×).

**Throughput medido en SM86** (`tests/proto/sm86_throughput.cu`, resultados por ciclo y SM, 27-09):

| Instrucción | por ciclo y SM | Nota |
|---|---|---|
| FFMA / FADD fp32 | ~128 | el pipe ancho |
| HFMA2 | 64 instrucciones = 128 resultados fp16 | |
| IADD3 | ~112 | corre en los dos caminos |
| IMAD, LOP3, SHF, PRMT, FMNMX | 64 | **medio ancho**: los enteros y el max comparten el camino con FP32 |
| I2FP (int32 → fp32) | 64 | en SM86 es rápido, no hace falta el truco del número mágico |
| F2FP pack (2 fp32 → f16x2) | 64 instrucciones = 128 valores | |
| **MUFU.EX2** | **16** | la unidad más angosta |
| FFMA + IADD mezclados | 46 pares | se pisan: comparten camino |
| FFMA + EX2 mezclados | 16 pares | la exp manda |

Tensor cores de la 3090 (spec, por ciclo y SM): fp16 con acumulador fp32, 256 FMA; con acumulador fp16, 512;
int8, 1024; int4, 2048 (medido 2× sobre s8).

**Presupuesto por par (query, key), hdim 256, ciclos por SM:**

| Qué | Costo |
|---|---|
| QKᵀ int8 | 0,25 |
| P·V fp16/acc16 | 0,5 |
| exp (MUFU) | 0,0625 |
| resto del softmax: I2F + FMNMX + FFMA + FADD + ½ F2F | ~0,055, en otro camino |

- **Total ~0,8–0,87** contra ~2 de FlashInfer. El softmax float pasa a ser ~13% una vez que los mma bajan: ya no es
  despreciable y hay que **esconderlo intercalando warps**.
- En SM86 no hay wgmma ni TMA (el solape asíncrono de FA3/FA4). Solo `mma.sync` + `cp.async`, así que el solape
  depende de tener ≥8 warps por SM.
- Con hdim 256, el acumulador O de un warp (16 queries × 256) en fp32 son 128 registros por hilo. Por eso FlashInfer
  está en 243 y SK-18h en 254. Salidas:
  - partir las 256 dimensiones entre 2 warps (O de 64 registros) y **recalcular QKᵀ en cada warp**: en int8 cuesta
    0,25, más barato que sincronizar por shared;
  - el acumulador fp16 por tile ocupa la mitad.
- El softmax entero de SK-18 (~15 operaciones de 64 por ciclo por par ≈ 0,23 ciclos, más dos pasadas de P·V) cuesta
  **~4× el ALU** del float y más registros.
- La exp emulada de FA4 en SM86 cuesta ~0,07 ciclos por exp (grado 3 + Cody-Waite con SHF a 64 por ciclo): igual que
  MUFU. Solo sirve para repartir carga si MUFU fuera el camino que limita.

**Diseño propuesto (literatura, ver abajo)**
1. **QKᵀ en int8** (IMMA): Q de `prep2` (Hadamard + int8 por token) y K int8 **leída directo de la caché paginada**, sin
   descuantizar. La escala de K por token se aplica al score (una FMUL por par). Opcional: el suavizado de K de Sage
   (restar la media de K, exacto bajo softmax), si el error lo pide; con la Hadamard puede no hacer falta.
2. **Softmax en fp32** con `exp2` de MUFU. Si algún día MUFU limita, la emulación de FA4 (polinomio grado 3 en FMA,
   Cody-Waite, 10–25% de las entradas).
3. **Reescalado condicional de FA4:** el acumulador O solo se reescala si el máximo crece más de τ = 8 (en log2).
   Es exacto porque todo se normaliza al final con el máximo y la suma verdaderos. Saca casi todo el ALU del reescalado.
4. **P·V en fp16 con acumulador fp16 dentro de cada tile** (Sage-B: 2× en GA102, sin pérdida medida), y el parcial de cada
   tile se suma a un O en fp32 en registros ("acumulación en dos niveles"). Evita que el fp16 acumule 60k términos y
   evita leer y escribir los acumuladores del mma entre tiles, que fue la carrera de SK-18
   (ptx-acumulador-mma-no-leer-escribir). V int8 → fp16 en registros; su escala por token se pliega en P
   (una FMUL por par).
5. **P en int8 NO** (variante Sage-vB): con contexto largo los pesos chicos se pierden (A0: 8 bits rompe a 22k).

**Cómo hacerlo**
- **Triton primero:** `tl.dot` int8 → int32 e `tl.dot(..., out_dtype=tl.float16)`, con paginación por tabla de bloques.
  1–2 semanas con la validación.
- Si Triton no llega, PTX con la base de SK-18h.

**Esperable**
- Sage reporta ~2× sobre FlashAttention2 en 3090/4090; realista **1,6–2× en la atención** y **sin `sk18h_decuant`**.
- A 60k (atención ~30% del prefill): **−10 a −15% del prefill**, más.
- Arregla de raíz los pasos mixtos si los que decodifican siguen yendo por SK-18h.

**Mismo razonamiento para el decode (paso P2):** SK-18h usa el mismo softmax entero con hi/lo y dos pasadas de P·V.
Está en 254 registros y 1 bloque de 4 warps por SM, al 35% del ancho de banda. Pasar su softmax a float con P·V fp16
bajaría registros y subiría la ocupación. Probarlo como variante del paso P2.

**Literatura**
- SageAttention, arXiv 2410.02367: QKᵀ int8 por warp, P·V fp16 con acumulador fp16, suavizado de K; ~2,1× sobre FA2
  en 4090.
- SageAttention2, arXiv 2411.10958: Q/K int4 por hilo, P·V fp8 con acumulación en dos niveles. En Ampere no hay fp8:
  solo sirve la idea de los dos niveles.
- FlashAttention-3, arXiv 2407.08608: la exp como cuello en H100 y su solape con los mma.
- FlashAttention-4, arXiv 2603.05451: exp2 emulada en FMA (grado 3, Cody-Waite, 10–25%) y reescalado condicional τ = 8.
- INT-FlashAttention, arXiv 2409.16997: todo int8, sin softmax entero; no aporta sobre Sage.

### P10 · Preparación de la atención de prefill — **−1–3% del prefill, costo bajo**

- **Qué pasa:** `torch.unique(bt[valid])` y sort en `sk18_attn.py` (l. 723 y 759) por capa de atención y chunk,
  ~0,5 ms con la GPU parada.
- El plan de FlashInfer ya se arma una vez por paso (`clave`). **Hacer:** calcular `ids/inv` una vez por paso,
  no por capa.

### Piso (no es código)

- **GEMM del target y lm_head:**
  - en decode están al techo de DRAM: 9,6 ms / 11,8 ms por paso;
  - en prefill, al 90% del pico int8 a 810 MHz.
- **Solo se mueven** con menos bits por peso o con más potencia: sin cap, +19,5% de prefill medido (memoria
  cap-de-potencia-costo-y-deriva).

### Total estimado

| | 1 pedido | 4 pedidos |
|---|---|---|
| Si todo sale al máximo, paso | 29,7 → ~20 ms (+45–50% tok/s) | 41,2 → ~26 ms (+55–60%) |
| Realista (la mitad) | **+20–25%** | **+20–25%** |

- **Prefill:** P9 y P10 dan ~−7–9%. P8 (PN136) y P7 (PN135) hay que medirlos sobre idiotSavant antes de estimar.
- **Orden:** ver la sección 4c, que mezcla mejoras de código y de capas.

---

## 4b. Capas que se pueden alinear mejor ahora que el residuo está rotado

Con noon, el int8 en estas entradas perdía calidad por los picos del residuo. Con la rotación, la entrada de
todo lo que lee el residuo tiene cresta ~2,7, así que vale la pena revisarlo:

| Capa | Hoy | Propuesta | Por qué ahora | Premio |
|---|---|---|---|---|
| **lm_head** del target y del borrador (compartido) | W4A16, HMMA fp16 | **W4A8** (IMMA s8) | lee la salida de la norma final, rotada. Con 36 filas el W4A16 limita por cómputo: 972 µs contra 402 con 9 | −0,5 ms (target) y −0,3 ms (borrador) con 4 pedidos. Medir el KL en la respuesta |
| **All-reduce de decode** | fp16 (PN120 solo con M ≥ 512) | **int8** también en decode, o grupo más grande en prefill | el residuo rotado tiene cresta baja: el error del int8 por grupo cae mucho respecto de noon | combinado con P3: menos bytes y menos latencia. En prefill, grupo 128/256 = menos escalas |
| **in_proj_a / in_proj_b** (GDN) | denso fp16 (cutlass + split-K, 6,2 µs por capa) | int8 × int8 o W8A8 | la entrada ya es la norma rotada. Son las compuertas: medir la sensibilidad por capa antes | −0,2 ms por paso; más importante, un kernel menos si se fusiona con in_proj_qkvz |
| **GEMM fp16 del borrador** (proyección K/V del contexto y otras, 298 µs) | fp16 sin cuantizar | W4A8 | el borrador está en su base original (PN149), así que hay que medir la cresta de sus entradas; si es alta, rotarlas como en el target | −0,2 ms |
| **o_proj / out_proj** | entrada sin rotar (cresta 12–21, A8 ~2%) | Hadamard online como down_proj (PN148) | el único sitio con error A8 apreciable que queda | calidad, no velocidad (el Hadamard fusionado cuesta ~0) |

El criterio es el mismo que en la cuantización: medir la cresta y el error A8 por lineal con
`entrenamiento/cuant/error_a8.py` y `sensibilidad.py` sobre idiotSavant, y el KL sobre las respuestas. Ver
`entrenamiento/idiotsavant/DECISIONES.md`.

## 4c. Plan unificado: código y capas del modelo, en un solo orden

Tipos: **código** (kernel o scheduling, salida idéntica), **capa** (cambia cómo se cuantiza o rota una lineal:
exige KL sobre respuestas y generación larga), **flag** (algo que ya existe, solo se mide).

El criterio es: primero lo barato y seguro; después juntar los cambios que tocan el mismo kernel, para
reescribirlo una sola vez; y los cambios de capa en tandas, porque cada tanda paga una validación de calidad.

| # | Paso | Tipo | Ahorro (1 / 4 pedidos, prefill) | Costo | Por qué va acá |
|---|---|---|---|---|---|
| 1 | **P1:** `_k_escribir` y `_k_salidas` en paralelo | código | −7% / −7% | bajo | el más barato; bit a bit; no depende de nada |
| 1b | **Pasos mixtos:** los que decodifican van por SK-18h; descuantizar solo las páginas del pedido en prefill | código | hasta −20 ms por paso mixto (4 pedidos) | bajo-medio | verificado en `is25_decode4`: con un prefill en el lote, SK-18h no corre y `sk18h_decuant` recorre las páginas de todos (16 → 72). Es el patrón de opencode: candidato a ir segundo |
| 2 | **Re-medir PN135 y PN136** sobre idiotSavant | flag | ? / ? / ? | un perfil cada uno | decide si 8 y 13 son fusiones nuevas o ya están hechas |
| 3 | **lm_head en W4A8** + **P6:** logits sin AllGather (gumbel-max por rango) | capa + código | −0,7% / −3% | bajo | los dos tocan la misma región (lm_head → muestreo). El lm_head no se re-cuantiza: mismos int4, cambia solo la activación; basta el KL. También acelera el lm_head del borrador |
| 4 | **P3 + AR int8 en decode:** all-reduce P2P crudo de una pasada, en int8 por grupo | código + capa | −6% / −11% | medio | el mayor de 4 pedidos. Con la cresta de 2,7 el int8 del residuo es seguro (medir KL); int8 = mitad de bytes = menos latencia en el mismo kernel nuevo |
| 5 | **P2 + Hadamard en o_proj:** reescribir la atención de decode (ocupación de `batch2`, `union4`+`salida` fusionados) y que la **salida escriba la entrada de o_proj ya rotada con Hadamard y en int8** | código + capa | −9% / −11% | medio-alto | `sk18h_salida` hay que reescribirla igual; ahí la Hadamard y el int8 salen gratis y desaparece el `_per_token_quant` antes de o_proj. Exige re-cuantizar o_proj como `W·H` (una etapa de `idiotsavant.py` para 16 capas) |
| 6 | **Hadamard en out_proj** (GDN): la norma con compuerta escribe la entrada rotada en int8 | capa + código | −0,5% | medio | mismo patrón que el paso 5, del lado GDN; va en la misma tanda de re-cuantización y de KL que o_proj |
| 7 | **P7:** norma que escribe int8 (SK-20 con g = 1), cableada como SK-23 | código | −2–4% | medio | según lo que dé el paso 2. Con 5–7 hechos, las entradas de todas las lineales ya llegan en int8 desde el kernel anterior |
| 8 | **in_proj_a/b en int8, concatenadas a in_proj_qkvz** (un solo Marlin) | capa + código | −1% | medio | se va el cutlass fp16 + split-K de cada capa GDN. Son las compuertas: medir la sensibilidad por capa antes; si alguna no aguanta W4, queda fp16 solo esa |
| 9 | **P4:** armado del árbol y verificación en 1–2 kernels / grafo | código | −5% / −2,5% | medio | independiente del modelo; conviene después de 3, que cambia la verificación |
| 10 | **Borrador:** GEMM fp16 → W4A8 (rotar sus entradas si la cresta lo pide), atención con KV partida o SK-18h, vocab podado | capa + código | −3% / −3% | medio | todo lo del borrador en una tanda: se valida con la aceptación y, si hace falta, se re-ajusta el DFlash2 con `dflash2.sh` |
| 11 | **P10:** páginas únicas una vez por paso | código | prefill −1–3% | bajo | puede ir en cualquier momento; es chico |
| 12 | **Grupo del AR int8 de prefill** 64 → 128/256 | capa (flag) | prefill −1–2% | bajo | menos escalas; la cresta baja lo permite. Mismo KL que el paso 4 |
| 13 | **PN136 o solape propio** del all-reduce de prefill | código | prefill 0 a −15% | según paso 2 | solo si el paso 2 muestra que el solape esconde la transferencia en idiotSavant |
| 14 | **P9:** atención de prefill estilo SageAttention sobre la KV int8 (QKᵀ int8, softmax fp32, P·V fp16/acc16 en dos niveles) | código | prefill −10–15% a 60k; sin descuantizar | medio-alto (Triton, 1–2 semanas) | antes, el intermedio: descuantizar solo las páginas nuevas y los pasos mixtos |

**Tandas de validación de calidad** (KL en respuestas + generación larga + banco de agente):
- **A:** paso 3 (lm_head) y paso 4 (AR int8 en decode). No re-cuantizan pesos.
- **B:** pasos 5, 6 y 8. Re-cuantizan o_proj, out_proj e in_proj_a/b: nuevo armado con `idiotsavant.py` (etapa
  `cuantizar` solo de esas lineales, el resto de la caché sirve) y nuevo informe por capa.
- **C:** paso 10 (borrador): aceptación y, si baja, re-ajuste.

**Acumulado aproximado** (máximos, sin contar PN135/PN136), paso con 1 pedido / 4 pedidos:

| Hasta el paso | 1 pedido | 4 pedidos |
|---|---|---|
| 1 | −7% | −7% |
| 3 | −8% | −10% |
| 4 | −14% | −21% |
| 6 | −24% | −33% |
| 10 | −35% | −42% |

La mitad realista de eso es **+20–25% de tokens por segundo**. El prefill suma −8–10% con los pasos 11, 12 y 14,
más lo que diga el paso 2 sobre el solape.

## 5. Protocolo para cada cambio

1. **Kernel aislado:** verificación bit a bit contra el camino actual con tensores reales (volcados de una
   corrida), y tiempo con el reloj fijo y grafos CUDA.
2. **Flag nueva** `GENESIS_ENABLE_PNxxx` apagada por omisión, registrada en `dispatcher.py` y
   `patches/apply_all.py`. Declararla en el `environment:` del compose u override.
3. **Perfil A/B:**
   - `perfil_idiotsavant.sh isNN_base` y `isNN_cambio`, con la variable en un override;
   - después `analizar_perfil.py` de las dos y `--comparar`;
   - la región tocada tiene que bajar lo esperado y ninguna otra subir.
4. **Punta a punta:** banco de agente (40 pedidos SWE-rebench) y prosa, con ≥2 arranques por brazo, y
   aceptación del borrador sin cambios.
5. **Calidad:** generación larga y fidelidad sobre las respuestas (piso de ruido KL ~0,019).
6. **Al terminar:** recrear `genesis-27b-idiotsavant` y actualizar el informe (artifact) y las memorias.

---

## 6. Archivos

| Archivo | Qué es |
|---|---|
| `tests/bench/medicion/perfil_idiotsavant.sh` | captura de las 3 fases |
| `tests/bench/medicion/analizar_perfil.py` | análisis decode / prefill / comparar |
| `tests/bench/medicion/analisis_perfil/is25_*.json` | línea base del 25-09 (decode1 r0/r1, decode4 r0, prefill r0) |
| `tests/bench/medicion/trazas/is25_{decode1,decode4,prefill}/` | trazas crudas (Perfetto / chrome://tracing) |
| `tests/bench/medicion/perfil_is25.log` | log de la corrida |
| `compose/ov-aislado.yml` | override de la instancia aislada (8361, `genesis-27b-pruebas`) |
| `docs/informe-tipos-sm86-2026-09-25.html` | informe visual (artifact v5 o posterior) |
