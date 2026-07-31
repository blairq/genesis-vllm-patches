# Análisis de flujo de datos, kernels y tipos de dato — 3 contenedores vLLM

**Fecha:** 2026-07-30
**Base de código analizada:** `assets/vllm` @ `0fc695f` (imagen `vllm/vllm-openai:v0.23.0`)
**Hardware objetivo:** 2× NVIDIA RTX 3090 (GA102, **compute capability 8.6**, 24 GiB, driver 580.173.02)
**Alcance:** sólo análisis. No se modificó ningún archivo de código.

---

## 0. Resumen ejecutivo

El hardware manda sobre todo lo demás. La **sm86 (Ampere)** define qué kernels son
siquiera alcanzables, y eso invalida varias suposiciones que están implícitas en los
`docker-compose`:

| Suposición en el compose | Realidad en sm86 |
|---|---|
| `--attention-backend FLASHINFER` acelera todo | Sólo la atención *full*. Las **capas GDN (75 % del modelo) usan Triton/FLA**, no FlashInfer |
| `--enable-flashinfer-autotune` | No toca el camino GDN; sin efecto en 3 de cada 4 capas |
| `--kv-cache-dtype fp8` = cómputo fp8 | La 3090 **no tiene tensor cores fp8**. Es sólo ahorro de memoria/ancho de banda; se convierte a fp16 dentro del kernel en cada lectura |
| `--performance-mode throughput` | **Inerte**: sólo actúa si `max_num_batched_tokens`/`max_num_seqs` NO están puestos, y los tres los ponen explícitamente |
| `VLLM_FLASHINFER_MOE_BACKEND=throughput` | Sólo aplica a KAT (único MoE), y el backend TRTLLM MoE requiere SM90+ |
| P100 "CUDAGraphs FULL para spec-decode" | En Fable y Heretic el decode a plena concurrencia **queda fuera del grafo** (ver hallazgo #2) |

Los **cinco hallazgos de mayor impacto**, ordenados por relación beneficio/riesgo,
están en la [§6](#6-hallazgos-priorizados). Los dos primeros son configuración pura
(cero cambios de código, cero riesgo numérico):

1. **KAT-Coder está prácticamente sin cuantizar fuera de los expertos** — 30 capas GDN,
   10 capas de atención, 40 expertos compartidos, routers y `lm_head` están en la lista
   `ignore` del checkpoint y corren en **fp16**. ~0,94 GiB/GPU sólo en las proyecciones GDN.
2. **Fable rellena su lote de decode de 60 a 66 tokens** por la retícula de tallas de
   CUDA Graph (~6 % del paso). Heretic captura exacto; KAT rellena poco. *(Corregido:
   una versión previa afirmaba que Fable y Heretic quedaban fuera del grafo — ver §6.2.)*
3. Un `.float()` en `GemmaRMSNorm` desactiva el kernel CUDA fusionado de RMSNorm
   **y bloquea explícitamente el pase de fusión norm+quant** en los tres contenedores.
4. El buffer `h` de FLA reserva **192 MiB por capa GDN** en prefill (18 GiB de tráfico por paso).
5. `in_proj_ba` se **replica** en Heretic por un `isinstance` que no comprueba si la capa
   está realmente cuantizada.

---

## 1. Los tres contenedores

Los tres son la **misma familia arquitectónica** (`Qwen3_5*ForConditionalGeneration`,
híbrido *linear attention* + *full attention*), lo que hace el análisis comparable
bloque a bloque. Lo que cambia es: densa vs MoE, y el esquema de cuantización.

| | **kat-coder** | **qwen27b-fable-fusion** | **qwen27b-heretic** |
|---|---|---|---|
| Contenedor | `genesis-kat-coder` | `genesis-27b-fable-fusion` | `genesis-27b-heretic` |
| Checkpoint | `cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4` | `lued/Qwen3.6-27B-Fable-Fusion-711-INT8-W8A16-MTP` | overlay GPTQ-Int4 (`llmfan46/...heretic-v2`) |
| Arquitectura | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| Capas | 40 (30 GDN + 10 full-attn) | 64 (48 GDN + 16 full-attn) | 64 (48 GDN + 16 full-attn) |
| `hidden_size` | 2048 | 5120 | 5120 |
| FFN | MoE 256 exp., top-8, `moe_inter=512` + shared 512 | denso `inter=17408` | denso `inter=17408` |
| Atención | 16 q / 2 kv, `head_dim=256`, gate | 24 q / 4 kv, `head_dim=256`, gate | 24 q / 4 kv, `head_dim=256`, gate |
| GDN | 16 k-heads×128 / 32 v-heads×128, conv 4 | 16 k / 48 v ×128, conv 4 | 16 k / 48 v ×128, conv 4 |
| Cuantización | compressed-tensors INT4 **g32 asimétrico** (zp int8) | compressed-tensors INT8 **g128 simétrico** | GPTQ INT4 **g128 simétrico**, `desc_act=false` |
| `--dtype` | float16 (checkpoint ya fp16) | float16 (**checkpoint bf16** → conversión en carga) | float16 (**checkpoint bf16** → conversión en carga) |
| KV cache | `fp8` (e4m3) | `fp8_e4m3` | `fp8` (e4m3) |
| Spec decode | ninguno | MTP, 2 tokens | MTP, 3 tokens |
| `VLLM_MARLIN_INPUT_DTYPE` | — | — | **`int8` (W4A8)** |
| TP / seqs / batched tok | 2 / 20 / 16384 | 2 / 20 / 16384 | 2 / 20 / 16384 |

> **Nota sobre `mamba_ssm_dtype`.** Los tres `config.json` declaran
> `mamba_ssm_dtype: "float32"`, y `models/config.py:536-560`
> (`Qwen3_5ForConditionalGenerationConfig`) lo propaga a `mamba_ssm_cache_dtype`.
> Resultado en los tres: **conv_state en fp16** (= `--dtype`) y **ssm_state en fp32**.

---

## 2. Flujo de datos: qué kernel toca cada bloque y por qué

### 2.1 El árbol de decisión del kernel GDN (75 % de las capas)

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:150-211`
(`_resolve_gdn_prefill_backend`):

```
¿CUDA?  sí
  ¿capability == 90 (Hopper)?            → FlashInfer GDN     ✗ (somos 86)
  ¿capability 10x + head_k==128 + cu13?  → FlashInfer/CuteDSL ✗ (somos 86)
  en cualquier otro caso                 → Triton / FLA       ✓  ← los tres contenedores
```

**Consecuencia directa:** todo el bloque `fi_chunk_gated_delta_rule`
(`qwen_gdn_linear_attn.py:241-287`), que incluye tres conversiones a fp32
(`initial_state.to(fp32)`, `g.to(fp32)`, `beta.to(fp32)`) y un `torch.exp`, es
**código muerto en esta flota**. No hay que optimizarlo; hay que saber que no se ejecuta.

### 2.2 Ruta de una capa GDN (idéntica en los tres, distinto dtype de pesos)

```
hidden_states (fp16, [T, hidden])
  │
  ├─► in_proj_qkvz  ──┐  ← ambos leen el MISMO tensor y (en Heretic) lo cuantizan
  └─► in_proj_ba    ──┘    a int8 por separado  → qwen_gdn_linear_attn.py:923-924
  │
  ├─ split [q|k|v] / z / b / a                          (views + 2 .contiguous())
  │
  ├─► causal_conv1d_fn (prefill) / _update (decode)     conv_state fp16 ✓ sin conversión
  │
  ├─► fused_post_conv_prep  (Triton, 1 kernel)          fused_gdn_prefill_post_conv.py
  │     q,k → l2norm en fp32 interno → store fp16
  │     v   → COPIA pura de layout (sin cómputo)        ← 100 MiB r+w por capa @T=16384
  │     g,beta → softplus/sigmoid en fp32 → store fp32
  │
  ├─► PREFILL: chunk_gated_delta_rule (FLA Triton, BT=64)
  │     cumsum(g) fp32 → A fp32 [T,H,64] → solve_tril → A fp16
  │     → recompute_w_u → w,u fp16
  │     → chunk_delta_h  → h fp16 [NT,H,V,K]  ← 192 MiB/capa
  │     → chunk_fwd_o    → o fp16
  │     initial_state = ssm_state[idx]  (gather fp32)   :1513
  │     ssm_state[idx] = final_state    (scatter fp32)  :1532  ← .to() es no-op ✓
  │
  ├─► DECODE: fused_sigmoid_gating_delta_rule_update (Triton, 1 kernel fusionado)
  │     grid = (1, V/32, N*HV), estado fp32 in-place    ✓ sin conversiones
  │
  ├─► RMSNormGated (Triton, forward_cuda → rmsnorm_fn)  ✓ camino rápido
  └─► out_proj → output
```

### 2.3 Ruta de una capa de atención *full* (1 de cada 4)

```
hidden_states fp16
  ├─► qkv_proj  →  [q | gate | k | v]     (attn_output_gate = True)
  ├─ chunk(q_gate, 2) → q, gate           ← views NO contiguos
  ├─ q.reshape(...)  → COPIA              ← qwen3_next.py:301-302
  ├─ gate.reshape(...) → COPIA            ← 2 copias de [T,3072] fp16 por capa
  ├─► q_norm / k_norm  (GemmaRMSNorm, peso fp32)  ← camino nativo, no el kernel C
  ├─► rotary_emb  (mrope, partial_rotary 0.25 → sólo 64 de 256 dims)
  ├─► Attention → FlashInfer BatchPrefill/Decode PagedKV
  │     query fp16, kv_cache fp8_e4m3, k_scale=v_scale=1.0
  │     → dequant fp8→fp16 DENTRO del kernel (sm86 no tiene fp8 nativo)
  ├─ sigmoid(gate) → tensor nuevo
  ├─ attn_output * gate → tensor nuevo     ← 2 pasadas elementwise separadas
  └─► o_proj
```

### 2.4 Selección del kernel de GEMM cuantizado

`vllm/model_executor/kernels/linear/__init__.py:347-357` prueba en orden:
`CutlassW4A8 → Machete → AllSpark → Marlin → Humming → Conch → Exllama → TritonW4A16`.

| Candidato | Requisito | ¿sm86? |
|---|---|---|
| CutlassW4A8 | capability 90 + act fp8 + bf16 out | ✗ |
| Machete | capability 90 | ✗ |
| **AllSpark** | capability 80-89, **`group_size == -1`**, `uint8b128` | ✗ *por group_size* |
| **Marlin** | capability ≥ 75, `N%64==0`, `K%128==0` | **✓ los tres** |

> **AllSpark es el kernel W8A16 optimizado para Ampere y está a un paso de Fable.**
> `allspark_utils.py:26-32` exige `group_size == -1` (per-channel). Fable está
> cuantizado a **g128**, así que se descarta y cae en Marlin. Si el checkpoint se
> recuantizara per-channel (`strategy: channel`), AllSpark quedaría disponible.
> Es una hipótesis a medir, no una certeza: hay que comparar AllSpark vs Marlin W8A16
> en 3090 antes de dar por hecho que gana.

### 2.5 Resultado final por contenedor

| Bloque | KAT-Coder | Fable-Fusion | Heretic |
|---|---|---|---|
| GDN `in_proj_qkvz` / `out_proj` | **fp16 sin cuantizar** (en `ignore`) | Marlin **W8A16** INT8 | Marlin **W4A8-int8** |
| GDN `in_proj_ba` | fp16, TP-shard | fp16 (en `ignore`), TP-shard | fp16, **replicado** ⚠ |
| Attn `qkv_proj` / `o_proj` | **fp16 sin cuantizar** (en `ignore`) | Marlin **W8A16** | Marlin **W4A8-int8** |
| FFN densa | — | Marlin **W8A16** | Marlin **W4A8-int8** |
| MoE expertos ruteados | **Marlin MoE INT4** (`check_moe_marlin_supports_layer` OK) | — | — |
| MoE shared expert / router | **fp16 sin cuantizar** (en `ignore`) | — | — |
| `lm_head` | fp16 → FP8 por parche PN77 | fp16 → FP8 por PN77 | fp16 → FP8 por PN77 |
| Capas MTP | n/a | bf16→fp16, + PN8 quant online | bf16→fp16, + PN8 quant online |
| Atención | FlashInfer, KV fp8_e4m3 | FlashInfer, KV fp8_e4m3 | FlashInfer, KV fp8_e4m3 |
| GDN prefill | Triton/FLA | Triton/FLA | Triton/FLA |
| GDN decode | Triton fusionado | Triton fusionado | Triton fusionado |
| Norms | GemmaRMSNorm, camino nativo ⚠ | idem ⚠ | idem ⚠ |

---

## 3. Tabla maestra: bloque × kernel × dtype × conversión

Leyenda de coste: **A** = alto, **M** = medio, **B** = bajo, **0** = sin coste.

| # | Bloque / archivo:línea | Kernel real | dtype entrada → salida | Conversión on-the-fly | Coste | Oportunidad |
|---|---|---|---|---|---|---|
| 1 | `layernorm.py:162` `GemmaRMSNorm` | **nativo/Inductor** (no `_C.rms_norm`) | x fp16, **w fp32** → fp16 | `weight.float()+1.0` **en cada llamada** | **A** | Precomputar `w+1` en fp16 al cargar → habilita `vllm_c` y el pase de fusión |
| 2 | `vllm_c.py:16-19` guard | — | — | `weight.dtype == x.dtype` falla | **A** | Consecuencia de #1 |
| 3 | `rms_quant_fusion.py:45-59` | pase de fusión | — | `_rms_input_weight_dtype_match` **bloquea la fusión** | **A** | Consecuencia de #1 |
| 4 | `chunk_delta_h.py:350` `h = k.new_empty(B,NT,H,V,K)` | Triton FLA | fp16 | ninguna | **A** (memoria) | 192 MiB/capa; escala lineal con `max-num-batched-tokens` |
| 5 | `chunk.py:41-51` `A` | `chunk_scaled_dot_kkt` → `solve_tril` | fp16 → **fp32** → fp16 | fp32 intermedio de 96 MiB | **M** | `output_dtype` es parámetro; se puede evaluar fp16 directo (riesgo numérico) |
| 6 | `fused_gdn_prefill_post_conv.py:121-125` | Triton | fp16 → fp16 | ninguna, **copia pura de layout** | **M** | 100 MiB r+w por capa; evitable si `causal_conv1d` escribiera v ya separado |
| 7 | `qwen_gdn_linear_attn.py:923-924` | 2× Marlin | fp16 → fp16 | en Heretic: **2× `per_token_quant_int8` del mismo tensor** | **M** | Cuantizar `hidden_states` una vez y reusar en ambas proyecciones |
| 8 | `marlin_utils.py:524-531` `marlin_quant_input` | Triton `per_token_quant_int8` | fp16 → int8 + fp32 scales | **sí, en cada GEMM** | **M** | Inherente a W4A8; ver #7 y #9 |
| 9 | `int8_utils.py:139` `BLOCK = next_power_of_2(N)` | Triton | — | — | **M** | K=8704 → BLOCK=16384 (88 % de padding). **Kernel especializado por tamaño de capa** |
| 10 | `qwen_gdn_linear_attn.py:616-632` `maybe_disable_tp` | — | — | — | **B** | `isinstance(AutoGPTQConfig)` replica `in_proj_ba` en Heretic aunque esté en fp16 |
| 11 | `qwen3_next.py:301-302` | `aten::copy` | fp16 → fp16 | 2 copias por capa full-attn | **M** | `q`/`gate` no contiguos tras `chunk`; fusionable en el norm |
| 12 | `qwen3_next.py:318-319` | 2 kernels elementwise | fp16 | `sigmoid` + `mul` separados | **B** | Fusionable en un kernel |
| 13 | `flashinfer.py:1580-1584` | FlashInfer PagedKV | q fp16, kv **fp8_e4m3** | **dequant fp8→fp16 en kernel** | inevitable | sm86 sin fp8 HW; el ahorro es de banda, no de FLOPs |
| 14 | `attention.py:97,111` `_k_scale=1.0` | — | fp16 → fp8_e4m3 | escala 1.0 sin calibrar | **B** | Escalas calibradas se pliegan en `bmm1/bmm2_scale`: **precisión gratis** |
| 15 | `qwen_gdn_linear_attn.py:1513-1514` | gather + máscara | fp32 | `.to()` es no-op ✓ | **B** | 1,5 MiB/seq/capa de gather+scatter; el dtype ya es correcto |
| 16 | `qwen_gdn_linear_attn.py:1532` | scatter | fp32 → fp32 | **no-op ✓** | **0** | Correcto: `final_state` ya nace fp32 (`chunk_delta_h.py:352`) |
| 17 | `fused_sigmoid_gating.py:208` `BV=min(pow2(V),32)` | Triton decode | fp16 q/k/v, fp32 estado | ninguna ✓ | **0** | Camino de decode limpio; `BV=32` fija 4 bloques por cabeza |
| 18 | Carga de pesos (Fable/Heretic) | — | **bf16 → fp16** | una vez, en carga | **0** | Correcto para Marlin en Ampere (bf16 no gana nada aquí) |
| 19 | `compilation.py:1474-1517` | CUDA Graph | — | — | **A** | Decode fuera de grafo en Fable/Heretic (§6.2) |
| 20 | `arg_utils.py:2513-2517` | — | — | — | **0** | `--performance-mode throughput` inerte |

---

## 4. Los dos regímenes de tamaño (prefill / decode)

Esto responde directamente a *"los tamaños no varían tanto, son dos variaciones"*. Es
exacto: cada kernel ve **exactamente dos formas** y ambas son conocidas de antemano,
así que ambas admiten especialización estática.

| | KAT-Coder | Fable-Fusion | Heretic |
|---|---|---|---|
| **M en prefill** | ≤ 16384 (chunked, `long_prefill` 8192) | ≤ 16384 | ≤ 16384 |
| **M en decode** | 20 × 1 = **20** | 20 × 3 = **60** | 20 × 4 = **80** |
| `max_cudagraph_capture_size` | 40 | 40 | 40 |
| Tallas capturadas | 1,2,4,8,16,24,32,40 | 3,6,9,18,24,33 | 4,8,16,24,32,40 |
| ¿Decode pleno capturado? | **✓ (20 → 24)** | **✗ 60 > 33** | **✗ 80 > 40** |

### 4.1 Formas de GEMM por rango TP (K × N)

**Densos (Fable / Heretic), `hidden=5120`, TP=2:**

| GEMM | K | N | Notas |
|---|---|---|---|
| `in_proj_qkvz` | 5120 | 8192 | ✓ N%64, K%128 |
| `in_proj_ba` | 5120 | 96 (repl.) / 48 (shard) | **N%64 ≠ 0** → nunca Marlin; queda fp16 |
| GDN `out_proj` | 3072 | 5120 | row-parallel |
| `qkv_proj` | 5120 | 7168 | 2·q + k + v (gate) |
| `o_proj` | 3072 | 5120 | |
| `gate_up_proj` | 5120 | 17408 | |
| `down_proj` | **8704** | 5120 | → `next_pow2` = 16384 en el quant int8 |

**KAT-Coder, `hidden=2048`, TP=2:**

| GEMM | K | N | Notas |
|---|---|---|---|
| `in_proj_qkvz` | 2048 | 6144 | **fp16** (ignore) |
| `in_proj_ba` | 2048 | 32/rank | fp16 |
| GDN `out_proj` | 2048 | 2048 | **fp16** (ignore) |
| `qkv_proj` | 2048 | 4608 | **fp16** (ignore) |
| `o_proj` | 2048 | 2048 | **fp16** (ignore) |
| MoE `w13` (×256 exp.) | 2048 | 512 | **INT4 Marlin MoE** |
| MoE `w2` (×256 exp.) | **256** | 2048 | **INT4 Marlin MoE**, K muy pequeño |
| shared exp. `gate_up`/`down` | 2048/256 | 512/2048 | **fp16** (ignore) |

### 4.2 Estado recurrente y memoria transitoria

| | KAT-Coder | Fable / Heretic |
|---|---|---|
| `ssm_state` fp32 / capa / seq | 1024 KiB | **1536 KiB** |
| `conv_state` fp16 / capa / seq | 24 KiB | 30 KiB |
| Total estado (20 seqs, todas las capas GDN) | **614 MiB/GPU** | **1468 MiB/GPU** |
| `h` (FLA) por capa GDN @ T=16384 | 128 MiB | **192 MiB** |
| `A` fp32 + `A` fp16 por capa | 64 + 32 MiB | 96 + 48 MiB |
| `w` / `u` / `v_new` / `o` por capa | 64 MiB c/u | 96 MiB c/u |
| **Pico transitorio por capa GDN** | ~544 MiB | **~816 MiB** |
| Tráfico de `h` por paso de prefill | 7,5 GiB | **18,0 GiB** |

> A ~800 GiB/s efectivos en una 3090, esos 18 GiB de tráfico sólo por `h` son
> **~22 ms por paso de prefill** en Fable/Heretic. Es el único término que domina
> claramente el prefill de las capas GDN.

---

## 5. Dónde ya hay cobertura en la suite Genesis

Varios de los puntos anteriores ya tienen infraestructura en
`genesis-vllm-patches/vllm/_genesis/kernels/`, lo cual conviene tener presente antes
de duplicar esfuerzo:

| Hallazgo | Kernel Genesis relacionado | ¿Activo en los composes? |
|---|---|---|
| `A` fp32 (#5) | `fla_kkt_buffer.py` | — (no hay flag explícito) |
| Buffers FLA (#4) | `gdn_scratch_pool.py`, `gdn_gating_buffer.py` | — |
| Doble proyección GDN (#7) | `pn50_gdn_fused_proj.py` | **sí** (`GENESIS_ENABLE_PN50_GDN_FUSED_PROJ=1` en los 3) |
| Reducción Marlin | `marlin_fp32_reduce.py`, `marlin_tuning.py` | — |
| Dequant | `dequant_buffer.py`, `fp8_dispatcher.py` | vía PN77 |
| MoE intermedio | `moe_intermediate_cache.py`, `router_softmax.py` | relevante sólo para KAT |
| SiLU+Mul (#12 análogo) | `silu_and_mul_customop.py` | — |
| CUDA Graphs spec-decode (#19) | P100 | **sí**, pero ver §6.2 |

---

## 6. Hallazgos priorizados

### 6.1 KAT-Coder: sólo los expertos están cuantizados

**Evidencia.** La lista `ignore` del `config.json` tiene 561 entradas. Desglose real:

| Entradas | Qué excluye |
|---|---|
| 110 | torre visual (irrelevante: `--language-model-only`) |
| 210 | **las 30 capas GDN completas** (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `norm`) |
| 40 | **`self_attn.q/k/v/o_proj` de las 10 capas full-attn** |
| 160 | **`shared_expert.gate/up/down` + `shared_expert_gate` de las 40 capas** |
| 40 | routers `mlp.gate` |
| 1 | `lm_head` |

Es decir: **lo único en INT4 son los expertos ruteados**. Todo el camino denso corre en
fp16. Sólo las proyecciones GDN suman **~0,94 GiB por GPU** que serían ~0,26 GiB en
INT4-g32 → **~0,68 GiB/GPU recuperables**, más lo de atención y shared experts.

**Qué hacer.** Esto no se arregla en vLLM: es una propiedad del checkpoint. Las opciones
son (a) recuantizar incluyendo el camino denso, o (b) aceptarlo y ajustar
`--gpu-memory-utilization` sabiendo que 0.91 ya está compensando este sobrecoste.
Conviene al menos **verificarlo en los logs de arranque**: debe aparecer
`Using CompressedTensorsWNA16MarlinMoEMethod` pero **no** `Using MarlinLinearKernel for
CompressedTensorsWNA16` para las capas densas.

### 6.2 CUDA Graphs: relleno de tallas (no pérdida del grafo)

> ⚠ **Corrección de una versión anterior de este informe.** Aquí se afirmaba que Fable
> y Heretic "pierden el CUDA Graph de decode" porque el techo era
> `min(max_num_seqs * 2, 512) = 40`. **Era falso**: esa fórmula está en el *docstring*
> de `_set_cudagraph_sizes`, no en el código. La implementación real
> (`config/vllm.py:1672-1680`) **sí multiplica por `num_speculative_tokens + 1`**:
>
> ```python
> decode_query_len = 1
> if self.speculative_config and self.speculative_config.num_speculative_tokens:
>     decode_query_len += self.speculative_config.num_speculative_tokens
> max_cudagraph_capture_size = min(
>     self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
> )
> ```
>
> Los tres contenedores **sí capturan** su lote de decode. Lo que queda es un problema
> menor de relleno.

Simulando fielmente `_set_cudagraph_sizes` + `adjust_cudagraph_sizes_for_spec_decode`:

| | techo | tallas generadas | decode pleno | grafo usado | relleno |
|---|---|---|---|---|---|
| **Heretic** | min(20·4·2, 512) = **160** | 4, 8, 16, 24, …, 160 | 80 | **80** | **0 % — exacto** |
| **Fable** | min(20·3·2, 512) = **120** | 3, 6, 9, 18, 24, 33, 42, 48, 57, 66, … | 60 | **66** | **10 %** |
| **KAT** | min(20·1·2, 512) = **40** | 1, 2, 4, 8, 16, 24, 32, 40 | 20 | **24** | 20 % |

**Pero el problema real no está en el decode pleno, sino en la baja concurrencia.** Las
listas por defecto avanzan a paso 8 mientras M avanza de `num_spec+1` en `num_spec+1`,
así que dejan huecos grandes justo en el rango de 1-10 secuencias, que es donde opera
esta flota:

| seqs | Heretic M → grafo | Fable M → grafo | KAT M → grafo |
|---|---|---|---|
| 3 | 12 → 16 (**+33 %**) | 9 → 9 | 3 → 4 (+33 %) |
| 4 | 16 → 16 | 12 → 18 (**+50 %**) | 4 → 4 |
| 5 | 20 → 24 (+20 %) | 15 → 18 (+20 %) | 5 → 8 (**+60 %**) |
| 9 | 36 → 40 (+11 %) | 27 → 33 (+22 %) | 9 → 16 (**+78 %**) |
| **relleno medio 1-10 seqs** | **7,9 %** | **11,7 %** | **27,9 %** |

Ese relleno se paga en las partes proporcionales a M —GEMM y all-reduce, ~60 % del paso
en los densos—. En KAT el impacto en tiempo es menor porque su decode lo domina la
lectura de pesos (independiente de M), pero el porcentaje de relleno es el peor.

**Qué se aplicó.** `--cudagraph-capture-sizes` explícito en los tres, denso en el rango
bajo y **manteniendo el techo por defecto** (160 / 120 / 40). Mantener el techo importa:
`v1/worker/gpu/cudagraph_utils.py:154` sólo emite grafos de *decode uniforme* hasta
`max_num_reqs × decode_query_len`, pero las tallas por encima siguen usándose para
grafos **mixtos** (prefill+decode), así que recortarlas habría reducido cobertura.

| | tallas | relleno medio 1-10 seqs |
|---|---|---|
| Heretic | 20 (4, 8, 12, …, 40, 48, …, 160) | 7,9 % → **0 %** |
| Fable | 20 (3, 6, 9, …, 30, 36, …, 120) | 11,7 % → **0 %** |
| KAT | 16 (1, 2, …, 8, 10, …, 16, 20, 24, 32, 40) | 27,9 % → **1,1 %** |

### 6.3 `GemmaRMSNorm` desactiva el kernel C y bloquea la fusión

**Evidencia, tres eslabones encadenados:**

1. `layernorm.py:162` — `weight = self.weight.float() + 1.0`, en `forward_native`,
   y `forward_cuda` (`:167-172`) simplemente delega en `forward_native`. Es decir,
   **no existe kernel CUDA propio para `GemmaRMSNorm`**.
2. `vllm_c.py:16-19` — el guard `rms_no_var_size` exige
   `weight.dtype == x.dtype`. Con peso fp32 y activación fp16, **no se selecciona
   `torch.ops._C.rms_norm`**.
3. `rms_quant_fusion.py:45-59` — `_rms_input_weight_dtype_match` está registrado como
   `extra_check` en todos los patrones RMSNorm+quant, con el comentario
   *"Prevent fusion when rms_norm input and weight dtypes differ"*.

`Qwen3_5RMSNorm` **es** `GemmaRMSNorm` (`qwen3_5.py:39-41`), y se usa en
`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` y `model.norm`.
En el modelo denso son **161 llamadas por forward**; en KAT, 101.

**Matiz honesto sobre el impacto.** El punto (3) sólo muerde de verdad si hay una
cuantización fp8 justo después del norm, y estos tres modelos son W*A16 / W4A8-int8,
no fp8-activación. Lo que sí duele en los tres es (1)+(2): se materializa un tensor
fp32 del peso en cada llamada y se cae al camino genérico de Inductor en lugar del
kernel C afinado. El coste en bytes es pequeño (20 KiB); el coste está en el número
de nodos y en perder el kernel fusionado.

### 6.4 El buffer `h` de FLA domina la memoria de prefill

`chunk_delta_h.py:350`: `h = k.new_empty(B, NT, H, V, K)`, con `NT = ceil(T/64)`.
Con T=16384, H=24, V=K=128 → **192 MiB en fp16 por capa GDN**, escrito y releído
íntegro (`chunk_fwd_o` lo consume). Sumando las 48 capas: **18 GiB de tráfico por
paso de prefill**.

Es intrínseco al algoritmo chunked de FLA, no un bug. Las dos palancas reales son:
- **Bajar `--max-num-batched-tokens`** (escala lineal: 8192 → 96 MiB/capa, 9 GiB).
  Contrapartida: menos solapamiento y prefills más largos en número de pasos.
- Reutilizar el buffer entre capas en lugar de dejarlo al caching allocator
  (es lo que apunta `gdn_scratch_pool.py` en la suite Genesis).

### 6.5 `in_proj_ba` replicado en Heretic sin necesidad

`qwen_gdn_linear_attn.py:616-632`:

```python
return (
    current_platform.is_cuda()
    and not self.gqa_interleaved_layout
    and isinstance(quant_config, (AWQMarlinConfig, AutoGPTQConfig, INCConfig))
)
```

En esta versión `"gptq_marlin"` mapea a **`AutoGPTQConfig`**
(`quantization/__init__.py:153`), así que Heretic entra por esta rama y `in_proj_ba`
se **replica** en ambos rangos. Pero el checkpoint de Heretic guarda
`in_proj_a.weight` / `in_proj_b.weight` **sin cuantizar** (verificado en el
`model.safetensors.index.json`: sólo `.weight`, sin `qweight`/`qzeros`/`scales`), luego
la capa nunca habría usado Marlin y la restricción `MIN_THREAD_N=64` no aplicaba.

Resultado: un GEMM fp16 de 5120×96 duplicado en cada capa GDN de cada rango, más el
`split_ba` que descarta la mitad (`:634-642`). El coste en FLOPs es pequeño (N=96),
pero son 48 lanzamientos redundantes por forward y por GPU. La condición debería
mirar si la capa está *efectivamente* cuantizada, no el tipo del `quant_config`.

### 6.6 W4A8-int8: por qué Heretic sí y los otros dos no

`marlin_utils.py:563-568` — cuando `input_dtype == torch.int8`:

```python
assert wtype == scalar_types.uint4b8, "W8A8-INT8 is not supported by marlin kernel."
```

| | Tipo de peso | ¿W4A8-int8 alcanzable? |
|---|---|---|
| Heretic | `uint4b8` (INT4 simétrico) | **✓ activo** |
| Fable | `uint8b128` (INT8) | ✗ el assert lo prohíbe (no hay W8A8 en Marlin) |
| KAT | `uint4` (INT4 **asimétrico**, con zero-points) | ✗ el assert exige `uint4b8` |

Para KAT existe `apply_awq_marlin_linear` (`marlin_utils.py:601-662`) que **sí** acepta
`uint4` con int8, pero `MarlinLinearKernel.apply_weights`
(`kernels/linear/mixed_precision/marlin.py:173-193`) llama siempre a
`apply_gptq_marlin_linear`, así que esa ruta no es alcanzable desde compressed-tensors.
Habría que recuantizar KAT a **simétrico** para desbloquearlo.

**Contrapartida de W4A8.** En una 3090 int8 rinde ~284 TOPS frente a ~71 TFLOPS en
fp16, de ahí la ganancia en prefill. W4A8 **añade** un `per_token_quant_int8` por cada
GEMM, y `VLLM_MARLIN_INPUT_DTYPE` es global: no se puede activar sólo en prefill. Con
CUDA Graphs el sobrecoste de lanzamiento se amortiza — pero justo Heretic es el
contenedor que **no** está capturando el decode (§6.2), así que hoy paga el coste completo.

> ⚠ **Corrección.** Una versión anterior de este párrafo afirmaba que en decode el GEMM
> está limitado por ancho de banda y que por tanto W4A8 no aporta ahí. **Es falso para
> Heretic**: con MTP=3 y 20 secuencias, M=80 da una intensidad de 310 FLOP/byte, muy por
> encima del *ridge point* fp16 de la 3090 (89), así que el GEMM de decode está limitado
> por **cómputo** y W4A8 lo acelera ~3,5× (28,9 ms → 8,3 ms). El desarrollo completo, con
> los umbrales de M por formato, está en
> `ANALISIS-ANCHOBANDA-MEMORIA-3-CONTENEDORES.md` §1.

### 6.7 Especialización por tamaño en `per_token_quant_int8` (sólo Heretic)

`int8_utils.py:139`: `BLOCK = triton.next_power_of_2(N)`, con `num_warps` topado a 8.

| Capa (Heretic, por rango) | K real | `BLOCK` | Desperdicio |
|---|---|---|---|
| `in_proj_qkvz`, `qkv_proj`, `gate_up_proj` | 5120 | 8192 | 60 % |
| GDN `out_proj`, `o_proj` | 3072 | 4096 | 33 % |
| **`down_proj`** | **8704** | **16384** | **88 %** |

Los `tl.load` van enmascarados, así que no hay tráfico de memoria extra; el coste está
en presión de registros y ocupación. Como sólo hay **tres valores distintos de K** en
todo el modelo, es el caso de libro para tener configuraciones fijadas por tamaño
(`BLOCK`/`num_warps` elegidos por K, o un `triton.autotune` con `key=["N"]`) en lugar
del heurístico `next_power_of_2`.

### 6.8 Escalas de KV cache sin calibrar

`attention.py:97,111` — `_k_scale` y `_v_scale` valen **1.0** salvo que el checkpoint
traiga escalas; ninguno de los tres las trae. Con `fp8_e4m3` (máx. 448) y escala 1.0,
K va bien (sale normalizado de `k_norm`) pero **V es una proyección cruda sin normalizar**
y puede perder precisión relativa en la cola baja del rango.

Lo relevante es que **corregirlo es gratis en tiempo de ejecución**: las escalas se
pliegan una sola vez en `bmm1_scale`/`bmm2_scale` (`flashinfer.py:1401-1410`), no hay
kernel extra. Es la mejora de precisión con mejor relación coste/beneficio del informe.

---

## 7. Recomendaciones ordenadas

### Coste cero, sin riesgo numérico — sólo configuración

| # | Acción | Contenedor | Efecto esperado |
|---|---|---|---|
| 1 | ✅ `--cudagraph-capture-sizes` con 60 exacto | Fable | Elimina el relleno 60→66 (~6 % del decode) (§6.2) |
| 2 | Quitar `--performance-mode throughput` o dejar de fijar `max-num-seqs`/`max-num-batched-tokens` | los 3 | Hoy el flag es inerte; clarifica la config |
| 3 | Quitar `--enable-flashinfer-autotune` del razonamiento sobre GDN | los 3 | No cambia rendimiento; evita conclusiones falsas al medir |
| 4 | ✅ `--max-num-batched-tokens 8192` + `--long-prefill-token-threshold 4096` | los 3 | Halva el pico de memoria de `h` (§6.4). Coste en velocidad ~nulo: el tráfico FLA total de dos chunks de 8192 es idéntico al de uno de 16384, y sólo supone el 3 % del prefill — ver informe de ancho de banda §5.3 |

### Coste bajo, requiere medición

| # | Acción | Contenedor | Riesgo |
|---|---|---|---|
| 5 | Escalas KV calibradas en el checkpoint | los 3 | Ninguno en velocidad; mejora precisión (§6.8) |
| 6 | `BLOCK`/`num_warps` fijados por K en `per_token_quant_int8` | Heretic | Bajo; 3 tamaños conocidos (§6.7) |
| 7 | Precomputar `weight + 1` en fp16 al cargar `GemmaRMSNorm` | los 3 | Medio: cambia el orden de operaciones en fp16 vs fp32. Hay que validar perplejidad (§6.3) |
| 8 | Corregir el `isinstance` de `maybe_disable_tp` | Heretic | Bajo (§6.5) |
| 9 | Cuantizar `hidden_states` una vez para `in_proj_qkvz` + `in_proj_ba` | Heretic | Bajo; parcialmente cubierto por PN50 (§6.7/#7) |

### Requiere recuantizar el checkpoint

| # | Acción | Contenedor | Ganancia estimada |
|---|---|---|---|
| ⏸ | Cuantizar también el camino denso | KAT | −0,64 GiB (INT8) / −0,94 (INT4), −5,5 % / −8,1 % del decode. **Descartado**: mal ratio esfuerzo/beneficio, y la receta "sólo expertos" cubre el 92 % de los parámetros — ver informe de ancho de banda §5.1 |
| 11 | Recuantizar a INT4 **simétrico** | KAT | Único cambio con premio real: mueve **todo el modelo** de `HMMA` a `IMMA` (§6.6). Exige rehacer los expertos (92 % de los pesos), y sólo gana en prefill |
| 12 | Recuantizar a INT8 **per-channel** (`group_size = -1`) | Fable | Desbloquea AllSpark, el kernel W8A16 de Ampere (§2.4) — **hipótesis a medir**. Conserva el INT8, que es el propósito del contenedor |

### No hacer

- No perseguir el camino FlashInfer GDN ni CuteDSL: son inalcanzables en sm86.
- No esperar aceleración de cómputo del KV cache fp8: en Ampere el beneficio es
  capacidad y ancho de banda, nunca FLOPs.
- No activar `VLLM_MARLIN_INPUT_DTYPE=fp8`: `marlin_utils.py:507-516` lo rechaza
  explícitamente fuera de SM89/SM12x.

---

## 8. Cómo verificar todo esto en el arranque

Con `GENESIS_LOG_LEVEL=DEBUG` ya activo, estas líneas confirman o refutan cada hallazgo:

| Qué buscar en el log | Confirma |
|---|---|
| `Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` | §2.1 |
| `Using MarlinLinearKernel for CompressedTensorsWNA16` | §2.4 (ausente en KAT para capas densas → §6.1) |
| `Using AutoGPTQLinearMethod` + `MarlinLinearKernel` | Heretic §2.5 |
| `Using CompressedTensorsWNA16MarlinMoEMethod` | KAT §2.4 |
| `Capturing CUDA graphs ... sizes=[...]` | §6.2 — comparar el máximo con 60/80 |
| Ausencia de `AllSparkLinearKernel` | §2.4 |

---

*Informe generado con asistencia de IA (Claude Opus 5) sobre lectura estática del
código en `assets/vllm` @ `0fc695f`, los tres `docker-compose` y los `config.json` /
`model.safetensors.index.json` de los tres checkpoints. Las cifras de memoria y de
formas de capa son cálculos derivados de esas configuraciones, no mediciones en
ejecución; conviene contrastarlas con un perfilado real antes de actuar sobre los
puntos que requieren cambios de código o recuantización.*
