# Análisis de ancho de banda y memoria — 3 contenedores vLLM

**Fecha:** 2026-07-30
**Complementa:** `ANALISIS-KERNELS-DTYPES-3-CONTENEDORES.md` (mismo directorio)
**Objetivo:** acelerar prefill, acelerar decode, reducir memoria estática
**Alcance:** sólo análisis. No se modificó código.

> **Naturaleza de las cifras.** Todo lo que sigue es un modelo analítico derivado de
> las configuraciones reales (formas de capa, dtypes, esquemas de cuantización) y de
> las especificaciones del hardware. **No son mediciones.** Sirven para ordenar
> prioridades y descartar callejones sin salida, no para prometer números. Cada
> recomendación indica qué habría que medir para confirmarla.

---

## 0. Las tres cifras que deciden todo

| Recurso | Valor | Comentario |
|---|---|---|
| **HBM (GDDR6X)** | 936 GB/s pico → **~796 GB/s efectivo** | por GPU |
| **Cómputo fp16 TC** | 71 TFLOPS (acum. fp32) | GA102 hace fp32-accum a mitad de ritmo |
| **Cómputo int8 TC** | 284 TOPS | 4× el fp16 |
| **PCIe entre GPUs** | **Gen4 x8 → ~11,8 GB/s útil** | **sin NVLink** |

```
$ nvidia-smi nvlink -s     → "all links are inActive"
$ nvidia-smi topo -m       → GPU0/GPU1 = PIX (un solo puente PCIe)
$ lspci -vvv               → LnkCap: Speed 16GT/s, Width x16
                             LnkSta: Speed 2.5GT/s (downgraded), Width x8 (downgraded)
```

El **ancho x8** es físico (placa AM5, bifurcación x8/x8) y no se recupera. La velocidad
Gen1 observada es el estado de reposo; bajo carga debería reentrenar a Gen4.
**Esto hay que verificarlo durante una petición real** con
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -l 1`,
porque si el enlace se queda en Gen3 o Gen1 todos los tiempos de comunicación de este
informe se multiplican por 2 u 8 respectivamente.

**La relación HBM:PCIe es de 67:1.** Cualquier byte que cruce entre GPUs cuesta 67 veces
más que leerlo de la memoria local. Con `--tensor-parallel-size 2` y 128 all-reduces por
paso, eso resulta ser el factor dominante del prefill (§4).

### Puntos de inflexión (roofline)

| Precisión | Ridge point | Interpretación |
|---|---|---|
| fp16 | **89 FLOP/byte** | por debajo → limitado por banda; por encima → por cómputo |
| int8 | **357 FLOP/byte** | activaciones int8 suben el techo 4× |

Para un GEMM con pesos cuantizados y M tokens, la intensidad es `2·M / bytes_por_parámetro`.
De ahí sale **el umbral exacto de M donde cada modelo cambia de régimen**:

| Formato de peso | B/par | Deja de estar limitado por banda si… (act. fp16) | …(act. int8) |
|---|---|---|---|
| INT4 g128 (Heretic) | 0,516 | **M > 23** | M > 92 |
| INT4 g32 (KAT expertos) | 0,563 | M > 25 | M > 100 |
| INT8 g128 (Fable) | 1,016 | **M > 45** | M > 181 |
| fp16 (KAT camino denso) | 2,000 | M > 89 | M > 357 |

Y los M reales de decode son **20 (KAT), 60 (Fable), 80 (Heretic)**. Es decir: los tres
caen justo a caballo de sus umbrales. Esto es lo que hace que la elección de dtype de
activación importe tanto aquí, y es la respuesta cuantitativa a lo de "dos variaciones
de tamaño".

---

## 1. Corrección a mi informe anterior

En el primer informe escribí que W4A8-int8 *"gana en prefill pero no aporta en decode,
donde el cuello es la banda de pesos"*. **Eso es incorrecto para Heretic**, y el motivo
es precisamente el spec decode:

| Heretic, decode M=80 | Tiempo |
|---|---|
| Leer pesos (6,15 GiB @ 796 GB/s) | 8,3 ms |
| Cómputo GEMM en **fp16** (2,05 TFLOP @ 71 TFLOPS) | **28,9 ms ← cuello** |
| Cómputo GEMM en **int8** (2,05 TFLOP @ 284 TOPS) | 7,2 ms |

Con MTP=3 y 20 secuencias, M=80 da una intensidad de **310 FLOP/byte**, muy por encima
del ridge fp16 (89). El GEMM de decode está **limitado por cómputo**, no por banda.
W4A8 lo devuelve al régimen de banda: **28,9 ms → 8,3 ms, un 3,5× en la parte de GEMM**.

`VLLM_MARLIN_INPUT_DTYPE=int8` en Heretic es, por tanto, más valioso de lo que dije:
ayuda en las dos fases. La observación que sí se mantiene es que el coste del
`per_token_quant_int8` se paga en ambas fases — pero sí se amortiza con CUDA Graph:
Heretic captura su lote de decode de 80 tokens de forma exacta (ver §6.2 corregido del
informe anterior).

**Corolario para Fable:** con M=60 e INT8 g128, la intensidad es 118 > 89 → su GEMM de
decode **también está limitado por cómputo** (21,6 ms frente a 16,3 ms de banda), y
**no tiene escapatoria**: Marlin no implementa W8A8 (`marlin_utils.py:564`). Fable está
atrapado en el peor cuadrante de los tres.

### 1.1 Todo lo anterior depende de la concurrencia real

Los números de arriba suponen `max-num-seqs` **saturado**. Como M = seqs × (spec+1), el
régimen cambia con la carga real:

| seqs | Heretic M | intensidad | régimen | Fable M | intensidad | régimen |
|---|---|---|---|---|---|---|
| 1 | 4 | 16 | banda | 3 | 6 | banda |
| 2 | 8 | 31 | banda | 6 | 12 | banda |
| 4 | 16 | 62 | banda | 12 | 24 | banda |
| **6** | 24 | 93 | **cómputo** | 18 | 35 | banda |
| 12 | 48 | 186 | cómputo | 36 | 71 | banda |
| **16** | 64 | 248 | cómputo | 48 | 95 | **cómputo** |
| 20 | 80 | 310 | cómputo | 60 | 118 | cómputo |

**Consecuencia práctica sobre W4A8 en Heretic:** sólo aporta en decode a partir de
**~6 secuencias concurrentes**. Por debajo, el GEMM ya está limitado por banda y el
`per_token_quant_int8` es coste neto sin contrapartida. Conviene mantenerlo igualmente
porque **en prefill el GEMM está limitado por cómputo siempre** (T = 8192 tokens), y ahí
gana ~4×; pero no hay que esperar mejora de decode en uso de baja concurrencia.

Y para Fable: su decode sólo se vuelve compute-bound a partir de ~16 secuencias. Por
debajo, su desventaja frente a Heretic es puramente de ancho de banda —13,3 GiB de pesos
por paso frente a 7,6— es decir, un factor 1,75 constante.

**Nota sobre el contexto disponible.** A baja concurrencia el KV deja de ser limitante, y
`max-num-seqs` apenas influye en el tamaño del pool (los pesos lo dominan):

| seqs | Heretic | Fable | KAT |
|---|---|---|---|
| 2 | 451 K/seq | 265 K/seq | 1 051 K/seq |
| 4 | 223 K/seq | 130 K/seq | 522 K/seq |
| 8 | 109 K/seq | 63 K/seq | 258 K/seq |
| 20 | 41 K/seq | 22 K/seq | 99 K/seq |

---

## 2. Memoria estática por GPU

| | **Heretic** | **Fable** | **KAT-Coder** |
|---|---|---|---|
| Pesos cuantizados | 12,16 G par → **5,84 GiB** | 12,16 G par → **11,50 GiB** | 16,11 G par → **8,44 GiB** |
| Pesos en fp16 | 0,65 G par → 1,21 GiB | 0,65 G par → 1,21 GiB | **0,98 G par → 1,82 GiB** |
| `lm_head` (con PN77 fp8) | 0,59 GiB | 0,59 GiB | 0,24 GiB |
| **Total pesos** | **7,64 GiB** | **13,30 GiB** | **10,50 GiB** |
| `--gpu-memory-utilization` | 0,96 → 23,04 GiB | 0,96 → 23,04 GiB | 0,91 → 21,84 GiB |
| Estado GDN (20 seqs) | 1,43 GiB | 1,43 GiB | 0,60 GiB |
| Margen activaciones/grafos | ~1,5 GiB | ~1,5 GiB | ~1,5 GiB |
| **Queda para KV cache** | **~12,5 GiB** | **~6,8 GiB** | **~9,2 GiB** |
| KV por token (fp8, por GPU) | 16,0 KiB | 16,0 KiB | 5,0 KiB |
| **Capacidad KV total** | ~817 K tokens | ~446 K tokens | ~1 938 K tokens |
| **Por secuencia con 20 seqs** | **~41 K** | **~22 K** | **~97 K** |

### 2.1 El `--max-model-len 262144` es nominal

Los tres anuncian 262 K de contexto, pero la memoria sólo permite servirlo a:

| | secuencias simultáneas a 262 K | a 32 K |
|---|---|---|
| Heretic | **3** | 25 |
| Fable | **1** | 13 |
| KAT | **7** | 59 |

Con `--max-num-seqs 20`, pedir contextos largos hará que el scheduler expulse
(*preempt*) secuencias. No es un fallo — es el comportamiento correcto — pero conviene
saber que **el número de 262 K y el de 20 secuencias son mutuamente excluyentes**.

### 2.2 Por qué el KV pesa tanto: `head_dim = 256`

Estos modelos usan `head_dim=256` (el doble de lo habitual). Por GPU y por token:
`2 (K,V) × 2 kv-heads × 256 × 1 byte (fp8) × 16 capas = 16 KiB`. Ya está en fp8; en
fp16 serían 32 KiB y no cabría nada. **El `--kv-cache-dtype fp8` no es un lujo aquí,
es lo que hace viable la configuración** — aunque en Ampere no aporte ni un FLOP (§0
del informe anterior).

KAT paga 3,2× menos (5 KiB/token) porque tiene 10 capas de atención en vez de 16 y
1 kv-head por rango en vez de 2. Es, con diferencia, el más cómodo en memoria de KV.

### 2.3 Fable: el más caro en memoria, y es deliberado

13,30 GiB de pesos sobre 23,04 GiB disponibles: **el 58 % del presupuesto se va en
pesos**, dejando 6,8 GiB de KV (~22 K tokens/seq con 20 secuencias). Sobre el papel, pasar
a INT4 g128 le daría **+84 % de contexto y ~2,6× en el GEMM de decode**:

| | INT8 (hoy) | INT4 (hipotético) |
|---|---|---|
| Pesos | 13,30 GiB | 7,64 GiB |
| KV disponible | 6,8 GiB | 12,5 GiB |
| Tokens/seq (20 seqs) | 22 K | 41 K |
| GEMM decode | 21,6 ms (limitado por cómputo) | 8,3 ms con W4A8 |

> ⚠ **Corrección de encuadre.** Una versión anterior concluía de esta tabla que *"el
> contenedor que conviene reemplazar es Fable"*. **Eso medía Fable contra el eje
> equivocado.**
>
> `lued/Qwen3.6-27B-Fable-Fusion-711-INT8-W8A16-MTP` deriva de
> `DavidAU/Qwen3.6-27B-Fable-Fusion-711-**Uncensored-Heretic**-NM-DAU-MTP`, y su ficha
> justifica el esquema exactamente con el mismo razonamiento de hardware que hace este
> informe: *"on RTX 3090 (sm_86) there is **no FP8 tensor-core path**"*, de ahí W8A16.
> El INT8 **es** lo que se está comprando: es el contenedor de calidad.
>
> Además, en la escalera que admite Marlin —**INT4 → INT8 → fp16**, sin escalones
> intermedios (`WNA16_SUPPORTED_TYPES_MAP = {4, 8}`)— y con bf16 fuera de alcance
> (27,78 G par × 2 B ≈ 28 GB/GPU con TP=2), **INT8 ya es el techo de calidad que cabe en
> esta máquina**. No hay nada mejor que ofrecerle.

Vistos como conjunto, los tres contenedores son coherentes y cubren ejes distintos:

| | Rol | Trade aceptado |
|---|---|---|
| **Heretic** INT4 W4A8 | uncensored **rápido** | `IMMA`, 41 K contexto, prefill ~4 800 tok/s |
| **Fable** INT8 W8A16 | uncensored **de calidad** | `HMMA`, 22 K contexto, prefill ~2 100 tok/s |
| **KAT** MoE INT4 | coder eficiente | el mejor de los tres en los tres ejes |

El "peor cuadrante" de Fable es el **precio de la calidad INT8**, no un error de
configuración.

---

## 3. Ancho de banda en decode

Desglose por paso de decode, 20 secuencias, contexto 32 K:

| Componente | Heretic (M=80) | Fable (M=60) | KAT (M=20) |
|---|---|---|---|
| GEMM (efectivo) | 8,3 ms *(banda)* | **21,6 ms** *(cómputo)* | 12,3 ms *(banda)* |
| Estado GDN (r+w) | 2,87 GiB → 3,9 ms | 2,87 GiB → 3,9 ms | 1,20 GiB → 1,6 ms |
| KV cache leído | 10,00 GiB → 13,5 ms | 10,00 GiB → 13,5 ms | 3,12 GiB → 4,2 ms |
| All-reduce PCIe | 100 MiB → 8,9 ms | 75 MiB → 6,7 ms | 6,2 MiB → 0,6 ms |
| **Paso total ~** | **~34,5 ms** | **~45,7 ms** | **~18,7 ms** |

### 3.1 Escalado con el contexto

| Contexto | Heretic | Fable | KAT |
|---|---|---|---|
| 8 K | 24,4 ms | 35,5 ms | 15,5 ms |
| 32 K | 34,5 ms | 45,7 ms | 18,7 ms |
| 128 K | 75,0 ms | 86,1 ms | 31,4 ms |

**La mitad buena de la arquitectura híbrida:** el estado GDN (2,87 GiB) **no crece con
el contexto**. Sólo las 16 capas de atención plena escalan linealmente. Por eso el
decode a 128 K "sólo" cuesta 2-3× lo que cuesta a 8 K, en vez de 16×.

**La mitad mala:** ese estado fijo de 2,87 GiB por paso equivale a leer **el 47 % de los
pesos de Heretic** en cada token generado, y se paga incluso con contextos cortos. Es el
precio de tener `ssm_state` en fp32 (§5.2).

### 3.2 KAT: los expertos se leen casi enteros

Con M=20 y top-8 se tocan **160 expertos únicos de 256** por capa → se leen 5,27 GiB de
los 8,44 GiB de expertos para generar 20 tokens. La consecuencia es contraintuitiva y
muy explotable:

| `max-num-seqs` | M | Expertos únicos | GiB leídos | GiB **por token** |
|---|---|---|---|---|
| 20 | 20 | 160/256 | 5,27 | 0,264 |
| 32 | 32 | 256/256 (saturado) | 8,44 | 0,264 |
| 64 | 64 | 256/256 | 8,44 | **0,132** |
| 128 | 128 | 256/256 | 8,44 | **0,066** |

Una vez saturados los 256 expertos, **cada token adicional es gratis en ancho de banda
de expertos**. KAT es el candidato obvio a subir `--max-num-seqs` de forma agresiva:
tiene el KV más barato (5 KiB/token), el all-reduce más barato (0,6 ms) y la curva de
expertos más favorable. Su limitación es memoria de KV, no banda.

### 3.3 `--disable-custom-all-reduce` cuesta en decode

`custom_all_reduce.py:54` fija `max_size = 8 MiB`. Los mensajes de decode
(800/600/80 KiB) **están todos por debajo**, así que el all-reduce personalizado
—de menor latencia que NCCL para mensajes pequeños— se aplicaría… si no estuviera
desactivado en los tres composes.

En prefill los mensajes son de 168 MiB, muy por encima del umbral, así que ahí el flag
es indiferente.

**Matiz importante antes de tocarlo:** vLLM sólo habilita el custom AR si detecta
acceso P2P entre las GPUs, y en tarjetas GeForce el P2P sobre PCIe es notoriamente
frágil según driver y placa. Es bastante probable que ese flag esté puesto justamente
por una inestabilidad observada. Si se prueba a quitarlo, hay que hacerlo con una
batería de peticiones larga y vigilando corrupción de salida, no sólo un smoke test.

---

## 4. Prefill: el PCIe se come el beneficio del int8

Chunk de T=16384 tokens:

| | Heretic (int8) | Fable (fp16) | KAT (fp16) |
|---|---|---|---|
| Cómputo GEMM | 1 477 ms | 5 907 ms | 1 188 ms |
| Tráfico FLA (buffers GDN) | 72 GiB → 97 ms | 72 GiB → 97 ms | 30 GiB → 41 ms |
| **All-reduce PCIe** | **20 GiB → 1 818 ms** | 20 GiB → 1 818 ms | 5 GiB → 454 ms |
| **Reparto** | cómputo 44 % / FLA 3 % / **PCIe 54 %** | cómputo 76 % / PCIe 23 % | cómputo 71 % / PCIe 27 % |
| Total estimado | ~3,4 s (**4 800 tok/s**) | ~7,8 s (2 100 tok/s) | ~1,7 s (9 700 tok/s) |

**Este es el hallazgo central de este informe.** El volumen de all-reduce en prefill es
`2 · L · T · hidden · 2 bytes` = **20 GiB por chunk** en los modelos densos. A 11,8 GB/s
son 1,8 segundos que no se solapan con nada.

La ironía: **cuanto mejor se optimiza el cómputo, peor se ve el PCIe**. En Fable
(fp16) el PCIe es el 23 % del tiempo; en Heretic, al bajar el cómputo 4× con W4A8, el
mismo PCIe pasa a ser el **54 %**. Cualquier trabajo adicional sobre kernels de GEMM en
Heretic tiene un techo duro: aunque el cómputo fuera instantáneo, el prefill no bajaría
de ~1,9 s.

Nótese también que el tráfico de los buffers FLA (§6.4 del informe anterior) resulta ser
sólo el **3 %** del prefill. Sigue importando por el **pico de memoria** (816 MiB por
capa GDN), no por el tiempo. Eso reordena su prioridad: es un problema de memoria, no de
velocidad.

### 4.1 La alternativa estructural: PP=2 en vez de TP=2

Para una caja de 2 GPUs sin NVLink, el paralelismo de tubería mueve **dos órdenes de
magnitud menos datos**:

| | TP=2 | PP=2 |
|---|---|---|
| Transferencias por forward | 128 all-reduce | **1 paso de activaciones** |
| Volumen en prefill (T=16384) | **20 GiB** | 168 MiB |
| Volumen en decode (M=80) | 100 MiB | 800 KiB |
| Tiempo PCIe en prefill | ~1 818 ms | **~14 ms** |

Memoria: equivalente (cada GPU guarda la mitad de las capas en vez de la mitad de cada
capa). vLLM soporta PP en estos modelos (`Qwen3_5ForCausalLMBase` implementa
`SupportsPP`) y la única incompatibilidad explícita con spec decode es EAGLE3
(`config/vllm.py:2018-2022`), que no es el caso aquí (usan `qwen3_next_mtp`).

**Contrapartidas honestas:**
- PP no reduce la latencia de un token: éste atraviesa las dos etapas en serie. Gana en
  *throughput* agregado, no en TTFT ni TPOT de una sola secuencia. Con 20 secuencias
  concurrentes y `--performance-mode throughput` como intención declarada, eso encaja.
- Hay burbujas de tubería si no hay suficientes micro-lotes en vuelo.
- Híbrido mamba + MTP + PP es un camino poco transitado en vLLM; hay que probarlo, no
  darlo por bueno.

Aun con todas esas reservas, **es la palanca de mayor recorrido del informe** para
prefill, y merece un experimento controlado antes que cualquier micro-optimización de
kernel.

---

## 5. Palancas de memoria, ordenadas

### 5.1 Cuantizar el camino denso de KAT — ⏸ DESCARTADO tras recalcular

> ⚠ **Corrección.** Una versión anterior de esta sección daba **−2,6 GiB/paso → −18 %**.
> Era un error de doble conteo: **los pesos se leen una sola vez por paso**, así que el
> ahorro de ancho de banda no puede superar al ahorro estático. El camino denso
> cuantizable son 1,31 GiB, no 2,6.

| Bloque | G par | fp16 | INT8 g128 | INT4 g32 |
|---|---|---|---|---|
| GDN `in_proj_qkvz` + `out_proj` | 503 M | 0,94 GiB | 0,48 GiB | 0,26 GiB |
| Atención `q/k/v/o_proj` (10 capas) | 136 M | 0,25 GiB | 0,13 GiB | 0,07 GiB |
| Shared experts (40) | 63 M | 0,12 GiB | 0,06 GiB | 0,03 GiB |
| **Total cuantizable** | | **1,31 GiB** | **0,66 GiB** | **0,37 GiB** |
| **Ahorro** (estático = banda/paso) | | | **0,64 GiB** | **0,94 GiB** |

Sobre un paso de decode de 11,66 GiB / 15,7 ms: **−5,5 %** con INT8, **−8,1 %** con INT4.
No el −18 % que decía antes.

Se dejarían fuera `linear_attn.in_proj_a/b` (2 M par; alimentan `softplus`/`sigmoid`
exponencialmente sensibles, y con N=64→32 por rango quedan por debajo de
`MIN_THREAD_N=64`, luego Marlin los rechazaría igual) y los routers `mlp.gate` (21 M par;
su error cambia *qué* expertos se eligen: discreto, no suave).

**Por qué se descarta.** Dos razones:

1. **La receta del checkpoint es defendible, no un descuido.** Los expertos ruteados son
   32,2 G de ~35 G de parámetros — el **92 %**. La lista `ignore` implementa la
   estrategia estándar de MoE: cuantizar el 92 % y dejar intacto el 8 % denso. Estaríamos
   persiguiendo ese último 8 %.
2. **La ganancia real estaría en la instrucción, no en los bytes** — y no es alcanzable
   sin rehacer también los expertos. Ver §5.1.1.

#### 5.1.1 Lo que sí valdría: `HMMA` → `IMMA`

Con cuantización *weight-only* (W4A16 / W8A16) **la GPU nunca hace aritmética entera**:
Marlin descomprime los pesos a fp16 en registros y emite `HMMA`, la misma instrucción que
si fueran fp16 de origen. El formato sólo cambia cuántos bytes cruzan la HBM.

| Esquema | Cadena por GEMM | Instrucción |
|---|---|---|
| fp16 (KAT denso hoy) | cargar → MMA | `HMMA.F32` — **cero conversiones** |
| INT4 asim (KAT expertos) | int4 → zero-point → × escala → fp16 → MMA | `HMMA.F32` |
| INT8 (Fable) | int8 → × escala → fp16 → MMA | `HMMA.F32` |
| **INT4 sim + W4A8** (Heretic) | act fp16→int8 ⟶ int4→int8 ⟶ MMA ⟶ int32→fp16 | **`IMMA`** (284 vs 71 TOPS) |

El premio de KAT no serían los 0,94 GiB del camino denso sino pasar **todo el modelo** de
`HMMA` a `IMMA`. Pero sus expertos están en INT4 **asimétrico** (`uint4`), y el assert de
`apply_gptq_marlin_linear` (`marlin_utils.py:564`) exige `uint4b8` (simétrico). Habría que
recuantizar el 92 % del modelo, no el 8 %.

Y aun así, la ganancia sería **sólo de prefill**: en decode a la concurrencia real de esta
flota (M=20, intensidad ~11 FLOP/B contra un ridge de 89) KAT está profundamente limitado
por banda, donde la instrucción es irrelevante.

### 5.2 `ssm_state` fp32 → 16 bits — ⏸ APLAZADO

> **Estado: analizado, no priorizado.** El ahorro es real y la opción existe sin tocar
> código, pero validarlo exige un estudio de degradación en contexto largo que no
> compensa frente a las otras palancas. Se documenta aquí para retomarlo cuando el resto
> esté agotado.

| | Heretic/Fable | KAT |
|---|---|---|
| Estático (20 seqs) | 1,43 → **0,72 GiB** | 0,60 → 0,30 GiB |
| Banda por paso de decode | 2,87 → **1,44 GiB** (−1,8 ms) | 1,20 → 0,60 GiB |
| Página mamba (fragmentación KV) | 1 566 → **798 KiB** | 1 048 → 536 KiB |

Hoy los tres heredan `float32` del `config.json` vía `models/config.py:546-549`.

**Los kernels GDN ya lo soportan sin cambios.** Verificado: el estado se maneja de forma
agnóstica al dtype en toda la ruta —acumulador `tl.float32` fijo, subida al cargar,
bajada al guardar— y no hay ningún `assert` sobre su tipo:

| Sitio | Código |
|---|---|
| `fla/ops/fused_sigmoid_gating.py:102` | `b_h = tl.zeros([BV, BK], dtype=tl.float32)` |
| `fla/ops/fused_sigmoid_gating.py:120` | `b_h += tl.load(p_h0, ...).to(tl.float32)` |
| `fla/ops/fused_sigmoid_gating.py:166,170` | `tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))` |
| `fla/ops/chunk_delta_h.py:113` | `b_h1 += tl.load(p_h0_1, ...).to(tl.float32)` |

#### Preferir `bfloat16` sobre `float16`

`MambaDType` (`config/cache.py:36`) acepta `"auto" | "float32" | "float16" | "bfloat16"`.
Para un **acumulador recurrente**, bf16 es la mejor de las dos opciones de 16 bits:

| | fp16 | bf16 |
|---|---|---|
| Bits de mantisa | 10 | 8 |
| Rango de exponente | subnormal < 6e-5, cero < 6e-8 | igual que fp32 (hasta ~1e-38) |
| Riesgo dominante | **precipicio de subnormales** | pérdida de precisión gradual |

El estado GDN decae geométricamente (`g = exp(-exp(A_log)·softplus(·))`, con g ∈ (0,1)),
así que pasa mucho tiempo en valores pequeños — justo donde fp16 se rompe. Además **los
tres checkpoints son nativamente bf16**, es decir, el modelo se entrenó en ese régimen
numérico. La ruta a probar es `--mamba-ssm-cache-dtype bfloat16`, no `float16`.

#### El *stochastic rounding* de vLLM no aplica a estos modelos

`config/mamba.py:64-73` restringe `--enable-mamba-cache-stochastic-rounding` a compute
capability 10.0 (Blackwell) porque usa la instrucción PTX `cvt.rs`. **Pero esa
restricción es irrelevante aquí: la funcionalidad no cubre la ruta GDN en ningún
hardware.**

```
cvt.rs.f16x2.f32       mamba_ssm.py:207-219   (convert_rs_fp16x2)
  └─ usado sólo en     _selective_scan_update_kernel:485
       └─ alcanzable   selective_state_update()
            └─ llamado mamba_mixer.py:431    (Mamba1)
                       mamba_mixer2.py:1030  (Mamba2)
```

`QwenGatedDeltaNetAttention` no llama a ninguno de los dos: usa
`fused_sigmoid_gating_delta_rule_update` (decode) y `chunk_gated_delta_rule` (prefill),
en `fla/ops/`, **donde no existe stochastic rounding en absoluto**. Incluso en una GPU
Blackwell, el flag no haría nada en estos tres contenedores.

#### Si algún día hiciera falta el parche

Emular `cvt.rs` en Ampere es trivial —sumar bits aleatorios a la parte descartada antes
de truncar—:

```python
@triton.jit
def convert_rs_fp16_emulado(x, rand):        # x: fp32, rand: uint32
    xi = x.to(tl.uint32, bitcast=True)
    xi = xi + (rand & 0x00001FFF)            # los 13 bits que se descartan (23→10)
    xi = xi & 0xFFFFE000                     # truncar
    return xi.to(tl.float32, bitcast=True).to(tl.float16)
```

Tras enmascarar, el fp32 conserva ≤10 bits de mantisa, así que la conversión final es
exacta y el modo de redondeo deja de importar. Son ~4 operaciones enteras por elemento:
gratis en un kernel limitado por memoria.

Habría que insertarlo en **dos sitios**, ninguno de los cuales es donde vive hoy:

| Sitio | Frecuencia del redondeo |
|---|---|
| `fla/ops/fused_sigmoid_gating.py:166,170` (store de decode) | **por token** ← aquí está el sesgo acumulado |
| `mamba/gdn/qwen_gdn_linear_attn.py:1532` (el `.to()` de prefill) | 1× por chunk, poco crítico |

Limitaciones que el hardware sí resuelve y la emulación no: **subnormales de fp16**
(donde el sesgo reaparece, y es justo el régimen del estado decaído), acarreo al
exponente cerca del tope del rango, y propagación de NaN/Inf. Todo ello es esquivable
usando bf16 desde el principio, que es el motivo por el que esta vía queda como último
recurso y no como primera opción.

### 5.3 Reducir `--max-num-batched-tokens`

El pico transitorio de los buffers FLA es de **816 MiB por capa GDN** a T=16384 y escala
linealmente. Bajar a 8192 lo deja en ~408 MiB. Como el tráfico FLA es sólo el 3 % del
tiempo de prefill (§4), **el coste en velocidad de este cambio es casi nulo y el alivio
de memoria es real** — al contrario de lo que sugerí en el primer informe, donde lo
presenté como un compromiso. No lo es: es prácticamente gratis.

---

## 6. Recomendaciones por eje

### Para acelerar prefill

| Prioridad | Acción | Ganancia estimada | Riesgo |
|---|---|---|---|
| 1 | **Medir el enlace PCIe bajo carga**; si no reentrena a Gen4, arreglar BIOS/ASPM | hasta 8× en el 54 % del tiempo de Heretic | ninguno (diagnóstico) |
| 2 | **Probar PP=2 en lugar de TP=2** | PCIe: 1 818 ms → ~14 ms | alto: camino poco transitado |
| 3 | Mantener W4A8 en Heretic | ya activo, 4× en cómputo | — |
| ✗ | Evaluar INT4 para Fable | 5 907 → 1 477 ms de cómputo | **no hacer**: INT8 es el punto del contenedor (§2.3) |
| 4 | Subir `num_speculative_tokens` a 3 en Fable | ficha del modelo reporta 96 % de aceptación | ✅ aplicado |

### Para acelerar decode

| Prioridad | Acción | Ganancia estimada | Riesgo |
|---|---|---|---|
| 1 | ✅ `--cudagraph-capture-sizes` con 60 exacto en Fable | elimina el relleno 60→66, ~6 % del paso (§6.2 anterior) | bajo |
| ⏸ | Cuantizar camino denso de KAT (§5.1) | −5,5 % (INT8) / −8,1 % (INT4) | **descartado**: mal ratio, ver §5.1 |
| 3 | Subir `--max-num-seqs` **en KAT** | expertos saturan a 256: banda/token cae hasta 4× | memoria de KV |
| 4 | Reevaluar `--disable-custom-all-reduce` | −8,9 ms de 34,5 en Heretic | P2P inestable en GeForce |
| ⏸ | `ssm_state` → bf16 (§5.2) | −1,8 ms/paso | **aplazado**: exige estudio de contexto largo |

### Para reducir memoria

| Prioridad | Acción | Ahorro | Riesgo |
|---|---|---|---|
| 1 | `--max-num-batched-tokens 8192` | ~400 MiB de pico por capa GDN | prácticamente nulo |
| ⏸ | Cuantizar camino denso de KAT (§5.1) | −0,64 GiB (INT8) / −0,94 (INT4) | **descartado**: mal ratio, ver §5.1 |
| 3 | Ajustar `--max-model-len` a lo realmente servible | evita expulsiones | ninguno |
| ✗ | Fable → INT4 | −5,7 GiB/GPU (+84 % de contexto) | **no hacer**: INT8 es el punto del contenedor, ver §2.3 |
| ⏸ | `ssm_state` → bf16 (§5.2) | −0,7 GiB/GPU | **aplazado**: exige estudio de contexto largo |

### Lo que NO va a ayudar

- **Optimizar los buffers FLA por velocidad**: son el 3 % del prefill. Sí importan por
  el pico de memoria.
- **Buscar kernels de atención más rápidos**: el KV cache es puro movimiento de datos a
  banda máxima; no hay FLOPs que ahorrar.
- **Subir `--max-num-seqs` en Heretic/Fable** sin resolver antes la captura de CUDA
  Graph y el KV: agravaría el all-reduce (lineal en M) y la presión de memoria.

---

## 7. Transferencias GPU↔CPU — 📋 REFERENCIA, sin acción

> **Estado: auditado, nada que aplicar.** Se documenta el resultado de rastrear los
> puntos de sincronización GPU→CPU en la ruta caliente, porque el mapa es útil para no
> romper nada al tocar kernels. **No hay ninguna acción derivada de esta sección.**

**Conclusión: no hay ningún caso en la ruta caliente donde un cálculo en GPU tenga que
bajar a CPU por un problema de tipo o tamaño de dato.** vLLM v1 mantiene espejos en CPU
de todos los metadatos, escenifica las copias H2D de forma asíncrona y solapa la única
D2H inevitable —los tokens muestreados— con el forward siguiente.

### 7.1 El único punto donde la presión existe de verdad

`model_executor/layers/fla/ops/index.py:22-30` (`prepare_chunk_indices`):

```python
indices = torch.cat([torch.arange(n)
    for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
```

Es un viaje redondo completo: `.tolist()` sincroniza, se construye la lista con un bucle
Python en CPU y se copia de vuelta. La causa es estructural — los índices de chunk son
**metadatos de forma** que el lanzador de Triton necesita en host.

Ya está esquivado en `v1/attention/backends/gdn_attn.py:369-383`, con comentario
explícito en el código: se le pasa el espejo CPU de `query_start_loc` (con lo que el
`.tolist()` no sincroniza nada) y se copia a GPU con `non_blocking=True`.

**⚠ Footgun latente.** `fla/ops/chunk_delta_h.py:339` conserva el camino de respaldo:

```python
if chunk_indices is None and cu_seqlens is not None:
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
```

Si algún parche llegara a llamar a `chunk_gated_delta_rule` **sin** pasar
`chunk_indices`, ese respaldo recalcularía desde el tensor **GPU** y sincronizaría. El
`@tensor_cache` (8 entradas, `fla/ops/utils.py:34-64`) haría que sólo la primera capa
pagase, pero es una parada completa del pipeline por paso. Relevante si se toca
`_genesis/kernels/` alrededor de GDN.

### 7.2 Los demás puntos de presión, todos resueltos

| Punto | Por qué querría ir a CPU | Cómo está resuelto |
|---|---|---|
| **GDN + MTP: `num_accepted_tokens`** | El bookkeeping del estado mamba necesita saber cuántos tokens se aceptaron, y eso lo decide el rejection sampler en GPU | `postprocess_mamba_align_gpu` — *"Fused GPU postprocess … without CPU-GPU sync"*; los metadatos se pre-cargan a buffers GPU en `_prepare_inputs` (`gpu_model_runner.py:1515-1534`) |
| **D2H de tokens muestreados** | Inevitable: los tokens tienen que llegar a la API | Copia en stream separado + evento; async scheduling la solapa con el forward siguiente (`gpu_model_runner.py:253-301`) |
| **`RejectionSampler.parse_output`** | `output_token_ids.cpu()` (`rejection_sampler.py:267`) | En la ruta asíncrona recibe `sampled_token_ids_cpu`, ya en host → el `.cpu()` es no-op |
| **`.item()` en capas lineales / MoE** | `param.weight_type = loaded_weight.item()` | Sólo en **carga de pesos**, nunca en el forward |

### 7.3 Sobre `--async-scheduling`

Es **redundante en los tres**: `async_scheduling` por defecto es `None` y vLLM lo
auto-activa salvo incompatibilidad (`config/vllm.py:957-997`). KAT lo obtiene aunque no
lleve el flag (no tiene spec decode, luego no cae en ninguna exclusión); Heretic y Fable
también lo tendrían, porque `qwen3_next_mtp ∈ MTPModelTypes ⊂ EagleModelTypes`
(`config/speculative.py:34-58`).

Conviene **dejar el flag explícito** de todas formas: si algún día se cambia a un método
de spec decode que no sea Eagle/MTP/Draft/NGram, el flag hace que el arranque **falle
ruidosamente** (`config/vllm.py:933-947`) en vez de desactivar el solapamiento en
silencio.

### 7.4 Dos cosas a vigilar (no a cambiar)

1. **`VLLM_COMPUTE_NANS_IN_LOGITS`** — hoy en 0 (valor por defecto, no está puesto en los
   compose). Si se activa para depurar, `gpu_model_runner.py:5512` ejecuta
   `logits.isnan().sum(dim=-1).cpu().numpy()`: **un sync completo por paso**. No dejarlo
   activado en producción.

2. **Salida estructurada** — el bitmask de gramática se calcula en numpy **en CPU**, se
   reordena en CPU y se copia H2D en cada paso (`v1/structured_output/utils.py:44-89`).
   No es un stall GPU→CPU del modelo, pero serializa. Aplica sólo a peticiones con
   `guided_json` / `response_format`; el `--tool-call-parser qwen3_coder` **no** lo
   activa, porque sólo parsea el texto ya generado.

### 7.5 Cómo verificarlo empíricamente

El análisis anterior es estático. Para confirmarlo en ejecución:

```bash
# Perfilado: buscar cudaStreamSynchronize / memcpy D2H entre kernels
nsys profile -t cuda,nvtx -o vllm_syncs --capture-range=cudaProfilerApi <proceso>

# Alternativa barata: en el worker, avisa en cada sync implícito
torch.cuda.set_sync_debug_mode("warn")
```

---

## 8. Qué medir para validar este modelo

| Medición | Comando / método | Contrasta |
|---|---|---|
| Enlace PCIe bajo carga | `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -l 1` durante un prefill | §0, §4 |
| Reparto real del prefill | `nsys profile` sobre un chunk; buscar `ncclAllReduce` vs GEMM | §4 |
| Ancho de banda HBM alcanzado | `dcgmi dmon` / `nvidia-smi dmon -s u` | supuesto de 796 GB/s |
| Tallas de CUDA Graph capturadas | log de arranque: `Capturing CUDA graphs ... sizes=[...]` | §6.2 anterior |
| Bloques KV asignados | log: `GPU KV cache size: N tokens` | §2.1 |
| Expertos únicos por paso (KAT) | contador en el router | §3.2 |

---

*Informe generado con asistencia de IA (Claude Opus 5). Modelo analítico construido
sobre las configuraciones reales de los tres contenedores, los `config.json` de los
checkpoints, el código de `assets/vllm` @ `0fc695f` y las especificaciones del hardware
consultadas en la máquina (`nvidia-smi`, `lspci`). Las cifras de tiempo son cotas
derivadas de anchos de banda y ritmos de cómputo teóricos, no medidas de ejecución.*
