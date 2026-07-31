# Propuesta: KAT-Coder en INT8 y sin censura

**Fecha:** 2026-07-30
**Estado:** 📋 **DOCUMENTO DE FUNDAMENTO — no planificado para ejecución**
**Contenedor afectado:** `genesis-kat-coder`
**Checkpoint actual:** `cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4`
**Relacionado:** `ANALISIS-KERNELS-DTYPES-3-CONTENEDORES.md`, `ANALISIS-ANCHOBANDA-MEMORIA-3-CONTENEDORES.md`

> Este documento explica **por qué** querríamos un KAT-Coder en INT8 y sin censura, qué
> haría falta, y por qué hoy no se hace. No es un plan de ejecución. Si en algún momento
> se retoma, acá está el razonamiento completo y los números.

---

## 1. Resumen

Se persiguen **dos objetivos que tiran en direcciones opuestas**:

| Objetivo | Efecto sobre la calidad |
|---|---|
| Pasar los expertos de INT4 a **INT8** | **sube** — duplica la precisión del 96 % de los pesos |
| Aplicar **abliteración** (uncensored / heretic) | **baja** — es una edición destructiva del espacio de pesos |

Esa tensión es el núcleo de la propuesta y la razón principal por la que conviene
ejecutarla **en dos fases separadas y medibles**, nunca de una sola vez.

---

## 2. Por qué INT8

### 2.1 El 96 % del modelo está en INT4

Recuento de parámetros por rango (TP=2):

| Bloque | G par | % del cuerpo | Precisión hoy |
|---|---|---|---|
| **Expertos ruteados** (40 × 256) | **16,11** | **96 %** | INT4 g32 asimétrico |
| GDN (30 capas) | 0,50 | 3 % | fp16 |
| Atención plena (10 capas) | 0,14 | 0,8 % | fp16 |
| Shared experts (40) | 0,06 | 0,4 % | fp16 |
| Routers `mlp.gate` | 0,02 | 0,1 % | fp16 |
| `lm_head` | 0,25 | — | fp16 → fp8 por PN77 |

El checkpoint actual aplica la receta estándar de MoE —cuantizar los expertos, dejar
intacto el camino denso— que es **defendible y no un descuido**. Pero implica que
prácticamente todo el modelo vive a 4 bits.

### 2.2 Por qué los MoE sufren más la cuantización

Un modelo denso ve **todos** los tokens de calibración en **todas** sus matrices. En KAT,
con `num_experts_per_tok = 8` de 256, **cada experto ve ~3 % de los tokens**. Eso tiene
dos consecuencias:

1. **Menos datos por matriz para ajustar la rejilla de cuantización.** El observer tiene
   ~32× menos evidencia por experto que en un denso equivalente.
2. **Sin redundancia que amortigüe el error.** En un denso, el error de una capa se diluye
   en las 63 restantes. Un experto raro que se degrada no tiene quien lo compense: cuando
   el router lo elige, su error entra entero al residual.

A esto se suma que **KAT es un modelo de código agéntico**. El código es frágil de una
forma que la prosa no: un identificador mal predicho, un off-by-one en un índice o un tipo
equivocado rompen la ejecución. No hay degradación elegante.

### 2.3 Lo que el INT4 actual ya hace bien

Es importante reconocerlo, porque acota cuánto hay para ganar. El checkpoint de `cyankiwi`
es **el de mayor calidad de los tres INT4 disponibles** para KAT:

| | cyankiwi (actual) | sahilchachra W4A16 | Ar4ikov ASYM |
|---|---|---|---|
| group_size | **32** | 128 | 128 |
| Simetría | **asimétrico** | simétrico | asimétrico |
| Observer | **mse** | memoryless_minmax | memoryless_minmax |
| Superficie cuantizada | sólo expertos | + atención + shared | expertos, salvo capas 0-6 |

g32 da 4× más resolución de escala que g128; asimétrico aprovecha el rango completo del
int4; el observer `mse` minimiza el error cuadrático real en vez de sólo cubrir el rango.
**La ganancia de pasar a INT8 es real pero no dramática** — se parte de un INT4 bien hecho,
no de uno ingenuo.

### 2.4 No existe: hay que fabricarlo

Inventario completo de cuantizaciones de KAT-Coder-V2.5-Dev usables en vLLM:

| Repo | Formato | Sirve |
|---|---|---|
| `cyankiwi/…-AWQ-INT4` | INT4 g32 asim | ✅ el actual |
| `sahilchachra/…-W4A16` | INT4 g128 sim | ✅ menor calidad |
| `Ar4ikov/…-AWQ-W4A16-ASYM` | INT4 g128 asim | ✅ menor calidad |
| `sakamakismile/…-NVFP4`, `sahilchachra/…-NVFP4A16`, `doth4580/…-NVFP4-MIXED` | NVFP4 | ❌ **exige SM100 (Blackwell)** |
| `bartowski`, `mradermacher`, `mudler` (APEX) | GGUF | ❌ vLLM no cubre esta arquitectura en GGUF |
| `Kwaipilot/KAT-Coder-V2.5-Dev` | bf16, ~70 GB | ❌ 31,6 GiB/GPU con TP=2 |

**No hay ningún INT8.** Y Marlin sólo admite 4 y 8 bits
(`WNA16_SUPPORTED_TYPES_MAP = {4, 8}`), así que no existe un escalón intermedio: la
escalera es INT4 → INT8 → fp16, y fp16 no entra.

---

## 3. Presupuesto de memoria

Calculado con `--gpu-memory-utilization 0.92`, TP=2, routers en fp16 y `lm_head` en fp8
(PN77), descontando estados GDN y ~1,5 GiB de margen para activaciones y grafos.

| Escenario | Pesos/GPU | KV a 20 seqs | KV a 4 seqs |
|---|---|---|---|
| **Hoy**: exp INT4 g32, denso fp16 | **10,02 GiB** | 104 K/seq | 547 K/seq |
| **A**: exp INT8 g128, denso fp16 | 16,82 GiB | 33 K/seq | 191 K/seq |
| **B**: exp INT8 g32, denso fp16 | 17,52 GiB | 26 K/seq | 154 K/seq |
| **C**: exp INT8 g128 **+ denso INT8** | **16,17 GiB** | **40 K/seq** | **225 K/seq** |
| D: todo bf16 (referencia) | 31,58 GiB | ❌ no entra | ❌ no entra |

**Entra, y con holgura a baja concurrencia.** KAT lo permite porque su KV es barato
—5 KiB/token contra 16 de los modelos densos, por tener 10 capas de atención en vez de 16
y 1 kv-head por rango en vez de 2.

### 3.1 Escenario C es el correcto, y es contraintuitivo

Una vez en INT8, **conviene cuantizar también el camino denso**: el escenario C usa
**0,65 GiB menos** que el A y a cambio pierde muy poco, porque INT8 por grupo es
prácticamente sin pérdida.

El camino denso está en fp16 hoy **para compensar la agresividad del INT4**. A 8 bits esa
compensación deja de tener sentido: se paga memoria por una protección que ya no hace
falta. Se dejarían fuera únicamente:

- `linear_attn.in_proj_a` / `in_proj_b` — 2 M par; alimentan `softplus`/`sigmoid`
  exponencialmente sensibles, y con N=64→32 por rango caen por debajo de
  `MIN_THREAD_N=64`, así que Marlin los rechazaría igual
- `mlp.gate` (routers) — 21 M par; su error cambia **qué** expertos se eligen: es un error
  discreto, no suave, y no se diluye
- `lm_head` — PN77 lo pasa a fp8 en runtime

### 3.2 Coste en velocidad

| | Hoy (INT4) | INT8 (escenario C) |
|---|---|---|
| Pesos leídos por paso de decode | ~9,1 GiB | **~15,3 GiB** |
| Paso de decode estimado | ~15,7 ms | **~23 ms** |
| Instrucción | `HMMA` | `HMMA` (sin cambio) |
| Contexto/seq a 20 seqs | 104 K | 40 K |

Se paga ~45 % de decode y 60 % de contexto. **Es aceptable** porque KAT es hoy el más
rápido de los tres contenedores por amplio margen (~15,7 ms contra 34,5 de Heretic y 45,7
de Fable), así que incluso degradado seguiría siendo competitivo.

Nótese que INT8 **no cambia la instrucción**: sigue siendo `HMMA` con dequantización en
registro. El INT8 compra precisión, no velocidad — igual que en Fable.

---

## 4. Por qué también sin censura

### 4.1 El problema es operativo, no ideológico

En un chat, un rechazo se resuelve reformulando. **En un bucle agéntico, un rechazo es una
parada en seco a mitad de tarea**: el agente deja el repositorio en un estado intermedio,
pierde el contexto de lo que estaba haciendo y hay que reiniciar la tarea completa. El
coste no es la molestia del rechazo, es la sesión perdida.

Esto es la misma clase de fallo que ya se combate en la suite Genesis con
`P69_LONG_CTX_TOOL_REMINDER` (recordatorio de formato de herramientas) y
`PN56_QWEN3CODER_XML_FALLBACK` (parseo alternativo de llamadas): **cosas que rompen la
navegación agéntica y hay que parchear fuera del modelo**. Una negativa espuria es
exactamente eso, pero no hay parche posible del lado del servidor.

### 4.2 Disparadores concretos en trabajo de ingeniería legítimo

Sobre trabajo real de código, los rechazos aparecen en:

- Herramientas de seguridad defensiva y análisis de malware
- Parsers para entrada hostil (fuzzing, validación de formatos)
- Código que manipula credenciales, tokens o secretos — aunque sea para rotarlos
- Scraping, automatización de navegadores, clientes de APIs
- Scripts de red-team y pruebas de penetración autorizadas
- Cualquier cosa cuyos identificadores o strings *suenen* alarmantes fuera de contexto

El último caso es el más frustrante: el modelo no evalúa la tarea, reacciona a un token.

### 4.3 Coherencia de la flota

Los otros dos contenedores ya son variantes sin censura:

- **Heretic** — `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`
- **Fable** — deriva de `DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-MTP`

KAT es el único que conserva el comportamiento de rechazo original, y es justamente el que
más se usa en modo agéntico.

---

## 5. La tensión, sin maquillar

Este es el punto que hay que tener presente antes de invertir esfuerzo:

**La abliteración es una edición destructiva del espacio de pesos.** Identifica la
dirección del residual asociada al rechazo y la proyecta fuera de las matrices que
escriben al residual. Eso **degrada capacidad de forma medible**, a veces de manera
apreciable en razonamiento y código — precisamente lo que se estaría intentando mejorar
con el INT8.

Y las dos operaciones **interactúan**: la abliteración cambia la distribución de los pesos
(proyectar fuera una dirección altera la estructura de outliers), y el error de
cuantización se comporta distinto sobre esa distribución modificada. No son efectos
independientes que se puedan sumar.

De ahí la consecuencia práctica: **hay que medirlos por separado o no se sabe cuál de los
dos produjo el resultado.**

---

## 6. Orden de operaciones

Sólo hay una secuencia válida:

```
Kwaipilot/KAT-Coder-V2.5-Dev (bf16, ~70 GB)
        │
        ├─ ① abliteración  (edición en espacio de pesos, requiere bf16)
        │
        └─ ② cuantización INT8  (RTN, sin calibración)
                │
                └─ checkpoint final
```

**No se puede invertir ni saltear:**

- ✗ *Cuantizar y después abliterar* — no se puede editar de forma significativa un tensor
  int8 empaquetado; la dirección a proyectar vive en el espacio continuo.
- ✗ *Partir del INT4 actual, descomprimir y abliterar* — quedaría horneado el error del
  INT4 en el punto de partida, que es exactamente lo que se quería eliminar.
- ✗ *Buscar un KAT ya abliterado* — no existe. El ecosistema de abliteración (huihui,
  heretic/MPOA, DavidAU) trabaja sobre la familia Qwen densa, no sobre KAT.

---

## 7. Complicaciones específicas de MoE

Este es el obstáculo técnico real, y la razón principal del aplazamiento.

La abliteración estándar ablaciona la dirección de rechazo de las matrices que **escriben
al residual**: `o_proj` en atención y `down_proj` en los MLP. En KAT eso significa:

| Objetivo | Cantidad | Forma |
|---|---|---|
| `self_attn.o_proj` | 10 | 4096 × 2048 |
| `linear_attn.out_proj` | 30 | 4096 × 2048 |
| `shared_expert.down_proj` | 40 | 512 × 2048 |
| **`experts.*.down_proj`** | **10 240** | **512 × 2048** |

**Diez mil doscientas cuarenta matrices diminutas**, una por experto por capa. Y de ahí
salen tres preguntas abiertas:

1. **¿La dirección de rechazo es uniforme entre expertos?** El router envía distribuciones
   de tokens distintas a cada uno. Es plausible que el comportamiento de rechazo esté
   **concentrado en expertos específicos** en vez de repartido. Ablacionar los 256 por
   igual podría degradar 250 innecesariamente para corregir 6.
2. **¿Alcanza la calibración?** Con 8 expertos activos de 256, hacen falta **~32× más
   datos** que en un denso para lograr cobertura equivalente por experto. Un conjunto de
   calibración típico de abliteración dejaría a la mayoría de los expertos con apenas un
   puñado de activaciones.
3. **¿Qué pasa con las capas GDN?** 30 de las 40 capas son atención lineal con estado
   recurrente. La dirección se identifica en el residual, que es común, pero el efecto de
   ablacionar `out_proj` sobre un estado que **se acumula a lo largo de toda la secuencia**
   no está caracterizado en la literatura de abliteración, que asume atención sin estado.

No hay, que sepamos, ninguna abliteración publicada de un MoE de 256 expertos con esta
forma. Sería trabajo exploratorio, no aplicación de una receta conocida.

---

## 8. Receta de cuantización (fase ②)

La parte fácil, documentada para cuando corresponda:

```
formato:     compressed-tensors pack-quantized  (o GPTQ)
bits:        8
strategy:    group,  group_size: 128
symmetric:   true          → uint8b128 → Marlin W8A16
desc_act:    false

excluir:  linear_attn.in_proj_a / in_proj_b
          mlp.gate            (routers)
          lm_head             (PN77 lo pasa a fp8 en runtime)
          visual.*            (irrelevante: se usa --language-model-only)
```

**Método: RTN, sin calibración.** A 8 bits con escalas por grupo, round-to-nearest es
prácticamente indistinguible de GPTQ — por eso la mayoría de los checkpoints W8A16 son RTN.
Eso convierte la fase ② en una transformación offline determinista sobre los safetensors:
leer tensor → absmax por grupo → escalar → redondear → empaquetar → escribir. Sin dataset,
sin pasadas forward, sin GPU.

**KAT no tiene cabeza MTP** (`mtp_num_hidden_layers: 0`), así que —a diferencia de Heretic
y Fable— no hay nada que preservar por ese lado, ni spec decode que se pueda romper.

**Verificación de formas** (todas cumplen Marlin con TP=2, `N % 64 == 0` y `K % 128 == 0`):

| GEMM | K | N |
|---|---|---|
| experto `w13` | 2048 | 512 |
| experto `w2` | 256 | 2048 |
| `qkv_proj` | 2048 | 4608 |
| `o_proj` | 2048 | 2048 |
| GDN `in_proj_qkvz` | 2048 | 6144 |
| GDN `out_proj` | 2048 | 2048 |

---

## 9. Plan de validación

Debe **separar los dos efectos**, o el resultado no es interpretable:

| Brazo | Checkpoint | Qué mide |
|---|---|---|
| **0** | INT4 actual | línea base |
| **1** | INT8 sin abliterar | **ganancia de la cuantización, aislada** |
| **2** | Abliterado bf16 | coste de la abliteración, aislado (no entra en esta máquina; requiere evaluación externa) |
| **3** | Abliterado + INT8 | el objetivo, y si hay interacción entre ambos |

Métricas:

- **Código**: evaluación agéntica sobre tareas reales del repositorio, más un benchmark
  estándar como referencia comparable
- **Rechazos**: tasa sobre un conjunto de prompts de ingeniería legítima que hoy se
  rechazan — hay que construirlo a partir de casos reales observados
- **Contexto largo**: la degradación del estado GDN sólo se manifiesta en secuencias
  largas, no en evaluaciones cortas (mismo problema que llevó a aplazar `ssm_state`→bf16)

---

## 10. Por qué no ahora

| Obstáculo | Detalle |
|---|---|
| **Abliteración de MoE sin precedente** | 10 240 matrices de expertos, cobertura de calibración 32× más exigente, interacción desconocida con el estado GDN (§7) |
| **No hay de dónde partir** | No existe KAT abliterado; habría que hacerlo desde cero sobre los ~70 GB en bf16 |
| **Los dos objetivos se contradicen** | INT8 sube calidad, abliteración la baja; sin medición separada no se sabe el saldo (§5) |
| **KAT ya es el mejor contenedor** | Es el más rápido y el más holgado en memoria de los tres; el margen de mejora es el más chico de la flota |
| **Coste de infraestructura** | 70 GB de descarga, pasada de abliteración con offload a CPU (no entra en 48 GB de VRAM), construcción del conjunto de calibración |

### Si se retomara, el orden correcto

**Empezar por la fase ② sola.** Fabricar el INT8 sin abliterar (brazo 1) es barato: RTN
mecánico, sin calibración, sin GPU, partiendo de los ~70 GB en bf16. Eso responde la
pregunta que gobierna todo lo demás:

> **¿Se nota la diferencia entre INT4 g32 y INT8 en tareas de código reales?**

Si no se nota —y es un desenlace plausible, porque se parte de un INT4 bien hecho (§2.3)—
el proyecto termina ahí y se ahorra por completo el trabajo exploratorio de abliteración de
MoE. Si se nota, recién entonces tiene sentido evaluar si vale la pena pagar la degradación
de la abliteración encima.

---

*Documento de fundamento generado con asistencia de IA (Claude Opus 5). Los recuentos de
parámetros y presupuestos de memoria son cálculos derivados de `config.json` del checkpoint
y de las especificaciones del hardware (2× RTX 3090, sm86, TP=2), no mediciones en
ejecución.*
