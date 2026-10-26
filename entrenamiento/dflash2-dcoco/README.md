# DFlash2-DCOCO: corrector de transiciones para el borrador DFlash2

Entrena, mide, cuantiza y publica la cabeza **DCOCO** ([arXiv 2610.12570](https://arxiv.org/abs/2610.12570),
[two-delta/DCOCO](https://github.com/two-delta/DCOCO)) sobre el borrador DFlash2 de
`qwen3.8_27b_idiotSavant_sm_86`, con las capturas que ya arma `idiotsavant/dflash2.sh`.

El resultado se sirve de dos formas:

- **con Genesis:** el parche PN178, `GENESIS_ENABLE_PN178_DCOCO=1`;
- **con [vllm-idiotsavant](https://github.com/blairq/vllm-idiotsavant):** la clave `"dcoco"` de
  `--speculative-config`.

## Qué es

DFlash2 propone 16 candidatos por posición. Su `CandidateSelector` arma la tabla
`scores[paso, previo, candidato]` con un bilineal de rango bajo:

- tiene codebooks por token;
- solo mira la oculta de la posición destino;
- vino con el borrador base, y su entrenamiento no es público.

Sobre esa tabla se recorre la cadena o se arma el árbol de 8 nodos.

DCOCO agrega un MLP chico (~2-3 M de parámetros) que vuelve a puntuar cada transición
`candidato a en p-1 → candidato b en p` con dos entradas:

- **el estado de par:** las ocultas de las dos posiciones;
- **una representación de cada candidato:** su embedding del target, proyectado y mezclado con la
  oculta de su posición.

Se suma a la tabla del selector (corrección residual) con la última capa en cero. **Sin entrenar,
el borrador es exactamente el de hoy.** La fila 0, que sale del ancla, no se toca.

La cuenta está en un solo lugar, [`vllm/_genesis/dcoco.py`](../../vllm/_genesis/dcoco.py). El
entrenamiento la importa por ruta y el servidor la usa tal cual.

## Archivos

| Archivo | Qué hace |
|---|---|
| `dflash2_dcoco.sh` | Pipeline completo y retomable (pasos 0 a 4, abajo) |
| `entrenar_dcoco.py` | Entrena la cabeza (`--modo cabeza`) o cabeza + LoRA (`--modo conjunto`). Evalúa y exporta |
| `evaluacion.py` | Cadena y árbol sobre una tabla fija o con filas que dependen del camino, aceptación golosa y con muestreo, calibración del árbol |
| `recurrente.py` | La cabeza con estado del camino (`--estado recurrente`, solo entrenamiento) |
| `selector.py` | El `CandidateSelector` y el top-k de vLLM en PyTorch, cargados del checkpoint (`../dflash2/` no carga el selector) |
| `subir_hf.sh`, `README_hf.md` | Validación y publicación en Hugging Face |
| `test_entrenar_dcoco.py` | Pruebas en CPU de la pérdida, los recorridos, el árbol, el muestreo, la calibración, la cabeza recurrente y el aprendizaje |

[`../dflash2/`](../dflash2/) no se toca: de ahí se importan los datos, la LoRA, la pérdida del
borrador y `guardar`.

## Requisitos

- **Capturas** de `idiotsavant/dflash2.sh` (su paso 3) en
  `tests/bench/medicion/trazas/captura/entrena_rot` (o la que se pase en `CAPTURAS`).
- **El borrador de producción** `qwen3.8_27b_idiotSavant_sm_86_dflash2` y el target
  `qwen3.8_27b_idiotSavant_sm_86` en el models-cache.
- **La imagen** `vllm/vllm-openai:v0.29.0`, una 3090 y el servidor apagado mientras se entrena.
- **Para el A/B (paso 4):** la compose `compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml` con
  el parche PN178 disponible (está en esta rama).

## Uso rápido

```bash
cd entrenamiento/dflash2-dcoco
PASOS="0 1" ./dflash2_dcoco.sh                               # base: ¿hay margen?
setsid nohup ./dflash2_dcoco.sh > dflash2_dcoco.log 2>&1 &   # todo, retomable
MODO=conjunto NOMBRE=conjunto ./dflash2_dcoco.sh             # cabeza + LoRA del borrador
huggingface-cli login && ./subir_hf.sh $MC/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco_cabeza
```

## `dflash2_dcoco.sh` paso a paso

Cada paso se saltea si su salida ya existe. Para rehacer uno, borrá esa salida o elegí los pasos con
`PASOS="..."`.

| Paso | Qué hace | Salida |
|---|---|---|
| 0 | Decuantiza producción a fp16 (`../dflash2/decuantizar.py`): exactamente los pesos servidos | `trazas/borrador_prod_fp16/` |
| 1 | Evaluación de base sobre los pedidos apartados, sin entrenar | `trazas/banco/dcoco_base.json` y `.log` |
| 2 | Entrenamiento (`MODO`, `EPOCAS`) | `trazas/borrador_dcoco_$NOMBRE/` (bf16, con `dcoco.*`) |
| 3 | Cuantización con las escalas de producción: lo no entrenado vuelve exacto a sus int4 y la cabeza queda en bf16 | `$MC/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco_$NOMBRE/` |
| 4 | A/B en vLLM: producción contra DCOCO, banco de agente (40 pedidos, concurrencia 4), dos arranques por brazo, PN178 prendido en los dos | `trazas/banco/dcoco_${NOMBRE}_{prod,dcoco}_r{1,2}.json` + resumen |

| Variable | Default | Para qué |
|---|---|---|
| `NOMBRE` | `cabeza` | Sufijo de las salidas: permite varias corridas en paralelo |
| `MODO` | `cabeza` | `cabeza` = borrador congelado; `conjunto` = cabeza + LoRA (con `--auf`) |
| `EPOCAS` | `2` | Épocas de entrenamiento |
| `CAPTURAS` | `entrena_rot` | Subdirectorio de `trazas/captura/` |
| `FLAGS` | `--vocab 65536` | Restricciones de servicio que se simulan al entrenar |
| `ENV_BORRADOR` | `GENESIS_PN157_BITS_CONV=0 GENESIS_PN159_VOCAB=65536` | Las mismas restricciones, para el servidor del A/B |
| `PASOS` | `0 1 2 3 4` | Qué pasos correr |
| `EXTRA` | (vacío) | Flags opcionales del paso 2 (ver [Opciones](#opciones-muestreo-calibración-y-estado-recurrente)). Con `--estado recurrente` se saltean 3 y 4 |
| `MC`, `IMG`, `PROD` | los de la máquina de siempre | models-cache, imagen y borrador de producción |

`FLAGS` y `ENV_BORRADOR` tienen que describir **lo mismo**. Si se sirve la conv en int8
(`GENESIS_PN157_BITS_CONV=8`), entrenar con `FLAGS="--conv_int8 --vocab 65536"`.

## Cómo leer los números

El paso 1 y cada época imprimen un JSON. En el log del paso 2 queda además un resumen de una línea
por época. Los campos de `registro.json`:

| Campo | Qué es |
|---|---|
| `largo_unario` | Largo aceptado (sin el bonus) tomando el top-1 del lm_head en cada posición |
| `largo_selector` | Ídem con el recorrido goloso sobre la tabla del selector (lo que hace hoy la cadena) |
| `largo_dcoco` | Ídem sobre la tabla corregida |
| `largo_arbol_selector`, `largo_arbol_dcoco` | Largo aceptado verificando el árbol best-first de `--nodos` nodos (8, como en producción) |
| `por_posicion_*` | Aceptación acumulada por posición en cadena |
| `recall_top16` | Si el token real está entre los 16 candidatos de esa posición |
| `ce_fila_selector`, `ce_fila_dcoco` | Entropía cruzada forzada por el maestro de la transición real (`-log p(real en p | real en p-1)`) en los pedidos apartados, sin peso por profundidad |
| `top1_fila_selector`, `top1_fila_dcoco` | Cuántas veces el real es el argmax de su fila |
| `perdida_entreno_dcoco`, `perdida_entreno_selector` | Pérdida media de la época en entrenamiento, con la cabeza y la misma cuenta sobre la tabla del selector |
| `largo_muestreo_T{T}_*` | Con `--temperaturas`: largo aceptado esperado con muestreo a esa temperatura (ver abajo) |

- **El margen (paso 1):** `recall_top16[p] − por_posicion_selector[p]` es la cota de lo que
  cualquier re-selección puede ganar en la posición `p`. Si es chico en todas las posiciones,
  DCOCO no tiene de dónde sacar.
- **El criterio para seguir:** que `largo_arbol_dcoco` supere a `largo_arbol_selector` en los pedidos
  apartados. Es lo que se va a servir, porque producción usa el árbol. Si no lo supera, no vale la
  pena el A/B.
- **¿Bug o falta de señal?** Si `ce_fila_dcoco` baja respecto de `ce_fila_selector` y
  `perdida_entreno_dcoco` respecto de `perdida_entreno_selector`, la cabeza aprende. Si las dos bajan
  pero el largo no se mueve, lo que aprende no cambia el argmax de las filas que importan
  (`top1_fila_*`). Si baja la de entreno y no la de los apartados, sobreajusta: menos épocas o un
  `--lr` más bajo.
- **Qué se exporta:** con `--quedarse mejor` (default), la mejor época por `--criterio`
  (`largo_arbol_dcoco`). La base también cuenta: si ninguna época la supera, se exporta la cabeza en
  cero (el borrador de hoy) y el log lo dice en la línea `MEJOR`.
- **El A/B (paso 4)** da `largo_aceptacion` por arranque. Los tokens por segundo de prosa y de
  código se miran después con `bateria.py` contra el servidor.

## `entrenar_dcoco.py`

Se puede llamar directo, con el mismo `docker run` de `../dflash2/README.md`:

```bash
.../entrenar_dcoco.py --datos /traces/captura/entrena_rot --borrador /traces/borrador_prod_fp16 \
  --noon /models/qwen3.8_27b_idiotSavant_sm_86 --salida /traces/borrador_dcoco --vocab 65536
```

| Argumento | Default | Para qué |
|---|---|---|
| `--datos`, `--salida`, `--borrador`, `--noon` | (obligatorios) | Como en `../dflash2/entrenar.py`. `--borrador` tiene que traer el selector (la salida de `decuantizar.py` lo trae) |
| `--modo` | `cabeza` | `cabeza` o `conjunto` |
| `--perdida` | `fila` | `fila`: entropía cruzada sobre la fila del previo real, que es como usa la tabla el árbol. `conjunta`: sobre las K×K transiciones, como en el paper |
| `--rho` | `4` | Peso `e^{-(p-1)/ρ}` por posición |
| `--peso_dcoco` | `1` | En modo conjunto, peso de la pérdida de DCOCO frente a la del borrador |
| `--emb`, `--state`, `--mlp` | `128`, `128`, `256` | Tamaños de la cabeza (los del paper; quedan en el config del checkpoint) |
| `--epocas`, `--lote`, `--lr` | `2`, `32`, `6e-4` | La cabeza usa AdamW con coseno y 4 % de calentamiento, como en el paper |
| `--lr_lora`, `--rango`, `--auf` | `1e-4`, `64`, apagado | LoRA del modo conjunto (igual que `dflash2/`) |
| `--nodos` | `8` | Nodos del árbol en la evaluación |
| `--apartar` | `0,03` | Igual que `dflash2.sh`: los mismos pedidos apartados que el borrador de producción |
| `--conv_int8`, `--conv_grupo`, `--vocab`, `--vocab_orden` | | Restricciones de servicio, como en `dflash2/` |
| `--quedarse`, `--criterio` | `mejor`, `largo_arbol_dcoco` | Exporta la mejor época (o la `ultima`) según esa métrica de la evaluación |
| `--temperaturas T...`, `--nucleo` | ninguna, `1` | Métrica con muestreo (opción 1, abajo) |
| `--calibrar`, `--calibrar_T` | `no`, la primera temperatura | Escala del árbol por profundidad (opción 3) |
| `--estado` | `pares` | `recurrente`: cabeza con estado del camino (opción 6) |
| `--solo_evaluar`, `--max_pedidos`, `--max_anclas_eval`, `--max_pasos` | | Medición sola y recortes para pruebas. `--solo_evaluar --calibrar ...` exporta un checkpoint solo calibrado (cabeza en cero) |

**La pérdida** es entropía cruzada forzada por el maestro:

- para cada transición `p-1 → p`, el previo es el candidato que coincide con el token real en `p-1`
  y se pide el token real en `p`;
- las transiciones con algún token real fuera del top-16 no cuentan;
- en modo conjunto se suma la pérdida de `dflash2/entrenar.py` (0,9 TV + 0,1 CE).

**El checkpoint de salida** es el borrador (con la LoRA fusionada en modo conjunto) más:

- la cabeza en bf16 como `dcoco.*`, 10 tensores con los mismos nombres que `DCOCOHead` del repo de
  DCOCO;
- en `dflash_config`: `dcoco_candidate_topk`, `dcoco_emb_dim`, `dcoco_state_dim`, `dcoco_mlp_dim`,
  `dcoco_estado` y, con `--calibrar`, `dcoco_escala_arbol`;
- un bloque `dcoco` con la receta.

`../dflash2/cuantizar_rtn.py` lo cuantiza dejando la cabeza intacta.

## Opciones: muestreo, calibración y estado recurrente

Las tres están apagadas por defecto; sin ellas el entrenamiento es el de antes.

### 1. Métrica con muestreo (`--temperaturas`)

La métrica golosa supone que el target elige su top-1. Sirviendo agentes de código con la
temperatura que recomienda el paper de Qwen3.8 para código, el target muestrea y la aceptación es
otra. Con `--temperaturas T1 T2 ...` (y `--nucleo` si se sirve con top-p) cada evaluación agrega
`largo_muestreo_T{T}_{unario,selector,dcoco,arbol_selector,arbol_dcoco}`:

- la propuesta es determinista y el token propuesto se acepta con `p_T(token)`;
- cadena: `Σ_k Π_{j≤k} p_T(x_j)`; árbol: la suma sobre nodos de P(llegar y aceptar);
- `p_T` sale del top-16 capturado: `p^(1/T)` renormalizado y, con `--nucleo`, cortado a top-p.

**Aproximación:** en cada profundidad se usa la distribución que el target dio con el prefijo
real; para prefijos que el target no generó no hay capturas. Sirve para comparar borradores entre
sí, no como número absoluto. Para elegir la época con esta métrica:
`--criterio largo_muestreo_T<T>_arbol_dcoco`.

### 3. Calibración del árbol (`--calibrar voraz|muestreo`)

El árbol best-first ordena por probabilidad de camino, `log_softmax` de cada fila. Con la cabeza
corregida esas probabilidades pueden quedar mal calibradas entre profundidades: el árbol se va a lo
hondo o a lo ancho cuando no conviene. Después de entrenar, se busca una escala por profundidad
(multiplica la fila antes del `log_softmax`):

- descenso por coordenadas sobre una grilla de 0,5 a 2, dos rondas;
- en la mitad par de los lotes apartados se busca y en la impar se informa
  (`CALIBRACION ... arbol dcoco antes -> despues`);
- `voraz` maximiza el largo goloso; `muestreo`, el esperado a `--calibrar_T` (default: la primera de
  `--temperaturas`).

La cadena no cambia, porque el argmax de una fila no depende de la escala. El log trae después la
línea `CALIBRADO {...}`, que es la evaluación completa con la escala. La escala queda en
`dflash_config.dcoco_escala_arbol`, y PN178 y vllm-idiotsavant la aplican al servir. Con
`--solo_evaluar --calibrar ...` se calibra el borrador sin entrenar la cabeza.

### 6. Estado recurrente del camino (`--estado recurrente`)

La cabeza de pares es una corrección de Markov de orden 1, igual que el selector: la fila de `p`
solo ve el candidato elegido en `p-1`. Los correctores que más le ganan a DFlash condicionan en todo
el prefijo elegido: xPress (arXiv 2608.02438) y la corrección RNN de DARTree (arXiv 2608.13524).
`recurrente.py` hace

```
s_a   = GRU(s_prefijo, [pe(a); l(a)])          estado después de elegir a en p-1
delta = MLP([s_a; l(a); pe(b); l(b)])          se suma a la fila a de la tabla del selector
```

con la misma `pe` que la cabeza de pares y la última capa en cero:

- **al entrenar** la tabla se arma forzada por el maestro, con el prefijo real. La pérdida solo
  cuenta las transiciones con todo el prefijo dentro del top-16;
- **al evaluar**, la cadena y el árbol calculan cada fila sobre el camino que van eligiendo, con
  estado por nodo.

**Todavía no se sirve.** PN178 y vllm-idiotsavant rechazan un checkpoint con
`dcoco_estado: recurrente` (salvo `dcoco=false` / `GENESIS_PN178_CORRECTOR=0`). Servirla pide un
constructor del árbol con estado por nodo en el servidor; `evaluacion.arbol` es la referencia.
`dflash2_dcoco.sh` saltea la cuantización y el A/B. Sirve para medir offline si condicionar en el
prefijo gana lo suficiente como para hacer ese trabajo.

```bash
# por ejemplo: T de código del paper, calibrar con muestreo y elegir la época con esa métrica
EXTRA="--temperaturas <T> --calibrar muestreo --criterio largo_muestreo_T<T>_arbol_dcoco" ./dflash2_dcoco.sh
EXTRA="--estado recurrente --temperaturas <T>" NOMBRE=recurrente PASOS="2" ./dflash2_dcoco.sh
```

## Relación con el repo de DCOCO

[two-delta/DCOCO](https://github.com/two-delta/DCOCO) solo trae el servidor (un overlay de vLLM
v0.31.0), el benchmark contra un servidor y los checkpoints de Qwen3-4B/8B. **No trae el
entrenamiento.** Lo de esta carpeta es propio, armado a partir del paper. La cabeza es la misma
cuenta que su `DCOCOHead` (mismos nombres de parámetros, misma partición de `trans_mlp[0]`), con
dos diferencias de diseño:

- **acá** corrige la tabla del selector de DFlash2: `scores = selector + delta`;
- **allá** el borrador es DFlash sin selector y la tabla es `logit crudo del siguiente + delta`, así
  que DCOCO es lo único que modela las transiciones. Por eso las ganancias del paper (contra DFlash
  sin selector) no se trasladan directo a DFlash2, que ya tiene un modelo de pares.

Ellos también usan logits crudos, sin escala ni soft cap, y un bloque de 15. Su recorrido es el
mismo que la cadena de acá (goloso prefijo-primero, o muestreado). El árbol de 8 nodos es nuestro,
y es lo que se sirve. Su `scripts/benchmark.py` se puede apuntar a nuestro servidor para un A/B de
punta a punta, pero sus checkpoints son para otros targets.

## Servir un borrador DFlash2-DCOCO

**Genesis.** En la compose idiotsavant-sm86:

```bash
GENESIS_ENABLE_PN178_DCOCO=1 docker compose -f compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d
```

Montá el checkpoint como borrador, igual que en el paso 4. `GENESIS_PN178_CORRECTOR` admite tres
valores:

- `auto` (default): usa la cabeza si el checkpoint la trae;
- `1`: la exige;
- `0`: la descarta y el borrador corre como DFlash2, útil para un A/B con el mismo checkpoint.

**Sin PN178**, un checkpoint DFlash2-DCOCO no carga: vLLM no sabe qué hacer con `dcoco.*`.

**vllm-idiotsavant:**

```bash
--speculative-config '{"method":"dflash","model":"BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco",
  "num_speculative_tokens":8,"tree_drafting":true,"draft_vocab_size":65536,"draft_residual_scale":32}'
```

Sin la clave `"dcoco"`, la cabeza se usa porque el checkpoint la trae. Con `"dcoco":true` se exige
(error si el borrador no es DFlash2-DCOCO) y con `"dcoco":false` se descarta.

**Costo al servir:** el MLP es despreciable. El embedding de los 16 × 8 candidatos pasa por
`embed_input_ids`, que con TP=2 hace un all-reduce de ~1,3 MB por pedido en cada paso. Si el A/B
muestra que pesa, se puede proyectar antes del all-reduce.

## Publicar

```bash
huggingface-cli login            # o HF_TOKEN con permiso de escritura
./subir_hf.sh <dir_cuantizado> [BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco]
```

El script hace cuatro cosas:

1. verifica que el config declare la cabeza y que estén los 10 tensores `dcoco.*`;
2. copia `README_hf.md` como tarjeta;
3. crea el repo si no existe;
4. sube.

Antes de subir, completá la tabla de resultados de `README_hf.md` con el `registro.json` y el A/B.
Para subidas largas, `../idiotsavant/vigia_subida_hf.sh` relanza la subida si se cuelga.

## Pruebas sin GPU

```bash
python3 -m pytest -q entrenamiento/dflash2-dcoco/test_entrenar_dcoco.py        # pérdida, árbol, muestreo, calibración, recurrente
PYTHONPATH=. python3 -m pytest -q vllm/_genesis/tests/test_pn178_dcoco.py \
    vllm/_genesis/tests/test_pn178_wiring.py                                    # cabeza y parche
```

## Problemas comunes

- **`no encuentro una sola clave terminada en 'candidate_selector...'`:** el `--borrador` no trae el
  selector. Usá la salida de `decuantizar.py` de producción o un bf16 completo, no la de
  `cargar_borrador`.
- **`checkpoint DFlash2-DCOCO incompleto` al servir:** faltan tensores `dcoco.*`. Volvé a exportar
  o a cuantizar; `subir_hf.sh` hace el mismo chequeo antes de publicar.
- **El paso 1 da `largo_selector` muy distinto del `largo_greedy` de `dflash2/`:** es esperable.
  `largo_greedy` cuenta el bonus y usa el top-1 unario, no el recorrido del selector.
- **Pérdida en 0:** ninguna transición tiene sus dos tokens reales en el top-16. Revisá que
  `--vocab` y `--conv_int8` coincidan con cómo se generaron las capturas y se sirve el borrador.
