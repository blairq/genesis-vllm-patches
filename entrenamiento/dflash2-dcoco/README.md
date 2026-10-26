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
| `selector.py` | El `CandidateSelector` y el top-k de vLLM en PyTorch, cargados del checkpoint (`../dflash2/` no carga el selector) |
| `subir_hf.sh`, `README_hf.md` | Validación y publicación en Hugging Face |
| `test_entrenar_dcoco.py` | Pruebas en CPU de la pérdida, los recorridos, el árbol y el aprendizaje |

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

- **El margen (paso 1):** `recall_top16[p] − por_posicion_selector[p]` es la cota de lo que
  cualquier re-selección puede ganar en la posición `p`. Si es chico en todas las posiciones,
  DCOCO no tiene de dónde sacar.
- **El criterio para seguir:** que `largo_arbol_dcoco` supere a `largo_arbol_selector` en los pedidos
  apartados. Es lo que se va a servir, porque producción usa el árbol. Si no lo supera, no vale la
  pena el A/B.
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
| `--solo_evaluar`, `--max_pedidos`, `--max_anclas_eval`, `--max_pasos` | | Medición sola y recortes para pruebas |

**La pérdida** es entropía cruzada forzada por el maestro:

- para cada transición `p-1 → p`, el previo es el candidato que coincide con el token real en `p-1`
  y se pide el token real en `p`;
- las transiciones con algún token real fuera del top-16 no cuentan;
- en modo conjunto se suma la pérdida de `dflash2/entrenar.py` (0,9 TV + 0,1 CE).

**El checkpoint de salida** es el borrador (con la LoRA fusionada en modo conjunto) más:

- la cabeza en bf16 como `dcoco.*`, 10 tensores con los mismos nombres que `DCOCOHead` del repo de
  DCOCO;
- en `dflash_config`: `dcoco_candidate_topk`, `dcoco_emb_dim`, `dcoco_state_dim` y `dcoco_mlp_dim`;
- un bloque `dcoco` con la receta.

`../dflash2/cuantizar_rtn.py` lo cuantiza dejando la cabeza intacta.

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
python3 -m pytest -q entrenamiento/dflash2-dcoco/test_entrenar_dcoco.py        # pérdida, recorridos, árbol, aprendizaje
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
