# DFlash2: entrenamiento del borrador

Herramientas para ajustar, rotar y cuantizar el borrador DFlash2 de **qwen3.8_27b_idiotSavant_sm_86**
(publicado como [BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2)).
El pipeline completo, de la captura al A/B, lo orquesta [`../idiotsavant/dflash2.sh`](../idiotsavant/dflash2.sh).
Esta carpeta tiene las piezas que ese script llama, y que también se pueden usar sueltas.

Todo corre en la imagen `vllm/vllm-openai:v0.29.0` (torch, CUDA y safetensors), en **una** 3090 y
**con el servidor apagado**. Con el repo montado en `/repo`, el models-cache en `/models` y las
trazas en `/traces`, igual que en los scripts:

```bash
R=/home/usuario/Proyectos/genesis-vllm-patches; MC=/home/usuario/Proyectos/models-cache; T=$R/tests/bench/medicion/trazas
IMG=vllm/vllm-openai:v0.29.0
docker run --rm --gpus '"device=0"' --ipc host --memory 16g --memory-swap 16g \
  -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG /repo/entrenamiento/dflash2/<script> ...
```

## Archivos

| Archivo | Qué hace |
|---|---|
| `dflash2_torch.py` | DFlash2 en PyTorch puro, línea por línea con `qwen3_dflash.py`/`qwen3_dflash2.py` de vLLM v0.29.0. Es el modelo que se entrena: el contexto entra como las features del `fc` ya capturadas. `cargar_borrador` carga el checkpoint bf16/fp16 (sin el selector ni el `fc`) y `cargar_target` carga el embedding y el lm_head efectivos del target (con la rotación de PN149 y el lm_head int4 de PN139). |
| `entrenar.py` | Ajuste fino con LoRA contra el target, sobre las capturas. Evalúa antes y después de cada época y exporta bf16. |
| `rotar_borrador.py` | Adapta un borrador bf16 a un target con el residuo rotado: pliega la rotación en el `fc` y deja `genesis_rotacion` en el config para PN149. |
| `cuantizar_rtn.py` | bf16 → W4A16 compressed-tensors (pack-quantized, g128, simétrico, escalas fp16) por redondeo, con el formato del checkpoint publicado. |
| `decuantizar.py` | W4A16 → fp16 con los nombres del bf16. Sirve para reajustar el borrador **servido**: el entrenador parte de los pesos exactos que se sirven. |
| `gptq.py` | GPTQ W4 g128 con el mismo formato de salida que `cuantizar_rtn.py` (biblioteca; la usan los experimentos de cuantización). |

## Los datos: capturas de `dflash2.sh`

El paso 3 de `dflash2.sh` sirve el modelo con `GENESIS_CAPTURA_BORRADOR=1`
(`vllm/_genesis/captura_borrador.py`): regenera 2.000 pedidos de agente y los vuelve a mandar como
prefill. Queda un directorio por pedido en `tests/bench/medicion/trazas/captura/<nombre>/`:

- **`<req>.<pos0>.npz`:** `pos` (int32), `ids` (int32) y `feat` (fp16 [n, 5120]), que son las
  features del `fc` servido para cada token. Se guarda desde 2.048 tokens antes de la respuesta.
- **`etiquetas.npz`:** `prompt_len`, `comp_ids` (la respuesta del target) y el top-16 del target
  por posición (`top_ids` y `top_lp`).

Son ~24 MB por pedido. `entrenar.py` descarta los pedidos con huecos en las posiciones.

## `entrenar.py`

Entrena **LoRA** (r=64 por defecto) sobre las 7 lineales de las 5 capas. Quedan congelados el
`fc`, las normas, el selector y el embedding y lm_head del target, y también las proyecciones de la
conv salvo con `--conv`.

La pérdida es, por cada posición de máscara, `0,9 × TV + 0,1 × CE` contra el top-16 del target.
Con `--auf`, se corta en el primer fallo del top-1 (Spec-AUF).

```bash
.../entrenar.py --datos /traces/captura/entrena_rot --salida /traces/borrador_ft \
  --borrador /traces/borrador_prod_fp16 --noon /models/qwen3.8_27b_idiotSavant_sm_86 \
  --epocas 1 --lote 32 --lr 1e-4 --rango 64 --apartar 0.03 --auf
```

| Argumento | Default | Para qué |
|---|---|---|
| `--datos` | (obligatorio) | Directorio de capturas, un subdirectorio por pedido |
| `--salida` | (obligatorio) | Ahí quedan `model.safetensors` (bf16, con la LoRA fusionada), `config.json`, `registro.json` y `particion.json` |
| `--borrador` | `/models/incoai-qwen3.8-27b-dflash2-bf16` | Punto de partida en bf16/fp16. Para reajustar producción: la salida de `decuantizar.py` |
| `--noon` | snapshot de noon | Target del que salen el embedding y el lm_head. Con `genesis_rotacion` en su config, se usan en la base del borrador |
| `--K` | `8` | Posiciones de máscara por bloque (`num_speculative_tokens`) |
| `--ventana` | `2048` | Contexto que ve cada ancla (la ventana deslizante del borrador) |
| `--lote` | `32` | Anclas por paso |
| `--epocas`, `--lr`, `--rango` | `1`, `1e-4`, `64` | Época(s), learning rate de la LoRA y su rango |
| `--auf` | apagado | Corta la pérdida en el primer fallo del top-1 |
| `--conv` | apagado | Entrena también las proyecciones de la conv |
| `--conv_int8 [--conv_grupo N]` | apagado | Simula la conv en int8 de PN157 (STE). Usarlo si se sirve con `GENESIS_PN157_BITS_CONV=8` |
| `--vocab N [--vocab_orden f.npy]` | `0` | Candidatos sobre los N tokens más frecuentes, como PN159 al servir |
| `--apartar` | `0,05` | Fracción de pedidos que no se entrenan y se usan para evaluar. Semilla fija: con el mismo valor salen los mismos pedidos |
| `--solo_evaluar` | apagado | Solo mide el borrador de `--borrador` |
| `--max_pedidos`, `--max_anclas_eval`, `--max_pasos` | `0` | Recortes para pruebas rápidas |

**Qué imprime:** `BASE {...}` antes de entrenar y `EPOCA n {...}` después de cada época, sobre los
pedidos apartados. Los mismos números quedan en `registro.json`:

- `largo_greedy`: 1 + posiciones seguidas que acierta el top-1, con el bonus incluido;
- `por_posicion`: aceptación por posición;
- `recall_top16`: si el token del target está entre los 16 candidatos, que es lo que aprovecha el
  árbol.

Memoria: con `--lote 32` alcanzan ~9 GB de RAM (medido en `dflash2.sh`, que sube el tope del
contenedor si el proceso muere por memoria).

## Rotar, cuantizar, decuantizar

```bash
# borrador bf16 original -> base de un target rotado (paso 1 de dflash2.sh)
python3 rotar_borrador.py <borrador_bf16> <target_bf16_original> <target_rotado> <salida_bf16>

# bf16 -> W4A16 (pasos 2 y 5). El tercer argumento da el quantization_config que se copia
python3 cuantizar_rtn.py <dir_bf16> <dir_salida> /models/qwen3.8_27b_idiotSavant_sm_86_dflash2
#   --escalas=<dir_w4a16>: reusa esas escalas. Lo que no se entreno vuelve EXACTO a sus int4

# W4A16 servido -> fp16 para reajustarlo (punto de partida de entrenar.py)
python3 decuantizar.py /models/qwen3.8_27b_idiotSavant_sm_86_dflash2 /traces/borrador_prod_fp16
```

`cuantizar_rtn.py` solo cuantiza las lineales de atención y MLP y el `fc`. Todo lo demás (normas,
conv, selector y, en un checkpoint DFlash2-DCOCO, la cabeza `dcoco.*`) pasa sin cambios.

## Recetas

- **Pipeline completo** (base de Inco → rotación → captura → entrenamiento → cuantización → A/B):

  ```bash
  cd ../idiotsavant
  DRY_RUN=1 ./dflash2.sh
  setsid nohup ./dflash2.sh > dflash2.log 2>&1 &
  ./dflash2.sh estado
  ```

- **Reajustar el borrador servido** con otra receta, reusando capturas: decuantizar producción,
  entrenar desde ahí y cuantizar con `--escalas` de producción. Los ejemplos están en
  [`../idiotsavant/destilar_conv8_vocab.sh`](../idiotsavant/destilar_conv8_vocab.sh) (variantes
  `--conv_int8`/`--vocab`, con el A/B incluido) y en
  [`../idiotsavant/destilar_F.sh`](../idiotsavant/destilar_F.sh).
- **Solo medir** un borrador en los pedidos apartados: `entrenar.py ... --solo_evaluar`, con las
  mismas restricciones con que se sirve (`--conv_int8`, `--vocab`).
- **Servirlo:** montar el directorio cuantizado como borrador de la compose idiotsavant. Por ejemplo,
  un override con
  `volumes: - <dir>:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro`, como
  hacen los scripts de A/B.

## Relación con dflash2-dcoco

[`../dflash2-dcoco/`](../dflash2-dcoco/) entrena el corrector DCOCO sobre este mismo borrador.
Importa de acá `dflash2_torch.py` y `entrenar.py` (datos, LoRA, pérdida, `guardar`) sin
modificarlos.
