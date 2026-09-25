# idiotSavant: reconstruir el modelo y su borrador

- **`qwen3.8_27b_idiotSavant_sm_86`**: Qwen3.8-27B (sin censura, orcarouter) cuantizado a W4 con el
  residuo rotado, para servirse como W4A8 en RTX 3090 (sm_86). Tiene la mitad de error que el cuant
  público (noon), con la misma velocidad.
- **`qwen3.8_27b_idiotSavant_sm_86_dflash2`**: su borrador DFlash2 para decodificación especulativa.

El porqué de cada decisión, con las mediciones que la respaldan, está en [DECISIONES.md](DECISIONES.md).

## Requisitos

- Driver de NVIDIA ≥ 580 (CUDA 13), 1 o 2 GPUs de 24 GB. Con dos, la calibración y la cuantización van
  en paralelo.
- ~12 GB de RAM libres.
- Disco: ~19 GB para el checkpoint, ~8 GB para el entorno y ~1 GB de trabajo. Guardar el estado oculto
  para retomar suma 10,7 GB, y conservar las Hessianas ~50 GB.
- Para el borrador: Docker y la imagen `vllm/vllm-openai:v0.29.0` con Genesis (ver `dflash2.sh`).

## Entorno (una vez)

```bash
./preparar.sh              # crea .venv/ y .cache/ ACÁ; no toca el sistema
./preparar.sh --verificar  # solo verifica versiones y GPUs
```

Para desinstalar, borrar `.venv/` y `.cache/`.

## Modelo

```bash
B=/home/usuario/Proyectos/models-cache/orcarouter-qwen3.8-27b-uncensored-bf16
C=../../tests/bench/medicion/trazas/cuant/calib_256x4096.npy
W=/home/usuario/Proyectos/cuant-cache/idiotsavant          # trabajo (caché, logs, marcas)
S=/home/usuario/Proyectos/models-cache/qwen3.8_27b_idiotSavant_sm_86

./correr.sh idiotsavant.py todo --dry-run --bf16 $B --calib $C --trabajo $W --salida $S   # ~20 s, no escribe capas
setsid nohup ./correr.sh idiotsavant.py todo --bf16 $B --calib $C --trabajo $W --salida $S > idiotsavant.log 2>&1 &
./correr.sh idiotsavant.py estado --trabajo $W --salida $S           # avance, en texto
./correr.sh idiotsavant.py estado --json --trabajo $W --salida $S    # avance, para scripts y LLMs
./correr.sh tui.py modelo --trabajo $W --salida $S                   # visor opcional (Ctrl-C para salir)
```

- **Retomar:** relanzar el mismo comando; saltea las capas terminadas.
- **Correr las etapas a mano:** `calibrar`, `cuantizar` y `armar` en lugar de `todo`. `calibrar` y
  `cuantizar` pueden correr a la vez: la segunda espera las Hessianas de la primera.
- **Logs:** `$W/logs/{calibrar,cuantizar,armar}.log`.
- **Informe por capa:** error local de la capa servible, error de pesos por lineal, cresta de las
  entradas antes y después de rotar, y canales con g = 0. Queda en `$W/capa_NN/informe.json` y, al
  final, junto en `$S/informe_idiotsavant.json`.

**Códigos de salida:** 0 = terminó; 1 = error; 2 = faltan recursos o el dry-run encontró un problema.
Cada línea del dry-run dice `OK` o `FALLA` y el motivo.

**Calibración:** un `.npy` entero `[N, L]` con tokens de tráfico real (se usó 256 × 4096 de pedidos de
agente de código). Cómo se armó: `tests/bench/medicion/trazas/banco/armar_calibracion.py`.

**Servir:** `compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml`. Necesita
`GENESIS_ENABLE_PN148_ROT_DOWN=1` (sin esa Hadamard el modelo da basura) y `VLLM_MARLIN_INPUT_DTYPE=int8`.
Al cambiar de checkpoint, vaciar el offload de KV (`/home/usuario/Proyectos/kv-offload`).

## Borrador DFlash2

```bash
DRY_RUN=1 ./dflash2.sh                    # valida todo sin crear nada
setsid nohup ./dflash2.sh > dflash2.log 2>&1 &
./dflash2.sh estado [--json]              # avance
./correr.sh tui.py borrador               # visor opcional
PASOS="6" ./dflash2.sh                    # solo el A/B (cada paso hecho se saltea solo)
```

Los pasos son: 1 plegar la rotación en el fc, 2 cuantizar la base, 3 captura (~15 s por pedido, frena
por disco), 4 entrenar (~75 min), 5 cuantizar, 6 A/B. Si el servidor idiotSavant está corriendo, se
para durante la captura y el A/B, y se vuelve a levantar al final. Se sirve con
`GENESIS_ENABLE_PN149_ROT_BORRADOR=1`.

## Notas para una LLM que lo opere

- Antes de lanzar, correr el dry-run y leer las líneas `FALLA`.
- Lanzar desacoplado (`setsid nohup … &`) y consultar con `estado --json`. Campos útiles:
  `cuantizadas`, `eta_minutos`, `procesos_vivos`, `marcas`, `recursos`, `capas[i].error_capa`.
- Un error local por capa de 0,5–10% es normal (la 0 ~1%, la meseta 23–51 ~13%). Un salto de 10× en una
  sola capa indica algo mal armado.
- No mezclar esta corrida con el servidor vLLM en las mismas GPUs. El dry-run chequea la memoria de GPU
  libre.
