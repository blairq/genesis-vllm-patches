# idiotSavant: reconstruir el modelo y su borrador

- **`qwen3.8_27b_idiotSavant_sm_86`**: Qwen3.8-27B (sin censura, orcarouter) en W4 con el residuo
  rotado, para servirse como W4A8 en RTX 3090 (sm_86). Tiene la mitad de error que el int4 público
  (noon), a la misma velocidad.
- **`qwen3.8_27b_idiotSavant_sm_86_dflash2`**: su borrador DFlash2.

Publicados en https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86 (con esta carpeta adentro,
en `reproducir/`) y https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2. El porqué de
cada decisión, con sus mediciones: [DECISIONES.md](DECISIONES.md).

## Qué hace falta

| | |
|---|---|
| GPU | 1 o 2 NVIDIA de 24 GB (se armó en 2× RTX 3090), driver ≥ 580 (CUDA 13). Con dos, calibración y cuantización van en paralelo: ~85 min |
| RAM | ~12 GB libres (también se mira el límite del cgroup si corre en un contenedor) |
| Disco | ~80 GB: BF16 55 + salida 19 + entorno 5 (+2 si la calibración se arma desde el dataset) |
| Software | Linux, `python3` (3.10+) con `venv` y `nvidia-smi`. No se instala nada en el sistema |

## Rápido: un solo comando

```bash
./reconstruir.sh --dry-run                               # entorno + BF16 (si falta, 55 GB) + validación
setsid nohup ./reconstruir.sh > reconstruir.log 2>&1 &   # la reconstrucción (larga)
./correr.sh idiotsavant.py estado --trabajo trabajo --salida qwen3.8_27b_idiotSavant_sm_86   # avance
```

- **Variables opcionales:** `BF16=…` (si ya lo tenés bajado, por ejemplo
  `/home/usuario/Proyectos/models-cache/orcarouter-qwen3.8-27b-uncensored-bf16`), `CALIB=…`
  (`desde-hf` para armarla del dataset), `TRABAJO=…` y `SALIDA=…`.
- **Retomar:** cada paso se saltea si ya está hecho; relanzar el mismo comando sigue donde quedó.

## Paso a paso (lo que hace `reconstruir.sh`)

### 1. Entorno virtual

Todo queda adentro de esta carpeta: `.venv/` para los paquetes y `.cache/` para las cachés (pip,
torch, Triton, HuggingFace). Borrar las dos deja el sistema como estaba.

```bash
./preparar.sh               # hace lo de abajo y verifica las GPUs
./preparar.sh --verificar   # solo verifica
```

o a mano:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` fija las versiones de la imagen de vLLM con la que se sirve. Otro transformers
podría cambiar el forward de Qwen3.5 que se calibra. Todo se corre con `./correr.sh <script> …`, que
activa el `.venv`, manda las cachés a `.cache/` y deja el Hub apagado.

### 2. Modelo base (55 GB)

```bash
HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 .venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored --local-dir modelos/orcarouter-qwen3.8-27b-uncensored-bf16
```

`HF_HUB_DISABLE_XET=1` es porque en esta red Xet se colgaba; por LFS anda siempre.

### 3. Calibración

| opción | cómo | resultado |
|---|---|---|
| **exacta** (default) | `calib_256x4096.npy` (en esta carpeta) | los mismos pesos que en HF (salvo el no-determinismo de la GPU) |
| desde el dataset público | `CALIB=desde-hf ./reconstruir.sh` o `./correr.sh calibracion.py --bf16 BF16 --desde-hf --salida calib.npy` | tráfico equivalente, pesos no idénticos |
| tráfico propio | `./correr.sh calibracion.py --bf16 BF16 --conversaciones charlas.jsonl --salida calib.npy [--template otro.jinja]` | ajustada a tu uso |

- **La exacta:** 256 × 4096 tokens. Cada muestra es la ventana final de un pedido de agente de código
  con tools: el prompt más la respuesta del modelo servido, con el chat template de producción. Los
  prompts salen de nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0); se armó con
  `tests/bench/medicion/trazas/banco/armar_calibracion.py`.
- **`--desde-hf`:** baja ese mismo dataset (2 GB, a `.cache/`), toma una trayectoria por repo y la
  corta justo después de un turno del asistente.
- **`--conversaciones`:** un JSONL en formato de chat de OpenAI, una conversación por línea:
  `{"messages": [...], "tools": [...]}`.

### 4. Dry-run (~20 s)

```bash
./correr.sh idiotsavant.py todo --dry-run --bf16 BF16 --calib calib_256x4096.npy --trabajo trabajo --salida qwen3.8_27b_idiotSavant_sm_86
```

Valida, sin escribir capas:
- el BF16, con todas las claves de cada capa;
- la calibración;
- RAM, GPU y disco;
- la matemática: ortogonalidad, GPTQ y empaquetado;
- una capa GDN y una de atención, completas en memoria.

Cada línea dice `OK` o `FALLA` y el motivo.

### 5. Reconstrucción

```bash
./correr.sh idiotsavant.py todo --bf16 BF16 --calib calib_256x4096.npy --trabajo trabajo --salida qwen3.8_27b_idiotSavant_sm_86
```

- **Dos procesos:** `calibrar` (la pasada BF16, que escribe las Hessianas de cada capa) y `cuantizar`
  (GPTQ rotado y medición), uno por GPU, que se hablan por `trabajo/`. Después `armar` escribe el
  resto del checkpoint y el config.
- **Recursos:** si falta RAM o disco, espera en vez de reventar.
- **Seguimiento:**
  - `idiotsavant.py estado [--json]`: el avance (`--json` es para scripts y LLMs);
  - `trabajo/logs/{calibrar,cuantizar,armar}.log`;
  - `./correr.sh tui.py modelo --trabajo trabajo --salida …`: visor opcional.
- **Informe por capa:** `trabajo/capa_NN/informe.json` y, al final,
  `…/informe_idiotsavant.json`. Un error local de 0,5–13% por capa es normal (bajo al principio,
  meseta en 23–51); una capa 10 veces por encima de sus vecinas indica que algo está mal.
- **Códigos de salida:** 0 = terminó; 1 = error; 2 = faltan recursos o el dry-run falló.

### 6. Servir

Hace falta la Hadamard antes de `down_proj` (PN148). El compose es
`../../compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml`, con cada ajuste explicado. Al cambiar
de checkpoint, vaciar el offload de KV.

## Borrador DFlash2 (`dflash2.sh`)

Se entrena contra el modelo mientras vLLM lo sirve, así que corre en la imagen de vLLM con los parches,
no en el `.venv`.

```bash
DRY_RUN=1 ./dflash2.sh                     # valida todo sin crear nada
setsid nohup ./dflash2.sh > dflash2.log 2>&1 &
./dflash2.sh estado [--json]               # avance
./correr.sh tui.py borrador                # visor opcional
PASOS="6" ./dflash2.sh                     # solo el A/B (cada paso hecho se saltea solo)
```

Los pasos son: 1 plegar la rotación en el fc, 2 cuantizar la base, 3 captura (~15 s por pedido, frena
por disco), 4 entrenar (~75 min), 5 cuantizar, 6 A/B. Arranca por defecto del DFlash2 original de Inco.

## Notas para una LLM que lo opere

- Antes de lanzar, correr `./reconstruir.sh --dry-run` y leer las líneas `FALLA`.
- Lanzar desacoplado y consultar `idiotsavant.py estado --json`. Campos útiles: `cuantizadas`,
  `eta_minutos`, `procesos_vivos`, `marcas`, `recursos`, `capas[i].error_capa`.
- No correrlo en las mismas GPUs que un servidor vLLM: el dry-run chequea la memoria de GPU libre.
