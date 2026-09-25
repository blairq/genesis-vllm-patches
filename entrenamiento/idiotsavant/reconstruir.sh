#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  Reconstruye qwen3.8_27b_idiotSavant_sm_86 de punta a punta, en UN comando:
#     entorno -> BF16 -> calibracion -> dry-run -> reconstruccion
#  Cada paso se saltea si ya esta hecho; relanzar retoma donde quedo.
#
#  Uso:
#     ./reconstruir.sh              todo (primero valida con el dry-run; si algo falla, frena)
#     ./reconstruir.sh --dry-run    solo prepara y valida, sin cuantizar
#  Para una corrida larga, desacoplado de la terminal:
#     setsid nohup ./reconstruir.sh > reconstruir.log 2>&1 &
#     ./correr.sh idiotsavant.py estado --trabajo trabajo --salida qwen3.8_27b_idiotSavant_sm_86
#
#  Variables (todas opcionales):
#     BF16     el modelo base; si no existe se baja de orcarouter/Qwen3.8-27B-Uncensored (55 GB)
#              [./modelos/orcarouter-qwen3.8-27b-uncensored-bf16]
#     CALIB    la calibracion [./calib_256x4096.npy, la EXACTA de idiotSavant, viene con este repo].
#              CALIB=desde-hf la arma desde el dataset publico (ver calibracion.py).
#     TRABAJO  cache de trabajo [./trabajo]      SALIDA  el checkpoint [./qwen3.8_27b_idiotSavant_sm_86]
#  Codigos: 0 ok, 1 error, 2 faltan recursos o el dry-run encontro un problema.
# ══════════════════════════════════════════════════════════════════════════════════════════════
set -euo pipefail
AQUI="$(cd "$(dirname "$0")" && pwd)"
cd "$AQUI"
BF16=${BF16:-$AQUI/modelos/orcarouter-qwen3.8-27b-uncensored-bf16}
CALIB=${CALIB:-$AQUI/calib_256x4096.npy}
TRABAJO=${TRABAJO:-$AQUI/trabajo}
SALIDA=${SALIDA:-$AQUI/qwen3.8_27b_idiotSavant_sm_86}
SOLO_VALIDAR=0; [ "${1:-}" = "--dry-run" ] && SOLO_VALIDAR=1
log() { echo "$(date +%T) [reconstruir] $*"; }
libre_gb() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; df --output=avail -BG "$d" | tail -1 | tr -dc 0-9; }

# 1. entorno local (.venv + .cache adentro de esta carpeta)
if [ ! -x .venv/bin/python ]; then log "1. creando el entorno (./preparar.sh)"; ./preparar.sh; else log "1. entorno listo"; fi
. ./entorno.sh

# 2. modelo base
if [ -f "$BF16/model.safetensors.index.json" ]; then log "2. BF16 listo ($BF16)"; else
  falta=$((55 + 19 + 5))
  [ "$(libre_gb "$BF16")" -ge $falta ] || { log "hacen falta ~$falta GB libres (BF16 55 + salida 19 + margen)"; exit 2; }
  log "2. bajando orcarouter/Qwen3.8-27B-Uncensored (55 GB) a $BF16"
  # Xet se colgo en la red donde se armo esto; por LFS clasico anda en todos lados
  HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1} .venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored --local-dir "$BF16"
fi

# 3. calibracion
if [ "$CALIB" = desde-hf ]; then
  CALIB=$AQUI/calib_desde_hf.npy
  if [ ! -f "$CALIB" ]; then log "3. armando la calibracion desde el dataset publico"
    ./correr.sh calibracion.py --bf16 "$BF16" --desde-hf --salida "$CALIB"; fi
fi
[ -f "$CALIB" ] || { log "no esta la calibracion $CALIB (CALIB=desde-hf para armarla)"; exit 2; }
log "3. calibracion: $CALIB"

# 4. dry-run: valida BF16, calibracion, recursos, matematica y dos capas enteras en memoria
log "4. dry-run"
if ! ./correr.sh idiotsavant.py todo --dry-run --bf16 "$BF16" --calib "$CALIB" --trabajo "$TRABAJO" --salida "$SALIDA"; then
  log "el dry-run encontro problemas: ver las lineas FALLA de arriba"; exit 2
fi
[ $SOLO_VALIDAR = 1 ] && { log "listo para correr (sin --dry-run)"; exit 0; }

# 5. reconstruccion (calibrar || cuantizar -> armar), retomable
log "5. reconstruyendo -> $SALIDA  (avance: ./correr.sh idiotsavant.py estado --trabajo $TRABAJO --salida $SALIDA)"
./correr.sh idiotsavant.py todo --bf16 "$BF16" --calib "$CALIB" --trabajo "$TRABAJO" --salida "$SALIDA"
log "LISTO: $SALIDA  (para servirlo hace falta la Hadamard antes de down_proj: ver README)"
