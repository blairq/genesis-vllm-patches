#!/bin/bash
# Prepara el entorno LOCAL del proyecto: .venv/ y .cache/ adentro de esta carpeta. No instala nada
# en el sistema ni escribe en ~/.cache. Borrar la carpeta .venv (y .cache) deja todo como estaba.
#   ./preparar.sh            instala (idempotente: si ya esta, solo verifica)
#   ./preparar.sh --verificar solo verifica
set -eu
AQUI="$(cd "$(dirname "$0")" && pwd)"
. "$AQUI/entorno.sh"
TORCH="torch==2.13.0"
INDICE="https://download.pytorch.org/whl/cu130"   # CUDA 13.0, como la imagen de vLLM (driver >= 580)

verificar() {
  "$AQUI/.venv/bin/python" - <<'PY'
import torch, transformers, safetensors, numpy
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, "
      f"safetensors {safetensors.__version__}, numpy {numpy.__version__}")
print(f"GPUs visibles: {torch.cuda.device_count()}" + "".join(
    f"\n  {i}: {torch.cuda.get_device_name(i)} sm_{''.join(map(str, torch.cuda.get_device_capability(i)))}"
    for i in range(torch.cuda.device_count())))
from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401  (el forward que se calibra)
print("OK")
PY
}
if [ "${1:-}" = "--verificar" ]; then verificar; exit; fi

libre=$(df --output=avail -BG "$AQUI" | tail -1 | tr -dc 0-9)
if [ "$libre" -lt 8 ]; then echo "hacen falta ~8 GB libres para el entorno (hay ${libre} GB)"; exit 2; fi
command -v nvidia-smi >/dev/null || { echo "no hay nvidia-smi: hace falta el driver de NVIDIA"; exit 2; }
if [ ! -x "$AQUI/.venv/bin/python" ]; then
  python3 -m venv "$AQUI/.venv"
fi
"$AQUI/.venv/bin/pip" install --upgrade pip
"$AQUI/.venv/bin/pip" install "$TORCH" --index-url "$INDICE"
"$AQUI/.venv/bin/pip" install -r "$AQUI/requirements.txt"
verificar
