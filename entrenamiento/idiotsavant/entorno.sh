# Variables del entorno LOCAL (lo usan preparar.sh y correr.sh): todo lo que torch, Triton,
# HuggingFace o pip quieran cachear queda en .cache/ de esta carpeta.
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$AQUI/.cache"
export PIP_NO_CACHE_DIR=1                # la rueda de torch son ~3 GB: no guardarla dos veces
export HF_HOME="$AQUI/.cache/huggingface"
export HF_HUB_OFFLINE=1                  # nunca baja nada: el BF16 y la calibracion son locales
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$AQUI/.cache/torch"
export TRITON_CACHE_DIR="$AQUI/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AQUI/.cache/inductor"
export CUDA_CACHE_PATH="$AQUI/.cache/nv"
export PYTHONPYCACHEPREFIX="$AQUI/.cache/pycache"
export TMPDIR="$AQUI/.cache/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
