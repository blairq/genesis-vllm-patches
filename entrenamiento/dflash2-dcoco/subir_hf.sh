#!/bin/bash
# Publica un checkpoint DFlash2-DCOCO cuantizado (salida del paso 3 de dflash2_dcoco.sh) en Hugging Face.
#   ./subir_hf.sh <dir_checkpoint> [repo, default BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco]
# Necesita `huggingface-cli login` (o HF_TOKEN) con permiso de escritura. Crea el repo si no existe, copia la
# tarjeta (README_hf.md, con los numeros del registro de entrenamiento) y sube todo.
set -eu
AQUI="$(cd "$(dirname "$0")" && pwd)"
DIR=${1:?"uso: $0 <dir_checkpoint> [repo]"}
REPO=${2:-BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco}
[ -f "$DIR/model.safetensors" ] && [ -f "$DIR/config.json" ] || { echo "falta model.safetensors/config.json en $DIR"; exit 1; }
python3 - "$DIR" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
assert "dcoco_emb_dim" in c.get("dflash_config", {}), "el config no trae la cabeza DCOCO (dcoco_emb_dim)"
from safetensors import safe_open
with safe_open(sys.argv[1] + "/model.safetensors", "pt") as f:
    n = sum(k.startswith("dcoco.") for k in f.keys())
assert n == 10, f"se esperaban 10 tensores dcoco.*, hay {n}"
print("checkpoint DFlash2-DCOCO ok")
PY
cp "$AQUI/README_hf.md" "$DIR/README.md"
huggingface-cli repo create "${REPO#*/}" --organization "${REPO%%/*}" --type model -y 2>/dev/null || true
huggingface-cli upload "$REPO" "$DIR" . --commit-message "DFlash2-DCOCO: borrador DFlash2 con el corrector DCOCO (PN178)"
echo "publicado: https://huggingface.co/$REPO"
