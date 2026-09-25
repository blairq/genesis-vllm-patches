#!/bin/bash
# Corre un script del proyecto con el entorno local (.venv + .cache de esta carpeta).
#   ./correr.sh idiotsavant.py todo --dry-run --bf16 ... --calib ... --trabajo ... --salida ...
#   ./correr.sh idiotsavant.py estado --json --trabajo ... --salida ...
#   ./correr.sh tui.py modelo --trabajo ... --salida ...
set -eu
AQUI="$(cd "$(dirname "$0")" && pwd)"
[ -x "$AQUI/.venv/bin/python" ] || { echo "falta el entorno: correr ./preparar.sh"; exit 2; }
. "$AQUI/entorno.sh"
guion="$1"; shift
exec "$AQUI/.venv/bin/python" "$AQUI/$guion" "$@"
