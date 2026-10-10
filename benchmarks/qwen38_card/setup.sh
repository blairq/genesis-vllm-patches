#!/usr/bin/env bash
# Prepara las dependencias de la suite del model card (una sola vez).
#  - venv con el verificador oficial de IFBench (nltk, spacy, emoji, syllapy...)
#  - clon de CharXiv (prompts oficiales de respuesta y calificacion)
#  - imagen python:3.12-slim para correr el codigo de LiveCodeBench sin red
# GPQA y HLE son datasets "gated": pedir acceso en HF con la cuenta del token
# (https://huggingface.co/datasets/Idavidrein/gpqa y .../cais/hle).
set -euo pipefail
C="${QWEN38_CARD_CACHE:-$HOME/.cache/qwen38_card}"
mkdir -p "$C"
[ -d "$C/IFBench" ] || git clone -q --depth 1 https://github.com/allenai/IFBench "$C/IFBench"
[ -d "$C/CharXiv" ] || git clone -q --depth 1 https://github.com/princeton-nlp/CharXiv "$C/CharXiv"
[ -x "$C/venv/bin/python" ] || python3 -m venv "$C/venv"
"$C/venv/bin/pip" install -q "$C/IFBench"
# evaluation_lib.py vive en la raiz del clon, fuera del paquete
SP=$("$C/venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')
echo "$C/IFBench" > "$SP/ifbench_root.pth"
NLTK_DATA="$HOME/nltk_data" "$C/venv/bin/python" -c "
import nltk
for p in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng']:
    nltk.download(p, quiet=True)"
docker image inspect python:3.12-slim >/dev/null 2>&1 || docker pull -q python:3.12-slim
python3 -c "import httpx, pandas, PIL, huggingface_hub, sympy" || \
  echo "falta alguna dependencia del python del sistema: pip install --user httpx pandas pillow huggingface_hub sympy"
echo "listo: $C"
