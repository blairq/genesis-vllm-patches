#!/bin/bash
# E: conv int8 simulada con la conv CONGELADA (sin --conv) + vocab 64k; A/B contra B (control) y A. Espera a ab_rotv.sh.
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas; MC=/home/usuario/Proyectos/models-cache
IMG=vllm/vllm-openai:v0.29.0; PROD=qwen3.8_27b_idiotSavant_sm_86_dflash2
until grep -q "TODO LISTO" $T/banco/rotv.log 2>/dev/null; do sleep 60; done
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
if [ ! -f $T/borrador_ft_E/model.safetensors ]; then
  echo "$(date +%T) entrenando E"
  docker run --rm --name ft-E --gpus '"device=0"' --ipc host --memory 16g --memory-swap 16g -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces \
    --entrypoint python3 $IMG /repo/entrenamiento/borrador/entrenar.py --datos /traces/captura/entrena_rot --salida /traces/borrador_ft_E \
    --borrador /traces/borrador_prod_fp16 --noon /models/qwen3.8_27b_idiotSavant_sm_86 --epocas 1 --lote 32 --lr 1e-4 --rango 64 \
    --apartar 0.03 --auf --conv_int8 --vocab 65536 > $T/banco/ft_E.log 2>&1
  grep -E "^BASE|^EPOCA" $T/banco/ft_E.log | cut -c1-60
fi
[ -f $MC/${PROD}_E/model.safetensors ] || docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
  "python3 /repo/entrenamiento/borrador/cuantizar_rtn.py /traces/borrador_ft_E /models/${PROD}_E /models/$PROD --escalas=/models/$PROD && chmod -R a+rX /models/${PROD}_E" 2>&1 | tail -1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
OV=$T/banco/ov-destilar; mkdir -p $OV
for r in 3 4; do for b in B E A; do
  out=$T/banco/destilar_${b}_r$r.json; [ -f $out ] && continue
  case $b in A) bor=$MC/$PROD; E="GENESIS_PN157_BITS_CONV=0";; B) bor=$MC/${PROD}_B; E="GENESIS_PN157_BITS_CONV=0";; E) bor=$MC/${PROD}_E; E="GENESIS_PN157_BITS_CONV=8";; esac
  printf 'services:\n  vllm-server:\n    volumes:\n      - %s:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro\n' $bor > $OV/$b.yml
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $E GENESIS_PN159_VOCAB=65536 docker compose $C -f $OV/$b.yml up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) A/B $b r$r"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder $out 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for b in A B D E; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/destilar_${b}_r*.json'))]
print('$b', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
