#!/bin/bash
# F: receta de B (produccion + 1 epoca, AUF) con el vocabulario recortado de PN159 (64k, mismo orden que el
# servidor) y la conv en fp16, sobre las capturas entrena_rot (25-09). Despues A/B corto contra A y B:
# 3 arranques de 20 pedidos por brazo (protocolo corto del 03-10).
# Uso: setsid nohup ./destilar_F.sh > ../../tests/bench/medicion/trazas/banco/destilar_F.log 2>&1 &   (retoma)
set -u
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas; MC=/home/usuario/Proyectos/models-cache
IMG=vllm/vllm-openai:v0.29.0; PROD=qwen3.8_27b_idiotSavant_sm_86_dflash2; MOD=/models/qwen3.8_27b_idiotSavant_sm_86
log() { echo "$(date +%T) [F] $*"; }
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
# 1. entrenar
if [ ! -f $T/borrador_ft_F/model.safetensors ]; then
  log "entrenando F (--vocab 65536)"
  docker run --rm --name ft-F --gpus '"device=0"' --ipc host --memory 16g --memory-swap 16g \
    -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG /repo/entrenamiento/borrador/entrenar.py \
    --datos /traces/captura/entrena_rot --salida /traces/borrador_ft_F --borrador /traces/borrador_prod_fp16 --noon $MOD \
    --epocas 1 --lote 32 --lr 1e-4 --rango 64 --apartar 0.03 --auf --vocab 65536 > $T/banco/ft_F.log 2>&1
  grep -E "^BASE|^EPOCA" $T/banco/ft_F.log | cut -c1-200
fi
[ -f $T/borrador_ft_F/model.safetensors ] || { log "el entrenamiento no dejo modelo: ver ft_F.log"; exit 1; }
# 2. cuantizar con las escalas de produccion
[ -f $MC/${PROD}_F/model.safetensors ] || docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
  "python3 /repo/entrenamiento/borrador/cuantizar_rtn.py /traces/borrador_ft_F /models/${PROD}_F /models/$PROD --escalas=/models/$PROD && chmod -R a+rX /models/${PROD}_F" 2>&1 | grep -v -i warn | tail -1
# 3. A/B corto: A, B, F x 3 arranques de 20 pedidos, intercalados
OV=$T/banco/ov-destilar; mkdir -p $OV
for r in 1 2 3; do for b in A B F; do
  out=$T/banco/destF_${b}_r$r.json; [ -f $out ] && continue
  case $b in A) bor=$MC/$PROD;; B) bor=$MC/${PROD}_B;; F) bor=$MC/${PROD}_F;; esac
  printf 'services:\n  vllm-server:\n    volumes:\n      - %s:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro\n' $bor > $OV/$b.yml
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml -f ov-aislado.yml -f $OV/$b.yml up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo20.jsonl coder /traces/banco/destF_${b}_r$r.json 4 1536 > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('$out'));print('$(date +%T) $b r$r acept %.3f ajeno %s errores %s' % (d['largo_aceptacion'], d['ajeno'], d['errores']))" 2>&1 | tail -1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for b in A B F; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/destF_${b}_r*.json'))]
print('$b', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
log "F LISTO"
