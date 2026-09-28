#!/bin/bash
# Reajuste del borrador SERVIDO con la conv int8 (PN157, BITS_CONV=8) y el vocabulario recortado (PN159) en el
# lazo, contra un control con la receta de siempre. Mismas capturas (entrena_rot) y misma particion que
# produccion (rotA: --apartar 0.03, semilla 0): los 25 apartados no los vio.
#   A  produccion tal cual                        -> servido conv densa, 64k
#   B  produccion + 1 epoca, receta de siempre    -> servido conv densa, 64k   (control de la epoca extra)
#   C  produccion + 1 epoca, conv int8 (STE, conv entrenable) + vocab 32k -> servido conv int8, 32k
#   D  idem con vocab 64k                         -> servido conv int8, 64k
# Uso: setsid nohup ./destilar_conv8_vocab.sh > destilar.log 2>&1 &   (retoma: saltea lo hecho)
set -u
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas; MC=/home/usuario/Proyectos/models-cache
IMG=vllm/vllm-openai:v0.29.0; PROD=qwen3.8_27b_idiotSavant_sm_86_dflash2; MOD=/models/qwen3.8_27b_idiotSavant_sm_86
log() { echo "$(date +%T) [destilar] $*"; }
ram_libre_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
docker stop genesis-27b-idiotsavant >/dev/null 2>&1

# 0. produccion en fp16 (exactamente los pesos servidos)
[ -f $T/borrador_prod_fp16/model.safetensors ] || docker run --rm -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint bash $IMG -c \
  "python3 /repo/entrenamiento/borrador/decuantizar.py /models/$PROD /traces/borrador_prod_fp16 && chmod -R a+rwX /traces/borrador_prod_fp16"

entrenar() {  # nombre salida flags...
  local n=$1; shift
  local sal=/traces/borrador_ft_$n
  if [ -f $T/borrador_ft_$n/model.safetensors ] || [ -f $T/banco/ft_$n.solo_eval ]; then log "$n ya hecho"; return; fi
  while [ "$(ram_libre_gb)" -lt 17 ]; do sleep 30; done
  log "$n: $*"
  docker run --rm --name ft-$n --gpus '"device=0"' --ipc host --memory 16g --memory-swap 16g \
    -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG /repo/entrenamiento/borrador/entrenar.py \
    --datos /traces/captura/entrena_rot --salida $sal --borrador /traces/borrador_prod_fp16 --noon $MOD \
    --epocas 1 --lote 32 --lr 1e-4 --rango 64 --apartar 0.03 --auf "$@" > $T/banco/ft_$n.log 2>&1
  grep -E "^BASE|^EPOCA" $T/banco/ft_$n.log | python3 -c "
import sys,json
for l in sys.stdin:
    t,j=l.split(' ',1) if l.startswith('BASE') else (l.split(' ',2)[0]+' '+l.split(' ',2)[1], l.split(' ',2)[2])
    d=json.loads(j); print('   ', t, 'largo greedy %.3f' % d['largo_greedy'], 'recall16 pos8 %.3f' % d['recall_top16'][-1])"
}

# 1. solo evaluar produccion con cada restriccion (lo que cae ANTES de reajustar)
for v in "e_nada:" "e_conv8:--conv_int8" "e_v32:--vocab 32768" "e_conv8_v32:--conv_int8 --vocab 32768" "e_conv8_v64:--conv_int8 --vocab 65536"; do
  n=${v%%:*}; f=${v#*:}
  if [ -f $T/banco/ft_$n.solo_eval ]; then continue; fi
  entrenar $n --solo_evaluar $f && touch $T/banco/ft_$n.solo_eval
done
# 2. entrenar
entrenar B
entrenar C --conv --conv_int8 --vocab 32768
entrenar D --conv --conv_int8 --vocab 65536
# 3. cuantizar con las escalas de produccion
for n in B C D; do
  [ -f $MC/${PROD}_$n/model.safetensors ] && continue
  [ -f $T/borrador_ft_$n/model.safetensors ] || { log "falta el entrenado $n"; continue; }
  docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
    "python3 /repo/entrenamiento/borrador/cuantizar_rtn.py /traces/borrador_ft_$n /models/${PROD}_$n /models/$PROD --escalas=/models/$PROD && chmod -R a+rX /models/${PROD}_$n" 2>&1 | grep -v -i warn | tail -1
done
# 4. A/B en vLLM: banco de agente (40 pedidos, conc 4), dos arranques por brazo, intercalados
OV=$T/banco/ov-destilar; mkdir -p $OV
for r in 1 2; do for b in A B C D; do
  out=$T/banco/destilar_${b}_r$r.json; [ -f $out ] && continue
  case $b in A) bor=$MC/$PROD; E="GENESIS_PN157_BITS_CONV=0 GENESIS_PN159_VOCAB=65536";;
             B) bor=$MC/${PROD}_B; E="GENESIS_PN157_BITS_CONV=0 GENESIS_PN159_VOCAB=65536";;
             C) bor=$MC/${PROD}_C; E="GENESIS_PN157_BITS_CONV=8 GENESIS_PN159_VOCAB=32768";;
             D) bor=$MC/${PROD}_D; E="GENESIS_PN157_BITS_CONV=8 GENESIS_PN159_VOCAB=65536";; esac
  [ -f $bor/model.safetensors ] || { log "falta $bor"; continue; }
  printf 'services:\n  vllm-server:\n    volumes:\n      - %s:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro\n' $bor > $OV/$b.yml
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $E docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml -f ov-aislado.yml -f $OV/$b.yml up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  log "A/B $b r$r ($E) listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder /traces/banco/destilar_${b}_r$r.json 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for b in A B C D; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/destilar_${b}_r*.json'))]
print('$b', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
(cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1)
log "TODO LISTO"
