#!/bin/bash
# DFlash2-DCOCO: la cabeza DCOCO (PN178) sobre el borrador DFlash2 de produccion, con las capturas de
# dflash2.sh. Mismo estilo que idiotsavant/destilar_conv8_vocab.sh: cada paso se saltea si ya esta hecho.
#
#   0  produccion en fp16 (dflash2/decuantizar.py): exactamente los pesos servidos
#   1  solo evaluar: largo aceptado unario / selector / arbol del selector en los pedidos apartados
#   2  entrenar la cabeza (MODO=cabeza, borrador congelado) o cabeza + LoRA (MODO=conjunto)
#   3  cuantizar con las escalas de produccion (dflash2/cuantizar_rtn.py): lo no entrenado vuelve exacto
#      a sus int4 y la cabeza queda en bf16
#   4  A/B en vLLM con PN178 prendido en los dos brazos (en el de produccion no hace nada: no trae cabeza)
#
# Uso:  setsid nohup ./dflash2_dcoco.sh > dflash2_dcoco.log 2>&1 &
#       MODO=conjunto NOMBRE=conjunto ./dflash2_dcoco.sh
#       PASOS="1" ./dflash2_dcoco.sh                  # solo la evaluacion de base
# Variables: NOMBRE [cabeza], MODO [cabeza|conjunto], EPOCAS [2], CAPTURAS [entrena_rot],
#            FLAGS [--vocab 65536], ENV_BORRADOR [GENESIS_PN157_BITS_CONV=0 GENESIS_PN159_VOCAB=65536],
#            PASOS [0 1 2 3 4], EXTRA [] (flags opcionales del paso 2: --temperaturas, --calibrar, --estado
#            recurrente, --criterio; con --estado recurrente no se corren 3 y 4). FLAGS y ENV_BORRADOR tienen que describir lo MISMO: como se sirve el borrador
#            (los defaults son los de la compose idiotsavant-sm86). Con la conv en int8 al servir
#            (GENESIS_PN157_BITS_CONV=8), entrenar con --conv_int8.
set -u
AQUI="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$AQUI/../.." && pwd)"
T=$R/tests/bench/medicion/trazas
MC=${MC:-/home/usuario/Proyectos/models-cache}
IMG=${IMG:-vllm/vllm-openai:v0.29.0}
PROD=${PROD:-qwen3.8_27b_idiotSavant_sm_86_dflash2}
MOD=/models/qwen3.8_27b_idiotSavant_sm_86
NOMBRE=${NOMBRE:-cabeza}
MODO=${MODO:-cabeza}
EPOCAS=${EPOCAS:-2}
CAPTURAS=${CAPTURAS:-entrena_rot}
FLAGS=${FLAGS:---vocab 65536}
ENV_BORRADOR=${ENV_BORRADOR:-GENESIS_PN157_BITS_CONV=0 GENESIS_PN159_VOCAB=65536}
PASOS=${PASOS:-0 1 2 3 4}
EXTRA=${EXTRA:-}
SALIDA=${PROD}_dcoco_$NOMBRE
log() { echo "$(date +%T) [dcoco] $*"; }
quiere() { [[ " $PASOS " == *" $1 "* ]]; }
ram_libre_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
mkdir -p $T/banco
[ -d $T/captura/$CAPTURAS ] || { log "faltan las capturas $T/captura/$CAPTURAS (paso 3 de idiotsavant/dflash2.sh)"; exit 1; }

entrenar() {  # salida flags...
  local sal=$1; shift
  while [ "$(ram_libre_gb)" -lt 17 ]; do sleep 30; done
  docker run --rm --name dcoco-$NOMBRE --gpus '"device=0"' --ipc host --memory 16g --memory-swap 16g \
    -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG \
    /repo/entrenamiento/dflash2-dcoco/entrenar_dcoco.py --datos /traces/captura/$CAPTURAS --salida $sal \
    --borrador /traces/borrador_prod_fp16 --noon $MOD $FLAGS "$@"
}

if quiere 0 && [ ! -f $T/borrador_prod_fp16/model.safetensors ]; then
  log "0. produccion -> fp16"
  docker stop genesis-27b-idiotsavant >/dev/null 2>&1
  docker run --rm -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint bash $IMG -c \
    "python3 /repo/entrenamiento/dflash2/decuantizar.py /models/$PROD /traces/borrador_prod_fp16 && chmod -R a+rwX /traces/borrador_prod_fp16"
fi

if quiere 1 && [ ! -f $T/banco/dcoco_base.json ]; then
  log "1. evaluacion de base"
  entrenar /traces/dcoco_base --solo_evaluar > $T/banco/dcoco_base.log 2>&1 && \
    cp $T/dcoco_base/registro.json $T/banco/dcoco_base.json
  grep ^BASE $T/banco/dcoco_base.log | cut -c1-200
fi

if quiere 2 && [ ! -f $T/borrador_dcoco_$NOMBRE/model.safetensors ]; then
  log "2. entrenando ($MODO, $EPOCAS epocas)"
  extra=""; [ $MODO = conjunto ] && extra="--auf"
  entrenar /traces/borrador_dcoco_$NOMBRE --modo $MODO --epocas $EPOCAS $extra $EXTRA > $T/banco/dcoco_$NOMBRE.log 2>&1
  grep -E "^BASE|^EPOCA|^CALIBRADO" $T/banco/dcoco_$NOMBRE.log | python3 -c "
import sys, json
for l in sys.stdin:
    t, j = (' '.join(l.split(' ', 2)[:2]), l.split(' ', 2)[2]) if l.startswith('EPOCA') else l.split(' ', 1)
    d = json.loads(j)
    m = ' '.join('%s %.3f' % (k[len('largo_muestreo_'):-len('_arbol_dcoco')], v) for k, v in d.items()
                 if k.startswith('largo_muestreo_') and k.endswith('_arbol_dcoco'))
    print('   %-9s cadena: unario %.3f selector %.3f dcoco %.3f | arbol: selector %.3f dcoco %.3f | ce fila: %.3f -> %.3f %s' % (
        t, d['largo_unario'], d['largo_selector'], d['largo_dcoco'], d['largo_arbol_selector'], d['largo_arbol_dcoco'],
        d['ce_fila_selector'], d['ce_fila_dcoco'], ('| arbol dcoco muestreo ' + m) if m else ''))"
  grep -E "^MEJOR|^CALIBRACION" $T/banco/dcoco_$NOMBRE.log
  [ -f $T/borrador_dcoco_$NOMBRE/model.safetensors ] || { log "fallo el paso 2"; exit 1; }
fi

if [[ "$EXTRA" == *recurrente* ]]; then
  log "cabeza recurrente: PN178 todavia no la sirve, no se cuantiza ni se hace el A/B"
  PASOS="${PASOS//3/}"; PASOS="${PASOS//4/}"
fi

if quiere 3 && [ ! -f $MC/$SALIDA/model.safetensors ]; then
  log "3. cuantizando -> models-cache/$SALIDA"
  docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
    "python3 /repo/entrenamiento/dflash2/cuantizar_rtn.py /traces/borrador_dcoco_$NOMBRE /models/$SALIDA /models/$PROD \
     --escalas=/models/$PROD && chmod -R a+rX /models/$SALIDA" 2>&1 | grep -v -i warn | tail -1
  [ -f $MC/$SALIDA/model.safetensors ] || { log "fallo el paso 3"; exit 1; }
fi

if quiere 4; then
  OV=$T/banco/ov-dcoco; mkdir -p $OV
  for r in 1 2; do for b in prod dcoco; do
    out=$T/banco/dcoco_${NOMBRE}_${b}_r$r.json; [ -f $out ] && continue
    [ $b = prod ] && bor=$MC/$PROD || bor=$MC/$SALIDA
    printf 'services:\n  vllm-server:\n    volumes:\n      - %s:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro\n' $bor > $OV/$b.yml
    docker rm -f genesis-27b-pruebas >/dev/null 2>&1
    (cd $R/compose && env $ENV_BORRADOR GENESIS_ENABLE_PN178_DCOCO=1 docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml \
       -f ov-aislado.yml -f $OV/$b.yml up -d --force-recreate >/dev/null 2>&1)
    t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
    log "4. A/B $b r$r listo en $(( $(date +%s) - t0 )) s"
    docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder $out 4 1536 > /dev/null 2>&1
  done; done
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  for b in prod dcoco; do python3 -c "
import json, glob
v = [json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/dcoco_${NOMBRE}_${b}_r*.json'))]
print('   $b', ' '.join('%.3f' % x for x in v), 'media %.3f' % (sum(v) / max(len(v), 1)))"; done
fi
log "LISTO. Subir: $AQUI/subir_hf.sh $MC/$SALIDA"
