#!/bin/bash
# PN159: aceptacion (banco de agente, 2 arranques por brazo: apagado / 32k / 64k) y perfil de decode (32k vs apagado)
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in ${BRAZOS:-off v32 v64 offb v32b v64b}; do
  case $brazo in off*) E="GENESIS_ENABLE_PN159_VOCAB_BORRADOR=0";;
                 v32*) E="GENESIS_ENABLE_PN159_VOCAB_BORRADOR=1 GENESIS_PN159_VOCAB=32768";;
                 v64*) E="GENESIS_ENABLE_PN159_VOCAB_BORRADOR=1 GENESIS_PN159_VOCAB=65536";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  env $E docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "== $brazo ($E) listo en $(( $(date +%s) - t0 )) s"; docker logs genesis-27b-pruebas 2>&1 | grep -E "PN159: (cand|no se)|Traceback" | head -2
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder /traces/banco/agente159_$brazo.json 4 1536 2>&1 | tail -1
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=0 ./perfil_idiotsavant.sh is28_159off
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=1 ./perfil_idiotsavant.sh is28_159on
for l in is28_159off is28_159on; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for f in decode1 decode4; do echo "== $f"; python3 analizar_perfil.py --comparar analisis_perfil/is28_159off_${f}_rank0.json analisis_perfil/is28_159on_${f}_rank0.json | sed -n 3,11p; done
cd $M/trazas/banco; for b in off v32 v64 offb v32b v64b; do python3 -c "
import json;d=json.load(open('agente159_$b.json'));print('$b', round(d['largo_aceptacion'],3), d['pasos'], d['aceptados'], d['errores'])"; done
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
