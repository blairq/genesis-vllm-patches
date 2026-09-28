#!/bin/bash
# aceptacion y tok/s punta a punta: v1 contra v2 (con batch2 de 8 warps los dos), prompts reales greedy conc 1
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in ${BRAZOS:-v1 v2 v1b v2b}; do
  case $brazo in v2*) E="IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_v2 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1";; *) E="";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  env $E VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; done
  echo "== $brazo listo en $(( $(date +%s) - t0 )) s"
  VLLM_API_KEY=x python3 $M/prompts_reales.py http://localhost:8361 n$brazo 0 1 2>&1 | tail -10
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "TODO LISTO"
