#!/bin/bash
# Aceptacion y tok/s con el lm_head A8 contra A16 (PN153 prendido en los dos), banco de prompts reales,
# greedy, 4 en paralelo; dos arranques por brazo, intercalados.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in a16 a8 a16b a8b; do
  case $brazo in a8*) A=1;; *) A=0;; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  GENESIS_ENABLE_PN153_LOGITS_RANGO=1 GENESIS_PN139_A8=$A VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; done
  echo "== $brazo (A8=$A) listo en $(( $(date +%s) - t0 )) s"
  VLLM_API_KEY=x python3 $M/prompts_reales.py http://localhost:8361 n$brazo 0 4 2>&1 | tail -12
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "TODO LISTO"
