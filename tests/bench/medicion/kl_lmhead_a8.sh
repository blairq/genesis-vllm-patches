#!/bin/bash
# kl_lmhead_a8.sh: KL en respuestas (48 ventanas contra BF16) con el lm_head W4A8 (PN139_A8=1).
# Base: el stack de siempre da 0,0193 (kl_ar_int8.sh, brazo ar8).
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas/cuant
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
GENESIS_PN139_A8=1 VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "no arranco"; docker logs --tail 30 genesis-27b-pruebas; exit 1; }; sleep 5; done
echo "$(date +%T) lmhead_a8 listo en $(( $(date +%s) - t0 )) s"
docker logs genesis-27b-pruebas 2>&1 | grep "PN139: lm_head" | head -2
python3 $R/entrenamiento/cuant/fidelidad.py http://127.0.0.1:8361 $T/ventanas_48x1024.npy $T/ref_bf16.npz $T/fid_lmhead_a8.json lmhead_a8 2>&1 | tail -2
echo "TODO LISTO"
