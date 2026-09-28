#!/bin/bash
# kl_ar_int8.sh: costo de calidad del all-reduce int8 (misma cuenta en PN120 de prefill y PN152 i8 de decode).
# El KL con prompt_logprobs solo pasa por el prefill, asi que se mide PN120 prendido contra apagado
# (M_MIN enorme = todo fp16 exacto), sobre las 48 ventanas de respuesta contra la referencia BF16.
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas/cuant
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in ${BRAZOS:-ar8 ar16 ar8b}; do
  case $brazo in ar16*) M=100000000;; *) M=512;; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  GENESIS_PN120_M_MIN=$M VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "no arranco"; exit 1; }; sleep 5; done
  echo "$(date +%T) $brazo (M_MIN=$M) listo en $(( $(date +%s) - t0 )) s"
  python3 $R/entrenamiento/cuant/fidelidad.py http://127.0.0.1:8361 $T/ventanas_48x1024.npy $T/ref_bf16.npz $T/fid_ar_$brazo.json $brazo 2>&1 | tail -3
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "TODO LISTO"
