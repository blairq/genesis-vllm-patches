#!/bin/bash
# PN156 (norma que escribe int8, SK-26): perfil de decode (1 y 4 pedidos, oraculo) y prefill (pp.py), apagado vs prendido
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1
GENESIS_ENABLE_PN156_NORMA_Q8=0 ./perfil_idiotsavant.sh is28_156off
GENESIS_ENABLE_PN156_NORMA_Q8=1 ./perfil_idiotsavant.sh is28_156on
for l in is28_156off is28_156on; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
R=/home/usuario/Proyectos/genesis-vllm-patches; C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in off on offb onb; do
  case $brazo in on*) P=1;; *) P=0;; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  GENESIS_ENABLE_PN156_NORMA_Q8=$P VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
  until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; done
  for r in 622 1266; do echo "== pp $brazo $r"; docker exec -i genesis-27b-pruebas python3 - p$brazo$r $r < $R/tests/bench/medicion/pp.py 2>&1 | tail -2; done
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
