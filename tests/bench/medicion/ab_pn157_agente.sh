#!/bin/bash
# PN157: aceptacion con la carga real (40 pedidos de agente, conc 4): apagado / solo K/V (conv densa) / K/V + conv W8
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in ${BRAZOS:-off kv kv8 offb kv8b}; do
  case $brazo in off*) E="GENESIS_ENABLE_PN157_BORRADOR_MARLIN=0";;
                 kv8*) E="GENESIS_ENABLE_PN157_BORRADOR_MARLIN=1 GENESIS_PN157_BITS_CONV=8";;
                 kv*) E="GENESIS_ENABLE_PN157_BORRADOR_MARLIN=1 GENESIS_PN157_BITS_CONV=0";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  env $E docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; done
  echo "== $brazo ($E) listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder /traces/banco/agente157_$brazo.json 4 1536 2>&1 | tail -4
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
