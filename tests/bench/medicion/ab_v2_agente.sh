#!/bin/bash
# v1 contra v2 con la carga real: 40 pedidos de agente (SWE-rebench, tools), perfil coder, conc 4
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in ${BRAZOS:-v1 v2 v1b v2b}; do
  case $brazo in v2*) E="IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1";;
                 *) E="IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_2509 GENESIS_ENABLE_PN155_BA_EN_QKVZ=0";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  env $E docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; done
  echo "== $brazo listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder /traces/banco/agente_$brazo.json 4 1536 2>&1 | tail -4
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
