#!/bin/bash
# prefill frio (pp.py, 28k y 57k): checkpoint del 25-09 contra el actual, mismo stack; 2 vueltas intercaladas
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
cd $R/compose
for brazo in 2509 2809 2509b 2809b; do
  case $brazo in 2509*) E="IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_2509 GENESIS_ENABLE_PN155_BA_EN_QKVZ=0";; *) E="";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  env $E VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1
  until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; done
  for r in 622 1266; do echo "== $brazo $r"; docker exec -i genesis-27b-pruebas python3 - p$brazo$r $r < $M/pp.py 2>&1 | tail -3; done
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
