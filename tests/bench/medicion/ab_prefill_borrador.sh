#!/bin/bash
# prefill (pp.py 28k/57k) con y sin el borrador DFlash2, en la instancia aislada: explica el -5% contra el 24-09?
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
cd $R/compose
for brazo in sin con sinb conb; do
  case $brazo in sin*) X="-f ov-sin-borrador.yml";; *) X="";; esac
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  VLLM_API_KEY= docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml -f ov-aislado.yml $X up -d --force-recreate >/dev/null 2>&1
  until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; done
  for r in 622 1266; do echo "== $brazo $r"; docker exec -i genesis-27b-pruebas python3 - p$brazo$r $r < $M/pp.py 2>&1 | tail -3; done
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
