#!/bin/bash
# Turnos consecutivos de una misma conversacion, SIN concurrencia: separa desalojo de mascara de retencion.
# ./par_solo.sh <etiqueta> [VAR=valor ...]
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas; et=$1; shift
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
B=/traces/banco_pedidos_reales
docker stop genesis-27b-idiotsavant >/dev/null 2>&1; docker rm -f genesis-27b-pruebas >/dev/null 2>&1
(cd $R/compose && env "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
echo "$(date +%T) == $et ($*)"
docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_214847_0001.json $B/1005_215125_0003.json \
  $B/1005_215223_0004.json $B/1005_215235_0005.json $B/1005_215315_0007.json
docker logs genesis-27b-pruebas 2>&1 | grep "PN165 req" | sed -E 's/.*PN165 //' | awk '!s[$1]++' | cut -c1-170
echo "PAR $et LISTO"
