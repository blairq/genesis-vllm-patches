#!/bin/bash
R=/home/usuario/Proyectos/genesis-vllm-patches; C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"; B=/traces/banco_pedidos_reales
docker stop genesis-27b-idiotsavant >/dev/null 2>&1; docker rm -f genesis-27b-pruebas >/dev/null 2>&1
(cd $R/compose && env GENESIS_PN168_DIAG=1 "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 16 $B/1005_214847_0001.json $B/1005_215125_0003.json $B/1005_215223_0004.json | cut -c1-90
echo "--- 0006 solo:"; docker exec genesis-27b-pruebas python3 /traces/banco/par_concurrente.py 1005_215241_0006.json:0
docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 16 $B/1005_215125_0003.json | cut -c1-90
echo "--- 0005 y 0006 a los 6 s:"; docker exec genesis-27b-pruebas python3 /traces/banco/par_concurrente.py 1005_215235_0005.json:0 1005_215241_0006.json:6
docker logs genesis-27b-pruebas 2>&1 | grep -E "PN165 req|PN168 diag.* g=6 " | grep -E "prompt=101250|prompt=45261" | sed -E 's/.*([0-9]{2}:[0-9]{2}:[0-9]{2}) [A-Z]+ +\[pid [0-9]+\] [a-z0-9_.]+: (PN16[58]) (req=[^ ]+|diag req=[^ ]+ g=[0-9]+ bloques [0-9.]+).*prompt=([0-9]+).*/\1 \2 \3 \4/' | tail -40
echo "PAR CONCURRENTE LISTO"
