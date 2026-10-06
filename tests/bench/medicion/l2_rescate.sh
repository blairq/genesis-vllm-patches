#!/bin/bash
# L2 rescata un turno? 0001 -> esperar los stores -> vaciar SOLO el prefix cache de la GPU -> 0003.
# ./l2_rescate.sh <etiqueta> [VAR=valor ...]   (instancia aislada; offload con L2 en RAM y sin disco)
R=/home/usuario/Proyectos/genesis-vllm-patches; et=$1; shift
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
B=/traces/banco_pedidos_reales; U=http://127.0.0.1:8391
docker stop genesis-27b-idiotsavant >/dev/null 2>&1; docker rm -f genesis-27b-pruebas >/dev/null 2>&1
(cd $R/compose && env GENESIS_KV_FLAG=--kv-transfer-config VLLM_SERVER_DEV_MODE=1 "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
t0=$(date +%s); until curl -sf -m 3 $U/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
echo "$(date +%T) == $et ($*)"
met() { curl -s -m 5 $U/metrics | grep -E "^vllm:(external_prefix_cache_hits_total|external_prefix_cache_queries_total|kv_offload_total_bytes_total)" | sed -E 's/engine="0",model_name="[^"]*",?//' | tr '\n' ' '; echo; }
docker exec -e GREEDY=${GREEDY:-0} genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_214847_0001.json
prev=x; for i in $(seq 1 40); do m=$(curl -s -m 5 $U/metrics | grep -E "^vllm:kv_offload_total_bytes_total" | tr -d '\n'); [ "$m" = "$prev" ] && break; prev=$m; sleep 5; done
echo "stores: $(met)"
for i in 1 2 3 4 5; do r=$(curl -s -m 30 -X POST "$U/reset_prefix_cache"); echo "reset GPU: $r"; echo "$r" | grep -q true && break; sleep 5; done
docker exec -e GREEDY=${GREEDY:-0} genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_215125_0003.json
echo "despues: $(met)"
if [ "${GREEDY:-0}" = 1 ]; then   # referencia: el mismo turno sin ningun cache (GPU y L2 vacios)
  sleep 5; for i in 1 2 3 4 5; do r=$(curl -s -m 30 -X POST "$U/reset_prefix_cache?reset_external=true"); echo "reset GPU+L2: $r"; echo "$r" | grep -q true && break; sleep 5; done
  docker exec -e GREEDY=1 genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_215125_0003.json
fi
docker logs genesis-27b-pruebas 2>&1 | grep -E "PN167|PN165 req" | sed -E 's/.*(PN16[57])/\1/' | awk '!s[$0]++' | cut -c1-220 | tail -14
echo "L2 $et LISTO"
