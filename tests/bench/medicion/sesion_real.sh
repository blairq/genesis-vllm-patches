#!/bin/bash
# Reproduce la sesion real capturada (banco_pedidos_reales, sin mis reproducciones) en la instancia aislada,
# con la variante de entorno que se pase: ./sesion_real.sh <etiqueta> [VAR=valor ...]
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas; et=$1; shift
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
(cd $R/compose && env "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
echo "$(date +%T) == $et ($*): listo en $(( $(date +%s) - t0 )) s"
F=$(cd $T/banco_pedidos_reales && ls 1*.json | grep -vE "_0011|_0014" | sed 's#^#/traces/banco_pedidos_reales/#' | tr '\n' ' ')
docker exec -e PLUGIN=${PLUGIN:-0} genesis-27b-pruebas python3 /traces/banco/reproducir_sesion.py /traces/banco/sesion_$et.json 512 ${ESCALA:-1} $F 2>&1 | tail -16
docker logs genesis-27b-pruebas 2>&1 | grep -E "Engine 000" | sed -E 's/.*GPU KV cache usage: ([0-9.]+)%.*/\1/' | sort -n | tail -1 | sed 's/^/KV max %: /'
docker logs genesis-27b-pruebas 2>&1 | grep "PN165 req" | sed -E 's/.*PN165 //' | awk '!s[$1]++' > $T/banco/pn165_$et.log
echo "SESION $et LISTA"
