#!/bin/bash
# El offload a disco rescata un turno? 0001 -> esperar que baje a disco -> reiniciar (GPU vacia, disco intacto) -> 0003.
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
B=/traces/banco_pedidos_reales
arrancar() { docker rm -f genesis-27b-pruebas >/dev/null 2>&1; (cd $R/compose && env "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done; }
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
arrancar "$@"; echo "$(date +%T) arriba ($*)"
docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_214847_0001.json
prev=-1; for i in $(seq 1 40); do s=$(du -sb /home/usuario/Proyectos/kv-offload | cut -f1); [ "$s" = "$prev" ] && break; prev=$s; sleep 15; done
echo "$(date +%T) disco: $(du -sh /home/usuario/Proyectos/kv-offload | cut -f1)"
arrancar "$@"; echo "$(date +%T) reiniciado (GPU vacia)"
docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 64 $B/1005_215125_0003.json
docker logs genesis-27b-pruebas 2>&1 | grep -E "PN167|PN165 req" | sed -E 's/.*(PN16[57])/\1/' | awk '!s[$0]++' | head -20
curl -s -m 5 http://127.0.0.1:8391/metrics | grep -E "^vllm:(external_prefix_cache_hits_total|kv_offload_total_bytes_total\{.*CPU_to_GPU)" | sed -E 's/engine="0",model_name="[^"]*",?//'
echo "OFFLOAD LISTO"
