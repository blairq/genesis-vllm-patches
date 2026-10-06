#!/bin/bash
# Por tamano de chunk: prefill de 0002 (125k) solo, y 0006 (cacheado) durante ese prefill.
R=/home/usuario/Proyectos/genesis-vllm-patches; C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"; B=/traces/banco_pedidos_reales
for u in "$@"; do
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  (cd $R/compose && env VLLM_SERVER_DEV_MODE=1 GENESIS_LONG_PREFILL=$u docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "=== chunk $u"
  docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 16 $B/1005_214916_0002.json | cut -c1-90
  curl -s -m 30 -X POST "http://127.0.0.1:8391/reset_prefix_cache?reset_external=true" >/dev/null
  docker exec genesis-27b-pruebas python3 /traces/banco/reproducir_pedidos.py 16 $B/1005_214847_0001.json $B/1005_215125_0003.json >/dev/null
  docker exec genesis-27b-pruebas python3 /traces/banco/linea_tiempo.py 1005_214916_0002.json 1005_215241_0006.json 6
done
echo "LINEA LISTO"
