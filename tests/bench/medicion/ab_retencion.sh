#!/bin/bash
# Checkpoints del estado GDN: densos (None, lo que vLLM fuerza con especulacion) contra uno cada 16 bloques (14080),
# con dos conversaciones de ~100k en paralelo, 4 rondas (conv_largas.py). Instancia aislada 127.0.0.1:8391.
R=/home/usuario/Proyectos/genesis-vllm-patches; T=$R/tests/bench/medicion/trazas
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
for v in ${BRAZOS:-None 14080}; do
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && GENESIS_RETENCION_GDN=$v docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://127.0.0.1:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == retencion $v: listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "GPU KV cache size|retention|dense checkpointing" | sed -E 's/.*\] //' | cut -c1-150 | head -3
  docker exec genesis-27b-pruebas python3 /traces/banco/conv_largas.py /traces/banco/retencion_$v.json 4 256 2>&1 | tail -8
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "RETENCION LISTO"
