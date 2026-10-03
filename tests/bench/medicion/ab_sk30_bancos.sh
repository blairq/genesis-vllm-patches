#!/bin/bash
# SK-30 (atencion de decode en una pasada): espera al A/B de la fase 2 de PN159 y mide apagado vs prendido:
# perfil de decode (1 y 4 pedidos, dos replicas en orden opuesto) y aceptacion (banco de agente, 2 arranques).
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas
until grep -q "COLA LISTA" $T/banco/cola_29.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
for r in 1 2; do for s in 0 1; do
  out=$T/banco/sk30_${s}_r$r.json; [ -f $out ] && continue
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && GENESIS_PN131_SK30=$s docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == SK30=$s r$r listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "Worker_TP0.*(SK-30|Traceback)" | cut -c40-200 | head -2
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for s in 0 1; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/sk30_${s}_r*.json'))]
print('SK30=$s aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
