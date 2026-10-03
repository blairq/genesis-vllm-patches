#!/bin/bash
# SK-31 (atencion del BORRADOR en una pasada): espera al A/B de la fase 2 de PN159 y mide apagado vs prendido:
# perfil de decode (1 y 4 pedidos, dos replicas en orden opuesto) y aceptacion (banco de agente, 2 arranques).
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas
until grep -q "TODO LISTO" $T/banco/cola_28.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
for r in 1 2; do for s in 0 1; do
  out=$T/banco/sk31_${s}_r$r.json; [ -f $out ] && continue
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && GENESIS_PN124_SK31=$s docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == SK31=$s r$r listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "Worker_TP0.*(SK-31|Traceback)" | cut -c40-200 | head -2
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
GENESIS_PN124_SK31=0 ./perfil_idiotsavant.sh is28_sk31off > /dev/null 2>&1
GENESIS_PN124_SK31=1 ./perfil_idiotsavant.sh is28_sk31on > /dev/null 2>&1
GENESIS_PN124_SK31=1 ./perfil_idiotsavant.sh is28_sk31on_b > /dev/null 2>&1
GENESIS_PN124_SK31=0 ./perfil_idiotsavant.sh is28_sk31off_b > /dev/null 2>&1
for l in is28_sk31off is28_sk31on is28_sk31on_b is28_sk31off_b; do for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== $f$s"; python3 analizar_perfil.py --comparar analisis_perfil/is28_sk31off${s}_${f}_rank0.json analisis_perfil/is28_sk31on${s}_${f}_rank0.json | sed -n 3,6p; done; done
for s in 0 1; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/sk31_${s}_r*.json'))]
print('SK31=$s aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
