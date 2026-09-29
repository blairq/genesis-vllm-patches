#!/bin/bash
# PN161 (norma q/k + rope + FWHT + KV int8 del borrador en un kernel, SK-33): aceptacion (banco de agente,
# 2 arranques) y paso (perfil de decode, dos replicas en orden opuesto). Brazos: base, PN161, PN160+PN161.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; P=${PUERTO:-8361}
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
env_de() { case $1 in base) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=0 GENESIS_ENABLE_PN160_BORRADOR_FUSION=0";;
                      q) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=1 GENESIS_ENABLE_PN160_BORRADOR_FUSION=0";;
                      qf) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=1 GENESIS_ENABLE_PN160_BORRADOR_FUSION=1";; esac; }
for r in 1 2; do for b in base q qf; do
  out=$T/banco/pn161_${b}_r$r.json; [ -f $out ] && continue
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $(env_de $b) docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:$P/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == $b r$r listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "Worker_TP0.*(PN161|Traceback)|PN161" | cut -c40-220 | head -3
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('$out'));print('   acept %.3f ajeno %s errores %s' % (d['largo_aceptacion'], d['ajeno'], d['errores']))" 2>&1 | tail -1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
for b in base q qf qf q base; do
  n=$(ls -d trazas/is29_pn161_${b}_*_decode1 2>/dev/null | wc -l); l=is29_pn161_${b}_$((n + 1))
  env $(env_de $b) ./perfil_idiotsavant.sh $l > /dev/null 2>&1
  for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done
done
for n in 1 2; do for b in q qf; do for f in decode1 decode4; do echo "== $b vs base, replica $n, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is29_pn161_base_${n}_${f}_rank0.json analisis_perfil/is29_pn161_${b}_${n}_${f}_rank0.json | sed -n 3,6p; done; done; done
for b in base q qf; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/pn161_${b}_r*.json'))]
print('$b aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "PN161 LISTO"
