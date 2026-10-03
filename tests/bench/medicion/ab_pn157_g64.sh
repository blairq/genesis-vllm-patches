#!/bin/bash
# PN157: proyeccion de la conv del borrador en int8 por grupo de 64 (GENESIS_PN157_GRUPO_CONV=64) contra fp16.
# 21,6 -> 11,8 us por conv (10 por paso); error del kernel de la conv 0,54% (por canal era 0,95% y restaba ~0,9%).
# Aceptacion (banco de agente, 2 arranques alternados) y paso (perfil de decode, dos replicas en orden opuesto).
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; P=${PUERTO:-8391}
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
env_de() { case $1 in f16) echo "GENESIS_PN157_BITS_CONV=0";; g64) echo "GENESIS_PN157_BITS_CONV=8 GENESIS_PN157_GRUPO_CONV=64";; esac; }
for rb in "1 f16" "1 g64" "2 g64" "2 f16"; do
  set -- $rb; r=$1; b=$2
  out=$T/banco/pn157g_${b}_r$r.json; [ -f $out ] && continue
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $(env_de $b) docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:$P/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == $b r$r listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "Worker_TP0.*(PN157|Traceback)" | cut -c40-220 | head -3
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('$out'));print('   acept %.3f ajeno %s errores %s' % (d['largo_aceptacion'], d['ajeno'], d['errores']))" 2>&1 | tail -1
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
for b in f16 g64 g64 f16; do
  n=$(ls -d trazas/is29_pn157g_${b}_*_decode1 2>/dev/null | wc -l); l=is29_pn157g_${b}_$((n + 1))
  env $(env_de $b) ./perfil_idiotsavant.sh $l > /dev/null 2>&1
  for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done
done
for n in 1 2; do for f in decode1 decode4; do echo "== g64 vs f16, replica $n, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is29_pn157g_f16_${n}_${f}_rank0.json analisis_perfil/is29_pn157g_g64_${n}_${f}_rank0.json | sed -n 3,6p; done; done
for b in f16 g64; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/pn157g_${b}_r*.json'))]
print('$b aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "PN157 G64 LISTO"
