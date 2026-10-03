#!/bin/bash
# 03-10: SK-30 + PN163 + PN164 prendidos juntos (nuevo default del compose) contra la base (los tres apagados).
# Perfil de decode (2 replicas en orden opuesto, oraculo en la primera) y aceptacion (2 arranques alternados).
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; P=${PUERTO:-8391}
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
env_de() { case $1 in base) echo "GENESIS_PN131_SK30=0 GENESIS_ENABLE_PN163_GDN_SALIDA=0 GENESIS_ENABLE_PN164_GDN_VISTAS=0";;
                      nuevo) echo "GENESIS_PN131_SK30=1 GENESIS_ENABLE_PN163_GDN_SALIDA=1 GENESIS_ENABLE_PN164_GDN_VISTAS=1";; esac; }
cd $M; export FASES="decode1 decode4"
env ORACULO=1 $(env_de base) ./perfil_idiotsavant.sh is31_base > /dev/null 2>&1
env ORACULO=1 $(env_de nuevo) ./perfil_idiotsavant.sh is31_nuevo > /dev/null 2>&1
env ORACULO=0 $(env_de nuevo) ./perfil_idiotsavant.sh is31_nuevo_b > /dev/null 2>&1
env ORACULO=0 $(env_de base) ./perfil_idiotsavant.sh is31_base_b > /dev/null 2>&1
grep -h "PN163\|PN164\|SK-30\|SK30\|Traceback" analisis_perfil/is31_nuevo_decode1.log | cut -c1-200 | head -4
for l in is31_base is31_nuevo is31_nuevo_b is31_base_b; do for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== nuevo vs base$s, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is31_base${s}_${f}_rank0.json analisis_perfil/is31_nuevo${s}_${f}_rank0.json | sed -n 3,6p; done; done
python3 -c "
import json
a=json.load(open('oraculo/is31_base.json')); b=json.load(open('oraculo/is31_nuevo.json'))
print('oraculo aceptados por borrador: base', round(a['aceptados_por_borrador'],3), 'nuevo', round(b['aceptados_por_borrador'],3))"
for rb in "1 base" "1 nuevo" "2 nuevo" "2 base"; do
  set -- $rb; r=$1; b=$2
  out=$T/banco/combo0310_${b}_r$r.json; [ -f $out ] && continue
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $(env_de $b) docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:$P/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == $b r$r listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('$out'));print('   acept %.3f ajeno %s errores %s' % (d['largo_aceptacion'], d['ajeno'], d['errores']))" 2>&1 | tail -1
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for b in base nuevo; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/combo0310_${b}_r*.json'))]
print('$b aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
echo "COMBO LISTO"
