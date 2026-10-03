#!/bin/bash
# Resto del A/B de PN161/162 tras la pausa del 29-09: faltan las replicas 2 de los perfiles (orden opuesto al de
# la 1: qfn qf q base). La aceptacion (8 arranques) ya esta.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
env_de() { case $1 in base) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=0 GENESIS_ENABLE_PN160_BORRADOR_FUSION=0";;
                      q) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=1 GENESIS_ENABLE_PN160_BORRADOR_FUSION=0";;
                      qf) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=1 GENESIS_ENABLE_PN160_BORRADOR_FUSION=1";;
                      qfn) echo "GENESIS_ENABLE_PN161_BORRADOR_QKKV=1 GENESIS_ENABLE_PN160_BORRADOR_FUSION=1 GENESIS_ENABLE_PN162_BORRADOR_NORMA=1";; esac; }
for b in qfn qf q base; do
  l=is29_pn161_${b}_2
  [ -f analisis_perfil/${l}_decode4_rank0.json ] && continue
  env $(env_de $b) ./perfil_idiotsavant.sh $l > /dev/null 2>&1
  for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done
  echo "$(date +%T) perfil $l"
  grep -h "PN16[012]\|Traceback" analisis_perfil/${l}_decode1.log | head -2
done
for n in 1 2; do for b in q qf qfn; do for f in decode1 decode4; do echo "== $b vs base, replica $n, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is29_pn161_base_${n}_${f}_rank0.json analisis_perfil/is29_pn161_${b}_${n}_${f}_rank0.json | sed -n 3,6p; done; done; done
for b in base q qf qfn; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/pn161_${b}_r*.json'))]
print('$b aceptacion', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "PN161 LISTO"
