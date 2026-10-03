#!/bin/bash
# SK-30 GMAX 128 contra 41 (los dos con el reparto balanceado BAL): perfil de decode con oraculo, un par.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=1
GENESIS_PN131_SK30_GMAX=128 ./perfil_idiotsavant.sh is31_g128 > /dev/null 2>&1
GENESIS_PN131_SK30_GMAX=41 ./perfil_idiotsavant.sh is31_g41 > /dev/null 2>&1
for l in is31_g128 is31_g41; do for f in decode1 decode4; do python3 analizar_perfil.py trazas/${l}_$f 0 > /dev/null 2>&1; done; done
for f in decode1 decode4; do echo "== g41 vs g128, $f"; python3 analizar_perfil.py --comparar analisis_perfil/is31_g128_${f}_rank0.json analisis_perfil/is31_g41_${f}_rank0.json | sed -n 3,6p; done
python3 -c "
import json
a=json.load(open('oraculo/is31_g128.json')); b=json.load(open('oraculo/is31_g41.json'))
print('oraculo aceptados por borrador: g128', round(a['aceptados_por_borrador'],3), 'g41', round(b['aceptados_por_borrador'],3))"
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "GMAX LISTO"
