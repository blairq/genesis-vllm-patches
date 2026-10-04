#!/bin/bash
# SK-30 actual (mascara sin saltos + escala de K una vez, bit a bit) contra el de 81fb9a6 (GMAX 41 + BAL):
# oraculo greedy (tiene que dar el MISMO texto) y perfil de decode, dos replicas en orden opuesto.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; K=$R/vllm/_genesis/kernels/cuda/sk30_decode_1pasada.cu
cp $K /tmp/sk30_actual_0410.cu
restaurar() { cp /tmp/sk30_actual_0410.cu $K; }
trap restaurar EXIT
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
cd $M; export FASES="decode1 decode4"
viejo() { git -C $R show 81fb9a6:vllm/_genesis/kernels/cuda/sk30_decode_1pasada.cu > $K; }
ORACULO=1 ./perfil_idiotsavant.sh is32_sk30_nuevo > /dev/null 2>&1
viejo; ORACULO=1 ./perfil_idiotsavant.sh is32_sk30_viejo > /dev/null 2>&1
ORACULO=0 ./perfil_idiotsavant.sh is32_sk30_viejo_b > /dev/null 2>&1
restaurar; ORACULO=0 ./perfil_idiotsavant.sh is32_sk30_nuevo_b > /dev/null 2>&1
cmp -s $K /tmp/sk30_actual_0410.cu && echo "kernel restaurado"
for l in is32_sk30_nuevo is32_sk30_viejo is32_sk30_viejo_b is32_sk30_nuevo_b; do for f in decode1 decode4; do python3 analizar_perfil.py trazas/${l}_$f 0 > /dev/null 2>&1; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== nuevo vs viejo$s, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is32_sk30_viejo${s}_${f}_rank0.json analisis_perfil/is32_sk30_nuevo${s}_${f}_rank0.json | sed -n 3,6p; done; done
python3 -c "
import json
a=json.load(open('oraculo/is32_sk30_viejo.json')); b=json.load(open('oraculo/is32_sk30_nuevo.json'))
print('oraculo: textos iguales', a['textos']==b['textos'], '| aceptados por borrador', round(a['aceptados_por_borrador'],3), round(b['aceptados_por_borrador'],3))"
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "SK30 0410 LISTO"
