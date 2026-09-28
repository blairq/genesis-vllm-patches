#!/bin/bash
# replica en orden invertido (prendido primero) para separar deriva de reloj del efecto
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=0
GENESIS_PN124_Q3D=16 ./perfil_idiotsavant.sh is28_q3don_b
GENESIS_PN124_Q3D=1 ./perfil_idiotsavant.sh is28_q3doff_b
for l in is28_q3doff_b is28_q3don_b; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for f in decode1 decode4; do echo "== $f"; python3 analizar_perfil.py --comparar analisis_perfil/is28_q3doff_b_${f}_rank0.json analisis_perfil/is28_q3don_b_${f}_rank0.json | sed -n 3,11p; done
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
