#!/bin/bash
# PN124 Q3D: atencion del borrador (9 queries, ventana 2048) por el kernel 3D recortado a la ventana, apagado vs prendido
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1
GENESIS_PN124_Q3D=1 ./perfil_idiotsavant.sh is28_q3doff
GENESIS_PN124_Q3D=16 ./perfil_idiotsavant.sh is28_q3don
for l in is28_q3doff is28_q3don; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for f in decode1 decode4; do echo "== $f"; python3 analizar_perfil.py --comparar analisis_perfil/is28_q3doff_${f}_rank0.json analisis_perfil/is28_q3don_${f}_rank0.json 2>&1 | head -40; done
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
