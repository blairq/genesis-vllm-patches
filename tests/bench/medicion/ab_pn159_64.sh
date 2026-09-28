#!/bin/bash
# PN159 con 64k: perfil de decode, dos replicas en orden opuesto
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=0 GENESIS_PN159_VOCAB=65536
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=0 ./perfil_idiotsavant.sh is28_64off
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=1 ./perfil_idiotsavant.sh is28_64on
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=1 ./perfil_idiotsavant.sh is28_64on_b
GENESIS_ENABLE_PN159_VOCAB_BORRADOR=0 ./perfil_idiotsavant.sh is28_64off_b
for l in is28_64off is28_64on is28_64on_b is28_64off_b; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== $f$s"; python3 analizar_perfil.py --comparar analisis_perfil/is28_64off${s}_${f}_rank0.json analisis_perfil/is28_64on${s}_${f}_rank0.json | sed -n 3,11p; done; done
echo "TODO LISTO"
