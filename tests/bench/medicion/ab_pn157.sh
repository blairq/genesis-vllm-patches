#!/bin/bash
# PN157: K/V de contexto y kernel_projection del borrador en Marlin, apagado vs prendido
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=0 ./perfil_idiotsavant.sh is28_157off
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=1 ./perfil_idiotsavant.sh is28_157on
for l in is28_157off is28_157on; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for f in decode1 decode4; do echo "== $f"; python3 analizar_perfil.py --comparar analisis_perfil/is28_157off_${f}_rank0.json analisis_perfil/is28_157on_${f}_rank0.json 2>&1 | head -40; done
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
