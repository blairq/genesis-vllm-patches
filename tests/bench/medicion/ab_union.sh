#!/bin/bash
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1 IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1
./perfil_idiotsavant.sh is27_v2u
for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/is27_v2u_$f $r > /dev/null 2>&1; done; done
echo "TODO LISTO"
