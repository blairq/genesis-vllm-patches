#!/bin/bash
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1 IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_v2 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1
GENESIS_PN131_CPG=64 ./perfil_idiotsavant.sh is27_v2s64
GENESIS_PN131_CPG=16 ./perfil_idiotsavant.sh is27_v2s16
for l in is27_v2s64 is27_v2s16; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
echo "TODO LISTO"
