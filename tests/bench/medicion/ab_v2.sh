#!/bin/bash
# base (v1, batch2 de 4 warps) | v1 con batch2 de 8 warps (bit a bit) | v2 (PN154 fusionado + PN155) con 8 warps
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="${FASES:-decode1 decode4}" ORACULO=1
GENESIS_SK18H_NW=4 ./perfil_idiotsavant.sh is27_v1b
GENESIS_SK18H_NW=8 ./perfil_idiotsavant.sh is27_nw8
IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_v2 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1 GENESIS_SK18H_NW=8 ./perfil_idiotsavant.sh is27_v2b
for l in is27_v1b is27_nw8 is27_v2b; do for f in $FASES; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
echo "TODO LISTO"
