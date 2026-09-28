#!/bin/bash
# base (armado del 25-09, batch2 de 4 warps) | idem con batch2 de 8 warps (bit a bit) | armado actual (PN154 + PN155) con 8 warps
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="${FASES:-decode1 decode4}" ORACULO=1
IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_2509 GENESIS_ENABLE_PN155_BA_EN_QKVZ=0 GENESIS_SK18H_NW=4 ./perfil_idiotsavant.sh is27_v1b
IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86_2509 GENESIS_ENABLE_PN155_BA_EN_QKVZ=0 GENESIS_SK18H_NW=8 ./perfil_idiotsavant.sh is27_nw8
IDIOTSAVANT_MODELO=qwen3.8_27b_idiotSavant_sm_86 GENESIS_ENABLE_PN155_BA_EN_QKVZ=1 GENESIS_SK18H_NW=8 ./perfil_idiotsavant.sh is27_v2b
for l in is27_v1b is27_nw8 is27_v2b; do for f in $FASES; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
echo "TODO LISTO"
