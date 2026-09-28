#!/bin/bash
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=1
GENESIS_ENABLE_PN153_LOGITS_RANGO=0 GENESIS_PN139_A8=0 ./perfil_idiotsavant.sh is27_153b_off
GENESIS_ENABLE_PN153_LOGITS_RANGO=1 GENESIS_PN139_A8=0 ./perfil_idiotsavant.sh is27_153b_rango
GENESIS_ENABLE_PN153_LOGITS_RANGO=1 GENESIS_PN139_A8=1 ./perfil_idiotsavant.sh is27_153b_a8
for l in is27_153b_off is27_153b_rango is27_153b_a8; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
echo "TODO LISTO"
