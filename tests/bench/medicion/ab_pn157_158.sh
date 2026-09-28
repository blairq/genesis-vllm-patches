#!/bin/bash
# PN157 (K/V de contexto del borrador en Marlin, conv densa) + PN158 (embedding por P2P): perfil de decode,
# dos replicas en orden opuesto
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
export FASES="decode1 decode4" ORACULO=0
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=0 GENESIS_ENABLE_PN158_EMBED_P2P=0 ./perfil_idiotsavant.sh is28_ppoff
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=1 GENESIS_ENABLE_PN158_EMBED_P2P=1 ./perfil_idiotsavant.sh is28_ppon
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=1 GENESIS_ENABLE_PN158_EMBED_P2P=1 ./perfil_idiotsavant.sh is28_ppon_b
GENESIS_ENABLE_PN157_BORRADOR_MARLIN=0 GENESIS_ENABLE_PN158_EMBED_P2P=0 ./perfil_idiotsavant.sh is28_ppoff_b
for l in is28_ppoff is28_ppon is28_ppon_b is28_ppoff_b; do for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $r > /dev/null 2>&1; done; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== $f$s"; python3 analizar_perfil.py --comparar analisis_perfil/is28_ppoff${s}_${f}_rank0.json analisis_perfil/is28_ppon${s}_${f}_rank0.json | sed -n 3,11p; done; done
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
