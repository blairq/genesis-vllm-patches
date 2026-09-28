#!/bin/bash
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
FASES="decode1 decode4" ORACULO=1 ./perfil_idiotsavant.sh is27_150fix
grep -h "PN150\] grafo" analisis_perfil/is27_150fix_decode*.log | sort -u > analisis_perfil/is27_150fix_capturas.log
for f in decode1 decode4; do for r in 0 1; do python3 analizar_perfil.py trazas/is27_150fix_$f $r > /dev/null 2>&1; done; done
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d --force-recreate >/dev/null 2>&1
echo "TODO LISTO"
