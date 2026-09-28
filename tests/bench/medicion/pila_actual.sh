#!/bin/bash
# una fase por arranque: con pilas el worker no libera la memoria del profiler entre capturas (OOM a los 10,6 GB)
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
FASES="${FASES_PILA:-decode4}" PROF_STACK=true PROF_ITER=${PROF_ITER:-25} ./perfil_idiotsavant.sh is27_pila
cd ../../../compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d --force-recreate >/dev/null 2>&1
echo "TODO LISTO"
