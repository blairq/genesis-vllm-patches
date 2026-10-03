#!/bin/bash
# Cola del 28-09 (noche), rearmada: la ruta de salida de correr_banco era la del HOST (fallaban los bancos).
C=/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion/trazas/banco/cola29
for s in ab_pn159_fase2 ab_sk31 ab_rotv ab_destilar_E ab_pn160; do
  echo "$(date +%T) ===== $s"
  bash $C/$s.sh 2>&1 | tail -14
done
echo "COLA LISTA"
