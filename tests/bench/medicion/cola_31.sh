#!/bin/bash
# retoma cola_30 tras la pausa del 29-09: resto de PN161/162, conv int8 g64, PN163/164
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
echo "$(date +%T) ===== ab_pn161_resto"; ./ab_pn161_resto.sh
echo "$(date +%T) ===== ab_pn157_g64"; ./ab_pn157_g64.sh
echo "$(date +%T) ===== ab_pn163"; ./ab_pn163.sh
echo "$(date +%T) COLA 31 LISTA"
