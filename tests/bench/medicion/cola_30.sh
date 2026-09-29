#!/bin/bash
# despues de las replicas de SK-30: verificacion de SK-29 (PN159) y A/B de PN161
cd /home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion
while pgrep -f "ab_sk30_rep.sh" >/dev/null; do sleep 60; done
echo "$(date +%T) ===== ab_pn159_verif"; ./ab_pn159_verif.sh
echo "$(date +%T) ===== ab_pn161"; ./ab_pn161.sh
echo "$(date +%T) COLA 30 LISTA"
echo "$(date +%T) ===== ab_pn157_g64"; ./ab_pn157_g64.sh
echo "$(date +%T) COLA 30b LISTA"
echo "$(date +%T) ===== ab_pn163"; ./ab_pn163.sh
echo "$(date +%T) COLA 30c LISTA"
