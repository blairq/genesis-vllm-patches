#!/bin/bash
# Tamano del chunk de prefill: interferencia con el decode de otro pedido. ./chunk_prefill.sh 880 1760 2640
for u in "$@"; do echo "=== chunk $u"; ./par_concurrente.sh GENESIS_LONG_PREFILL=$u 2>&1 | grep -E "json|---"; done
echo "CHUNK LISTO"
