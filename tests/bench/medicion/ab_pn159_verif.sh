#!/bin/bash
# PN159: F1s (SK-29 sin anillo) acepto ~3% menos que F1 en fase 2, pero SK-29 da exacto offline y F2a/F2b (mismo
# kernel, con anillo) aceptan igual que F1. Dos replicas mas de cada uno, en orden alternado, con
# GENESIS_PN159_VERIFICAR=1 en F1s: cuenta en el servidor las filas en que SK-29 y flashinfer difieren.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; P=${PUERTO:-8391}
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
env_de() { case $1 in F1) echo "GENESIS_PN159_VOCAB=65536 GENESIS_PN159_SK28=0 GENESIS_PN159_DINAMICO=0";;
                      F1s) echo "GENESIS_PN159_VOCAB=65536 GENESIS_PN159_SK28=1 GENESIS_PN159_DINAMICO=0 GENESIS_PN159_VERIFICAR=1";; esac; }
for rb in "3 F1s" "3 F1" "4 F1" "4 F1s"; do
  set -- $rb; r=$1; b=$2
  out=$T/banco/fase2_${b}_r$r.json; [ -f $out ] && continue
  E=$(env_de $b)
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $E docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:$P/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == $b r$r listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
  sleep 35
  docker logs genesis-27b-pruebas 2>&1 | grep "PN159 verificar" | tail -2 | cut -c1-200
  python3 -c "import json;d=json.load(open('$out'));print('   acept %.3f ajeno %s' % (d['largo_aceptacion'], d['ajeno']))"
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "$(date +%T) PN159 VERIF LISTO"
