#!/bin/bash
# PN159 fase 2: espera a que termine destilar_conv8_vocab.sh y mide con el borrador de produccion:
#   F1   64k fijo, top-k de flashinfer (lo de hoy)
#   F1s  64k fijo, SK-29
#   F2a  32k fijo + anillo de 512 (SK-27/28/29)
#   F2b  16k fijo + anillo de 512
# Aceptacion: banco de agente, 2 arranques por brazo. Paso: perfil de decode, dos replicas en orden opuesto.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas
until grep -q "TODO LISTO" $T/banco/destilar.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) arranca (la destilacion termino)"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
env_de() { case $1 in F1) echo "GENESIS_PN159_VOCAB=65536 GENESIS_PN159_SK28=0 GENESIS_PN159_DINAMICO=0";;
                      F1s) echo "GENESIS_PN159_VOCAB=65536 GENESIS_PN159_SK28=1 GENESIS_PN159_DINAMICO=0";;
                      F2a) echo "GENESIS_PN159_VOCAB=32768 GENESIS_PN159_SK28=1 GENESIS_PN159_DINAMICO=512";;
                      F2b) echo "GENESIS_PN159_VOCAB=16384 GENESIS_PN159_SK28=1 GENESIS_PN159_DINAMICO=512";; esac; }
for r in 1 2; do for b in F1 F1s F2a F2b; do
  out=$T/banco/fase2_${b}_r$r.json; [ -f $out ] && continue
  E=$(env_de $b)
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && env $E docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == $b r$r ($E) listo en $(( $(date +%s) - t0 )) s"
  docker logs genesis-27b-pruebas 2>&1 | grep -E "Worker_TP0.*(PN159|Traceback)" | cut -c40-200 | head -3
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder ${out/$T\/banco\//\/traces\/banco\/} 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4" ORACULO=0
for b in F1 F1s F2a F2b F2b F2a F1s F1; do
  n=$(ls -d trazas/is28_f2_${b}_* 2>/dev/null | grep -c decode1); l=is28_f2_${b}_$((n + 1))
  env $(env_de $b) ./perfil_idiotsavant.sh $l > /dev/null 2>&1
  for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done
done
python3 - <<'PY'
import json, glob
T = "/home/usuario/Proyectos/genesis-vllm-patches/tests/bench/medicion"
for b in ("F1", "F1s", "F2a", "F2b"):
    ac = [json.load(open(f))["largo_aceptacion"] for f in sorted(glob.glob(f"{T}/trazas/banco/fase2_{b}_r*.json"))]
    pasos = {}
    for f in ("decode1", "decode4"):
        v = [json.load(open(x))["paso_pared_us"] for x in sorted(glob.glob(f"{T}/analisis_perfil/is28_f2_{b}_*_{f}_rank0.json"))]
        pasos[f] = sum(v) / max(len(v), 1)
    print(b, "aceptacion", " ".join("%.3f" % x for x in ac), "media %.3f" % (sum(ac) / max(len(ac), 1)),
          "| paso 1 pedido %.0f us, 4 pedidos %.0f us" % (pasos["decode1"], pasos["decode4"]))
PY
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
