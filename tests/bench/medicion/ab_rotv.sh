#!/bin/bash
# ROTV (V rotada en d): espera al A/B de SK-31. Brazos con SK-30 prendido, ROTV 0 / 1:
#   KL contra BF16 (fidelidad.py: el prefill lee V por decuant, que des-rota) y aceptacion (banco de agente).
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; TC=$T/cuant
until grep -q "TODO LISTO" $T/banco/sk31.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
for r in 1 2; do for v in 0 1; do
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'      # el formato de la KV cambia
  (cd $R/compose && GENESIS_PN131_SK30=1 GENESIS_PN131_ROTV=$v VLLM_API_KEY= docker compose $C up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) == ROTV=$v r$r listo en $(( $(date +%s) - t0 )) s"
  [ -f $TC/fid_rotv_${v}_r$r.json ] || python3 $R/entrenamiento/cuant/fidelidad.py http://127.0.0.1:8361 $TC/ventanas_48x1024.npy $TC/ref_bf16.npz $TC/fid_rotv_${v}_r$r.json rotv$v 2>&1 | tail -2
  [ -f $T/banco/rotv_${v}_r$r.json ] || docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder $T/banco/rotv_${v}_r$r.json 4 1536 > /dev/null 2>&1
done; done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for v in 0 1; do python3 -c "
import json,glob
a=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/rotv_${v}_r*.json'))]
print('ROTV=$v aceptacion', ' '.join('%.3f'%x for x in a), 'media %.3f'%(sum(a)/max(len(a),1)))"; done
cd $R/compose && docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d >/dev/null 2>&1
echo "TODO LISTO"
