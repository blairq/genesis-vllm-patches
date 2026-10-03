#!/bin/bash
# Cola del 28-09: espera la destilacion (C en GPU0) y D (en GPU1, a mano), y despues:
#   1. D: cuantizar con las escalas de produccion + A/B (2 arranques, servido conv8 + 64k)
#   2. SK-30 (ab_sk30.sh sin la espera)   3. PN159 fase 2 (ab_pn159_fase2.sh sin la espera)
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion; T=$M/trazas; MC=/home/usuario/Proyectos/models-cache
PROD=qwen3.8_27b_idiotSavant_sm_86_dflash2; IMG=vllm/vllm-openai:v0.29.0
until grep -q "TODO LISTO" $T/banco/destilar.log 2>/dev/null; do sleep 60; done
while docker ps --format '{{.Names}}' | grep -q '^ft-D$'; do sleep 60; done
echo "$(date +%T) destilacion y D terminados"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
if [ -f $T/borrador_ft_D/model.safetensors ] && [ ! -f $MC/${PROD}_D/model.safetensors ]; then
  docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
    "python3 /repo/entrenamiento/borrador/cuantizar_rtn.py /traces/borrador_ft_D /models/${PROD}_D /models/$PROD --escalas=/models/$PROD && chmod -R a+rX /models/${PROD}_D" 2>&1 | grep -v -i warn | tail -1
fi
grep -E "^BASE|^EPOCA" $T/banco/ft_D.log | cut -c1-60
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
OV=$T/banco/ov-destilar; mkdir -p $OV
printf 'services:\n  vllm-server:\n    volumes:\n      - %s:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro\n' $MC/${PROD}_D > $OV/D.yml
for r in 1 2; do
  out=$T/banco/destilar_D_r$r.json; [ -f $out ] && continue
  [ -f $MC/${PROD}_D/model.safetensors ] || { echo "falta D"; break; }
  docker rm -f genesis-27b-pruebas >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload:/k alpine sh -c 'rm -rf /k/*'
  (cd $R/compose && GENESIS_PN157_BITS_CONV=8 GENESIS_PN159_VOCAB=65536 docker compose $C -f $OV/D.yml up -d --force-recreate >/dev/null 2>&1)
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8391/health >/dev/null 2>&1; do sleep 5; [ $(( $(date +%s) - t0 )) -gt 900 ] && break; done
  echo "$(date +%T) A/B D r$r listo en $(( $(date +%s) - t0 )) s"
  docker exec genesis-27b-pruebas python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder $out 4 1536 > /dev/null 2>&1
done
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
for b in A B C D; do python3 -c "
import json,glob
v=[json.load(open(f))['largo_aceptacion'] for f in sorted(glob.glob('$T/banco/destilar_${b}_r*.json'))]
print('destilacion $b', ' '.join('%.3f'%x for x in v), 'media %.3f'%(sum(v)/max(len(v),1)))"; done
echo "== SK-30"
sed '/until grep -q "TODO LISTO" \$T\/banco\/fase2.log/d' $M/ab_sk30.sh > /tmp/claude-1000/ab_sk30_ya.sh 2>/dev/null || sed '/fase2.log/d' $M/ab_sk30.sh > $T/banco/ab_sk30_ya.sh
bash $(ls /tmp/claude-1000/ab_sk30_ya.sh 2>/dev/null || echo $T/banco/ab_sk30_ya.sh) 2>&1 | tail -20
echo "== PN159 fase 2"
sed '/until grep -q "TODO LISTO" \$T\/banco\/destilar.log/d' $M/ab_pn159_fase2.sh > $T/banco/ab_fase2_ya.sh
bash $T/banco/ab_fase2_ya.sh 2>&1 | tail -8
echo "TODO LISTO"
