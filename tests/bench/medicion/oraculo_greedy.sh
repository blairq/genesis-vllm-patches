#!/usr/bin/env bash
# oraculo_greedy.sh <etiqueta> [VAR=valor ...]: arranca la instancia aislada con esas variables, genera en greedy
# 3 respuestas (una larga, que cruza varios bloques de KV) y las guarda. Dos corridas con cambios bit a bit
# exactos tienen que dar el MISMO texto (y la misma aceptacion del borrador). Correr con setsid nohup.
set -uo pipefail
LABEL=$1; shift
R=/home/usuario/Proyectos/genesis-vllm-patches
MED=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
N=genesis-27b-pruebas
cd $R/compose
docker stop genesis-27b-idiotsavant >/dev/null 2>&1; docker rm -f $N >/dev/null 2>&1
env "$@" docker compose $C up -d --force-recreate >/dev/null 2>&1
t0=$(date +%s)
until [ "$(docker inspect -f '{{.State.Health.Status}}' $N 2>/dev/null)" = healthy ]; do
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "no arranco"; exit 1; }; sleep 15; done
mkdir -p $MED/oraculo
docker exec -i $N python3 - > $MED/oraculo/$LABEL.json <<'PY'
import json, os, urllib.request
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
def chat(txt, mt):
    b = json.dumps({"model": "qwen3.8", "messages": [{"role": "user", "content": txt}], "max_tokens": mt,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/v1/chat/completions", b, H), timeout=900))
chat("Explicame en detalle la paginacion de memoria virtual, con TLB y fallos de pagina.", 32)
ps = ["Escribi una funcion en Python que resuelva el problema de las N reinas con backtracking, con tests.",
      "Conta la historia de la computacion desde Babbage hasta los transformers, con detalle tecnico.",
      "Explica paso a paso como funciona un compilador: lexer, parser, AST, SSA, optimizaciones y emision."]
out = [chat(p, n)["choices"][0]["message"]["content"] for p, n in zip(ps, (600, 2000, 600))]
m = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8320/metrics", headers=H), timeout=60).read().decode()
acc = {l.split()[0]: l.split()[1] for l in m.splitlines() if l.startswith("vllm:spec_decode_num_accepted_tokens_total")
       or l.startswith("vllm:spec_decode_num_drafts_total")}
print(json.dumps({"textos": out, "spec": acc}, ensure_ascii=False))
PY
docker rm -f $N >/dev/null 2>&1
echo "$(date +%T) ORACULO $LABEL LISTO"
