#!/usr/bin/env bash
# perfil_idiotsavant.sh <label>: trazas torch del stack de idiotSavant TAL CUAL se sirve (compose de
# produccion + instancia aislada: sin trafico ajeno). Tres fases:
#   decode1  un pedido, ~55k de contexto ya cacheado, DFlash2 + arbol, 60 pasos
#   decode4  cuatro pedidos a la vez (el patron real: un hilo y subagentes)
#   prefill  un prefill frio de ~16k tokens (chunks como en produccion)
# Deja las trazas en tests/bench/medicion/trazas/<label>_<fase>. NO levanta produccion al final
# (lo hace quien lo llama). Correr con setsid nohup.
set -uo pipefail
LABEL=$1
R=/home/usuario/Proyectos/genesis-vllm-patches
MED=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
N=genesis-27b-pruebas
cd $R/compose
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
for FASE in ${FASES:-decode1 decode4 prefill}; do
  docker rm -f $N >/dev/null 2>&1
  case $FASE in decode*) DEL=8; MAX=60 ;; prefill) DEL=0; MAX=12 ;; esac
  rm -rf $MED/trazas/${LABEL}_$FASE
  PROF_KIND='"torch"' PROF_DIR=/traces/${LABEL}_$FASE PROF_DELAY=$DEL PROF_MAX=$MAX docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s)
  until [ "$(docker inspect -f '{{.State.Health.Status}}' $N 2>/dev/null)" = healthy ]; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "[$FASE] no arranco"; docker logs --tail 20 $N; exit 1; }
    sleep 15
  done
  echo "$(date +%T) [$LABEL/$FASE] listo"
  docker exec -i -e FASE=$FASE $N python3 - <<'PY'
import glob, json, os, threading, time, urllib.request
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
U = "http://127.0.0.1:8320"
def post(path, cuerpo=None, t=900):
    b = json.dumps(cuerpo).encode() if cuerpo is not None else b""
    return json.load(urllib.request.urlopen(urllib.request.Request(U + path, b, H, method="POST"), timeout=t)) if cuerpo is not None \
        else urllib.request.urlopen(urllib.request.Request(U + path, b, H, method="POST"), timeout=t).read()
def chat(txt, mt):
    return post("/v1/chat/completions", {"model": "qwen3.8", "messages": [{"role": "user", "content": txt}], "max_tokens": mt,
                                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
fs = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/_genesis/**/*.py", recursive=True))
txt = "".join(open(f, errors="ignore").read() for f in fs)
fase = os.environ["FASE"]
chat("Explicame en detalle la paginacion de memoria virtual, con TLB y fallos de pagina.", 32)   # calienta (prompt largo primero)
if fase == "decode1":
    largo = txt[:200000] + "\n\nEscribi un resumen largo y detallado de este codigo, modulo por modulo."
    chat(largo, 8)                                   # deja el prefijo en la cache
    post("/start_profile"); r = chat(largo, 400); print("decode1", r["usage"])
elif fase == "decode4":
    ps = [txt[i * 60000:i * 60000 + 50000] + "\n\nExplica en detalle que hace este codigo." for i in range(4)]
    for p in ps: chat(p, 8)
    post("/start_profile")
    hs = [threading.Thread(target=lambda p=p: print("decode4", chat(p, 400)["usage"])) for p in ps]
    [h.start() for h in hs]; [h.join() for h in hs]
else:
    corto = txt[300000:356000] + "\n\nDeci solo OK."
    post("/start_profile"); t = time.time(); r = chat(corto, 1); print("prefill", r["usage"], round(time.time() - t, 2), "s")
post("/stop_profile", t=900)
PY
  for i in $(seq 1 60); do n=$(ls $MED/trazas/${LABEL}_$FASE 2>/dev/null | grep -c "json"); [ "$n" -ge 2 ] && break; sleep 10; done
  sleep 20; echo "$(date +%T) [$LABEL/$FASE] trazas: $(ls $MED/trazas/${LABEL}_$FASE 2>/dev/null | tr '\n' ' ')"
done
docker rm -f $N >/dev/null 2>&1
echo "$(date +%T) LISTO"
