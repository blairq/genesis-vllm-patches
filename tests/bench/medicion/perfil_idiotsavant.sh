#!/usr/bin/env bash
# perfil_idiotsavant.sh <etiqueta>: trazas torch del stack de idiotSavant TAL CUAL se sirve (compose de
# produccion + instancia aislada: sin trafico ajeno). Fases:
#   decode1  un pedido, ~62k de contexto ya cacheado, DFlash2 + arbol, 60 pasos
#   decode4  cuatro pedidos a la vez (el patron real: un hilo y subagentes)
#   prefill  un prefill frio de ~15k tokens (chunks como en produccion)
# Las fases de decode (y el oraculo, con ORACULO=1) corren sobre UN SOLO arranque del servidor: el
# profiler se prende y apaga por fase y las trazas se mueven a su carpeta. El prefill necesita otra
# configuracion del profiler (sin iteraciones de retardo) y va en un arranque aparte.
#
# Variables: FASES (default "decode1 decode4 prefill"), ORACULO=1 (texto greedy + aceptacion del borrador
# en tests/bench/medicion/oraculo/<etiqueta>.json), y cualquier GENESIS_* declarada en el compose.
# Deja las trazas en tests/bench/medicion/trazas/<etiqueta>_<fase> y los avisos de Genesis en
# analisis_perfil/<etiqueta>_<fase>.log. NO levanta idiotSavant al final (lo hace quien lo llama).
# Tiempos (27-09): ~1 min de arranque + ~1,5 min por fase. Correr con setsid nohup.
set -uo pipefail
LABEL=$1
R=/home/usuario/Proyectos/genesis-vllm-patches
MED=$R/tests/bench/medicion
C="-f $R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml -f $R/compose/ov-aislado.yml"
N=genesis-27b-pruebas
FASES=${FASES:-decode1 decode4 prefill}
mkdir -p $MED/analisis_perfil $MED/oraculo
cd $R/compose

arrancar() {   # $1 = dir de trazas (dentro del contenedor), $2 = delay, $3 = max iteraciones
  docker rm -f $N >/dev/null 2>&1
  PROF_KIND='"torch"' PROF_DIR=$1 PROF_DELAY=$2 PROF_MAX=$3 docker compose $C up -d --force-recreate >/dev/null 2>&1
  local t0=$(date +%s)
  # /health cada 5 s: el healthcheck de docker del override es cada 60 s y hacia perder hasta un minuto
  until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "no arranco"; docker logs --tail 20 $N; exit 1; }
    sleep 5
  done
  echo "$(date +%T) [$LABEL] servidor listo en $(( $(date +%s) - t0 )) s"
}

mover_trazas() {   # $1 = dir temporal, $2 = fase: espera las dos trazas y las mueve a <etiqueta>_<fase>
  for i in $(seq 1 90); do
    n=$(docker exec $N sh -c "ls $1 2>/dev/null | grep -c 'pt.trace.json'" 2>/dev/null || echo 0)
    [ "${n:-0}" -ge 2 ] && break; sleep 2
  done
  sleep 3   # que terminen de escribirse
  # las trazas son de root: un rm del host fallaba en silencio y analizar_perfil leia las VIEJAS
  docker exec $N sh -c "rm -rf /traces/${LABEL}_$2; mkdir -p /traces/${LABEL}_$2 && mv $1/* /traces/${LABEL}_$2/"
  echo "$(date +%T) [$LABEL/$2] trazas: $(ls $MED/trazas/${LABEL}_$2 2>/dev/null | tr '\n' ' ')"
}

cargar() {   # $1 = fase: la carga de trabajo, dentro del contenedor
  docker exec -i -e FASE=$1 $N python3 - <<'PY'
import glob, json, os, threading, time, urllib.request
H = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]}
U = "http://127.0.0.1:8320"
def post(path, cuerpo=None, t=900):
    b = json.dumps(cuerpo).encode() if cuerpo is not None else b""
    r = urllib.request.urlopen(urllib.request.Request(U + path, b, H, method="POST"), timeout=t)
    return json.load(r) if cuerpo is not None else r.read()
def chat(txt, mt):
    return post("/v1/chat/completions", {"model": "qwen3.8", "messages": [{"role": "user", "content": txt}], "max_tokens": mt,
                                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
def spec():
    m = urllib.request.urlopen(urllib.request.Request(U + "/metrics", headers=H), timeout=60).read().decode()
    d = {}
    for l in m.splitlines():
        for k in ("num_accepted_tokens_total", "num_drafts_total"):
            if l.startswith("vllm:spec_decode_" + k):
                d[k] = float(l.split()[1])
    return d
fs = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/_genesis/**/*.py", recursive=True))
txt = "".join(open(f, errors="ignore").read() for f in fs)
fase = os.environ["FASE"]
chat("Explicame en detalle la paginacion de memoria virtual, con TLB y fallos de pagina.", 32)   # calienta (prompt largo primero)
if fase == "decode1":
    largo = txt[:200000] + "\n\nEscribi un resumen largo y detallado de este codigo, modulo por modulo."
    chat(largo, 8)                                   # deja el prefijo en la cache
    post("/start_profile"); r = chat(largo, 400); print("decode1", r["usage"]["completion_tokens"], "tokens")
    post("/stop_profile", t=900)
elif fase == "decode4":
    ps = [txt[i * 60000:i * 60000 + 50000] + "\n\nExplica en detalle que hace este codigo." for i in range(4)]
    for p in ps: chat(p, 8)
    post("/start_profile")
    hs = [threading.Thread(target=lambda p=p: chat(p, 400)) for p in ps]
    [h.start() for h in hs]; [h.join() for h in hs]
    post("/stop_profile", t=900); print("decode4 ok")
elif fase == "prefill":
    corto = txt[300000:356000] + "\n\nDeci solo OK."
    post("/start_profile"); t = time.time(); r = chat(corto, 1)
    print("prefill", r["usage"]["prompt_tokens"], "tokens", round(time.time() - t, 2), "s")
    post("/stop_profile", t=900)
elif fase == "ttft":
    # prefill frio SIN profiler: 3 prompts distintos de ~15k (no comparten prefijo), max_tokens 1
    ts = []
    for i in range(3):
        # nonce al PRINCIPIO: el tier de KV en disco sobrevive entre arranques y sin esto la segunda
        # corrida acertaria el prefijo (TTFT falso; ver la memoria offload-kv-envenenado-por-cambio-de-config)
        p = f"[{os.urandom(8).hex()}-{i}] " + txt[400000 + i * 70000: 400000 + i * 70000 + 56000] + "\n\nDeci solo OK."
        t = time.time(); r = chat(p, 1); ts.append(round(time.time() - t, 3))
        print("ttft", r["usage"]["prompt_tokens"], "tokens", ts[-1], "s")
    print("TTFT_JSON " + json.dumps(ts))
elif fase == "oraculo":
    # Dos corridas con cambios bit a bit exactos tienen que dar el MISMO texto; con cambios que reordenan
    # sumas el texto diverge y lo que tiene que quedar igual es la aceptacion del borrador (delta de los
    # contadores alrededor de estos pedidos, sin lo que hicieron las fases anteriores).
    ps = ["Escribi una funcion en Python que resuelva el problema de las N reinas con backtracking, con tests.",
          "Conta la historia de la computacion desde Babbage hasta los transformers, con detalle tecnico.",
          "Explica paso a paso como funciona un compilador: lexer, parser, AST, SSA, optimizaciones y emision."]
    s0 = spec()
    out = [chat(p, n)["choices"][0]["message"]["content"] for p, n in zip(ps, (600, 2000, 600))]
    s1 = spec()
    acc, dr = s1["num_accepted_tokens_total"] - s0["num_accepted_tokens_total"], s1["num_drafts_total"] - s0["num_drafts_total"]
    print("ORACULO_JSON " + json.dumps({"textos": out, "aceptados": acc, "borradores": dr,
                                        "aceptados_por_borrador": acc / max(dr, 1)}, ensure_ascii=False))
PY
}

guardar_avisos() {
  docker logs $N 2>&1 | grep -iE "genesis|warning|error|no aplica" > $MED/analisis_perfil/${LABEL}_$1.log
}

DECODE=$(echo $FASES | tr ' ' '\n' | grep -E '^decode' | tr '\n' ' ')
if [ -n "$DECODE" ] || [ "${ORACULO:-0}" = 1 ]; then
  arrancar /traces/${LABEL}_tmp ${PROF_DELAY_DEC:-8} ${PROF_ITER:-60}     # con PROF_STACK=true usar ~25: la traza con pilas se come 10 GB y el cgroup (26 GB) mata al worker
  for FASE in $DECODE; do
    cargar $FASE
    mover_trazas /traces/${LABEL}_tmp $FASE
    guardar_avisos $FASE
  done
  if [ "${ORACULO:-0}" = 1 ]; then
    cargar oraculo | grep '^ORACULO_JSON ' | sed 's/^ORACULO_JSON //' > $MED/oraculo/$LABEL.json
    echo "$(date +%T) [$LABEL] oraculo: $(python3 -c "import json;d=json.load(open('$MED/oraculo/$LABEL.json'));print(round(d['aceptados_por_borrador'],3),'aceptados por borrador')")"
    guardar_avisos oraculo
  fi
fi
if echo " $FASES " | grep -q ' ttft '; then
  docker rm -f $N >/dev/null 2>&1
  docker compose $C up -d --force-recreate >/dev/null 2>&1
  t0=$(date +%s); until curl -sf -m 3 http://localhost:8361/health >/dev/null 2>&1; do sleep 5; done
  echo "$(date +%T) [$LABEL] servidor (sin profiler) listo en $(( $(date +%s) - t0 )) s"
  cargar ttft | tee /dev/stderr | grep '^TTFT_JSON ' | sed 's/^TTFT_JSON //' > $MED/analisis_perfil/${LABEL}_ttft.json
  guardar_avisos ttft
fi
if echo " $FASES " | grep -q ' prefill '; then
  arrancar /traces/${LABEL}_tmp 0 12
  cargar prefill
  mover_trazas /traces/${LABEL}_tmp prefill
  guardar_avisos prefill
fi
docker exec $N sh -c "rmdir /traces/${LABEL}_tmp" 2>/dev/null
docker rm -f $N >/dev/null 2>&1
echo "$(date +%T) LISTO"
