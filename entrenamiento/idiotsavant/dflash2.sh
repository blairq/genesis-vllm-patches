#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  qwen3.8_27b_idiotSavant_sm_86_dflash2 — reajuste del borrador DFlash2 contra idiotSavant
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  El borrador (5 capas, bloque de 8, selector top-16 para el arbol) propone tokens que el modelo
#  grande verifica: cuanto mas se parece su distribucion a la del modelo SERVIDO, mas tokens se
#  aceptan por paso. Por eso se ajusta contra el modelo tal cual corre (W4A8 rotado, arbol, perfil
#  de agente de codigo), no contra el BF16. Decisiones y mediciones: DECISIONES.md.
#
#  A diferencia de idiotsavant.py esto NO corre en el virtualenv: la captura necesita a vLLM
#  sirviendo el modelo con los parches Genesis (PN148, PN149, captura_borrador), y eso vive en la
#  imagen vllm/vllm-openai:v0.29.0. Para no mezclar dos entornos, todos los pasos (tambien los de
#  torch puro) corren en esa misma imagen con --rm: tampoco ensucian el sistema.
#
#  Uso:
#     ./dflash2.sh                      corre los pasos que falten (retoma lo hecho)
#     DRY_RUN=1 ./dflash2.sh            valida todo sin crear nada (codigo 2 si algo falla)
#     PASOS="4 5 6" ./dflash2.sh        solo esos pasos
#     ./dflash2.sh estado [--json]      avance (lo mismo que lee la TUI: ./correr.sh tui.py borrador)
#  Lanzarlo desacoplado para corridas largas:  setsid nohup ./dflash2.sh > dflash2.log 2>&1 &
#
#  Variables (defaults entre corchetes):
#     BASE      borrador BF16 de partida [/models/incoai-qwen3.8-27b-dflash2-bf16: el DFlash2 original
#               de Inco]. Se ajusta DIRECTO contra idiotSavant, sin modelos intermedios (asi es el
#               publicado: 5,70 -> 6,13 greedy offline; 5,63 aceptados/paso y ~216 tok/s servido).
#     NOMBRE    nombre de la corrida [idiotsavant_dflash2]: dirs de captura, entrenamiento y estado
#     SALIDA    borrador W4A16 final en models-cache [qwen3.8_27b_idiotSavant_sm_86_dflash2_$NOMBRE]
#     MIN_DISCO_GB  la captura frena por debajo de esto [12] (~24 MB por pedido)
#     REPLICAS  replicas del A/B [3]
#
#  Pasos: 1 plegar la rotacion en el fc   2 cuantizar la base   3 captura   4 entrenar
#         5 cuantizar el ajustado          6 A/B (base contra ajustado)
#  Codigos de salida: 0 ok, 1 error en un paso, 2 recursos o dry-run.
# ══════════════════════════════════════════════════════════════════════════════════════════════
set -u
AQUI="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$AQUI/../.." && pwd)"
T=$R/tests/bench/medicion/trazas
MC=/home/usuario/Proyectos/models-cache
IMG=vllm/vllm-openai:v0.29.0
COMPOSE=$R/compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml
MODELO=/models/qwen3.8_27b_idiotSavant_sm_86
BASE=${BASE:-/models/incoai-qwen3.8-27b-dflash2-bf16}
NOMBRE=${NOMBRE:-idiotsavant_dflash2}
SALIDA=${SALIDA:-qwen3.8_27b_idiotSavant_sm_86_dflash2_$NOMBRE}
MIN_DISCO_GB=${MIN_DISCO_GB:-12}
REPLICAS=${REPLICAS:-3}
PASOS=${PASOS:-1 2 3 4 5 6}
DRY_RUN=${DRY_RUN:-0}
PRUEBAS=genesis-27b-pruebas
PROD=genesis-27b-idiotsavant
ESTADO=$T/banco/$NOMBRE.estado.json
OVS=$T/banco/ov-$NOMBRE
log() { echo "$(date +%T) [dflash2] $*"; }
en_host() { echo "$1" | sed "s#^/traces#$T#; s#^/models#$MC#"; }

estado() {  # paso detalle  -> archivo de estado (JSON) que leen `estado` y la TUI
  python3 - "$ESTADO" "$1" "$2" "$NOMBRE" <<'PY'
import json, os, sys, time
ruta, paso, detalle, nombre = sys.argv[1:5]
d = json.load(open(ruta)) if os.path.exists(ruta) else {"nombre": nombre, "inicio": time.time(), "historia": []}
d.update({"paso": paso, "detalle": detalle, "actualizado": time.time(), "pid": os.getppid()})
d["historia"].append([time.strftime("%H:%M:%S"), paso, detalle])
d["historia"] = d["historia"][-50:]
json.dump(d, open(ruta + ".tmp", "w"), indent=1)
os.replace(ruta + ".tmp", ruta)
PY
}

if [ "${1:-}" = "estado" ]; then
  [ -f "$ESTADO" ] || { echo "no hay corrida '$NOMBRE' ($ESTADO)"; exit 1; }
  if [ "${2:-}" = "--json" ]; then cat "$ESTADO"; exit 0; fi
  python3 - "$ESTADO" "$T/banco" "$NOMBRE" <<'PY'
import json, os, sys, time, glob
d = json.load(open(sys.argv[1])); B, N = sys.argv[2], sys.argv[3]
vivo = os.path.exists(f"/proc/{d.get('pid')}")
print(f"corrida {d['nombre']}: paso {d['paso']} - {d['detalle']} ({'corriendo' if vivo else 'sin proceso'}, "
      f"hace {(time.time() - d['actualizado']) / 60:.0f} min)")
cap = os.path.join(B, f"captura_{N}.log")
if os.path.exists(cap):
    print(f"  captura: {sum(1 for l in open(cap) if ' ok:' in l)} pedidos")
ft = os.path.join(B, f"ft_{N}.log")
if os.path.exists(ft):
    ult = [l.strip() for l in open(ft) if l.startswith(("BASE", "EPOCA", "ep "))]
    print("  entrenamiento: " + (ult[-1][:120] if ult else "arrancando"))
for f in sorted(glob.glob(os.path.join(B, f"{N}_*_r*.json"))):
    x = json.load(open(f)); P = x["pedidos"]
    dec = sum(p["decode_s"] or 0 for p in P); comp = sum((p["completion_tokens"] or 1) - 1 for p in P)
    print(f"  {os.path.basename(f)}: largo {x['largo_aceptacion']:.3f}, tok/s decode {comp / dec:.1f}")
PY
  exit 0
fi

# ── recursos ────────────────────────────────────────────────────────────────────────────────────
ram_libre_gb() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
disco_libre_gb() { df --output=avail -BG "$1" | tail -1 | tr -dc 0-9; }
gpus_libres() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 < 1000' | wc -l; }
esperar_ram() {  # GB
  local t0=$(date +%s)
  while [ "$(ram_libre_gb)" -lt "$1" ]; do
    [ $(( ($(date +%s) - t0) % 300 )) -lt 30 ] && log "esperando RAM: $(ram_libre_gb)/$1 GB libres"
    sleep 30
  done
}

# ── dry-run: validar sin crear nada ─────────────────────────────────────────────────────────────
problemas=()
chq() { if eval "$2"; then log "OK    $1"; else log "FALLA $1"; problemas+=("$1"); fi; }
chq "docker y la imagen $IMG"      "docker image inspect $IMG >/dev/null 2>&1"
chq "compose de idiotSavant valido" "docker compose -f $COMPOSE -f $R/compose/ov-aislado.yml config -q 2>/dev/null"
chq "modelo $(en_host $MODELO)"     "[ -f $(en_host $MODELO)/config.json ] && grep -q genesis_rotacion $(en_host $MODELO)/config.json"
chq "borrador base $(en_host $BASE)" "[ -f $(en_host $BASE)/model.safetensors ] && [ -f $(en_host $BASE)/config.json ]"
chq "BF16 original (para rotar el fc)" "[ -f $MC/orcarouter-qwen3.8-27b-uncensored-bf16/model.safetensors.index.json ]"
chq "bancos de entrenamiento y evaluacion" "[ -f $T/banco/entrena2000.jsonl ] && [ -f $T/banco/codigo40.jsonl ]"
chq "referencia de cuantizacion del borrador" "[ -f $MC/qwen3.8_27b_idiotSavant_sm_86_dflash2/config.json ]"
chq "RAM libre >= 12 GB ($(ram_libre_gb) GB)" "[ $(ram_libre_gb) -ge 12 ]"
chq "disco >= $((MIN_DISCO_GB + 6)) GB ($(disco_libre_gb $T) GB; la captura usa lo que sobre de $MIN_DISCO_GB)" "[ $(disco_libre_gb $T) -ge $((MIN_DISCO_GB + 6)) ]"
if [ "$(docker inspect -f '{{.State.Running}}' $PROD 2>/dev/null)" = true ]; then
  log "nota  $PROD esta corriendo: se para durante los pasos 3 y 6 y se vuelve a levantar al final"
else
  chq "dos GPUs libres ($(gpus_libres))" "[ $(gpus_libres) -ge 2 ]"
fi
if [ ${#problemas[@]} -gt 0 ]; then log "${#problemas[@]} problemas: ${problemas[*]}"; exit 2; fi
if [ "$DRY_RUN" = 1 ]; then
  for p in $PASOS; do log "haria el paso $p"; done
  log "DRY-RUN OK"; exit 0
fi
mkdir -p $OVS

ESTABA=0
parar_prod() {
  if [ "$(docker inspect -f '{{.State.Running}}' $PROD 2>/dev/null)" = true ]; then
    ESTABA=1; log "parando $PROD (se vuelve a levantar al final)"; docker stop $PROD >/dev/null
  fi
}
volver() {
  docker rm -f $PRUEBAS >/dev/null 2>&1
  docker run --rm -v /home/usuario/Proyectos/kv-offload-ab:/k --entrypoint bash $IMG -c "rm -rf /k/$NOMBRE" 2>/dev/null
  if [ $ESTABA = 1 ]; then log "levantando $PROD de nuevo"; docker start $PROD >/dev/null; fi
}
trap volver EXIT

# Instancia AISLADA (puerto 8361, red propia): ni Hermes ni opencode le pegan, asi no contaminan la
# captura ni los contadores de aceptacion (que son globales del servidor).
arrancar() {  # $1 = borrador W4A16 (ruta del host), $2.. = overrides extra
  local bor=$1; shift
  parar_prod
  docker run --rm -v /home/usuario/Proyectos/kv-offload-ab:/k --entrypoint bash $IMG -c "rm -rf /k/$NOMBRE; mkdir -p /k/$NOMBRE"
  cat > $OVS/borrador.yml <<EOF
services:
  vllm-server:
    environment:
      # tope chico del offload de KV en disco: con el del compose (40 GB) y el disco justo, una
      # prueba lo llena en minutos
      - GENESIS_KV_DISK_MAX_GB=5
      - GENESIS_KV_DISK_CHECK_SECS=15
    volumes:
      - /home/usuario/Proyectos/kv-offload-ab/$NOMBRE:/kv-offload
      - $bor:/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2:ro
EOF
  local extra=""; for o in "$@"; do extra="$extra -f $o"; done
  docker rm -f $PRUEBAS >/dev/null 2>&1
  docker compose -f $COMPOSE -f $R/compose/ov-aislado.yml -f $OVS/borrador.yml $extra up -d --force-recreate >/dev/null 2>&1
  local t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 2400 ]; do
    [ "$(docker inspect -f '{{.State.Health.Status}}' $PRUEBAS 2>/dev/null)" = healthy ] && return 0
    [ "$(docker inspect -f '{{.RestartCount}}' $PRUEBAS 2>/dev/null)" -gt 0 ] && break
    sleep 20
  done
  log "NO arranco:"; docker logs --tail 20 $PRUEBAS 2>&1 | cut -c1-200; return 1
}
quiere() { echo " $PASOS " | grep -q " $1 "; }
BASE_ROT=/traces/${NOMBRE}_base_rot
BASE_Q=$MC/${NOMBRE}_base_rot_w4a16
FT=/traces/borrador_ft_$NOMBRE

# ── 1. plegar la rotacion del target en el fc ──────────────────────────────────────────────────
# idiotSavant tiene el residuo rotado (h Rt). El fc del borrador lee 5 hidden states del target: se
# pliega W_c Rt en cada trozo de 5120 y las features salen IDENTICAS. El resto del borrador queda en su
# base original (su k/v sirve a dos normas distintas y no se puede rotar); embedding y lm_head
# compartidos los convierte PN149 en linea con los parametros que quedan en el config.
if quiere 1; then
  if [ -f $(en_host $BASE_ROT)/model.safetensors ]; then log "1. ya hecho ($BASE_ROT)"; else
    estado 1 "plegando la rotacion en el fc"; log "1. plegando la rotacion en el fc de $BASE"
    docker run --rm --memory 12g -v $R:/repo -v $MC:/models -v $T:/traces --entrypoint bash $IMG -c \
      "cd /repo/entrenamiento/borrador && python3 rotar_borrador.py $BASE /models/orcarouter-qwen3.8-27b-uncensored-bf16 \
       $MODELO $BASE_ROT && chmod -R a+rwX $BASE_ROT" 2>&1 | grep -v -i warn | tail -1
    [ -f $(en_host $BASE_ROT)/model.safetensors ] || { log "fallo el paso 1"; exit 1; }
  fi
fi

# ── 2. cuantizar la base rotada (RTN W4A16 g128) ─────────────────────────────────────────────────
# Se sirve DURANTE la captura. Las features que se capturan son la salida del fc SERVIDO (cuantizado);
# el fc no se entrena y el final se cuantiza igual (RTN es determinista): coinciden exactamente.
if quiere 2; then
  if [ -f $BASE_Q/model.safetensors ]; then log "2. ya hecho ($BASE_Q)"; else
    estado 2 "cuantizando la base rotada"; log "2. cuantizando la base rotada"
    docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint python3 $IMG \
      /repo/entrenamiento/borrador/cuantizar_rtn.py $BASE_ROT /models/${NOMBRE}_base_rot_w4a16 \
      /models/qwen3.8_27b_idiotSavant_sm_86_dflash2 2>&1 | grep -v -i warn | tail -1
    docker run --rm -v $MC:/models --entrypoint bash $IMG -c "chmod -R a+rX /models/${NOMBRE}_base_rot_w4a16"
    [ -f $BASE_Q/model.safetensors ] || { log "fallo el paso 2"; exit 1; }
  fi
fi

# ── 3. captura ───────────────────────────────────────────────────────────────────────────────────
# Banco de ENTRENAMIENTO: 2000 pedidos de agente con tools (SWE-rebench; excluye los repos del banco
# de evaluacion). Perfil coder de opencode, hasta 2048 tokens. Por pedido: regenera (logprobs top-16 =
# etiquetas) y re-envia prompt+respuesta con max_tokens=1 para volcar las features del prefill.
# Retomable: regenerar_y_capturar saltea los pedidos que ya tienen etiquetas. Frena solo con el disco.
if quiere 3; then
  if [ -f $T/captura/$NOMBRE/CAPTURA_TERMINADA ]; then log "3. ya hecho ($(ls $T/captura/$NOMBRE | wc -l) pedidos)"; else
    estado 3 "captura (frena con < $MIN_DISCO_GB GB libres)"; log "3. captura -> captura/$NOMBRE"
    arrancar $BASE_Q $R/compose/ov-captura.yml || exit 1
    docker exec -e MIN_DISCO_GB=$MIN_DISCO_GB $PRUEBAS python3 /traces/banco/regenerar_y_capturar.py \
      /traces/banco/entrena2000.jsonl /traces/captura/$NOMBRE coder 2048 >> $T/banco/captura_$NOMBRE.log 2>&1
    n=$(grep -c ' ok:' $T/banco/captura_$NOMBRE.log)
    log "   $n pedidos capturados"
    docker rm -f $PRUEBAS >/dev/null 2>&1
    rm -f $T/captura/CAPTURAR $T/captura/*.npz
    [ "$n" -ge 200 ] || { log "muy pocos pedidos ($n): no alcanza para entrenar"; exit 1; }
    touch $T/captura/$NOMBRE/CAPTURA_TERMINADA
  fi
fi

# ── 4. entrenar ─────────────────────────────────────────────────────────────────────────────────
# LoRA r=64 en las 7 lineales de las 5 capas; conv, fc, normas y selector congelados. 0,9 TV + 0,1 CE
# contra el top-16 del modelo, con AUF. El entrenador ve el embedding y el lm_head EFECTIVOS del
# borrador servido con PN149 (cargar_target detecta genesis_rotacion). 3% de los pedidos apartados.
if quiere 4; then
  if [ -f $(en_host $FT)/model.safetensors ]; then log "4. ya hecho ($FT)"; else
    # tope de memoria del contenedor segun la RAM que haya (medido: ~9 GB alcanzan, 16 da holgura)
    esperar_ram 11
    [ "$(ram_libre_gb)" -ge 18 ] && TOPE=16g || TOPE=9g
    estado 4 "entrenando (tope $TOPE)"; log "4. entrenando (tope de memoria $TOPE)"
    docker run --rm --name ft-$NOMBRE --gpus '"device=0"' --ipc host --memory $TOPE --memory-swap $TOPE \
      -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG /repo/entrenamiento/borrador/entrenar.py \
      --datos /traces/captura/$NOMBRE --salida $FT --borrador $BASE_ROT \
      --noon $MODELO --epocas 1 --lote 32 --lr 1e-4 --rango 64 --apartar 0.03 --auf > $T/banco/ft_$NOMBRE.log 2>&1
    rc=$?
    if [ $rc = 137 ]; then log "   murio por memoria: reintento con 18 GB"; esperar_ram 20
      docker run --rm --name ft-$NOMBRE --gpus '"device=0"' --ipc host --memory 18g --memory-swap 18g \
        -v $R:/repo:ro -v $MC:/models:ro -v $T:/traces --entrypoint python3 $IMG /repo/entrenamiento/borrador/entrenar.py \
        --datos /traces/captura/$NOMBRE --salida $FT --borrador $BASE_ROT \
        --noon $MODELO --epocas 1 --lote 32 --lr 1e-4 --rango 64 --apartar 0.03 --auf >> $T/banco/ft_$NOMBRE.log 2>&1
    fi
    grep -E "^BASE|^EPOCA" $T/banco/ft_$NOMBRE.log | cut -c1-60 | sed 's/^/   /'
    [ -f $(en_host $FT)/model.safetensors ] || { log "fallo el paso 4"; exit 1; }
  fi
fi

# ── 5. cuantizar el ajustado ───────────────────────────────────────────────────────────────────
if quiere 5; then
  if [ -f $MC/$SALIDA/model.safetensors ]; then log "5. ya hecho ($SALIDA)"; else
    estado 5 "cuantizando el ajustado"; log "5. cuantizando -> models-cache/$SALIDA"
    docker run --rm --memory 9g --memory-swap 9g -v $R:/repo:ro -v $MC:/models -v $T:/traces --entrypoint python3 $IMG \
      /repo/entrenamiento/borrador/cuantizar_rtn.py $FT /models/$SALIDA \
      /models/qwen3.8_27b_idiotSavant_sm_86_dflash2 2>&1 | grep -v -i warn | tail -1
    docker run --rm -v $MC:/models --entrypoint bash $IMG -c "chmod -R a+rX /models/$SALIDA"
    [ -f $MC/$SALIDA/model.safetensors ] || { log "fallo el paso 5"; exit 1; }
  fi
fi

# ── 6. A/B en vLLM ──────────────────────────────────────────────────────────────────────────────
# Banco de EVALUACION (40 pedidos de agente de repos que no estan en el de entrenamiento), arbol,
# perfil coder, una concurrencia. Metricas TOTALES (la mediana de tok/s por pedido engana entre
# borradores: textos distintos -> pedidos cortos/largos distintos). Cada replica ya hecha se saltea.
if quiere 6; then
  for brazo in base ajustado; do
    [ $brazo = base ] && bor=$BASE_Q || bor=$MC/$SALIDA
    faltan=""; for r in $(seq 1 $REPLICAS); do [ -f $T/banco/${NOMBRE}_${brazo}_r$r.json ] || faltan="$faltan $r"; done
    [ -z "$faltan" ] && { log "6. $brazo ya medido"; continue; }
    estado 6 "A/B: brazo $brazo, replicas$faltan"; log "6. A/B brazo $brazo (replicas$faltan)"
    arrancar $bor || continue
    for r in $faltan; do
      docker exec $PRUEBAS python3 /traces/banco/correr_banco.py /traces/banco/codigo40.jsonl coder \
        /traces/banco/${NOMBRE}_${brazo}_r$r.json 1 1536 > /dev/null 2>&1
    done
    docker rm -f $PRUEBAS >/dev/null 2>&1
  done
  NOMBRE=$NOMBRE "$0" estado | sed 's/^/   /'
fi
estado fin "LISTO"
log "LISTO. Para servir el ajustado: montar models-cache/$SALIDA como el borrador del compose idiotSavant"
