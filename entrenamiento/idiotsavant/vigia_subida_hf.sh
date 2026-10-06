#!/bin/bash
# Vigia de las subidas a HF (04-10). Cada 30 min: si el repo ya tiene el commit nuevo, listo; si la subida murio, la
# relanza; si sigue viva pero en 30 min subio < 5 MB, la mata y la relanza (Xet reaprovecha lo subido).
# Procesos por PID (nada de pgrep: una linea de comando que contiene el patron se encuentra a si misma).
# Estado (pids, logs) y staging: VIGIA_DIR. El borrador se sube desde $VIGIA_DIR/hf_borrador y el README del target
# desde $VIGIA_DIR/README_target.md (prepararlos antes). Uso: setsid nohup ./vigia_subida_hf.sh >> $VIGIA_DIR/vigia.log &
S=${VIGIA_DIR:-$HOME/.cache/vigia_hf}; mkdir -p $S
MC=/home/usuario/Proyectos/models-cache
BOR=BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2; TGT=BlairQ/qwen3.8_27b_idiotSavant_sm_86
MARCA_BOR="Update in place (2026-10-03)"; MARCA_TGT="2026-09-28 weights"; MARCA_README="README: 2026-09-28"
log() { echo "$(date '+%m-%d %T') $*"; }
# Bytes escritos por la subida viva (sockets incluidos), no el trafico de toda la placa: con los demas servicios la
# placa pasaba el umbral aunque la subida estuviera colgada en CLOSE-WAIT (06-10).
tx() { local p; p=$(cat $S/pid_tgt 2>/dev/null); local h; h=$(pgrep -P "$p" -f "hf upload" 2>/dev/null | head -1); [ -z "$h" ] && h=$p
  awk '/^wchar/ {print $2}' /proc/$h/io 2>/dev/null || echo 0; }
remoto() {  # repo marca -> 0 si el ultimo commit lleva la marca
  python3 -c "
import sys
from huggingface_hub import HfApi
t=[c.title for c in HfApi().list_repo_commits('$1')[:3]]
sys.exit(0 if any('$2' in x for x in t) else 1)" 2>/dev/null; }
vivo() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
lanzar_bor() {
  (cd $S/hf_borrador && setsid nohup hf upload $BOR . . --commit-message "$MARCA_BOR: +1 epoch, served acceptance +0.9% / +1.1%" >> $S/subida_borrador.log 2>&1 < /dev/null & echo $! > $S/pid_bor)
  log "borrador: lanzado pid $(cat $S/pid_bor)"
  sleep 60; vivo "$(cat $S/pid_bor)" || log "borrador: MURIO AL ARRANCAR: $(tr '\r' '\n' < $S/subida_borrador.log | grep -iE 'error|exception|usage' | tail -2 | tr '\n' ' ' | cut -c1-240)"; }
lanzar_tgt() {
  (cd $MC/qwen3.8_27b_idiotSavant_sm_86 && setsid nohup hf upload $TGT . . --include "capa_*.safetensors" --include "config.json" --include "informe_idiotsavant.json" --include "model.safetensors.index.json" --include "quantization_config.json" \
     --commit-message "Update in place ($MARCA_TGT): per-head Hadamard on o_proj/out_proj, GDN gates in int4 (KL 0.0194 -> 0.0178)" >> $S/subida_target.log 2>&1 < /dev/null & echo $! > $S/pid_tgt)
  log "target: lanzado pid $(cat $S/pid_tgt)"
  sleep 60; vivo "$(cat $S/pid_tgt)" || log "target: MURIO AL ARRANCAR: $(tr '\r' '\n' < $S/subida_target.log | grep -iE 'error|exception|usage' | tail -2 | tr '\n' ' ' | cut -c1-240)"; }
matar() { vivo "$1" && { local h; h=$(pgrep -P "$1" 2>/dev/null); kill $h "$1" 2>/dev/null; sleep 5; kill -9 $h "$1" 2>/dev/null; log "matado pid $1 e hijos $h (colgado)"; }; }
t_prev=$(tx); primera=1
while true; do
  t_act=$(tx); subido=$(( (t_act - t_prev) / 1048576 )); t_prev=$t_act
  [ $subido -lt 0 ] && subido=999                         # proceso nuevo tras un relanzamiento: contador desde cero
  [ -n "$primera" ] && { subido=999; primera=; }          # la primera vuelta no mide cuelgues
  if ! remoto $BOR "$MARCA_BOR"; then
    pid=$(cat $S/pid_bor 2>/dev/null)
    if ! vivo "$pid"; then log "borrador: no esta en HF y no hay subida viva; ultimo error: $(tr '\r' '\n' < $S/subida_borrador.log | grep -iE 'error|exception' | tail -1 | cut -c1-160)"; lanzar_bor
    elif [ $subido -lt 5 ]; then log "borrador: ${subido} MB en 30 min"; matar "$pid"; lanzar_bor
    else log "borrador: subiendo (${subido} MB en 30 min)"; fi
  elif ! remoto $TGT "$MARCA_TGT"; then
    pid=$(cat $S/pid_tgt 2>/dev/null)
    if ! vivo "$pid"; then log "borrador en HF; target: sin subida viva; ultimo error: $(tr '\r' '\n' < $S/subida_target.log | grep -iE 'error|exception' | tail -1 | cut -c1-160)"; lanzar_tgt
    elif [ $subido -lt 5 ]; then log "target: ${subido} MB en 30 min"; matar "$pid"; lanzar_tgt
    else log "target: subiendo (${subido} MB en 30 min)"; fi
  else
    remoto $TGT "$MARCA_README" || { hf upload $TGT $S/README_target.md README.md --commit-message "$MARCA_README weights (PN154/PN155 required, KL 0.0178)" >> $S/subida_target.log 2>&1 && log "README del target subido"; }
    remoto $TGT "$MARCA_README" && { log "TODO SUBIDO"; exit 0; }
  fi
  sleep 1800
done
