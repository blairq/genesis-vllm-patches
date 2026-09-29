#!/bin/bash
# PN163 (GDN sin cero ni copia de core_attn_out en pasos solo-spec): no cambia ninguna fila real, asi que el
# oraculo greedy tiene que dar el MISMO texto; paso: perfil de decode, dos replicas en orden opuesto.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4"
ORACULO=1 GENESIS_ENABLE_PN163_GDN_SALIDA=0 ./perfil_idiotsavant.sh is29_pn163off > /dev/null 2>&1
ORACULO=1 GENESIS_ENABLE_PN163_GDN_SALIDA=1 ./perfil_idiotsavant.sh is29_pn163on > /dev/null 2>&1
ORACULO=0 GENESIS_ENABLE_PN163_GDN_SALIDA=1 ./perfil_idiotsavant.sh is29_pn163on_b > /dev/null 2>&1
ORACULO=0 GENESIS_ENABLE_PN163_GDN_SALIDA=0 ./perfil_idiotsavant.sh is29_pn163off_b > /dev/null 2>&1
grep -h "PN163\|Traceback" analisis_perfil/is29_pn163on_decode1.log | head -3
for l in is29_pn163off is29_pn163on is29_pn163on_b is29_pn163off_b; do for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done; done
for s in "" _b; do for f in decode1 decode4; do echo "== $f$s"; python3 analizar_perfil.py --comparar analisis_perfil/is29_pn163off${s}_${f}_rank0.json analisis_perfil/is29_pn163on${s}_${f}_rank0.json | sed -n 3,6p; done; done
python3 - <<'PY'
import json
a = json.load(open("oraculo/is29_pn163off.json")); b = json.load(open("oraculo/is29_pn163on.json"))
print("oraculo: textos iguales", a["textos"] == b["textos"], "| aceptados por borrador", round(a["aceptados_por_borrador"], 3), round(b["aceptados_por_borrador"], 3))
PY
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "PN163 LISTO"
