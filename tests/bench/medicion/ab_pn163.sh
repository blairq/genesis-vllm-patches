#!/bin/bash
# PN163 (GDN sin cero ni copia de core_attn_out) y PN164 (entradas del GDN como vistas de in_proj, sin la copia
# por capa): no cambian ninguna fila real, asi que el oraculo greedy tiene que dar el MISMO texto en los tres
# brazos; paso: perfil de decode, dos replicas en orden opuesto. Brazos: base, s = PN163, sv = PN163 + PN164.
R=/home/usuario/Proyectos/genesis-vllm-patches; M=$R/tests/bench/medicion
echo "$(date +%T) arranca"
docker stop genesis-27b-idiotsavant >/dev/null 2>&1
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
cd $M; export FASES="decode1 decode4"
env_de() { case $1 in base) echo "GENESIS_ENABLE_PN163_GDN_SALIDA=0 GENESIS_ENABLE_PN164_GDN_VISTAS=0";;
                      s) echo "GENESIS_ENABLE_PN163_GDN_SALIDA=1 GENESIS_ENABLE_PN164_GDN_VISTAS=0";;
                      sv) echo "GENESIS_ENABLE_PN163_GDN_SALIDA=1 GENESIS_ENABLE_PN164_GDN_VISTAS=1";; esac; }
for b in base s sv; do env ORACULO=1 $(env_de $b) ./perfil_idiotsavant.sh is29_gdn_${b} > /dev/null 2>&1; done
for b in sv s base; do env ORACULO=0 $(env_de $b) ./perfil_idiotsavant.sh is29_gdn_${b}_b > /dev/null 2>&1; done
grep -h "PN16[34]\|Traceback" analisis_perfil/is29_gdn_sv_decode1.log | head -4
for l in is29_gdn_base is29_gdn_s is29_gdn_sv is29_gdn_base_b is29_gdn_s_b is29_gdn_sv_b; do for f in decode1 decode4; do for rk in 0 1; do python3 analizar_perfil.py trazas/${l}_$f $rk > /dev/null 2>&1; done; done; done
for s in "" _b; do for b in s sv; do for f in decode1 decode4; do echo "== $b vs base$s, $f"
  python3 analizar_perfil.py --comparar analisis_perfil/is29_gdn_base${s}_${f}_rank0.json analisis_perfil/is29_gdn_${b}${s}_${f}_rank0.json | sed -n 3,6p; done; done; done
python3 - <<'PY'
import json
o = {b: json.load(open(f"oraculo/is29_gdn_{b}.json")) for b in ("base", "s", "sv")}
for b in ("s", "sv"):
    print(f"oraculo {b}: textos iguales a base", o[b]["textos"] == o["base"]["textos"],
          "| aceptados por borrador", round(o["base"]["aceptados_por_borrador"], 3), round(o[b]["aceptados_por_borrador"], 3))
PY
docker rm -f genesis-27b-pruebas >/dev/null 2>&1
echo "PN163/164 LISTO"
