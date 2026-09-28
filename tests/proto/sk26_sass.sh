#!/bin/bash
# SASS del .cu (las 3 variantes), para contar instrucciones como con Triton
cd /k
for v in 0 1 2; do
  nvcc -arch=sm_86 -cubin -O3 -DSUMA=$v -DH=5120 -o /tmp/s$v.cubin sk26_norma_q8.cu 2>&1 | grep -v warning
  cuobjdump -sass /tmp/s$v.cubin > /tmp/s$v.sass
  python3 - $v <<'PY'
import re, sys
v = sys.argv[1]; s = open(f"/tmp/s{v}.sass").read()
ins = [i.split(".")[0] for i in re.findall(r"^\s+/\*[0-9a-f]+\*/\s+([A-Z0-9_.]+)", s, re.M)]
sel = {a: ins.count(a) for a in ("FFMA","FMUL","FADD","FMNMX","IMAD","IDP","IADD3","HADD2","HFMA2","HMUL2","HMNMX2","LDG","STG","SHFL","BAR","F2I","F2F","I2F","MUFU")}
print(f"SASS .cu S{v}: {len(ins)} instrucciones: {sel}")
PY
  cuobjdump -res-usage /tmp/s$v.cubin 2>/dev/null | grep -o "REG:[0-9]*" | head -1
done
