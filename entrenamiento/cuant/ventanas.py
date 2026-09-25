#!/usr/bin/env python3
"""Ventanas de evaluacion para la prueba de FIDELIDAD contra el BF16.

Salen de pedidos capturados que NO entraron en la calibracion (armar_calibracion.py tomo los
primeros N pedidos que calificaban, en orden del banco): se saltean esos y se toman los siguientes.
Cada ventana son los ultimos L tokens de prompt + respuesta, sacados directo de los ids de la
captura (que son exactamente los que proceso produccion), sin volver a tokenizar.

Uso: ventanas.py <dir_capturas> <n_calib_usadas> <L_calib> <salida.npy> [n] [L]
"""
import glob
import os
import sys

import numpy as np

capt, n_cal, L_cal, salida = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 48
L = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
dirs = sorted(d for d in glob.glob(os.path.join(capt, "*")) if os.path.exists(os.path.join(d, "etiquetas.npz")))
usadas, out, resp = 0, [], []
for d in dirs:
    e = np.load(os.path.join(d, "etiquetas.npz"))
    califica = int(e["prompt_len"]) + len(e["comp_ids"]) >= L_cal
    if usadas < n_cal:                       # mismo filtro y orden que la calibracion: se saltean
        usadas += califica
        continue
    zs = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "cmpl-*.npz")))]
    pos = np.concatenate([z["pos"] for z in zs])
    ids = np.concatenate([z["ids"] for z in zs])[np.argsort(pos)]
    if len(ids) < L:
        continue
    out.append(ids[-L:])
    resp.append(min(len(e["comp_ids"]), L))
    if len(out) >= N:
        break
np.save(salida, np.stack(out).astype(np.int32))
np.save(salida.replace(".npy", "_resp.npy"), np.asarray(resp, np.int32))
print(f"{len(out)} ventanas de {L} tokens (respuesta media {np.mean(resp):.0f}) -> {salida}")
