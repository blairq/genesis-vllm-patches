"""PN159: orden del vocabulario por frecuencia en la carga real (capturas de dflash2.sh), para el lm_head
recortado del borrador. Cuenta los tokens generados (comp_ids), el top-4 del target en cada posicion
(las alternativas que el arbol necesita poder proponer) y los del contexto (ids del prefill). Escribe
el orden completo (int32 [V]): el tamano del recorte se elige al servir.

  python3 tests/proto/pn159_frecuencias.py <dir de capturas> vllm/_genesis/datos/pn159_orden.npy
Con --evaluar mide la cobertura contando la mitad de los pedidos y evaluando en la otra.
"""
import glob, sys
import numpy as np

V = 248320
ESPECIALES = 2048


def contar(dirs):
    c = np.zeros(V, np.float64)
    for d in dirs:
        try:
            e = np.load(d + "/etiquetas.npz")
        except OSError:
            continue
        np.add.at(c, e["comp_ids"], 1.0)
        np.add.at(c, e["top_ids"][:, :4].ravel(), 0.5)
        for f in glob.glob(d + "/cmpl*.npz"):
            try:
                np.add.at(c, np.load(f)["ids"], 0.25)
            except Exception:
                pass
    return c


def orden(c):
    c = c.copy(); c[:ESPECIALES] = np.inf
    return np.argsort(-c, kind="stable").astype(np.int32)


dirs = sorted(d for d in glob.glob(sys.argv[1] + "/*/*") if glob.os.path.isdir(d))
print(len(dirs), "pedidos")
if "--evaluar" in sys.argv:
    o = orden(contar(dirs[0::2]))
    ev = np.concatenate([np.load(d + "/etiquetas.npz")["comp_ids"] for d in dirs[1::2]])
    ev4 = np.concatenate([np.load(d + "/etiquetas.npz")["top_ids"][:, :4].ravel() for d in dirs[1::2]])
    for n in (16384, 24576, 32768, 49152, 65536):
        s = np.zeros(V, bool); s[o[:n]] = True
        print(f"{n:6d}: generados {s[ev].mean():.4f}  top-4 del target {s[ev4].mean():.4f}")
else:
    np.save(sys.argv[2], orden(contar(dirs)))
    print("escrito", sys.argv[2])
