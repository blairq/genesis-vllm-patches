"""Compara capa por capa dos checkpoints INT4 de Qwen3.8-27B en compressed-tensors (pack-quantized).

Pensado para noon (AutoRound W4A16, fine-tune "uncensored") contra RedHat (AWQ + GPTQ W4A16 sobre
el base). Corre en CPU, lee los safetensors por mmap sin depender de `safetensors`, y va capa
por capa: el pico de RAM es de unos pocos GB.

Lo que hay que tener en cuenta para que la comparacion tenga sentido:
- RedHat plegó las escalas de AWQ: qkv/z/a/b y gate/up tienen columnas * s y la RMSNorm previa / s;
  up tiene filas / s y down columnas * s. Se compara en el espacio INVARIANTE: W * (1 + g) para las
  entradas (las normas son GemmaRMSNorm, peso centrado en cero) y up/down normalizados por la
  norma de cada fila de up.
- La diferencia entre dos int4 nunca es cero: se estima el piso de ruido de cuantizacion de los dos
  (escala^2 / 12 por elemento, llevada al espacio invariante) y se reporta diff / piso.
- Un "uncensored" por abliteracion es una edicion de rango 1 sobre las matrices que ESCRIBEN el
  residuo (o_proj, out_proj, down_proj, embed). Se mide la energia del primer valor singular de
  la diferencia y se compara su direccion con la que sale del embed (que es bf16 en los dos).

Uso: python comparar_checkpoints_int4.py <dir_noon> <dir_redhat> <salida.json>
"""
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("HILOS", "12")))
torch.manual_seed(0)
G = 128
N_CAPAS = int(os.environ.get("CAPAS", "64"))

_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64,
       "I8": np.int8, "U8": np.uint8}


class Checkpoint:
    def __init__(self, d):
        self.d = d
        self.mapa = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._hdr, self._mm = {}, {}

    def _archivo(self, f):
        if f not in self._hdr:
            p = os.path.join(self.d, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._hdr[f] = (json.loads(fh.read(n)), 8 + n)
            self._mm[f] = np.memmap(p, dtype=np.uint8, mode="r")
        return self._hdr[f], self._mm[f]

    def tiene(self, k):
        return k in self.mapa

    def dtype(self, k):
        (h, _), _mm = self._archivo(self.mapa[k])
        return h[k]["dtype"]

    def get(self, k, filas=None):
        """Tensor fp32 (o int para los empaquetados). `filas` = slice sobre la dim 0."""
        (h, base), mm = self._archivo(self.mapa[k])
        e = h[k]
        dt, shape = _DT[e["dtype"]], e["shape"]
        a, b = e["data_offsets"]
        arr = mm[base + a: base + b].view(dt).reshape(shape)
        if filas is not None:
            arr = arr[filas]
        arr = np.ascontiguousarray(arr)
        if e["dtype"] == "BF16":
            return torch.from_numpy(arr.astype(np.uint32) << 16).view(torch.float32)
        t = torch.from_numpy(arr)
        return t.float() if e["dtype"] in ("F16", "F32") else t


def deq(ck, pref):
    """-> (W fp32 [out,in], var del ruido de cuantizacion por elemento [out,in], meta)."""
    if not ck.tiene(pref + ".weight_packed"):
        w = ck.get(pref + ".weight")
        return w, torch.zeros_like(w), {"fmt": ck.dtype(pref + ".weight")}
    pk = ck.get(pref + ".weight_packed")                     # int32 [out, in/8]
    sc = ck.get(pref + ".weight_scale")                      # [out, in/G]
    out, inn = [int(x) for x in ck.get(pref + ".weight_shape").tolist()]
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((pk.unsqueeze(-1) >> sh) & 0xF).reshape(out, -1)[:, :inn].float() - 8.0
    gsz = inn // sc.shape[1]
    s_exp = sc.repeat_interleave(gsz, dim=1)
    w = q * s_exp
    meta = {
        "fmt": "int4", "group": gsz, "scale_dtype": ck.dtype(pref + ".weight_scale"),
        "g_idx": ck.tiene(pref + ".weight_g_idx"), "zp": ck.tiene(pref + ".weight_zero_point"),
        "frac_escala_neg": float((sc < 0).float().mean()),
        "frac_q_extremo": float(((q == -8) | (q == 7)).float().mean()),
        "frac_q_menos8": float((q == -8).float().mean()),
        "q_abs_medio": float(q.abs().mean()),
    }
    return w, s_exp.square() / 12.0, meta


def gemma(ck, k):
    return 1.0 + ck.get(k)


def comparar(A, B, varA, varB, svd=True):
    """A = noon, B = redhat, ya en el espacio invariante. Reducciones en fp64: el norm() fp32 de
    torch en CPU erra ~1% en matrices de 90M elementos (daba cosenos > 1)."""
    A, B, varA, varB = A.double(), B.double(), varA.double(), varB.double()
    D = A - B
    nB, nD = float(B.norm()), float(D.norm())
    r = {
        "rel": nD / max(nB, 1e-30),
        "cos": float((A * B).sum() / (A.norm() * B.norm() + 1e-30)),
        "piso": math.sqrt(float(varA.sum() + varB.sum())) / max(nB, 1e-30),
        "normA_normB": float(A.norm()) / max(nB, 1e-30),
    }
    r["exceso"] = r["rel"] / r["piso"] if r["piso"] > 0 else float("inf")
    if svd and nD > 0:
        U, S, V = torch.svd_lowrank(D.float(), q=8, niter=4)
        S, U, V = S.double(), U.double(), V.double()
        r["sv1_frac"] = float(S[0] ** 2) / nD ** 2
        r["sv4_frac"] = float((S[:4] ** 2).sum()) / nD ** 2
        r["_u1"], r["_v1"] = U[:, 0].clone(), V[:, 0].clone()
    return r


def direccion_embed(ckA, ckB, k, bloque=16384):
    """Direccion dominante (en el espacio del residuo) de la diferencia de una matriz bf16 enorme."""
    n = ckA.get(k, filas=slice(0, 1)).shape[1]
    C = torch.zeros(n, n, dtype=torch.float64)
    total, filas = 0.0, 248320
    for i in range(0, filas, bloque):
        d = (ckA.get(k, filas=slice(i, i + bloque)) - ckB.get(k, filas=slice(i, i + bloque))).double()
        C += d.T @ d
        total += float(d.square().sum())
    ev, evec = torch.linalg.eigh(C)
    ref = 0.0
    for i in range(0, filas, bloque):
        ref += float(ckB.get(k, filas=slice(i, i + bloque)).double().square().sum())
    return {"rel": math.sqrt(total / ref), "sv1_frac": float(ev[-1] / ev.sum()),
            "sv4_frac": float(ev[-4:].sum() / ev.sum())}, evec[:, -1].float()


def main():
    ckA, ckB, salida = Checkpoint(sys.argv[1]), Checkpoint(sys.argv[2]), sys.argv[3]
    res = {"capas": [], "bf16": {}, "globales": {}}
    t0 = time.time()

    # 1. embed y lm_head: bf16 en los dos, sin ruido de cuantizacion
    dirs = {}
    for k in ([] if os.environ.get("SIN_EMBED") else
              ["model.language_model.embed_tokens.weight", "lm_head.weight"]):
        r, u = direccion_embed(ckA, ckB, k)
        res["globales"][k] = r
        dirs[k] = u
        print(f"{k}: {r}  ({time.time()-t0:.0f}s)", flush=True)
    ref_dir = dirs.get("model.language_model.embed_tokens.weight",
                       torch.nn.functional.normalize(torch.ones(5120), dim=0))

    # 2. capas
    P = "model.language_model.layers."
    for L in range(N_CAPAS):
        p = f"{P}{L}."
        atn = (L % 4 == 3)
        fila = {"capa": L, "tipo": "atencion" if atn else "gdn", "mats": {}}
        gA_in, gB_in = gemma(ckA, p + "input_layernorm.weight"), gemma(ckB, p + "input_layernorm.weight")
        gA_pa, gB_pa = gemma(ckA, p + "post_attention_layernorm.weight"), \
            gemma(ckB, p + "post_attention_layernorm.weight")
        # AWQ implicito: s = g_noon / g_redhat (si el fine-tune no toco la norma)
        for nom, a, b in [("input_layernorm", gA_in, gB_in), ("post_attention_layernorm", gA_pa, gB_pa)]:
            s = a / b
            fila[nom] = {"s_mediana": float(s.median()), "s_min": float(s.min()), "s_max": float(s.max()),
                         "raw_rel": float((ckA.get(p + nom + ".weight") - ckB.get(p + nom + ".weight")).norm()
                                          / (ckB.get(p + nom + ".weight").norm() + 1e-30))}
        entradas = (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if atn else
                    ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                     "linear_attn.in_proj_a", "linear_attn.in_proj_b"])
        salida_mix = "self_attn.o_proj" if atn else "linear_attn.out_proj"

        def reg(nombre, A, B, vA, vB, mA, mB, escribe):
            r = comparar(A, B, vA, vB)
            u1, v1 = r.pop("_u1", None), r.pop("_v1", None)
            if u1 is not None and escribe:
                rd = ref_dir.double()
                D = A.double() - B.double()
                r["cos_dir_embed"] = float(abs(u1 @ rd))
                # fraccion de la diferencia que cae en la direccion del embed
                r["frac_en_dir_embed"] = float((rd @ D).square().sum() / D.square().sum())
                del D
            r["noon"], r["redhat"] = mA, mB
            fila["mats"][nombre] = r

        for m in entradas:
            wA, vA, mA = deq(ckA, p + m)
            wB, vB, mB = deq(ckB, p + m)
            reg(m, wA * gA_in, wB * gB_in, vA * gA_in.square(), vB * gB_in.square(), mA, mB, False)
            del wA, wB, vA, vB
        wA, vA, mA = deq(ckA, p + salida_mix)
        wB, vB, mB = deq(ckB, p + salida_mix)
        reg(salida_mix, wA, wB, vA, vB, mA, mB, True)
        del wA, wB, vA, vB

        # MLP: gate y up con columnas * (1+g); up/down normalizados por la norma de cada fila de up
        wA, vA, mA = deq(ckA, p + "mlp.gate_proj")
        wB, vB, mB = deq(ckB, p + "mlp.gate_proj")
        reg("mlp.gate_proj", wA * gA_pa, wB * gB_pa, vA * gA_pa.square(), vB * gB_pa.square(), mA, mB, False)
        del wA, wB, vA, vB
        uA, uvA, umA = deq(ckA, p + "mlp.up_proj")
        uB, uvB, umB = deq(ckB, p + "mlp.up_proj")
        uA, uB, uvA, uvB = uA * gA_pa, uB * gB_pa, uvA * gA_pa.square(), uvB * gB_pa.square()
        nA = uA.double().norm(dim=1, keepdim=True).float()
        nB = uB.double().norm(dim=1, keepdim=True).float()
        fila["awq_up_down"] = {"s_mediana": float((nA / nB).median()),
                               "s_min": float((nA / nB).min()), "s_max": float((nA / nB).max())}
        reg("mlp.up_proj", uA / nA, uB / nB, uvA / nA.square(), uvB / nB.square(), umA, umB, False)
        del uA, uB, uvA, uvB
        dA, dvA, dmA = deq(ckA, p + "mlp.down_proj")
        dB, dvB, dmB = deq(ckB, p + "mlp.down_proj")
        reg("mlp.down_proj", dA * nA.T, dB * nB.T, dvA * nA.T.square(), dvB * nB.T.square(), dmA, dmB, True)
        del dA, dB, dvA, dvB

        # tensores chicos bf16 de la capa (sin plegado)
        chicos = ([p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"] if atn else
                  [p + "linear_attn.conv1d.weight", p + "linear_attn.A_log", p + "linear_attn.dt_bias",
                   p + "linear_attn.norm.weight"])
        for k in chicos:
            a, b = ckA.get(k), ckB.get(k)
            a, b = a.double(), b.double()
            res["bf16"][k] = {"rel": float((a - b).norm() / (b.norm() + 1e-30)),
                              "igual": bool(torch.equal(a, b))}
        if atn:
            fila["kv_scale_redhat"] = [float(ckB.get(p + "self_attn.k_scale")),
                                       float(ckB.get(p + "self_attn.v_scale"))] \
                if ckB.tiene(p + "self_attn.k_scale") else None
        res["capas"].append(fila)
        m = fila["mats"]
        print(f"L{L:02d} {fila['tipo']:8s} " + " ".join(
            f"{k.split('.')[-1]}={v['rel']:.3f}/x{v['exceso']:.2f}/sv1={v.get('sv1_frac', 0):.2f}"
            for k, v in m.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(salida, "w"), indent=1)

    # 3. MTP: bf16 en los dos
    for k in sorted(set(k for k in ckA.mapa if k.startswith("mtp."))):
        if ckB.tiene(k):
            a, b = ckA.get(k), ckB.get(k)
            if a.shape == b.shape:
                res["bf16"][k] = {"rel": float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)),
                                  "igual": bool(torch.equal(a, b))}
    for k in ["model.language_model.norm.weight"]:
        a, b = ckA.get(k), ckB.get(k)
        res["bf16"][k] = {"rel": float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)),
                          "igual": bool(torch.equal(a, b))}
    json.dump(res, open(salida, "w"), indent=1)
    print(f"listo en {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
