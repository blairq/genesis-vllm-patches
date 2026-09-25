#!/usr/bin/env python3
"""ETAPA B: cuantiza capas con el cache de la etapa A y escribe UN archivo por capa y por variante.

Cada archivo `<salida>/<variante>/capa_NN.safetensors` es AUTOCONTENIDO: tiene todos los tensores
de esa capa con los nombres del checkpoint (las lineales cuantizadas en compressed-tensors
pack-quantized y el resto en bf16). Asi un modelo es solo una lista de archivos (etapa C), y una
capa se puede rehacer sola sin tocar las demas.

Metodo por defecto (fase 1): GPTQ W4 simetrico, grupo 128, escalas fp16, damp 0,01, sin
reordenamiento (sin g_idx: Marlin lee los grupos contiguos). Las Hessianas salen del cache, asi que
una capa tarda segundos.

Variantes de ejemplo (--spec JSON):
  {"metodo": "gptq", "bits": 4, "grupo": 128}                      la base
  {"metodo": "gptq", ..., "a_b": "int4"}                           in_proj_a/b del GDN tambien en int4
  {"metodo": "rtn", ...}                                           redondeo, para comparar
  {"metodo": "gptq_rot", "bloque": 1024, "bloque_down": 512}       residuo rotado + Hadamard en down
                                                                   (ver rotacion.py; H~ exacta del cache)
  {"capas": {"0": {"grupo": 64}, "63": {"grupo": 64}}}             sobreescrituras por capa

Error local: con la muestra de la etapa A, la capa se corre con los pesos DEQUANTIZADOS y se
compara su salida con la de BF16 (error relativo). Queda en `<variante>/informe.json`.

Uso: variante.py --modelo <bf16> --cache <etapa A> --salida <dir> --nombre base [--capas 0,1,2] [--spec '{...}']
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import ENTRADAS, Pesos  # noqa: E402
from gptq import empaquetar, gptq  # noqa: E402
import rotacion as ro  # noqa: E402

G = 128

# lineal -> entrada (Hessiana) por tipo de capa
LINEALES = {
    "full_attention": {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in",
                       "self_attn.o_proj": "attn_out", "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in",
                       "mlp.down_proj": "mlp_down"},
    "linear_attention": {"linear_attn.in_proj_qkv": "attn_in", "linear_attn.in_proj_z": "attn_in",
                         "linear_attn.out_proj": "attn_out", "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in",
                         "mlp.down_proj": "mlp_down"},
}
OPCIONALES = {"linear_attn.in_proj_a": "attn_in", "linear_attn.in_proj_b": "attn_in"}


def cargar_H(dirc, nombre, dev):
    d = torch.load(os.path.join(dirc, f"H_{nombre}.pt"))
    n = d["n"]
    m = torch.ones(n, n, dtype=torch.bool, device=dev).triu_()
    H = torch.zeros(n, n, dtype=torch.float32, device=dev)
    H.masked_scatter_(m, d["triu"].to(dev))
    return H + H.triu(1).T


@torch.no_grad()
def rtn(W, bits=4, grupo=G):
    out, inn = W.shape
    qmax = 2 ** (bits - 1) - 1
    wf = W.float().view(out, inn // grupo, grupo)
    s = (wf.abs().amax(-1) / (qmax + 0.5)).clamp_min(1e-10).half()
    q = (wf / s.float().unsqueeze(-1)).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q.view(out, inn), s


_EXT = {}


def externo_leer(d, base, dev):
    """(q int8 en [-8,7], escala) de una lineal pack-quantized de otro checkpoint."""
    if d not in _EXT:
        _EXT[d] = Pesos(d)
    E = _EXT[d]
    pk = E.get(base + ".weight_packed", dev, None)
    s = E.get(base + ".weight_scale", dev, None)
    out, inn = [int(x) for x in E.get(base + ".weight_shape", "cpu", None).tolist()]
    sh = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)
    q = (((pk.unsqueeze(-1) >> sh) & 0xF).reshape(out, -1)[:, :inn].to(torch.int8) - 8)
    return q, s.half()


def dequant(q, s, grupo=G):
    out, inn = q.shape
    return (q.float().view(out, inn // grupo, grupo) * s.float().unsqueeze(-1)).view(out, inn)


@torch.no_grad()
def normalizada_de_muestra(P, tc, i, ent, dirc, dev):
    """La entrada de la lineal ANTES del peso de la RMSNorm (n = x / rms(x)) sobre la muestra apartada
    de la etapa A, con la capa en BF16: para las filas de la Hessiana que el cache no puede tener."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    pref = f"model.language_model.layers.{i}."
    with torch.device("meta"):
        capa = m.Qwen3_5DecoderLayer(tc, i)
    capa.load_state_dict({k[len(pref):]: P.get(k, dev, torch.float32) for k in P.claves(pref)},
                         assign=True, strict=True)
    norma = capa.post_attention_layernorm if ent == "mlp_in" else capa.input_layernorm
    capt = []
    norma.register_forward_pre_hook(lambda m_, a_: capt.append(a_[0].float().reshape(-1, a_[0].shape[-1])))
    mu = torch.load(os.path.join(dirc, "muestra.pt"))
    x = mu["entrada"].to(dev, torch.float32)
    L = x.shape[1]
    rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None]
    for s in range(len(x)):
        capa(x[s:s + 1], position_embeddings=rot(x[s:s + 1], pos), attention_mask=None, position_ids=pos)
    h = torch.cat(capt)
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + tc.rms_norm_eps)
    return n, n.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", default="base")
    ap.add_argument("--capas", default="")
    ap.add_argument("--spec", default="{}")
    ap.add_argument("--dispositivo", default="cuda")
    ap.add_argument("--externo", default="", help="checkpoint compressed-tensors del que se toman las lineales (metodo externo)")
    a = ap.parse_args()
    dev = a.dispositivo
    spec = {"metodo": "gptq", "bits": 4, "grupo": G, "damp": 0.01, "a_b": "bf16"}
    spec.update(json.loads(a.spec))
    por_capa = spec.pop("capas", {})
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    P = Pesos(a.modelo)
    dest = os.path.join(a.salida, a.nombre)
    os.makedirs(dest, exist_ok=True)
    json.dump({"spec": spec, "capas": por_capa}, open(os.path.join(dest, "spec.json"), "w"), indent=1)
    capas = [int(x) for x in a.capas.split(",")] if a.capas else list(range(meta["capas"]))
    ruta_inf = os.path.join(dest, "informe.json")
    informe = json.load(open(ruta_inf)) if os.path.exists(ruta_inf) else {}

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg_full = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = Qwen3_5TextConfig(**cfg_full["text_config"])
    tc._attn_implementation = "sdpa"

    for i in capas:
        t0 = time.time()
        sp = dict(spec)
        sp.update(por_capa.get(str(i), {}))
        tipo = meta["layer_types"][i]
        dirc = os.path.join(a.cache, f"capa_{i:02d}")
        pref = f"model.language_model.layers.{i}."
        lin = dict(LINEALES[tipo])
        if tipo == "linear_attention" and sp["a_b"] != "bf16":
            lin.update(OPCIONALES)
        tensores, deq, errs = {}, {}, {}
        Hs = {}
        if sp["metodo"] == "gptq_rot":
            D = tc.hidden_size
            Rt = ro.Rt_de(D, sp.get("bloque", 1024), dev)
            R = Rt.T.contiguous()
            gs = {n: 1 + P.get(pref + n + ".weight", dev, torch.float32)
                  for n in ("input_layernorm", "post_attention_layernorm")}
            tensores[pref + "genesis_rot.signos"] = ro.signos(D)
            tensores[pref + "genesis_rot.bloques"] = torch.tensor([sp.get("bloque", 1024), sp.get("bloque_down", 512)])
        for k in P.claves(pref):
            nombre = k[len(pref):]
            base = nombre[: -len(".weight")] if nombre.endswith(".weight") else None
            if base in lin:
                W = P.get(k, dev)
                ent = lin[base]
                if sp["metodo"] == "externo":
                    # las lineales YA cuantizadas de otro checkpoint (p. ej. noon, que es AutoRound del
                    # mismo BF16): comparacion directa, capa por capa, con el mismo error local
                    q, s = externo_leer(a.externo, k[: -len(".weight")], dev)
                elif sp["metodo"] == "gptq":
                    if ent not in Hs:
                        Hs = {ent: cargar_H(dirc, ent, dev)}          # una Hessiana viva por vez
                    q, s = gptq(W, Hs[ent], damp=sp["damp"])
                elif sp["metodo"] == "gptq_rot":
                    c = ro.clase(base, sp.get("bloque_down", 512))
                    if ent not in Hs:
                        H = cargar_H(dirc, ent, dev)
                        if c == "entrada":
                            g = gs[ro.norma_de(base)]
                            gg = ro.g_segura(g)
                            H = H / gg[:, None] / gg[None, :]
                            if bool((g == 0).any()):
                                # Canal muerto (g=0 exacto; la capa 7 tiene uno, el de la activacion
                                # masiva): en el cache x = g n = 0, pero en el modelo rotado la entrada
                                # es n Rt y n_j es ENORME. Sin su fila en la Hessiana, GPTQ no ve esa
                                # energia y el error de cuantizacion la amplifica (MLP x7 en la capa 7).
                                # Su fila sale de la muestra apartada, reescalada a los tokens del cache.
                                muertos = (g == 0).nonzero().flatten()
                                n_m, tok_m = normalizada_de_muestra(P, tc, i, ent, dirc, dev)
                                tok_H = torch.load(os.path.join(dirc, f"stats_{ent}.pt"))["tokens"]
                                # solo la diagonal (la energia de n_j): mezclar una fila de 8k tokens con
                                # una Hessiana de 1M deja de ser definida positiva
                                H[muertos, :] = 0
                                H[:, muertos] = 0
                                H[muertos, muertos] = n_m[:, muertos].pow(2).sum(0) * (tok_H / tok_m)
                                print(f"capa {i} {ent}: canales muertos {muertos.tolist()}, rms de n_j "
                                      f"{n_m[:, muertos].pow(2).mean(0).sqrt().tolist()} vs mediana "
                                      f"{n_m.pow(2).mean(0).sqrt().median().item():.3f}", flush=True)
                            H = R @ H @ Rt
                        elif c == "bajada":
                            bd = sp.get("bloque_down", 512)
                            H = ro.bloques(ro.bloques(H, bd).T.contiguous(), bd)
                        Hs = {ent: H}
                        del H
                    Wf = W.float()
                    if c == "entrada":
                        g = gs[ro.norma_de(base)]
                        A = (Wf * g[None, :]) @ Rt
                    elif c == "escribe":
                        A = R @ Wf
                    else:
                        bd = sp.get("bloque_down", 512)
                        A = ro.bloques(R @ Wf, bd)
                    q, s = gptq(A, Hs[ent], damp=sp["damp"])
                else:
                    q, s = rtn(W, sp["bits"], sp["grupo"])
                Wd = dequant(q, s)
                if sp["metodo"] == "gptq_rot":                  # peso efectivo en la base original
                    if c == "entrada":
                        Wd = (Wd @ R) / ro.g_segura(g)[None, :]
                        Wd[:, g == 0] = 0
                    elif c == "escribe":
                        Wd = Rt @ Wd
                    else:
                        Wd = ro.bloques(Rt @ Wd, bd)
                errs[base] = float((Wd - W.float()).norm() / W.float().norm())
                p, s2, shp = empaquetar(q, s)
                tensores[k[: -len(".weight")] + ".weight_packed"] = p
                tensores[k[: -len(".weight")] + ".weight_scale"] = s2
                tensores[k[: -len(".weight")] + ".weight_shape"] = shp
                deq[nombre] = Wd.to(torch.bfloat16)
                del W
            else:
                t = P.get(k, "cpu", None)                      # tal cual, con su dtype
                tensores[k] = t.contiguous()
                deq[nombre] = t.to(dev)
        del Hs
        save_file({k: v.contiguous() for k, v in tensores.items()}, os.path.join(dest, f"capa_{i:02d}.safetensors"))

        # error local de la capa entera con los pesos dequantizados
        mu = torch.load(os.path.join(dirc, "muestra.pt"))
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        capa.load_state_dict(deq, assign=True, strict=True)
        capa.eval()
        x, y = mu["entrada"].to(dev), mu["salida"].to(dev).float()
        L = x.shape[1]
        rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
        pos = torch.arange(L, device=dev)[None].expand(x.shape[0], -1)
        with torch.no_grad():
            o = capa(x, position_embeddings=rot(x, pos), attention_mask=None, position_ids=pos)
            o = (o[0] if isinstance(o, tuple) else o).float()
        # el residuo domina la salida: se mide el error sobre (salida - entrada), o sea lo que la capa agrega
        d_ref, d_q = y - x.float(), o - x.float()
        e_capa = float((d_q - d_ref).norm() / d_ref.norm())
        informe[str(i)] = {"tipo": tipo, "spec": sp, "error_capa": e_capa, "error_pesos": errs,
                           "segundos": round(time.time() - t0, 1)}
        json.dump(informe, open(ruta_inf, "w"), indent=1)
        del capa, deq, mu, x, y
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
        print(f"capa {i:02d} {tipo:16s} error de la capa {e_capa:.4f}  "
              f"(pesos {min(errs.values()):.3f}-{max(errs.values()):.3f})  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
