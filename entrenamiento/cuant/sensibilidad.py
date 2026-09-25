#!/usr/bin/env python3
"""Sensibilidad POR CAPA: cuanto cambia la salida del modelo si SOLO la capa i va cuantizada.

Por que: en ventanas que arrancan como un chat real (`<|im_start|>system` + tools), cuantizar una
sola capa del medio (la 32, GDN) ya deja el top-1 en 90% del BF16, y la 0 en 97,8%. El error de
la capa aislada (etapa B) no dice cuanto importa: hay que propagarlo por el resto del modelo.

Una sola pasada por las capas: se llevan en paralelo el flujo BF16 y, por cada variante y cada capa
i, un flujo que recibe la capa i cuantizada y todas las demas en BF16. El flujo i nace en la capa i
(copia del BF16 que entra), asi que cada capa se lee del disco una vez. Al final, KL exacta sobre
el vocabulario completo contra el BF16, acuerdo del top-1 y |dif de logprob| del token real.

Sale <salida>.json con, por ventana, variante y capa: kl, kl_desde16, top1, dlp_abs, dlp_medio,
err_local (error relativo de la salida de la capa sobre su aporte al residuo, con la entrada BF16).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import Pesos  # noqa: E402
from referencia import capa_de_variante  # noqa: E402


def q8(x, partes=1):
    """int8 simetrico por token, como Marlin W4A8; `partes`=2 en las lineales de fila de TP=2 (cada
    GPU cuantiza su mitad de K con su propia escala)."""
    xs = x.reshape(*x.shape[:-1], partes, -1)
    e = xs.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return ((xs / e).round().clamp(-127, 127) * e).reshape(x.shape)


def a8_en(capa, ruta, pref):
    """Engancha la activacion int8 en las lineales que la variante trae cuantizadas (las que son W4A8
    en vLLM). Si la variante es rotada (rotacion.py), la cuantizacion se hace en la base rotada: la
    entrada del residuo como (x/g) Rt, y la de down_proj con la Hadamard en linea."""
    from safetensors import safe_open
    import rotacion as ro
    with safe_open(ruta, "pt") as f:
        claves = list(f.keys())
        bases = {k[len(pref):-len(".weight_packed")] for k in claves if k.endswith(".weight_packed")}
        rot = None
        if pref + "genesis_rot.signos" in claves:
            b, bd = [int(x) for x in f.get_tensor(pref + "genesis_rot.bloques").tolist()]
            sg = f.get_tensor(pref + "genesis_rot.signos").float()
            rot = (b, bd, sg)
    assert bases, ruta
    dev = next(capa.parameters()).device
    if rot is not None:
        b, bd, sg = rot
        Rt = sg.to(dev)[:, None] * ro.por_bloques(len(sg), b, dev)
    for n in bases:
        mod = capa.get_submodule(n)
        partes = 2 if n.endswith(ro.FILA) else 1
        c = ro.clase(n, rot[1] if rot is not None else 512)
        if rot is None or c == "escribe":
            f = (lambda p: lambda m_, a_: (q8(a_[0], p),) + tuple(a_[1:]))(partes)
        elif c == "entrada":
            g = ro.g_segura(1 + capa.get_submodule(ro.norma_de(n)).weight.float())
            f = (lambda g_: lambda m_, a_: ((q8((a_[0] / g_) @ Rt) @ Rt.T) * g_,) + tuple(a_[1:]))(g)
        else:
            f = (lambda p: lambda m_, a_: (ro.bloques(q8(ro.bloques(a_[0], bd), p), bd),) + tuple(a_[1:]))(partes)
        mod.register_forward_pre_hook(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--ventanas", required=True, nargs="+", help="npy [n, L] (se usan todas las filas)")
    ap.add_argument("--variantes", required=True, nargs="+", help="nombre=dir_etapa_B")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--capas", default="", help="a-b: limitar las capas estudiadas")
    a = ap.parse_args()
    dev, DT = "cuda", torch.float32
    torch.set_grad_enabled(False)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = Qwen3_5TextConfig(**cfg["text_config"])
    tc._attn_implementation = "sdpa"
    P = Pesos(a.modelo)
    ids = np.concatenate([np.load(v) for v in a.ventanas])
    N, L = ids.shape
    variantes = dict(v.split("=", 1) for v in a.variantes)
    nL = tc.num_hidden_layers
    c0, c1 = (int(x) for x in a.capas.split("-")) if a.capas else (0, nL - 1)
    idt = torch.from_numpy(ids.astype(np.int64)).to(dev)
    E = P.get("model.language_model.embed_tokens.weight", dev, DT)
    base = E[idt]                                  # [N, L, D] flujo BF16
    del E
    rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None]
    pe = rot(base[:1], pos)
    flujos: dict[tuple[str, int], torch.Tensor] = {}   # (variante, capa) -> [N, L, D]
    err_local: dict[tuple[str, int], list] = {}

    def correr(capa, h):
        for s in range(len(h)):
            o = capa(h[s:s + 1], position_embeddings=pe, attention_mask=None, position_ids=pos)
            h[s:s + 1] = o[0] if isinstance(o, tuple) else o

    t0 = time.time()
    for i in range(nL):
        pref = f"model.language_model.layers.{i}."
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        capa.load_state_dict({k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else DT)
                              for k in P.claves(pref)}, assign=True, strict=True)
        entrada = base.clone() if c0 <= i <= c1 else None
        correr(capa, base)
        for h in flujos.values():
            correr(capa, h)
        del capa
        if entrada is not None:
            for nom, d in variantes.items():
                with torch.device("meta"):
                    cq = m.Qwen3_5DecoderLayer(tc, i)
                ruta = os.path.join(d, f"capa_{i:02d}.safetensors")
                cq.load_state_dict(capa_de_variante(ruta, pref, dev, DT), assign=True, strict=True)
                if nom.endswith("_a8"):
                    a8_en(cq, ruta, pref)
                h = entrada.clone()
                correr(cq, h)
                del cq
                aporte = (base - entrada).flatten(1).norm(dim=1)
                err_local[(nom, i)] = ((h - base).flatten(1).norm(dim=1) / aporte).tolist()
                flujos[(nom, i)] = h
            del entrada
        print(f"capa {i} ({tc.layer_types[i]}) lista, {len(flujos)} flujos, {time.time() - t0:.0f} s", flush=True)

    norma = m.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
    norma.weight.data = P.get("model.language_model.norm.weight", dev, torch.float32)
    W = P.get("lm_head.weight", dev, DT)
    real = idt[:, 1:]
    torch.cuda.empty_cache()
    out = {"ventanas": a.ventanas, "tipos": tc.layer_types, "por_ventana": []}
    kl_pos = {}                                    # "nom_capa_ventana" -> KL por posicion
    B = 128                                        # posiciones por trozo: los logits son 1 GB por 1000
    for s in range(N):
        acum = {k: {"kl": [], "top1": [], "d": []} for k in flujos}
        lpr_s = []
        for p0 in range(0, L - 1, B):
            p1 = min(p0 + B, L - 1)
            r = (norma(base[s, p0:p1]) @ W.T).log_softmax(-1)
            lpr = r.gather(-1, real[s, p0:p1, None]).squeeze(-1)
            top_r, pr = r.argmax(-1), r.exp()
            lpr_s.append(lpr)
            for k, h in flujos.items():
                q = (norma(h[s, p0:p1]) @ W.T).log_softmax(-1)
                acum[k]["kl"].append((pr * (r - q)).sum(-1))
                acum[k]["top1"].append(q.argmax(-1) == top_r)
                acum[k]["d"].append(q.gather(-1, real[s, p0:p1, None]).squeeze(-1) - lpr)
                del q
            del r, pr
        filas = {"lp_bf16_medio": float(torch.cat(lpr_s).mean()), "variantes": {}}
        for (nom, i), v in acum.items():
            kl, t1, d = torch.cat(v["kl"]), torch.cat(v["top1"]), torch.cat(v["d"])
            kl_pos[f"{nom}_{i}_{s}"] = kl.half().cpu().numpy()
            filas["variantes"].setdefault(nom, {})[i] = {
                "kl": float(kl.mean()), "kl_desde16": float(kl[16:].mean()),
                "top1": float(t1.float().mean()), "dlp_abs": float(d.abs().mean()),
                "dlp_medio": float(d.mean()), "err_local": err_local[(nom, i)][s]}
        out["por_ventana"].append(filas)
        print(f"ventana {s}: lp BF16 {filas['lp_bf16_medio']:.3f}", flush=True)
    json.dump(out, open(a.salida, "w"), indent=1)
    np.savez(a.salida.replace(".json", "_pos.npz"), **kl_pos)
    for s, f in enumerate(out["por_ventana"]):
        for nom, capas in f["variantes"].items():
            print(f"ventana {s} {nom}: " + " ".join(f"{i}:{v['kl']:.3f}" for i, v in sorted(capas.items())), flush=True)


if __name__ == "__main__":
    main()
