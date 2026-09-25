#!/usr/bin/env python3
"""Referencia BF16 para la prueba de fidelidad: top-K de la distribucion del proximo token en cada
posicion de las ventanas, con el forward capa por capa de la etapa A (validado: 90,5% de acierto
sobre las respuestas de noon).

Sale `<salida>.npz` con, para cada ventana y posicion p (predice el token p+1):
  top_ids [n, L-1, K] int32, top_lp [n, L-1, K] float16, lp_real [n, L-1] float32.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import Pesos  # noqa: E402


def capa_de_variante(ruta, pref, dev, dt=torch.bfloat16):
    """Los tensores de una capa de la etapa B, con las lineales dequantizadas a bf16."""
    from safetensors.torch import load_file
    t = load_file(ruta, device=str(dev))
    sd = {}
    rot = None
    if pref + "genesis_rot.signos" in t:              # variante rotada: ver rotacion.py
        import rotacion as ro
        b, bd = [int(x) for x in t.pop(pref + "genesis_rot.bloques").tolist()]
        sg = t.pop(pref + "genesis_rot.signos").float()
        Rt = sg[:, None] * ro.por_bloques(len(sg), b, dev)
        rot = (ro, Rt, bd)
    for k, v in t.items():
        n = k[len(pref):]
        if n.endswith(".weight_packed"):
            base = n[: -len(".weight_packed")]
            out, inn = [int(x) for x in t[pref + base + ".weight_shape"].tolist()]
            s = t[pref + base + ".weight_scale"].float()
            sh = torch.arange(0, 32, 4, dtype=torch.int32, device=v.device)
            q = ((v.unsqueeze(-1) >> sh) & 0xF).reshape(out, -1)[:, :inn].float() - 8
            Wd = q * s.repeat_interleave(inn // s.shape[1], 1)
            if rot is not None:
                ro, Rt, bd = rot
                c = ro.clase(base, bd)
                if c == "entrada":
                    g = 1 + t[pref + ro.norma_de(base) + ".weight"].float()
                    Wd = (Wd @ Rt.T) / ro.g_segura(g)[None, :]
                    Wd[:, g == 0] = 0
                elif c == "escribe":
                    Wd = Rt @ Wd
                elif c == "bajada":
                    Wd = ro.bloques(Rt @ Wd, bd)
            sd[base + ".weight"] = Wd.to(dt)
        elif n.endswith((".weight_scale", ".weight_shape")):
            continue
        else:
            sd[n] = v if n.endswith("A_log") else v.to(dt)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--ventanas", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--variante", default="", help="dir de etapa B: las capas salen de ahi (dequantizadas)")
    ap.add_argument("--solo", default="",
                    help="rango a-b: la variante solo en esas capas, el resto BF16 (para localizar)")
    ap.add_argument("--capas_de", type=int, default=-1,
                    help="guardar el flujo residual despues de cada capa para esta ventana (diagnostico)")
    ap.add_argument("--ref", default="", help="npz BF16 de referencia: calcula KL(BF16||este) sobre su top-K")
    ap.add_argument("--rot_down", type=int, default=0,
                    help="checkpoint servible rotado (armar_rot.py): Hadamard por bloques de N antes de down_proj")
    ap.add_argument("--a8", action="store_true",
                    help="activacion int8 por token en las lineales cuantizadas (W4A8; rotada si la variante lo es)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="dtype del computo y del residuo (produccion corre en fp16)")
    a = ap.parse_args()
    dev = "cuda"
    DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = Qwen3_5TextConfig(**cfg["text_config"])
    tc._attn_implementation = "sdpa"
    P = Pesos(a.modelo)
    ids = np.load(a.ventanas)
    N, L = ids.shape
    E = P.get("model.language_model.embed_tokens.weight", dev, DT)
    idt = torch.from_numpy(ids.astype(np.int64)).to(dev)
    h = E[idt]
    del E
    rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None]
    pe = rot(h[:1], pos)
    capas = [] if a.capas_de >= 0 else None
    for i in range(tc.num_hidden_layers):
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        pref = f"model.language_model.layers.{i}."
        lo, hi = (int(x) for x in a.solo.split("-")) if a.solo else (0, 10 ** 9)
        if a.variante and lo <= i <= hi:
            sd = capa_de_variante(os.path.join(a.variante, f"capa_{i:02d}.safetensors"), pref, dev, DT)
        else:
            sd = {k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else DT) for k in P.claves(pref)}
        capa.load_state_dict(sd, assign=True, strict=True)
        if a.rot_down:
            import rotacion as ro
            capa.mlp.down_proj.register_forward_pre_hook(
                lambda m_, a_: (ro.bloques(a_[0], a.rot_down),) + tuple(a_[1:]))
        if a.a8 and a.variante and lo <= i <= hi:
            from sensibilidad import a8_en
            a8_en(capa, os.path.join(a.variante, f"capa_{i:02d}.safetensors"), pref)
        with torch.no_grad():
            for s in range(N):
                o = capa(h[s:s + 1], position_embeddings=pe, attention_mask=None, position_ids=pos)
                h[s:s + 1] = o[0] if isinstance(o, tuple) else o
        del capa, sd
        if capas is not None:
            capas.append(h[a.capas_de].float().half().cpu().numpy())
        if i % 8 == 7:
            print(f"capa {i} lista", flush=True)
    if capas is not None:
        np.save(a.salida.replace(".npz", "_capas.npy"), np.stack(capas))
        print("capas guardadas", flush=True)
    norma = m.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
    norma.weight.data = P.get("model.language_model.norm.weight", dev, torch.float32)
    W = P.get("lm_head.weight", dev, DT)
    K = a.k
    top_ids = np.zeros((N, L - 1, K), np.int32)
    top_lp = np.zeros((N, L - 1, K), np.float16)
    lp_real = np.zeros((N, L - 1), np.float32)
    REF = np.load(a.ref) if a.ref else None
    kl = np.zeros((N, L - 1), np.float32)
    with torch.no_grad():
        for s in range(N):
            x = norma(h[s, :-1].float()).to(DT)
            lp = (x @ W.T).float().log_softmax(-1)
            v, ix = lp.topk(K, -1)
            top_ids[s], top_lp[s] = ix.cpu().numpy(), v.half().cpu().numpy()
            lp_real[s] = lp.gather(-1, idt[s, 1:, None]).squeeze(-1).cpu().numpy()
            if REF is not None:
                ri = torch.from_numpy(REF["top_ids"][s].astype(np.int64)).to(dev)
                rl = torch.from_numpy(REF["top_lp"][s].astype(np.float32)).to(dev)
                kl[s] = (rl.exp() * (rl - lp.gather(-1, ri))).sum(-1).cpu().numpy()
    np.savez(a.salida, top_ids=top_ids, top_lp=top_lp, lp_real=lp_real, kl=kl)
    if REF is not None:
        t1 = np.mean(top_ids[:, 16:, 0] == REF["top_ids"][:, 16:, 0])
        print(f"KL(BF16||este) desde pos 16: {kl[:, 16:].mean():.4f} (por ventana {' '.join(f'{x:.4f}' for x in kl[:, 16:].mean(1))}); "
              f"top-1 acuerdo {t1:.4f}; ppl {np.exp(-lp_real[:, 16:].mean()):.4f} vs BF16 {np.exp(-REF['lp_real'][:, 16:].mean()):.4f}", flush=True)
    print(f"referencia: {N} x {L - 1} posiciones, perplejidad {np.exp(-lp_real.mean()):.3f}, "
          f"top-1 = token real {np.mean(top_ids[:, :, 0] == ids[:, 1:]):.3f}", flush=True)


if __name__ == "__main__":
    main()
