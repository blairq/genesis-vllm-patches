#!/usr/bin/env python3
"""Ajuste fino del borrador DFlash2 contra noon, con los datos de captura_borrador.

Corre adentro de la imagen de vLLM (torch + CUDA + safetensors), en UNA 3090 y con el servidor
apagado. Ejemplo:

  docker run --rm --gpus '"device=0"' --ipc host -v <repo>:/repo -v <models-cache>:/models \\
      -v <trazas>:/traces vllm/vllm-openai:v0.29.0 --entrypoint python3 \\
      /repo/entrenamiento/borrador/entrenar.py --datos /traces/captura/entrena --salida /traces/borrador_ft

Que se entrena
--------------
LoRA (r=64) sobre q/k/v/o/gate/up/down de las 5 capas, mas las proyecciones del conv (chicas).
Congelados: fc (las features ya vienen proyectadas), hidden_norm, norm, el selector, y el
embed/lm_head del target.

La perdida
----------
Por cada posicion de mascara j = 1..K: 0,9 x TV + 0,1 x CE contra la distribucion top-16 de noon
en esa posicion (la de su regeneracion). TV conserva calibradas las alternativas, que es de lo que
vive el arbol; CE sola afila el top-1. Con --auf, la perdida se corta en la primera posicion donde
el top-1 del borrador falla (Spec-AUF, arXiv 2607.01893): lo que viene despues se descarta igual.

Evaluacion
----------
Sobre pedidos apartados (por pedido, no por ancla): largo de aceptacion greedy en cadena (cuantas
posiciones seguidas acierta el top-1), aceptacion por posicion, y el recall del top-16 (que el
token de noon este entre los 16 candidatos), que es lo que usa el arbol. Se mide ANTES de entrenar
y despues de cada epoca, en los mismos pedidos.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dflash2_torch import cargar_borrador, cargar_target  # noqa: E402


# ───────────────────────────── datos ─────────────────────────────

def cargar_pedido(d):
    fs = sorted(f for f in glob.glob(os.path.join(d, "*.npz")) if not f.endswith("etiquetas.npz"))
    if not fs or not os.path.exists(os.path.join(d, "etiquetas.npz")):
        return None
    zs = [np.load(f) for f in fs]
    pos = np.concatenate([z["pos"] for z in zs])
    orden = np.argsort(pos, kind="stable")
    pos, ids = pos[orden], np.concatenate([z["ids"] for z in zs])[orden]
    feat = np.concatenate([z["feat"] for z in zs])[orden]
    if len(pos) < 2 or not (np.diff(pos) == 1).all():
        return None
    e = np.load(os.path.join(d, "etiquetas.npz"))
    L, C = int(e["prompt_len"]), e["comp_ids"]
    fin = L + len(C)
    if pos[-1] != fin - 1 or not np.array_equal(ids[pos >= L], C[: (pos >= L).sum()]):
        return None
    return {"pos": pos, "ids": ids, "feat": feat, "L": L, "C": C,
            "top_ids": e["top_ids"], "top_lp": e["top_lp"]}


def anclas_de(p, K):
    """Anclas a con bonus = token en a (de la respuesta) y al menos una mascara con etiqueta."""
    L, n = p["L"], len(p["C"])
    return list(range(L, L + n - 1))


def armar_lote(p, anclas, K, W, mask_id, dev):
    pos0 = int(p["pos"][0])
    L = p["L"]
    B, T = len(anclas), K + 1
    b_ids = np.full((B, T), mask_id, np.int64)
    b_pos = np.zeros((B, T), np.int64)
    idx = np.full((B, W), -1, np.int64)
    lab = np.full((B, K), -1, np.int64)
    top_i = np.zeros((B, K, 16), np.int64)          # sin etiqueta: indice 0 con probabilidad 0
    top_p = np.zeros((B, K, 16), np.float32)
    for r, a in enumerate(anclas):
        b_ids[r, 0] = p["ids"][a - pos0]
        b_pos[r] = np.arange(a, a + T)
        lo = max(pos0, a - W)
        n = a - lo
        idx[r, W - n:] = np.arange(lo - pos0, a - pos0)
        for j in range(1, T):
            ci = a + j - L                           # indice en la respuesta de la posicion a+j
            if ci < len(p["C"]):
                lab[r, j - 1] = p["C"][ci]
                ti, tl = p["top_ids"][ci], p["top_lp"][ci]
                pr = np.exp(tl - tl.max())
                pr[ti < 0] = 0
                top_i[r, j - 1] = np.where(ti < 0, 0, ti)
                top_p[r, j - 1] = pr / pr.sum() if pr.sum() > 0 else pr
    t = lambda x: torch.from_numpy(x).to(dev)
    return t(b_ids), t(b_pos), t(idx), t(lab), t(top_i), t(top_p)


# ───────────────────────────── LoRA ─────────────────────────────

class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b) * self.s).to(x.dtype)

    @torch.no_grad()
    def fusionar(self):
        w = self.base.weight
        w.copy_((w.float() + (self.b @ self.a) * self.s).to(w.dtype))
        return self.base


def poner_lora(m, r, alpha, conv=False):
    """LoRA en las 7 lineales de cada capa. Las proyecciones del conv (rms 0,025) quedan
    congeladas salvo con conv=True: Adam las mueve ~lr por paso sin importar su escala, y a 3e-4
    siete pasos bastaron para bajar el largo greedy de 4,2 a 2,8 en la prueba."""
    ps = []
    for capa in m.layers:
        for nom in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            lo = LoRA(getattr(capa, nom), r, alpha)
            setattr(capa, nom, lo)
            ps += [lo.a, lo.b]
        for nom in ("attention_conv_proj", "mlp_conv_proj") if conv else ():
            getattr(capa, nom).float()                   # en bf16 los pasos chicos se pierden
            getattr(capa, nom).weight.requires_grad_(True)
            ps.append(getattr(capa, nom).weight)
    return ps


# ───────────────────────────── perdida y metricas ─────────────────────────────

def perdida(logits, lab, top_i, top_p, auf):
    """logits [B, K, V] fp32. TV + CE contra el top-16 de noon; con AUF se corta en el primer fallo."""
    valido = lab >= 0
    lq = logits.log_softmax(-1)
    q_top = lq.gather(-1, top_i).exp()                                  # [B, K, 16]
    tv = 0.5 * ((top_p - q_top).abs().sum(-1) + (1 - q_top.sum(-1)).clamp(min=0))
    ce = -(top_p * lq.gather(-1, top_i)).sum(-1)
    peso = valido.float()
    if auf:
        with torch.no_grad():
            acierta = (logits.argmax(-1) == lab) & valido
            # posiciones hasta el primer fallo inclusive
            fallos_previos = torch.cumsum((~acierta).int(), -1) - (~acierta).int()
            peso = peso * (fallos_previos == 0).float()
    n = peso.sum().clamp(min=1)
    return ((0.9 * tv + 0.1 * ce) * peso).sum() / n


@torch.no_grad()
def metricas(logits, lab, acum):
    valido = lab >= 0
    top1 = logits.argmax(-1)
    acierta = (top1 == lab) & valido
    cad = torch.cumprod(acierta.int(), -1)                               # acierta seguido desde j=1
    top16 = logits.topk(16, -1).indices
    rec16 = ((top16 == lab[..., None]).any(-1) & valido)
    acum["n"] += int(valido[:, 0].sum())
    acum["cadena"] += cad.sum(0).double().cpu()
    acum["valido"] += valido.sum(0).double().cpu()
    acum["rec16"] += rec16.sum(0).double().cpu()
    acum["largo"] += float(cad.sum())


@torch.no_grad()
def evaluar(m, peds, K, W, lote, dev, max_anclas=0):
    m.eval()
    acum = {"n": 0, "cadena": torch.zeros(K, dtype=torch.float64), "valido": torch.zeros(K, dtype=torch.float64),
            "rec16": torch.zeros(K, dtype=torch.float64), "largo": 0.0}
    for d in peds:
        p = cargar_pedido(d)
        if p is None:
            continue
        feat = torch.from_numpy(p["feat"]).to(dev, torch.bfloat16)
        fpos = torch.from_numpy(p["pos"].astype(np.int64)).to(dev)
        an = anclas_de(p, K)
        if max_anclas:
            an = an[:max_anclas]
        for i in range(0, len(an), lote):
            b_ids, b_pos, idx, lab, _, _ = armar_lote(p, an[i:i + lote], K, W, m.mask_id, dev)
            with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16):
                h = m(b_ids, b_pos, feat, fpos, idx)
            metricas(m.logits(h[:, 1:]), lab, acum)
    n = max(acum["n"], 1)
    return {"anclas": acum["n"], "largo_greedy": 1 + acum["largo"] / n,
            "por_posicion": (acum["cadena"] / n).tolist(),
            "recall_top16": (acum["rec16"] / acum["valido"].clamp(min=1)).tolist()}


# ───────────────────────────── principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--borrador", default="/models/incoai-qwen3.8-27b-dflash2-bf16")
    ap.add_argument("--noon", default=None, help="snapshot de noon (embed y lm_head)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ventana", type=int, default=2048)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--epocas", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rango", type=int, default=64)
    ap.add_argument("--auf", action="store_true")
    ap.add_argument("--conv", action="store_true", help="entrenar tambien las proyecciones del conv")
    ap.add_argument("--apartar", type=float, default=0.05)
    ap.add_argument("--solo_evaluar", action="store_true")
    ap.add_argument("--max_pedidos", type=int, default=0)
    ap.add_argument("--max_anclas_eval", type=int, default=0)
    ap.add_argument("--dispositivo", default="cuda")
    ap.add_argument("--max_pasos", type=int, default=0, help="corta el entrenamiento (pruebas)")
    a = ap.parse_args()
    dev = a.dispositivo
    torch.manual_seed(0)
    random.seed(0)
    os.makedirs(a.salida, exist_ok=True)
    peds = sorted(d for d in glob.glob(os.path.join(a.datos, "*")) if os.path.isdir(d))
    if a.max_pedidos:
        peds = peds[: a.max_pedidos]
    rng = random.Random(0)
    orden = peds[:]
    rng.shuffle(orden)
    n_ap = max(1, int(len(orden) * a.apartar))
    evalp, trainp = sorted(orden[:n_ap]), orden[n_ap:]
    json.dump({"eval": evalp, "train": trainp}, open(os.path.join(a.salida, "particion.json"), "w"))
    print(f"{len(peds)} pedidos: {len(trainp)} para entrenar, {len(evalp)} apartados", flush=True)

    noon = a.noon or sorted(glob.glob("/models/hub/models--noon-at-cgn--Qwen3.8-27B-Uncensored-W4A16-AutoRound/snapshots/*"))[0]
    m, cfg, _ = cargar_borrador(a.borrador, device=dev)
    m.embed, m.lm_head = cargar_target(noon, device=dev)
    for q in m.parameters():
        q.requires_grad_(False)
    if dev.startswith("cuda"):
        print(f"cargado; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    base = evaluar(m, evalp, a.K, a.ventana, a.lote, dev, a.max_anclas_eval)
    print("BASE", json.dumps(base), flush=True)
    registro = {"base": base, "epocas": []}
    if a.solo_evaluar:
        json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
        return

    ps = poner_lora(m, a.rango, 2 * a.rango, conv=a.conv)
    opt = torch.optim.AdamW(ps, lr=a.lr, weight_decay=0.0)
    paso = 0
    for ep in range(a.epocas):
        m.train()
        rng.shuffle(trainp)
        t0 = time.time()
        for k, d in enumerate(trainp):
            p = cargar_pedido(d)
            if p is None:
                continue
            feat = torch.from_numpy(p["feat"]).to(dev, torch.bfloat16)
            fpos = torch.from_numpy(p["pos"].astype(np.int64)).to(dev)
            an = anclas_de(p, a.K)
            rng.shuffle(an)
            for i in range(0, len(an), a.lote):
                b_ids, b_pos, idx, lab, top_i, top_p = armar_lote(p, an[i:i + a.lote], a.K, a.ventana, m.mask_id, dev)
                with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16):
                    h = m(b_ids, b_pos, feat, fpos, idx)
                logits = m.logits(h[:, 1:])
                loss = perdida(logits, lab, top_i, top_p, a.auf)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                opt.step()
                paso += 1
                if a.max_pasos and paso >= a.max_pasos:
                    break
                if paso % 50 == 0 or a.max_pasos:
                    print(f"ep {ep} ped {k}/{len(trainp)} paso {paso} loss {loss.item():.4f} "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if a.max_pasos and paso >= a.max_pasos:
                break
        ev = evaluar(m, evalp, a.K, a.ventana, a.lote, dev, a.max_anclas_eval)
        print(f"EPOCA {ep}", json.dumps(ev), flush=True)
        registro["epocas"].append(ev)
        json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)

    # Fusionar LoRA y exportar con los nombres del checkpoint original (para GPTQ y vLLM).
    guardar(m, a.borrador, a.salida)


@torch.no_grad()
def guardar(m, dir_orig, salida):
    from safetensors.torch import load_file, save_file
    sd = load_file(os.path.join(dir_orig, "model.safetensors"))
    inv = {"q_proj": "self_attn.q_proj.weight", "k_proj": "self_attn.k_proj.weight",
           "v_proj": "self_attn.v_proj.weight", "o_proj": "self_attn.o_proj.weight",
           "gate_proj": "mlp.gate_proj.weight", "up_proj": "mlp.up_proj.weight", "down_proj": "mlp.down_proj.weight",
           "attention_conv_proj": "attention_conv.kernel_projection.weight",
           "mlp_conv_proj": "mlp_conv.kernel_projection.weight"}
    for i, capa in enumerate(m.layers):
        for nom, clave in inv.items():
            mod = getattr(capa, nom)
            if isinstance(mod, LoRA):
                mod = mod.fusionar()
            sd[f"layers.{i}.{clave}"] = mod.weight.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(sd, os.path.join(salida, "model.safetensors"))
    import shutil
    for f in ("config.json", "README.md"):
        if os.path.exists(os.path.join(dir_orig, f)):
            shutil.copy(os.path.join(dir_orig, f), salida)
    print("guardado en", salida, flush=True)


if __name__ == "__main__":
    main()
