#!/usr/bin/env python3
"""Entrena la cabeza DCOCO (PN178) del borrador DFlash2 con las capturas de dflash2.sh.

Corre en la imagen de vLLM en UNA 3090, con el servidor apagado, igual que dflash2/entrenar.py:

  docker run --rm --gpus '"device=0"' --ipc host -v <repo>:/repo:ro -v <models-cache>:/models:ro \\
      -v <trazas>:/traces --entrypoint python3 vllm/vllm-openai:v0.29.0 \\
      /repo/entrenamiento/dflash2-dcoco/entrenar_dcoco.py --datos /traces/captura/entrena_rot \\
      --borrador /traces/borrador_prod_fp16 --noon /models/qwen3.8_27b_idiotSavant_sm_86 \\
      --salida /traces/borrador_dcoco --conv_int8 --vocab 65536

Que se entrena
--------------
La cabeza de ``vllm/_genesis/dcoco.py`` (~2-3 M de parametros) como correccion residual de la tabla del
selector, que queda congelado. Dos modos:

* ``cabeza`` (default): el borrador congelado; solo la cabeza. Barato (el borrador va sin gradiente).
* ``conjunto``: ademas LoRA sobre el borrador con la perdida de dflash2/entrenar.py (0,9 TV + 0,1 CE,
  AUF opcional). En el paper, entrenar el borrador junto con el corrector da bastante mas que
  congelarlo (tabla de ablaciones: -16,5% de largo aceptado con el borrador congelado).

La perdida de DCOCO
-------------------
Forzado por el maestro: para cada transicion p-1 -> p se toma como previo el candidato que es el token
real en p-1 y se pide el token real en p. ``--perdida fila`` (default): entropia cruzada sobre la fila
del previo (softmax sobre el siguiente), que es como arbol_borrador.construir usa la tabla.
``--perdida conjunta``: la del paper, softmax sobre las K x K transiciones juntas. Peso e^{-(p-1)/rho}
con rho = 4; las transiciones con algun token real fuera del top-K no cuentan.

Evaluacion
----------
Sobre los mismos pedidos apartados que dflash2/entrenar.py (misma semilla y --apartar): largo aceptado
(sin el bonus) en cadena con el top-1 unario, con el recorrido del selector y con el de DCOCO, y con el
arbol de --nodos nodos (arbol_borrador.construir) sobre la tabla del selector y sobre la corregida;
aceptacion por posicion y recall del top-16. Antes de entrenar y despues de cada epoca.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "entrenamiento", "dflash2"))
sys.path.insert(0, AQUI)

import entrenar as E  # noqa: E402  (dflash2/entrenar.py: datos, LoRA, perdida del borrador, guardar)
from dflash2_torch import cargar_borrador, cargar_target  # noqa: E402
from selector import candidatos, cargar_selector, procesador  # noqa: E402


def _modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# La cuenta de la cabeza y el constructor del arbol son los del servidor (solo dependen de torch).
D = _modulo("g178_dcoco", os.path.join(REPO, "vllm", "_genesis", "dcoco.py"))
AB = _modulo("g_arbol_borrador", os.path.join(REPO, "vllm", "_genesis", "arbol_borrador.py"))


# ───────────────────────────── perdida ─────────────────────────────

def indices_reales(cand, lab):
    """cand [B, S, K], lab [B, S] -> (indice del token real entre los candidatos [B, S], esta [B, S])."""
    igual = cand == lab[..., None]
    return igual.float().argmax(-1), igual.any(-1) & (lab >= 0)


def perdida_dcoco(sc, cand, lab, rho=4.0, modo="fila"):
    B, S, K = cand.shape
    idx, esta = indices_reales(cand, lab)
    valido = (esta[:, 1:] & esta[:, :-1]).float()                                   # [B, S-1]
    if modo == "fila":
        fila = sc[:, 1:].gather(2, idx[:, :-1, None, None].expand(-1, -1, 1, K)).squeeze(2)
        lp = fila.log_softmax(-1).gather(-1, idx[:, 1:, None]).squeeze(-1)
    else:
        lp = sc[:, 1:].flatten(2).log_softmax(-1).gather(-1, (idx[:, :-1] * K + idx[:, 1:])[..., None]).squeeze(-1)
    w = torch.exp(-torch.arange(S - 1, device=sc.device, dtype=torch.float32) / rho)
    pw = valido * w
    return -(lp * pw).sum() / pw.sum().clamp(min=1)


# ───────────────────────────── metricas ─────────────────────────────

def recorrido(sc, cand):
    """Cadena golosa prefijo-primero sobre la tabla: tokens [B, S]."""
    B, S, K = cand.shape
    i = sc[:, 0, 0].argmax(-1)
    idx = [i]
    for s in range(1, S):
        i = sc[:, s].gather(1, i[:, None, None].expand(-1, 1, K)).squeeze(1).argmax(-1)
        idx.append(i)
    return cand.gather(2, torch.stack(idx, 1)[..., None]).squeeze(-1)


def aceptados_cadena(tok, lab):
    ok = (tok == lab) & (lab >= 0)
    return torch.cumprod(ok.int(), -1)                                              # [B, S] 1 mientras acierta


def aceptados_arbol(sc, cand, lab, n):
    """Largo aceptado verificando el arbol best-first de n nodos (sin el bonus): [B]."""
    tok, padre, prof, _ = AB.construir(cand, sc, n)
    B = cand.shape[0]
    ok = torch.zeros(B, n, dtype=torch.bool, device=cand.device)
    for j in range(n):
        real = lab.gather(1, prof[:, j:j + 1]).squeeze(1)
        p = padre[:, j]
        previo = torch.where(p < 0, torch.ones_like(p, dtype=torch.bool),
                             ok.gather(1, p.clamp(min=0)[:, None]).squeeze(1))
        ok[:, j] = (tok[:, j] == real) & (real >= 0) & previo
    return torch.where(ok, prof + 1, torch.zeros_like(prof)).max(1).values


class Acum:
    def __init__(self, S):
        self.n = 0
        self.suma = {k: 0.0 for k in ("unario", "selector", "dcoco", "arbol_selector", "arbol_dcoco")}
        self.pos = {k: torch.zeros(S, dtype=torch.float64) for k in ("unario", "selector", "dcoco")}
        self.rec16 = torch.zeros(S, dtype=torch.float64)
        self.valido = torch.zeros(S, dtype=torch.float64)

    def sumar(self, nombre, cad):
        self.suma[nombre] += float(cad.sum())
        if nombre in self.pos:
            self.pos[nombre] += cad.sum(0).double().cpu()

    def resumen(self):
        n = max(self.n, 1)
        r = {"anclas": self.n}
        r.update({f"largo_{k}": v / n for k, v in self.suma.items()})
        r.update({f"por_posicion_{k}": (v / n).tolist() for k, v in self.pos.items()})
        r["recall_top16"] = (self.rec16 / self.valido.clamp(min=1)).tolist()
        return r


# ───────────────────────────── paso comun ─────────────────────────────

class Borrador:
    """El DFlash2 de dflash2/ mas lo que el servidor hace despues del lm_head (candidatos y selector)."""

    def __init__(self, a, dev):
        self.m, self.cfg, _ = cargar_borrador(a.borrador, device=dev)
        self.m.embed, self.m.lm_head = cargar_target(a.noon, device=dev)
        for q in self.m.parameters():
            q.requires_grad_(False)
        if a.conv_int8:
            E.poner_conv_int8(self.m, a.conv_grupo)
        if a.vocab:
            E.recortar_vocab(self.m, a.vocab_orden, a.vocab, dev)
        self.sel, _ = cargar_selector(a.borrador, dev)
        self.escala, self.softcap = procesador(self.cfg)
        self.K = self.sel.top_k

    def paso(self, p, anclas, a, dev, cab, con_grad_borrador):
        feat = torch.from_numpy(p["feat"]).to(dev, torch.bfloat16)
        fpos = torch.from_numpy(p["pos"].astype(np.int64)).to(dev)
        b_ids, b_pos, idx, lab, top_i, top_p = E.armar_lote(p, anclas, a.K, a.ventana, self.m.mask_id, dev)
        with torch.set_grad_enabled(con_grad_borrador), torch.autocast(torch.device(dev).type, dtype=torch.bfloat16):
            h = self.m(b_ids, b_pos, feat, fpos, idx)
        hs = h[:, 1:]                                                               # [B, S, H] posiciones de mascara
        logits = self.m.logits(hs)                                                  # [B, S, V] fp32
        cand, val = candidatos(logits, self.K, self.escala, self.softcap)
        if not con_grad_borrador:
            hs, val = hs.detach(), val.detach()
        sc = self.sel(cand, val, hs, b_ids[:, 0])
        emb = self.m.embed[cand]                                                    # base original (como PN149)
        with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16):
            sc2 = cab(sc, emb, val, hs)
        return dict(lab=lab, top_i=top_i, top_p=top_p, logits=logits, cand=cand, sc=sc, sc2=sc2.float())


@torch.no_grad()
def evaluar(bor, cab, peds, a, dev):
    cab.eval()
    bor.m.eval()
    ac = Acum(a.K)
    for d in peds:
        p = E.cargar_pedido(d)
        if p is None:
            continue
        an = E.anclas_de(p, a.K)
        if a.max_anclas_eval:
            an = an[: a.max_anclas_eval]
        for i in range(0, len(an), a.lote):
            r = bor.paso(p, an[i:i + a.lote], a, dev, cab, False)
            lab, cand = r["lab"], r["cand"]
            ac.n += int((lab[:, 0] >= 0).sum())
            ac.sumar("unario", aceptados_cadena(cand[..., 0], lab))
            ac.sumar("selector", aceptados_cadena(recorrido(r["sc"], cand), lab))
            ac.sumar("dcoco", aceptados_cadena(recorrido(r["sc2"], cand), lab))
            ac.suma["arbol_selector"] += float(aceptados_arbol(r["sc"], cand, lab, a.nodos).sum())
            ac.suma["arbol_dcoco"] += float(aceptados_arbol(r["sc2"], cand, lab, a.nodos).sum())
            _, esta = indices_reales(cand, lab)
            ac.rec16 += esta.sum(0).double().cpu()
            ac.valido += (lab >= 0).sum(0).double().cpu()
    return ac.resumen()


def contar_anclas(d):
    """Anclas de un pedido sin cargar sus features (E.anclas_de: una por token de la respuesta menos uno)."""
    f = os.path.join(d, "etiquetas.npz")
    return max(0, len(np.load(f)["comp_ids"]) - 1) if os.path.exists(f) else 0


# ───────────────────────────── exportar ─────────────────────────────

@torch.no_grad()
def guardar(bor, cab, a, hp):
    """Checkpoint DFlash2-DCOCO bf16: el del borrador (con la LoRA fusionada si hubo) mas ``dcoco.*`` y
    los hiperparametros en dflash_config. Despues va por dflash2/cuantizar_rtn.py, que deja la cabeza
    tal cual (no es una de sus lineales objetivo)."""
    from safetensors.torch import load_file, save_file
    os.makedirs(a.salida, exist_ok=True)
    if a.modo == "conjunto":
        E.guardar(bor.m, a.borrador, a.salida)                                      # fusiona la LoRA
        origen = a.salida
    else:
        origen = a.borrador
    sd = load_file(os.path.join(origen, "model.safetensors"))
    for n, t in cab.state_dict().items():
        sd[D.PREFIJO + n] = t.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(sd, os.path.join(a.salida, "model.safetensors"))
    c = json.load(open(os.path.join(a.borrador, "config.json")))
    c["dflash_config"].update({"dcoco_candidate_topk": bor.K, "dcoco_emb_dim": hp["emb"],
                               "dcoco_state_dim": hp["state"], "dcoco_mlp_dim": hp["mlp"]})
    c["dcoco"] = {"metodo": "DCOCO residual sobre el CandidateSelector (arXiv 2610.12570)",
                  "modo": a.modo, "perdida": a.perdida, "rho": a.rho, "vocab": a.vocab, "conv_int8": a.conv_int8}
    json.dump(c, open(os.path.join(a.salida, "config.json"), "w"), indent=2)
    print("guardado en", a.salida, flush=True)


# ───────────────────────────── principal ─────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datos", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--borrador", required=True, help="DFlash2 bf16/fp16 con el selector (p. ej. decuantizar.py de produccion)")
    ap.add_argument("--noon", required=True, help="target: embed y lm_head efectivos (con la rotacion de PN149)")
    ap.add_argument("--modo", choices=("cabeza", "conjunto"), default="cabeza")
    ap.add_argument("--perdida", choices=("fila", "conjunta"), default="fila")
    ap.add_argument("--rho", type=float, default=4.0)
    ap.add_argument("--peso_dcoco", type=float, default=1.0, help="en modo conjunto, peso de la perdida de DCOCO")
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--state", type=int, default=128)
    ap.add_argument("--mlp", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ventana", type=int, default=2048)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4, help="cabeza (el paper: 6e-4, coseno, 4%% de calentamiento)")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--rango", type=int, default=64)
    ap.add_argument("--auf", action="store_true")
    ap.add_argument("--nodos", type=int, default=8, help="nodos del arbol en la evaluacion (el de produccion: 8)")
    ap.add_argument("--apartar", type=float, default=0.03, help="igual que dflash2.sh: mismos pedidos apartados")
    ap.add_argument("--solo_evaluar", action="store_true")
    ap.add_argument("--max_pedidos", type=int, default=0)
    ap.add_argument("--max_anclas_eval", type=int, default=0)
    ap.add_argument("--max_pasos", type=int, default=0)
    ap.add_argument("--dispositivo", default="cuda")
    ap.add_argument("--conv_int8", action="store_true")
    ap.add_argument("--conv_grupo", type=int, default=-1)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--vocab_orden", default=os.path.join(REPO, "vllm", "_genesis", "datos", "pn159_orden.npy"))
    a = ap.parse_args()
    dev = a.dispositivo
    torch.manual_seed(0)
    os.makedirs(a.salida, exist_ok=True)

    # misma particion que dflash2/entrenar.py
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

    bor = Borrador(a, dev)
    hp = dict(emb=a.emb, state=a.state, mlp=a.mlp)
    H = bor.m.c["hidden_size"]
    cab = D.CabezaDCOCO(H, bor.K, **hp).to(dev)
    print(f"cabeza DCOCO: {sum(p.numel() for p in cab.parameters()) / 1e6:.2f} M parametros; K={bor.K}, "
          f"escala {bor.escala}, soft cap {bor.softcap}", flush=True)

    base = evaluar(bor, cab, evalp, a, dev)
    print("BASE", json.dumps(base), flush=True)
    registro = {"args": vars(a), "base": base, "epocas": []}
    json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
    if a.solo_evaluar:
        return

    grupos = [{"params": list(cab.parameters()), "lr": a.lr}]
    if a.modo == "conjunto":
        grupos.append({"params": E.poner_lora(bor.m, a.rango, 2 * a.rango), "lr": a.lr_lora})
    opt = torch.optim.AdamW(grupos, weight_decay=0.0)
    total = a.max_pasos or sum(math.ceil(n / a.lote) for n in map(contar_anclas, trainp) if n) * a.epocas
    calor = max(1, int(0.04 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / calor) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    paso = 0
    for ep in range(a.epocas):
        cab.train()
        if a.modo == "conjunto":
            bor.m.train()
        rng.shuffle(trainp)
        t0 = time.time()
        for k, d in enumerate(trainp):
            p = E.cargar_pedido(d)
            if p is None:
                continue
            an = E.anclas_de(p, a.K)
            rng.shuffle(an)
            for i in range(0, len(an), a.lote):
                r = bor.paso(p, an[i:i + a.lote], a, dev, cab, a.modo == "conjunto")
                loss = perdida_dcoco(r["sc2"], r["cand"], r["lab"], a.rho, a.perdida)
                if a.modo == "conjunto":
                    loss = E.perdida(r["logits"], r["lab"], r["top_i"], r["top_p"], a.auf) + a.peso_dcoco * loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([q for g in grupos for q in g["params"]], 1.0)
                opt.step()
                sched.step()
                paso += 1
                if paso % 50 == 0 or a.max_pasos:
                    print(f"ep {ep} ped {k}/{len(trainp)} paso {paso}/{total} loss {loss.item():.4f} "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)
                if a.max_pasos and paso >= a.max_pasos:
                    break
            if a.max_pasos and paso >= a.max_pasos:
                break
        ev = evaluar(bor, cab, evalp, a, dev)
        print(f"EPOCA {ep}", json.dumps(ev), flush=True)
        registro["epocas"].append(ev)
        json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
        if a.max_pasos and paso >= a.max_pasos:
            break

    guardar(bor, cab, a, hp)


if __name__ == "__main__":
    main()
