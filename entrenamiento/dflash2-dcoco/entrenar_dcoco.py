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
import evaluacion as EV  # noqa: E402
import recurrente as RC  # noqa: E402


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


def perdida_dcoco(sc, cand, lab, rho=4.0, modo="fila", prefijo=False):
    """``prefijo``: la transicion p-1 -> p cuenta solo si TODO el prefijo real 0..p esta en el top-K (la
    cabeza recurrente arma el estado con esos candidatos)."""
    B, S, K = cand.shape
    idx, esta = indices_reales(cand, lab)
    if prefijo:
        esta = torch.cumprod(esta.int(), -1).bool()
    valido = (esta[:, 1:] & esta[:, :-1]).float()                                   # [B, S-1]
    if modo == "fila":
        fila = sc[:, 1:].gather(2, idx[:, :-1, None, None].expand(-1, -1, 1, K)).squeeze(2)
        lp = fila.log_softmax(-1).gather(-1, idx[:, 1:, None]).squeeze(-1)
    else:
        lp = sc[:, 1:].flatten(2).log_softmax(-1).gather(-1, (idx[:, :-1] * K + idx[:, 1:])[..., None]).squeeze(-1)
    w = torch.exp(-torch.arange(S - 1, device=sc.device, dtype=torch.float32) / rho)
    pw = valido * w
    return -(lp * pw).sum() / pw.sum().clamp(min=1)


def ce_forzada(sc, cand, lab):
    """Diagnostico: suma y cantidad de -log p(real en p | real en p-1) sobre la fila (sin peso por
    profundidad), y cuantas veces el real es el argmax de su fila."""
    B, S, K = cand.shape
    idx, esta = indices_reales(cand, lab)
    valido = esta[:, 1:] & esta[:, :-1]
    fila = sc[:, 1:].gather(2, idx[:, :-1, None, None].expand(-1, -1, 1, K)).squeeze(2).float()
    lp = fila.log_softmax(-1).gather(-1, idx[:, 1:, None]).squeeze(-1)
    top1 = fila.argmax(-1) == idx[:, 1:]
    return float(-(lp * valido).sum()), int(valido.sum()), int((top1 & valido).sum())


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
    """Sumas por metrica. Las que tienen ``por_posicion`` son cadenas golosas."""

    def __init__(self, S):
        self.S = S
        self.n = 0
        self.suma = {}
        self.pos = {}
        self.ce = {}
        self.rec16 = torch.zeros(S, dtype=torch.float64)
        self.valido = torch.zeros(S, dtype=torch.float64)

    def sumar(self, nombre, v, por_posicion=False):
        self.suma[nombre] = self.suma.get(nombre, 0.0) + float(v.sum())
        if por_posicion:
            self.pos[nombre] = self.pos.get(nombre, torch.zeros(self.S, dtype=torch.float64)) + v.sum(0).double().cpu()

    def resumen(self):
        n = max(self.n, 1)
        r = {"anclas": self.n}
        r.update({f"largo_{k}": v / n for k, v in self.suma.items()})
        r.update({f"por_posicion_{k}": (v / n).tolist() for k, v in self.pos.items()})
        for k, (suma, cnt, top1) in self.ce.items():
            r[f"ce_fila_{k}"] = suma / max(cnt, 1)
            r[f"top1_fila_{k}"] = top1 / max(cnt, 1)
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
            if isinstance(cab, RC.CabezaRecurrente):
                sc2 = cab.tabla_forzada(sc, emb, val, hs, indices_reales(cand, lab)[0])
            else:
                sc2 = cab(sc, emb, val, hs)
        return dict(lab=lab, top_i=top_i, top_p=top_p, logits=logits, cand=cand, sc=sc, sc2=sc2.float(),
                    emb=emb, val=val, hs=hs)


def _puntuadores(cab, r):
    """(selector, dcoco) para evaluacion.cadena / arbol. Con la cabeza recurrente, las filas de dcoco se
    calculan sobre el camino elegido (no la tabla forzada por el maestro, que ve las etiquetas)."""
    sel = EV.Tabla(r["sc"])
    if isinstance(cab, RC.CabezaRecurrente):
        return sel, RC.Puntuador(cab, r["sc"], r["emb"], r["val"], r["hs"])
    return sel, EV.Tabla(r["sc2"])


def medir_lote(ac, cab, r, a, escala=None):
    lab, cand = r["lab"], r["cand"]
    ac.n += int((lab[:, 0] >= 0).sum())
    p_sel, p_dc = _puntuadores(cab, r)
    cadenas = {"unario": cand[..., 0], "selector": EV.cadena(p_sel, cand), "dcoco": EV.cadena(p_dc, cand)}
    arboles = {"arbol_selector": EV.arbol(p_sel, cand, a.nodos), "arbol_dcoco": EV.arbol(p_dc, cand, a.nodos, escala)}
    for k, tok in cadenas.items():
        ac.sumar(k, EV.aceptados_cadena(tok, lab), por_posicion=True)
    for k, (tok, pa, pr) in arboles.items():
        ac.sumar(k, EV.aceptados_arbol(tok, pa, pr, lab))
    for T in a.temperaturas:
        pT = EV.dist_muestreo(r["top_p"], T, a.nucleo)
        for k, tok in cadenas.items():
            ac.sumar(f"muestreo_T{T:g}_{k}", EV.muestreo_cadena(tok, r["top_i"], pT))
        for k, (tok, pa, pr) in arboles.items():
            ac.sumar(f"muestreo_T{T:g}_{k}", EV.muestreo_arbol(tok, pa, pr, r["top_i"], pT))
    for k, t in (("selector", r["sc"]), ("dcoco", r["sc2"])):
        v = ce_forzada(t, cand, lab)
        ac.ce[k] = [x + y for x, y in zip(ac.ce.get(k, (0.0, 0, 0)), v)]
    _, esta = indices_reales(cand, lab)
    ac.rec16 += esta.sum(0).double().cpu()
    ac.valido += (lab >= 0).sum(0).double().cpu()


@torch.no_grad()
def evaluar(bor, cab, peds, a, dev, escala=None, cache=None):
    """Metricas sobre los pedidos apartados. Con ``cache`` (lista) guarda lo que necesita la calibracion."""
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
            medir_lote(ac, cab, r, a, escala)
            if cache is not None:
                c = {k: r[k].cpu() for k in ("lab", "cand", "sc", "top_i", "top_p")}
                if isinstance(cab, RC.CabezaRecurrente):
                    c["pe"] = cab.pe(r["emb"], r["hs"]).half().cpu()
                    c["val"] = r["val"].float().cpu()
                else:
                    c["sc2"] = r["sc2"].cpu()
                cache.append(c)
    return ac.resumen()


def calibrar_arbol(cab, cache, a, dev):
    """Escala por profundidad del arbol corregido (paso 3 del README): se busca en la mitad par de los lotes
    apartados y se informa en la impar, antes y despues."""
    T = a.calibrar_T if a.calibrar_T is not None else (a.temperaturas[0] if a.temperaturas else 1.0)

    def objetivo(c, escala):
        r = {k: v.to(dev) for k, v in c.items()}
        if isinstance(cab, RC.CabezaRecurrente):
            punt = RC.Puntuador.desde(cab, r["sc"], r["pe"].float(), r["val"])
        else:
            punt = EV.Tabla(r["sc2"])
        tok, pa, pr = EV.arbol(punt, r["cand"], a.nodos, escala)
        if a.calibrar == "muestreo":
            pT = EV.dist_muestreo(r["top_p"], T, a.nucleo)
            return float(EV.muestreo_arbol(tok, pa, pr, r["top_i"], pT).sum())
        return float(EV.aceptados_arbol(tok, pa, pr, r["lab"]).sum())

    with torch.no_grad():
        busq, control = cache[0::2], cache[1::2]
        n_ctl = max(1, sum(int((c["lab"][:, 0] >= 0).sum()) for c in control))
        antes = sum(objetivo(c, None) for c in control) / n_ctl
        esc = EV.calibrar(busq, objetivo, a.K, log=lambda m: print(m, flush=True))
        despues = sum(objetivo(c, esc) for c in control) / n_ctl
    obj = "muestreo T=%g" % T if a.calibrar == "muestreo" else "voraz"
    print(f"CALIBRACION ({obj}) en la mitad de control: arbol dcoco {antes:.4f} -> {despues:.4f}", flush=True)
    return esc, {"objetivo": obj, "control_antes": antes, "control_despues": despues, "escala": esc}


def contar_anclas(d):
    """Anclas de un pedido sin cargar sus features (E.anclas_de: una por token de la respuesta menos uno)."""
    f = os.path.join(d, "etiquetas.npz")
    return max(0, len(np.load(f)["comp_ids"]) - 1) if os.path.exists(f) else 0


# ───────────────────────────── exportar ─────────────────────────────

@torch.no_grad()
def guardar(bor, cab, a, hp, escala=None):
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
                               "dcoco_state_dim": hp["state"], "dcoco_mlp_dim": hp["mlp"],
                               "dcoco_estado": a.estado})
    if escala is not None:
        c["dflash_config"]["dcoco_escala_arbol"] = [float(x) for x in escala]
    c["dcoco"] = {"metodo": "DCOCO residual sobre el CandidateSelector (arXiv 2610.12570)",
                  "modo": a.modo, "estado": a.estado, "perdida": a.perdida, "rho": a.rho, "vocab": a.vocab,
                  "conv_int8": a.conv_int8, "calibracion": a.calibrar, "entrenada": not a.solo_evaluar}
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
    ap.add_argument("--estado", choices=("pares", "recurrente"), default="pares",
                    help="pares: la cabeza de PN178; recurrente: estado GRU del camino elegido (solo entrenamiento)")
    ap.add_argument("--temperaturas", type=float, nargs="*", default=[],
                    help="ademas de la golosa, aceptacion esperada con muestreo a estas temperaturas")
    ap.add_argument("--nucleo", type=float, default=1.0, help="top-p del muestreo en las metricas (1 = sin corte)")
    ap.add_argument("--calibrar", choices=("no", "voraz", "muestreo"), default="no",
                    help="escala por profundidad del arbol, buscada en los pedidos apartados; va al checkpoint")
    ap.add_argument("--calibrar_T", type=float, default=None, help="temperatura de --calibrar muestreo [la primera]")
    ap.add_argument("--quedarse", choices=("mejor", "ultima"), default="mejor",
                    help="que se exporta: la mejor epoca por --criterio (la base cuenta) o la ultima")
    ap.add_argument("--criterio", default="largo_arbol_dcoco",
                    help="metrica de la evaluacion para --quedarse mejor (p. ej. largo_muestreo_T0.6_arbol_dcoco)")
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
    if a.estado == "recurrente":
        cab = RC.CabezaRecurrente(H, bor.K, **hp).to(dev)
    else:
        cab = D.CabezaDCOCO(H, bor.K, **hp).to(dev)
    print(f"cabeza DCOCO: {sum(p.numel() for p in cab.parameters()) / 1e6:.2f} M parametros; K={bor.K}, "
          f"escala {bor.escala}, soft cap {bor.softcap}", flush=True)

    base = evaluar(bor, cab, evalp, a, dev)
    print("BASE", json.dumps(base), flush=True)
    registro = {"args": vars(a), "base": base, "epocas": []}
    json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
    if a.solo_evaluar:
        if a.calibrar != "no":
            terminar(bor, cab, evalp, a, dev, hp, registro)                         # checkpoint solo calibrado
        return

    grupos = [{"params": list(cab.parameters()), "lr": a.lr}]
    if a.modo == "conjunto":
        grupos.append({"params": E.poner_lora(bor.m, a.rango, 2 * a.rango), "lr": a.lr_lora})
    opt = torch.optim.AdamW(grupos, weight_decay=0.0)
    entrenables = [q for g in grupos for q in g["params"]]
    if a.criterio not in base:
        raise SystemExit(f"--criterio {a.criterio}: no es una metrica de la evaluacion ({sorted(base)})")
    mejor = (base[a.criterio], -1, [q.detach().clone() for q in entrenables])
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
        suma_dc = suma_sel = 0.0
        n_pasos = 0
        for k, d in enumerate(trainp):
            p = E.cargar_pedido(d)
            if p is None:
                continue
            an = E.anclas_de(p, a.K)
            rng.shuffle(an)
            for i in range(0, len(an), a.lote):
                r = bor.paso(p, an[i:i + a.lote], a, dev, cab, a.modo == "conjunto")
                loss = perdida_dcoco(r["sc2"], r["cand"], r["lab"], a.rho, a.perdida, a.estado == "recurrente")
                with torch.no_grad():
                    suma_sel += float(perdida_dcoco(r["sc"], r["cand"], r["lab"], a.rho, a.perdida,
                                                    a.estado == "recurrente"))
                suma_dc += float(loss.detach())
                n_pasos += 1
                if a.modo == "conjunto":
                    loss = E.perdida(r["logits"], r["lab"], r["top_i"], r["top_p"], a.auf) + a.peso_dcoco * loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(entrenables, 1.0)
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
        ev["perdida_entreno_dcoco"] = suma_dc / max(n_pasos, 1)
        ev["perdida_entreno_selector"] = suma_sel / max(n_pasos, 1)
        print(f"EPOCA {ep}", json.dumps(ev), flush=True)
        registro["epocas"].append(ev)
        if a.quedarse == "mejor" and ev[a.criterio] > mejor[0]:
            mejor = (ev[a.criterio], ep, [q.detach().clone() for q in entrenables])
        json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
        if a.max_pasos and paso >= a.max_pasos:
            break

    if a.quedarse == "mejor":
        with torch.no_grad():
            for q, v in zip(entrenables, mejor[2]):
                q.copy_(v)
        cual = "la base (ninguna epoca la supera: la cabeza queda en cero)" if mejor[1] < 0 else f"la epoca {mejor[1]}"
        print(f"MEJOR por {a.criterio}: {cual}, {mejor[0]:.4f}", flush=True)
        registro["exportada"] = mejor[1]
    terminar(bor, cab, evalp, a, dev, hp, registro)


def terminar(bor, cab, evalp, a, dev, hp, registro):
    """Calibra el arbol si se pidio, mide con la escala y exporta."""
    escala = None
    if a.calibrar != "no":
        cache = []
        evaluar(bor, cab, evalp, a, dev, cache=cache)
        escala, info = calibrar_arbol(cab, cache, a, dev)
        ev = evaluar(bor, cab, evalp, a, dev, escala=escala)
        print("CALIBRADO", json.dumps(ev), flush=True)
        registro["calibracion"] = dict(info, metricas=ev)
        json.dump(registro, open(os.path.join(a.salida, "registro.json"), "w"), indent=1)
    if a.estado == "recurrente":
        print("AVISO: la cabeza recurrente se exporta, pero PN178 todavia no la sirve", flush=True)
    guardar(bor, cab, a, hp, escala)


if __name__ == "__main__":
    main()
