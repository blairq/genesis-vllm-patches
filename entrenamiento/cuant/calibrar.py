#!/usr/bin/env python3
"""ETAPA A de la cuantizacion propia: pasar la calibracion por el modelo BF16, capa por capa, y
dejar en disco TODO lo que las variantes necesitan, para no volver a correr el modelo nunca mas.

Por que capa por capa: el BF16 pesa 55 GB y la maquina tiene 32 GB de RAM. Se carga UNA capa por
vez (tensor por tensor, del safetensors al GPU), se pasa la calibracion y se descarta.

Que guarda por capa (en <salida>/capa_NN/):
  H_<entrada>.pt      Hessiana X^T X en fp32, solo el triangulo superior (la de down_proj pesa
                      0,6 GB asi en vez de 1,2). Con ella sale CUALQUIER variante GPTQ (bits, grupo,
                      simetria), y tambien las que cambian la entrada: suavizado (H' = S^-1 H S^-1)
                      o rotacion (H' = R^T H R), sin recalibrar.
  stats_<entrada>.pt  por canal: max|x|, suma de x^2, y max|x| por grupo de 128 (para suavizado por
                      grupo y escalas estaticas), mas el numero de tokens.
  muestra.pt          entrada y salida de la capa para las primeras M muestras (error local de una
                      variante de la capa entera, sin correr el resto del modelo).
  qkv.pt              (solo atencion) Q, K, V despues de las normas y de RoPE, de una muestra: para
                      probar offline esquemas de KV (int4, SmoothAttention, rotaciones).
  LISTA               marca de capa terminada.

La calibracion propaga la salida BF16 de cada capa, NO la cuantizada: asi las Hessianas no
dependen de que variante se elija en las capas de antes y el cache vale para cualquier mezcla.

Al final: norma final + estadisticas de la entrada del lm_head, y un control de cordura de punta a
punta (precision top-1 del proximo token sobre la calibracion). Si el forward estuviera mal armado
(RoPE, mascara, GDN), eso se desploma.

Retomable: guarda el estado oculto cada --cada capas y saltea las capas con LISTA.

Uso (adentro de la imagen de vLLM, con UNA GPU):
  calibrar.py --modelo /models/orcarouter-... --calib /traces/cuant/calib_256x4096.npy --salida /cache
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np
import torch
from safetensors import safe_open

G = 128


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class Pesos:
    """Lectura perezosa de tensores del checkpoint sharded (un tensor por vez)."""

    def __init__(self, d):
        self.d = d
        self.mapa = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(self, k, dev, dtype=torch.bfloat16):
        """dtype=None conserva el del checkpoint (para los tensores que se copian tal cual)."""
        with safe_open(os.path.join(self.d, self.mapa[k]), "pt", device="cpu") as f:
            t = f.get_tensor(k)
        return t.to(dev) if dtype is None else t.to(dev, dtype)

    def claves(self, pref):
        return [k for k in self.mapa if k.startswith(pref)]


def estado_guardar(h, ruta):
    """El estado oculto (10,7 GB con 256 x 4096) a un memmap, muestra por muestra: torch.save de un
    tensor de GPU lo copiaria entero a RAM de una vez."""
    N, L, D = h.shape
    mm = np.lib.format.open_memmap(ruta, mode="w+", dtype=np.int16, shape=(N, L, D))
    for s in range(N):
        mm[s] = h[s].view(torch.int16).cpu().numpy()
    mm.flush()
    del mm


def estado_cargar(ruta, dev):
    mm = np.load(ruta, mmap_mode="r")
    h = torch.empty(mm.shape, dtype=torch.bfloat16, device=dev)
    for s in range(mm.shape[0]):
        h[s] = torch.from_numpy(np.array(mm[s])).view(torch.bfloat16).to(dev)
    return h


def triu_guardar(H, ruta):
    n = H.shape[0]
    m = torch.ones(n, n, dtype=torch.bool, device=H.device).triu_()
    torch.save({"n": n, "triu": H.masked_select(m).cpu()}, ruta)


class Acumulador:
    """Hessiana y estadisticas por canal de UNA entrada (compartida por varias lineales)."""

    def __init__(self, n, dev):
        self.H = torch.zeros(n, n, dtype=torch.float32, device=dev)
        self.amax = torch.zeros(n, dtype=torch.float32, device=dev)
        self.sq = torch.zeros(n, dtype=torch.float64, device=dev)
        self.gmax = torch.zeros(n // G, dtype=torch.float32, device=dev) if n % G == 0 else None
        self.tokens = 0

    @torch.no_grad()
    def sumar(self, x):
        x = x.reshape(-1, x.shape[-1])
        for i in range(0, x.shape[0], 4096):
            xf = x[i:i + 4096].float()
            self.H.addmm_(xf.T, xf)
            a = xf.abs()
            self.amax = torch.maximum(self.amax, a.amax(0))
            self.sq += xf.double().square().sum(0)
            if self.gmax is not None:
                self.gmax = torch.maximum(self.gmax, a.view(a.shape[0], -1, G).amax(-1).amax(0))
        self.tokens += x.shape[0]

    def guardar(self, dirc, nombre):
        triu_guardar(self.H, os.path.join(dirc, f"H_{nombre}.pt"))
        torch.save({"amax": self.amax.cpu(), "sq": self.sq.cpu(), "gmax": None if self.gmax is None else self.gmax.cpu(),
                    "tokens": self.tokens}, os.path.join(dirc, f"stats_{nombre}.pt"))


# Que lineal representa a cada entrada (las que comparten entrada se calibran una vez).
ENTRADAS = {
    "full_attention": {"attn_in": "self_attn.q_proj", "attn_out": "self_attn.o_proj",
                       "mlp_in": "mlp.gate_proj", "mlp_down": "mlp.down_proj"},
    "linear_attention": {"attn_in": "linear_attn.in_proj_qkv", "attn_out": "linear_attn.out_proj",
                         "mlp_in": "mlp.gate_proj", "mlp_down": "mlp.down_proj"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--muestras", type=int, default=0, help="usar solo las primeras N (pruebas)")
    ap.add_argument("--largo", type=int, default=0, help="recortar cada muestra (pruebas)")
    ap.add_argument("--capas", type=int, default=0, help="solo las primeras N capas (pruebas)")
    ap.add_argument("--micro", type=int, default=2, help="muestras por micro-lote")
    ap.add_argument("--guardar_m", type=int, default=2,
                    help="muestras apartadas: se guardan como muestra.pt y NO entran en la Hessiana (multiplo de --micro)")
    ap.add_argument("--cada", type=int, default=8, help="guardar el estado oculto cada N capas")
    ap.add_argument("--dispositivo", default="cuda")
    a = ap.parse_args()
    dev = a.dispositivo
    os.makedirs(a.salida, exist_ok=True)
    torch.manual_seed(0)

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    cfg_full = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = Qwen3_5TextConfig(**cfg_full["text_config"])
    tc._attn_implementation = "captura"
    capturas = {}

    def atencion_captura(module, q, k, v, mask, **kw):
        if capturas.get("activa"):
            capturas["qkv"] = {"q": q[:1].detach().cpu(), "k": k[:1].detach().cpu(), "v": v[:1].detach().cpu()}
            capturas["activa"] = False
        return ALL_ATTENTION_FUNCTIONS["sdpa"](module, q, k, v, mask, **kw)

    ALL_ATTENTION_FUNCTIONS["captura"] = atencion_captura

    P = Pesos(a.modelo)
    ids = np.load(a.calib)
    if a.muestras:
        ids = ids[: a.muestras]
    if a.largo:
        ids = ids[:, -a.largo:]
    N, L = ids.shape
    n_capas = a.capas or tc.num_hidden_layers
    print(f"calibracion {N} x {L} = {N * L / 1e6:.2f} M tokens, {n_capas} capas, dispositivo {dev}", flush=True)
    meta = {"N": N, "L": L, "modelo": a.modelo, "calib": a.calib, "capas": n_capas,
            "layer_types": tc.layer_types[:n_capas]}
    json.dump(meta, open(os.path.join(a.salida, "meta.json"), "w"), indent=1)

    # estado oculto: el de la ultima capa guardada, o los embeddings
    inicio = 0
    for i in range(n_capas):
        if os.path.exists(os.path.join(a.salida, f"estado_{i:02d}.npy")):
            inicio = i
    ruta_estado = os.path.join(a.salida, f"estado_{inicio:02d}.npy")
    if inicio and os.path.exists(ruta_estado):
        h = estado_cargar(ruta_estado, dev)
        print(f"retomando desde la capa {inicio}", flush=True)
    else:
        E = P.get("model.language_model.embed_tokens.weight", dev)
        h = torch.empty(N, L, tc.hidden_size, dtype=torch.bfloat16, device=dev)
        for s in range(N):
            h[s] = E[torch.from_numpy(ids[s].astype(np.int64)).to(dev)]
        del E
        inicio = 0

    rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None].expand(a.micro, -1)
    cos, sin = rot(h[: a.micro], pos)

    for i in range(inicio, n_capas):
        dirc = os.path.join(a.salida, f"capa_{i:02d}")
        if os.path.exists(os.path.join(dirc, "LISTA")):
            continue
        os.makedirs(dirc, exist_ok=True)
        t0 = time.time()
        tipo = tc.layer_types[i]
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        pref = f"model.language_model.layers.{i}."
        sd = {k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else torch.bfloat16)
              for k in P.claves(pref)}
        faltan = set(capa.state_dict()) - set(sd)
        if faltan:
            raise KeyError(f"capa {i}: faltan {sorted(faltan)[:5]}")
        capa.load_state_dict(sd, assign=True, strict=True)
        del sd
        capa.eval()

        # Las primeras guardar_m muestras son de EVALUACION: pasan por la capa (hay que propagarlas)
        # pero no entran en la Hessiana, asi el error local de una variante no se mide sobre los
        # mismos tokens con los que GPTQ se ajusto (eso lo favorece).
        acc, hooks, estado_hooks = {}, [], {"sumar": True}
        for nombre, lin in ENTRADAS[tipo].items():
            mod = capa.get_submodule(lin)
            acc[nombre] = Acumulador(mod.in_features, dev)
            hooks.append(mod.register_forward_pre_hook(
                lambda mo, args, nm=nombre: acc[nm].sumar(args[0]) if estado_hooks["sumar"] else None))

        entrada_m = h[: a.guardar_m].detach().cpu().clone()   # en CPU .cpu() NO copia: h se pisa despues
        with torch.no_grad():
            for s in range(0, N, a.micro):
                x = h[s:s + a.micro]
                pe = (cos[: x.shape[0]], sin[: x.shape[0]])
                capturas["activa"] = (s == 0 and tipo == "full_attention")
                estado_hooks["sumar"] = s >= a.guardar_m
                out = capa(x, position_embeddings=pe, attention_mask=None,
                           position_ids=pos[: x.shape[0]])
                out = out[0] if isinstance(out, tuple) else out
                h[s:s + a.micro] = out
        for hk in hooks:
            hk.remove()
        for nombre, ac in acc.items():
            ac.guardar(dirc, nombre)
        torch.save({"entrada": entrada_m, "salida": h[: a.guardar_m].detach().cpu().clone()}, os.path.join(dirc, "muestra.pt"))
        if tipo == "full_attention" and "qkv" in capturas:
            torch.save(capturas.pop("qkv"), os.path.join(dirc, "qkv.pt"))
        del capa, acc
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
        if not torch.isfinite(h[: a.micro]).all():
            raise FloatingPointError(f"capa {i}: el estado oculto tiene NaN/inf")
        open(os.path.join(dirc, "LISTA"), "w").write(f"{time.time() - t0:.1f}s\n")
        if (i + 1) % a.cada == 0 and i + 1 < n_capas:
            for f in os.listdir(a.salida):
                if f.startswith("estado_"):
                    os.remove(os.path.join(a.salida, f))
            estado_guardar(h, os.path.join(a.salida, f"estado_{i + 1:02d}.npy"))
        print(f"capa {i:02d} {tipo:16s} {time.time() - t0:6.1f}s  |h| {h[:1].float().abs().max().item():8.1f}"
              f"  RAM max {rss_gb():.1f} GB"
              + (f"  GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if dev.startswith("cuda") else ""),
              flush=True)

    if n_capas < tc.num_hidden_layers:
        print("capas recortadas (prueba): sin lm_head ni control de cordura", flush=True)
        return

    # norma final + entrada del lm_head + control de cordura
    dirc = os.path.join(a.salida, "lm_head")
    os.makedirs(dirc, exist_ok=True)
    norma = m.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
    norma.weight.data = P.get("model.language_model.norm.weight", dev, torch.float32)
    W = P.get("lm_head.weight", dev)
    ac = Acumulador(tc.hidden_size, dev)
    aciertos = total = 0
    nll = 0.0
    with torch.no_grad():
        for s in range(N):
            x = norma(h[s].float()).to(torch.bfloat16)
            ac.sumar(x)
            if s < 32:                               # control: ultimos 512 tokens de 32 muestras
                lg = (x[-513:-1] @ W.T).float()
                obj = torch.from_numpy(ids[s, -512:].astype(np.int64)).to(dev)
                aciertos += int((lg.argmax(-1) == obj).sum())
                total += obj.numel()
                nll += float(torch.nn.functional.cross_entropy(lg, obj, reduction="sum"))
    ac.guardar(dirc, "lm_head_in")
    control = {"top1": aciertos / total, "ppl": float(np.exp(nll / total)), "tokens": total}
    json.dump(control, open(os.path.join(dirc, "control.json"), "w"), indent=1)
    open(os.path.join(dirc, "LISTA"), "w").write("ok\n")
    print(f"CONTROL de punta a punta: top-1 {control['top1']:.3f}, perplejidad {control['ppl']:.2f} "
          f"sobre {total} tokens", flush=True)


if __name__ == "__main__":
    main()
