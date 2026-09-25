#!/usr/bin/env python3
"""Separa "error mio al armar las capas" de "diferencia de transformers con vLLM", con Qwen3.5-0.8B.

Corre el mismo modelo chico por dos vias sobre las mismas ventanas:
  1. OFICIAL: Qwen3_5ForConditionalGeneration.from_pretrained (el modelo completo de transformers);
  2. MIO: el armado capa por capa de la etapa A (config desde text_config, Qwen3_5DecoderLayer,
     rotary, norma final), igual que calibrar.py y referencia.py.
Reporta la perplejidad de los ultimos 255 tokens con contexto 256 y 1024 y la diferencia entre vias.
Si MIO != OFICIAL, el error es del armado; si coinciden y los dos se degradan con el largo, la
diferencia con vLLM es de transformers (o del modelo).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import Pesos  # noqa: E402

D = sorted(glob.glob("/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
V = np.load(sys.argv[1])[:8]
torch.set_grad_enabled(False)
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


def lp_de(logits, ids):
    return logits.float().log_softmax(-1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)


# 1. oficial
mod = m.Qwen3_5ForConditionalGeneration.from_pretrained(D, dtype=torch.float32).eval()
ids = torch.from_numpy(V.astype(np.int64))
of = {}
for ctx in (1024, 256):
    x = ids[:, -ctx:]
    of[ctx] = torch.cat([lp_de(mod(input_ids=x[s:s + 1]).logits, x[s:s + 1]) for s in range(len(x))])
W_lm = mod.lm_head.weight.detach()
del mod

# 2. mio (capa por capa)
cfg = json.load(open(os.path.join(D, "config.json")))
tc = Qwen3_5TextConfig(**cfg["text_config"])
tc._attn_implementation = "sdpa"
P = Pesos(D)
E = P.get("model.language_model.embed_tokens.weight", "cpu", torch.float32)
mio = {}
for ctx in (1024, 256):
    x = ids[:, -ctx:]
    h = E[x]
    rot = m.Qwen3_5TextRotaryEmbedding(tc)
    pos = torch.arange(ctx)[None]
    pe = rot(h[:1], pos)
    for i in range(tc.num_hidden_layers):
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        pref = f"model.language_model.layers.{i}."
        capa.load_state_dict({k[len(pref):]: P.get(k, "cpu", torch.float32) for k in P.claves(pref)}, assign=True, strict=True)
        for s in range(len(h)):
            o = capa(h[s:s + 1], position_embeddings=pe, attention_mask=None, position_ids=pos)
            h[s:s + 1] = o[0] if isinstance(o, tuple) else o
    norma = m.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
    norma.weight.data = P.get("model.language_model.norm.weight", "cpu", torch.float32)
    # ventana por ventana: los logits de las 8 juntas son 8 GB en fp32
    mio[ctx] = torch.cat([lp_de(norma(h[s:s + 1]) @ W_lm.T, x[s:s + 1]) for s in range(len(h))])

f = lambda t: float(torch.exp(-t.mean()))
for ctx in (256, 1024):
    print(f"contexto {ctx:4d}: ppl ultimos 255  OFICIAL {f(of[ctx][:, -255:]):7.2f}   MIO {f(mio[ctx][:, -255:]):7.2f}   "
          f"|dif logprob| medio {float((of[ctx] - mio[ctx]).abs().mean()):.2e}", flush=True)
