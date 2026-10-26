"""Lo que DFlash2 hace despues del lm_head al servir, en PyTorch: candidatos y tabla del selector.

Replica ``qwen3_dflash2.py`` de vLLM v0.29.0 (``CandidateSelector`` / ``_score_edges``) y el top-k de
``LogitsProcessor.get_top_k_tokens`` (top-k, despues escala y soft cap sobre los k valores), para que la
cabeza DCOCO se entrene sobre la MISMA tabla que va a corregir en el servidor.

El entrenamiento de dflash2/ no carga el selector (cargar_borrador lo descarta: queda congelado y no
entra en su perdida); aca hace falta, porque DCOCO es una correccion residual de su tabla.
"""
from __future__ import annotations

import json
import os

import torch
from torch import nn


class Selector(nn.Module):
    """CandidateSelector congelado: scores[b, l, p, c] = l_unario[b, l, c] + <pred[p] * W h[l], succ[c]>."""

    def __init__(self, pred: torch.Tensor, succ: torch.Tensor, w_hidden: torch.Tensor, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.register_buffer("pred", pred, persistent=False)
        self.register_buffer("succ", succ, persistent=False)
        self.register_buffer("w_hidden", w_hidden, persistent=False)

    @torch.no_grad()
    def forward(self, cand, unario, hidden, ancla):
        """cand [B, S, K] ids del target, unario [B, S, K], hidden [B, S, H], ancla [B] -> [B, S, K, K] fp32."""
        h = torch.nn.functional.linear(hidden.to(self.w_hidden.dtype), self.w_hidden)          # [B, S, r]
        previos = torch.cat((ancla[:, None, None].expand(-1, 1, self.top_k), cand[:, :-1]), 1)
        p = self.pred[previos].float() * h.float()[:, :, None]                                  # [B, S, K, r]
        s = self.succ[cand].float()                                                             # [B, S, K, r]
        return unario.float()[:, :, None] + torch.einsum("blpr,blcr->blpc", p, s)


def _buscar(sd_keys, sufijo):
    c = [k for k in sd_keys if k.endswith(sufijo)]
    if len(c) != 1:
        raise KeyError(f"no encuentro una sola clave terminada en {sufijo!r}: {c[:4]}")
    return c[0]


def cargar_selector(dir_borrador: str, device="cuda") -> tuple[Selector, dict]:
    """Codebooks y hidden_projection del checkpoint (bf16 o fp16, sin cuantizar: el selector no se
    cuantiza) y el dflash_config del borrador."""
    from safetensors import safe_open
    c = json.load(open(os.path.join(dir_borrador, "config.json")))
    dc = c["dflash_config"]
    with safe_open(os.path.join(dir_borrador, "model.safetensors"), "pt", device="cpu") as f:
        ks = list(f.keys())
        t = {s: f.get_tensor(_buscar(ks, s)) for s in ("candidate_selector.predecessor_codebook",
                                                       "candidate_selector.successor_codebook",
                                                       "candidate_selector.hidden_projection.weight")}
    sel = Selector(t["candidate_selector.predecessor_codebook"].to(device),
                   t["candidate_selector.successor_codebook"].to(device),
                   t["candidate_selector.hidden_projection.weight"].to(device),
                   int(dc["selector_top_k"]))
    return sel, c


def procesador(config: dict):
    """(escala, soft_cap) del candidate_logits_processor de DFlash2Qwen3ForCausalLM."""
    dc = config["dflash_config"]
    softcap = float(dc.get("final_logit_softcapping") or 0.0)
    return float(dc.get("output_multiplier", 1.0)), (softcap if softcap > 0 else None)


def candidatos(logits: torch.Tensor, k: int, escala: float, soft_cap):
    """logits [.., V] fp32 (con el vocabulario recortado ya en -inf) -> ids [.., k], valores [.., k] como
    get_top_k_tokens: top-k primero, escala y soft cap despues (monotonos: misma seleccion)."""
    v, ids = logits.topk(k, -1)
    if escala != 1.0:
        v = v * escala
    if soft_cap is not None:
        v = torch.tanh(v / soft_cap) * soft_cap
    return ids, v
