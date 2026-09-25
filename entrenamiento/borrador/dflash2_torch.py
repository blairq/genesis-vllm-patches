"""DFlash2 en PyTorch puro, para ENTRENAR el borrador (vLLM solo tiene la inferencia).

Replica linea por linea ``vllm/model_executor/models/qwen3_dflash.py`` y ``qwen3_dflash2.py`` de
v0.29.0, con dos diferencias deliberadas:

* El contexto entra como FEATURES YA PROYECTADAS por ``fc`` (lo que vuelca
  ``vllm/_genesis/captura_borrador.py``). El ``fc`` queda congelado y fuera del grafo.
* Se entrena sobre bloques de queries por ancla: ``[bonus, mask x K]`` en las posiciones
  ``a .. a+K``, que atienden (NO causal) al contexto ``[a-W, a)`` y a todo el bloque.

El selector de candidatos no entra en la perdida: queda congelado (de ahi sale el top-16 del
arbol, y su entrenamiento no es publico). La perdida va sobre los logits "unarios" del lm_head.

Lo que usa del TARGET (noon): ``embed_tokens`` y ``lm_head``. El lm_head se simula igual que en
el runtime: int4 simetrico por grupo de 128 con RTN (PN139), dequantizado.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F
from torch import nn


def rms(x, w, eps):
    """RMSNorm de Qwen3 (peso multiplicativo, sin el +1 de Gemma), en fp32 por dentro."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def rope(x, pos, theta, head_dim):
    """RoPE estilo neox (mitades), como get_rope(is_neox_style=True). x [..., n, hd], pos [...]"""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * inv                      # [..., half]
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)  # [..., 1, half]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


def conv_bloque(h, delta, base, taps, G, gs):
    """_grouped_conv de vLLM sobre bloques ya separados: h [B, T, H], delta [B, T, taps, G].

    En vLLM la posicion dentro del bloque sale de ``idx % block_size`` sobre la fila aplanada;
    aca el bloque es una dimension, asi que el corrimiento no cruza bloques por construccion.
    """
    B, T, H = h.shape
    blocks = h.view(B, T, G, gs)
    coef = base.view(1, 1, taps, G, gs) + delta.unsqueeze(-1)          # [B, T, taps, G, gs]
    out = coef[:, :, 0] * blocks
    for tap in range(1, taps):
        shifted = F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))        # corre tap filas hacia abajo
        out = out + coef[:, :, tap] * shifted
    return out.view(B, T, H)


class Capa(nn.Module):
    def __init__(self, c):
        super().__init__()
        H, I = c["hidden_size"], c["intermediate_size"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.q_proj = nn.Linear(H, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(H, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, H, bias=False)
        self.q_norm = nn.Parameter(torch.ones(self.hd))
        self.k_norm = nn.Parameter(torch.ones(self.hd))
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.input_layernorm = nn.Parameter(torch.ones(H))
        self.post_attention_layernorm = nn.Parameter(torch.ones(H))
        dc = c["dflash_config"]
        self.taps, self.gs = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.G = H // self.gs
        for nom in ("attention_conv", "mlp_conv"):
            setattr(self, nom + "_base", nn.Parameter(torch.zeros(2, self.taps, H)))
            setattr(self, nom + "_proj", nn.Linear(H, 2 * self.taps * self.G, bias=False))

    def _conv_prep(self, nom, h):
        B, T, _ = h.shape
        co = getattr(self, nom + "_proj")(h).view(B, T, 2, self.taps, self.G)
        base = getattr(self, nom + "_base")
        return conv_bloque(h, co[:, :, 0], base[0], self.taps, self.G, self.gs), (co[:, :, 1], base[1])

    def _conv_fin(self, h, estado):
        delta, base = estado
        return conv_bloque(h, delta, base, self.taps, self.G, self.gs)

    def kv_contexto(self, feat_normed, pos, theta):
        """K/V del contexto a partir de hidden_norm(feat): [N, nkv, hd] cada uno."""
        N = feat_normed.shape[0]
        k = self.k_proj(feat_normed).view(N, self.nkv, self.hd)
        v = self.v_proj(feat_normed).view(N, self.nkv, self.hd)
        k = rope(rms(k, self.k_norm, self.eps), pos, theta, self.hd)
        return k, v

    def forward(self, h, residual, pos, ctx_k, ctx_v, mask, theta):
        """h [B, T, H]; ctx_k/v [B, W, nkv, hd]; mask [B, T, W+T] (True = la query ve la clave)."""
        B, T, H = h.shape
        if residual is None:
            residual = h
            x = rms(h, self.input_layernorm, self.eps)
        else:
            residual = h + residual
            x = rms(residual, self.input_layernorm, self.eps)
        x, est = self._conv_prep("attention_conv", x)
        q = self.q_proj(x).view(B, T, self.nh, self.hd)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd)
        q = rope(rms(q, self.q_norm, self.eps), pos, theta, self.hd)
        k = rope(rms(k, self.k_norm, self.eps), pos, theta, self.hd)
        K_ = torch.cat([ctx_k, k], 1)                                  # [B, W+T, nkv, hd]
        V_ = torch.cat([ctx_v, v], 1)
        rep = self.nh // self.nkv
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), K_.transpose(1, 2).repeat_interleave(rep, 1),
            V_.transpose(1, 2).repeat_interleave(rep, 1),
            attn_mask=mask[:, None], scale=self.hd ** -0.5)             # no causal: todo el bloque
        x = self.o_proj(att.transpose(1, 2).reshape(B, T, self.nh * self.hd))
        x = self._conv_fin(x, est)
        residual = x + residual
        x = rms(residual, self.post_attention_layernorm, self.eps)
        x, est = self._conv_prep("mlp_conv", x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = self._conv_fin(x, est)
        return x, residual


class DFlash2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        H = c["hidden_size"]
        self.eps = c["rms_norm_eps"]
        self.theta = float(c["rope_parameters"]["rope_theta"])
        self.mask_id = int(c["dflash_config"]["mask_token_id"])
        self.layers = nn.ModuleList([Capa(c) for _ in range(c["num_hidden_layers"])])
        self.hidden_norm = nn.Parameter(torch.ones(H))
        self.norm = nn.Parameter(torch.ones(H))
        # Del target (noon): no se entrenan.
        self.embed = None
        self.lm_head = None

    def forward(self, bloque_ids, bloque_pos, feat, feat_pos, idx_ctx):
        """Un pedido, varias anclas.

        bloque_ids/pos [B, T]: [bonus, mask x K] de cada ancla y sus posiciones.
        feat [N, H], feat_pos [N]: las features del fc de TODO el tramo capturado del pedido.
        idx_ctx [B, W]: para cada ancla, los indices en feat de su ventana de contexto (-1 = nada).

        Los K/V del contexto se calculan UNA vez por capa para todo el tramo y cada ancla toma su
        ventana por indice: las anclas de un pedido comparten casi todo el contexto.
        Devuelve el hidden final del bloque [B, T, H] (antes del lm_head)."""
        B, T = bloque_ids.shape
        W = idx_ctx.shape[1]
        valido = idx_ctx >= 0
        idx = idx_ctx.clamp(min=0)
        vent = int(self.c.get("sliding_window") or 10 ** 9)
        pos_ctx = feat_pos[idx]                                          # [B, W]
        ven_ctx = valido[:, None, :] & (pos_ctx[:, None, :] > bloque_pos[:, :, None] - vent)
        mask = torch.cat([ven_ctx, torch.ones(B, T, T, dtype=torch.bool, device=feat.device)], 2)
        fn = rms(feat, self.hidden_norm, self.eps)
        h = F.embedding(bloque_ids, self.embed)
        residual = None
        for capa in self.layers:
            k, v = capa.kv_contexto(fn, feat_pos, self.theta)             # [N, nkv, hd]
            h, residual = capa(h, residual, bloque_pos, k[idx], v[idx], mask, self.theta)
        return rms(h + residual, self.norm, self.eps)

    def logits(self, hidden):
        if hidden.device.type == "cuda":
            return F.linear(hidden, self.lm_head).float()
        # En CPU el GEMM bf16 pasa el peso entero a fp32 (5 GB mas): por tramos de vocabulario.
        return torch.cat([F.linear(hidden, self.lm_head[i:i + 16384]).float()
                          for i in range(0, self.lm_head.shape[0], 16384)], -1)


# ───────────────────────────── carga de pesos ─────────────────────────────

_MAPA = {
    "self_attn.q_proj.weight": "q_proj.weight", "self_attn.k_proj.weight": "k_proj.weight",
    "self_attn.v_proj.weight": "v_proj.weight", "self_attn.o_proj.weight": "o_proj.weight",
    "self_attn.q_norm.weight": "q_norm", "self_attn.k_norm.weight": "k_norm",
    "mlp.gate_proj.weight": "gate_proj.weight", "mlp.up_proj.weight": "up_proj.weight",
    "mlp.down_proj.weight": "down_proj.weight",
    "input_layernorm.weight": "input_layernorm", "post_attention_layernorm.weight": "post_attention_layernorm",
    "attention_conv.base_kernel": "attention_conv_base", "mlp_conv.base_kernel": "mlp_conv_base",
    "attention_conv.kernel_projection.weight": "attention_conv_proj.weight",
    "mlp_conv.kernel_projection.weight": "mlp_conv_proj.weight",
}


def cargar_borrador(dir_bf16, dtype=torch.bfloat16, device="cuda"):
    from safetensors.torch import load_file
    c = json.load(open(os.path.join(dir_bf16, "config.json")))
    with torch.device("meta"):
        m = DFlash2(c)
    sd = load_file(os.path.join(dir_bf16, "model.safetensors"), device="cpu")
    nuevo = {}
    for k, v in sd.items():
        if k.startswith("layers."):
            _, i, resto = k.split(".", 2)
            if resto in _MAPA:
                nuevo[f"layers.{i}.{_MAPA[resto]}"] = v
        elif k == "hidden_norm.weight":
            nuevo["hidden_norm"] = v
        elif k == "norm.weight":
            nuevo["norm"] = v
    faltan = [k for k in m.state_dict() if k not in nuevo]
    if faltan:
        raise KeyError(f"faltan pesos: {faltan[:6]} ...")
    m.load_state_dict({k: v.to(dtype) for k, v in nuevo.items()}, assign=True)
    del sd, nuevo                      # el selector y el fc no se usan: que no ocupen RAM
    return m.to(device), c, None


def cargar_target(dir_noon, device="cuda", dtype=torch.bfloat16, lm_head_int4=True):
    """embed_tokens y lm_head de noon; el lm_head pasa por el RTN int4 g128 de PN139.

    Se lee por tramos de filas (get_slice): cargar el lm_head entero y despues cuantizarlo dejaba
    tres copias de 2,5 GB vivas a la vez y tumbaba el contenedor por memoria."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(dir_noon, "model.safetensors.index.json")))["weight_map"]
    # Target con el residuo ROTADO (entrenamiento/cuant/armar_rot.py): se devuelven el embedding y el
    # lm_head EFECTIVOS en la base original del borrador, exactamente lo que ve el borrador servido con
    # PN149: e = e_rot R ; logits = ((h / g) Rt) Q(L')^T  =>  lm_head_ef = Q(L') R diag(1/g).
    rc = json.load(open(os.path.join(dir_noon, "config.json"))).get("genesis_rotacion")
    if rc:
        import sys as _s
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuant"))
        import rotacion as ro
        assert rc["semilla"] == ro.SEMILLA
        sg = ro.signos(len(rc["g_final"])).to(device, torch.float32)
        g = torch.tensor(rc["g_final"], device=device, dtype=torch.float32)
        a_original = lambda x: ro.bloques(x.float(), rc["bloque"]) * sg  # noqa: E731  (x R)
    out = {}
    for nom, clave in (("embed", "model.language_model.embed_tokens.weight"), ("lm_head", "lm_head.weight")):
        with safe_open(os.path.join(dir_noon, idx[clave]), "pt", device="cpu") as f:
            sl = f.get_slice(clave)
            N, Kd = sl.get_shape()
            w = torch.empty(N, Kd, dtype=dtype, device=device)
            for lo in range(0, N, 16384):
                t = sl[lo:lo + 16384].to(device)
                if nom == "lm_head" and lm_head_int4:
                    t = rtn_int4_g128(t)
                if rc:
                    t = a_original(t)
                    if nom == "lm_head":
                        t = t / g[None, :]
                w[lo:lo + t.shape[0]] = t.to(dtype)
                del t
        out[nom] = w
    return out["embed"], out["lm_head"]


@torch.no_grad()
def rtn_int4_g128(w):
    """Lo mismo que vllm/_genesis/lm_head_int4.cuantizar_int4_grupo, dequantizado en fp16."""
    N, Kd = w.shape
    out = torch.empty(N, Kd, dtype=torch.float16, device=w.device)
    for lo in range(0, N, 16384):
        wf = w[lo:lo + 16384].float().view(-1, Kd // 128, 128)
        esc = (wf.abs().amax(dim=2).clamp_min(1e-8) / 7.0).half().float()
        q = (wf / esc.unsqueeze(2)).round().clamp(-8, 7)
        out[lo:lo + 16384] = (q * esc.unsqueeze(2)).view(-1, Kd).half()
    return out
