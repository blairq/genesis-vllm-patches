"""Rotacion del flujo residual (estilo QuaRot) y Hadamard en linea de down_proj, para W4A8.

Por que: las entradas que salen del residuo (qkv, in_proj del GDN, gate/up) tienen cresta ~56 por
token (un canal con un pico enorme) y el int8 por token de Marlin las aplasta: k_proj/v_proj pierden
hasta 37-48% de la salida (error_a8.py). Rotar el residuo reparte el pico y deja el error en <1%.

Convencion (filas, como torch: y = x @ W.T):
  Rt = diag(signos) @ Hb       ortogonal, Hb = Hadamard por bloques de `bloque` (5120 = 5 x 1024)
  R  = Rt.T
  residuo rotado  h~ = h @ Rt ;  RMSNorm(h~) = RMSNorm(h) @ Rt  (la norma no cambia con la rotacion,
  por eso el peso g=(1+w) de la RMSNorm se pliega en la lineal siguiente y la norma queda en 1).

Pesos que se cuantizan (los que irian en el checkpoint rotado), y el peso EFECTIVO equivalente en
la base original (lo que usa la evaluacion en torch, que no cambia de forma):
  entrada del residuo (q/k/v, in_proj_qkv, in_proj_z, gate, up):
        A = W diag(g) Rt            H~ = R diag(1/g) H diag(1/g) Rt      W_ef = deq(A) R diag(1/g)
  escriben al residuo (o_proj, out_proj):
        A = R W                     H~ = H                              W_ef = Rt deq(A)
  down_proj (escribe y ademas Hadamard en linea Hd por bloques de 512 sobre K; K/2 = 17 x 512, asi
  los bloques no cruzan la particion de TP=2):
        A = R W Hd                  H~ = Hd H Hd                        W_ef = Rt deq(A) Hd
"""
from __future__ import annotations

import torch

SEMILLA = 20260924
ENTRADA = ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z", "gate_proj", "up_proj")
ESCRIBEN = ("o_proj", "out_proj")
BAJADA = ("down_proj",)
FILA = ("o_proj", "down_proj", "out_proj")      # row-parallel en TP=2: K partida en 2


def hadamard(n, dev="cpu", dt=torch.float32):
    h = torch.ones(1, 1, device=dev, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n, n
    return (h / n ** 0.5).to(dt)


def por_bloques(n, b, dev="cpu", dt=torch.float32):
    assert n % b == 0, (n, b)
    return torch.block_diag(*[hadamard(b, dev, dt)] * (n // b))


def signos(n):
    g = torch.Generator().manual_seed(SEMILLA)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.int8)


def Rt_de(n, bloque, dev, dt=torch.float32):
    return signos(n).to(dev, dt)[:, None] * por_bloques(n, bloque, dev, dt)


def clase(base, bloque_down=512):
    """bloque_down=0: down_proj sin Hadamard en linea (solo escribe al residuo rotado)."""
    hoja = base.split(".")[-1]
    if hoja in BAJADA and not bloque_down:
        return "escribe"
    if hoja in ENTRADA:
        return "entrada"
    if hoja in ESCRIBEN:
        return "escribe"
    if hoja in BAJADA:
        return "bajada"
    return None


def norma_de(base):
    """Que RMSNorm precede a la lineal (su peso se pliega)."""
    return "post_attention_layernorm" if base.startswith("mlp.") else "input_layernorm"


def bloques(X, b):
    """X @ Hd, con Hd Hadamard por bloques de b sobre la ultima dim (simetrica: Hd = Hd.T = Hd^-1)."""
    return (X.reshape(*X.shape[:-1], -1, b) @ hadamard(b, X.device, X.dtype)).reshape(X.shape)


def g_segura(g):
    """g=(1+w) de la RMSNorm con los canales muertos (g=0 exacto; la capa 7 tiene uno) en 1: esa
    entrada es 0 igual, y asi no aparecen 0/0 al pasar de x = g n a n."""
    return torch.where(g == 0, torch.ones_like(g), g)
