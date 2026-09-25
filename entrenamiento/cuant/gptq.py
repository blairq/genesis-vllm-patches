"""GPTQ (Frantar et al., 2022) para W4 simetrico por grupo de 128, con escalas fp16.

Mismo formato de salida que cuantizar_rtn.py (compressed-tensors pack-quantized): la escala de cada
grupo se fija al entrar al grupo con max|w| / 7,5 sobre los pesos YA corregidos, como hace
llm-compressor con el observer minmax. Sin reordenamiento por activacion (actorder): asi no hace
falta g_idx y Marlin lee los grupos contiguos.

H = X^T X de las ENTRADAS de la lineal (las del borrador en su propio forward, sin mezclar la KV
de contexto: syv-ai midio que mezclarla daba 7% peor aceptacion).
"""
from __future__ import annotations

import torch

G = 128


@torch.no_grad()
def gptq(W: torch.Tensor, H: torch.Tensor, damp: float = 0.01, bloque: int = 128):
    """W [out, in] (cualquier dtype), H [in, in] fp32. Devuelve (q int [-8,7], s fp16 [out, in/G])."""
    W = W.float().clone()
    out, inn = W.shape
    H = H.float().clone()
    muertos = torch.diag(H) == 0
    H[muertos, muertos] = 1
    W[:, muertos] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(inn, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(out, inn, dtype=torch.int8, device=W.device)
    S = torch.zeros(out, inn // G, dtype=torch.float16, device=W.device)
    for i1 in range(0, inn, bloque):
        i2 = min(i1 + bloque, inn)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % G == 0:
                g = col // G
                # los pesos de este grupo que estan en el bloque ya fueron corregidos: usar W1
                ini = col - i1
                s = (torch.cat([W1[:, ini:], W[:, i2:col + G]], 1)[:, :G].abs().amax(1) / 7.5).clamp_min(1e-10).half()
                S[:, g] = s
            sc = S[:, col // G].float()
            w = W1[:, i]
            q = (w / sc).round().clamp(-8, 7)
            Q[:, col] = q.to(torch.int8)
            err = (w - q * sc) / Hi1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, S


def empaquetar(Q: torch.Tensor, S: torch.Tensor):
    out, inn = Q.shape
    q = (Q.to(torch.int32) + 8).view(out, inn // 8, 8).cpu()
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    p = (q << sh).sum(-1, dtype=torch.int64)
    p = torch.where(p >= 2 ** 31, p - 2 ** 32, p).to(torch.int32)
    return p.contiguous(), S.cpu().contiguous(), torch.tensor([out, inn], dtype=torch.int64)
