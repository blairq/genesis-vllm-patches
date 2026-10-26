"""Cabeza DCOCO con estado RECURRENTE del camino (``--estado recurrente``). Solo entrenamiento.

La cabeza de pares (vllm/_genesis/dcoco.py) puntua "a en p-1 -> b en p" con un estado de PAR, las
ocultas de las dos posiciones: es una correccion de Markov de orden 1, como el selector. Los correctores
que mas ganan sobre DFlash condicionan cada posicion en TODO el prefijo elegido: xPress (arXiv
2608.02438, mezclador causal sobre el prefijo + embedding del token previo) y la "correccion RNN" de
DARTree (arXiv 2608.13524). Aca:

    s_a    = GRU(s_prefijo, [pe[p-1, a]; l[p-1, a]])        estado despues de elegir a en p-1
    delta  = MLP([s_a; l[p-1, a]; pe[p, b]; l[p, b]])       se suma a la fila a de la tabla del selector

con ``pe`` la misma representacion de candidato que la cabeza de pares y la ultima capa en cero (sin
entrenar, la tabla es la del selector). El estado depende del camino, asi que la tabla ya no es fija:

* al entrenar, forzado por el maestro: ``tabla_forzada`` arma la tabla con el prefijo REAL;
* al evaluar (cadena y arbol), ``Puntuador`` calcula las filas a medida que se elige cada nodo.

PN178 todavia no la sirve: dcoco.crear rechaza un checkpoint con ``dcoco_estado: recurrente``. Servirla
pide un constructor del arbol con estado por nodo en el servidor (evaluacion.arbol es la referencia).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CabezaRecurrente(nn.Module):
    def __init__(self, hidden: int, topk: int, emb: int = 128, state: int = 128, mlp: int = 256):
        super().__init__()
        self.topk, self.emb, self.state = topk, emb, state
        self.cand_proj = nn.Linear(hidden, emb, bias=False)
        self.ctx_h_proj = nn.Linear(hidden, emb, bias=False)
        self.cand_ctx_proj = nn.Linear(2 * emb, emb, bias=False)
        self.ctx_out_norm = nn.LayerNorm(emb)
        self.gru = nn.GRUCell(emb + 1, state)
        # entrada de trans_mlp[0]: [s_a, l_a, pe_b, l_b]
        self.trans_mlp = nn.Sequential(nn.Linear(state + 1 + emb + 1, mlp), nn.GELU(), nn.Linear(mlp, 1))
        nn.init.zeros_(self.trans_mlp[2].weight)
        nn.init.zeros_(self.trans_mlp[2].bias)

    def pe(self, emb_cand, hidden):
        """Representacion de cada candidato, igual que la cabeza de pares: [B, S, K, E] fp32."""
        dt = self.cand_proj.weight.dtype
        e = self.emb
        base = F.layer_norm(self.cand_proj(emb_cand.to(dt)).float(), (e,))
        ctx = F.layer_norm(self.ctx_h_proj(hidden.to(dt)).float(), (e,))
        pe = self.cand_ctx_proj(torch.cat((base, ctx.unsqueeze(-2).expand_as(base)), -1).to(dt))
        return F.layer_norm(pe.float(), (e,), self.ctx_out_norm.weight.float(), self.ctx_out_norm.bias.float())

    def avanzar(self, s, pe_a, l_a):
        """Estado despues de elegir el candidato (pe_a [N, E], l_a [N]) con prefijo s [N, state] (o None)."""
        x = torch.cat((pe_a, l_a.unsqueeze(-1)), -1).float()
        if s is None:
            s = torch.zeros(x.shape[0], self.state, device=x.device)
        return self.gru(x, s.float())

    def delta(self, s_a, l_a, pe_b, l_b):
        """s_a [B, (A,) state], l_a [B, (A)], pe_b [B, K, E], l_b [B, K] -> [B, (A,) K]."""
        w = self.trans_mlp[0].weight.float()
        st = self.state
        izq = F.linear(torch.cat((s_a, l_a.unsqueeze(-1)), -1).float(), w[:, :st + 1], self.trans_mlp[0].bias.float())
        der = F.linear(torch.cat((pe_b, l_b.unsqueeze(-1)), -1).float(), w[:, st + 1:])
        if izq.dim() == 3:                                                   # [B, A, M] x [B, K, M]
            h = self.trans_mlp[1](izq.unsqueeze(2) + der.unsqueeze(1))
        else:                                                                # [B, M] x [B, K, M]
            h = self.trans_mlp[1](izq.unsqueeze(1) + der)
        return F.linear(h, self.trans_mlp[2].weight.float(), self.trans_mlp[2].bias.float()).squeeze(-1)

    def tabla_forzada(self, sc, emb_cand, logits, hidden, idx_real):
        """Tabla [B, S, K, K] con el prefijo REAL (forzado por el maestro): la fila a de la profundidad p es
        la de haber elegido a en p-1 despues del prefijo real 0..p-2. idx_real [B, S]."""
        B, S, K = logits.shape
        pe = self.pe(emb_cand, hidden)
        l = logits.float()
        filas, s = [sc[:, 0]], None
        ar = torch.arange(B, device=sc.device)
        for p in range(1, S):
            s_rep = None if s is None else s.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            s_a = self.avanzar(s_rep, pe[:, p - 1].reshape(B * K, -1), l[:, p - 1].reshape(B * K)).view(B, K, -1)
            filas.append(sc[:, p] + self.delta(s_a, l[:, p - 1], pe[:, p], l[:, p]).to(sc.dtype))
            s = s_a[ar, idx_real[:, p - 1]]
        return torch.stack(filas, 1)


class Puntuador:
    """Filas dependientes del camino para evaluacion.cadena / evaluacion.arbol."""

    def __init__(self, cab: CabezaRecurrente, sc, emb_cand, logits, hidden, pe=None):
        self.cab, self.sc = cab, sc
        self.pe = cab.pe(emb_cand, hidden) if pe is None else pe
        self.l = logits.float()
        self.filas = torch.arange(sc.shape[0], device=sc.device)
        self.S = sc.shape[1]

    @classmethod
    def desde(cls, cab: CabezaRecurrente, sc, pe, logits):
        """Con ``pe`` ya calculado (la cache de la calibracion guarda pe en vez de embeddings y ocultas)."""
        return cls(cab, sc, None, logits, None, pe=pe)

    def raiz(self):
        return self.sc[:, 0, 0]

    def nodo(self, est_padre, d, i):
        return self.cab.avanzar(est_padre, self.pe[self.filas, d, i], self.l[self.filas, d, i])

    def fila(self, est, d, i):
        d1 = (d + 1).clamp(max=self.S - 1)
        return self.sc[self.filas, d1, i] + self.cab.delta(est, self.l[self.filas, d, i], self.pe[self.filas, d1],
                                                            self.l[self.filas, d1]).to(self.sc.dtype)
