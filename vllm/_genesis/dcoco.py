# SPDX-License-Identifier: Apache-2.0
"""PN178: corrector DCOCO sobre el selector de candidatos del borrador DFlash2.

DFlash2 propone por cada posicion del bloque 16 candidatos y una tabla de transiciones
``scores[paso, candidato previo, candidato]`` que sale de su ``CandidateSelector``: un bilineal de rango
bajo con codebooks por token, que solo mira la oculta de la posicion destino. Sobre esa tabla se
recorre la cadena o se arma el arbol (arbol_borrador.construir).

DCOCO (Eisenstadt et al., arXiv 2610.12570; github.com/two-delta/DCOCO) puntua cada transicion
"candidato a en p -> candidato b en p+1" con un MLP chico sobre un estado de PAR (las ocultas de las
dos posiciones) y una representacion por candidato (su embedding del target proyectado y mezclado
con la oculta de su posicion). Aca va como correccion RESIDUAL de la tabla del selector:

    scores[p](a, b) += delta(st[p-1], pe[p-1, a], pe[p, b], l[p-1, a], l[p, b])      p >= 1
    pe[s, i] = LN(cand_ctx_proj([LN0(cand_proj(E[s, i])), LN0(ctx_h_proj(h[s]))]))
    st[p]    = pair_proj([h[p], h[p+1]])

con la ultima capa del MLP en cero al crearla: sin entrenar, la tabla queda exactamente como la del
selector. La fila 0 (desde el ancla) no se toca. Si el checkpoint trae ``dcoco_escala_arbol`` (la
calibracion de entrenar_dcoco.py --calibrar), la fila s se multiplica por escala[s]: no cambia el argmax
de la cadena, solo las probabilidades de camino con que se arma el arbol. Los nombres de los parametros son los de DCOCOHead
del repo de DCOCO, asi una cabeza entrenada alla carga aca.

Solo depende de torch: el entrenamiento (entrenamiento/dflash2-dcoco/) lo importa por ruta y usa la
misma cuenta que el servidor.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

PREFIJO = "dcoco."


def config_de(draft_config) -> dict | None:
    """Hiperparametros de la cabeza desde el ``dflash_config`` del checkpoint, o None si no trae."""
    if not draft_config or "dcoco_emb_dim" not in draft_config:
        return None
    return dict(
        topk=int(draft_config.get("dcoco_candidate_topk", draft_config.get("selector_top_k", 16))),
        emb=int(draft_config["dcoco_emb_dim"]),
        state=int(draft_config["dcoco_state_dim"]),
        mlp=int(draft_config["dcoco_mlp_dim"]),
        escala=draft_config.get("dcoco_escala_arbol"),
    )


class CabezaDCOCO(nn.Module):
    def __init__(self, hidden: int, topk: int, emb: int = 128, state: int = 128, mlp: int = 256,
                 escala=None):
        super().__init__()
        self.topk, self.emb, self.state = topk, emb, state
        self.cand_proj = nn.Linear(hidden, emb, bias=False)
        self.ctx_h_proj = nn.Linear(hidden, emb, bias=False)
        self.cand_ctx_proj = nn.Linear(2 * emb, emb, bias=False)
        self.ctx_out_norm = nn.LayerNorm(emb)
        self.pair_proj = nn.Linear(2 * hidden, state, bias=False)
        # entrada de trans_mlp[0]: [state, pe_prev, pe_cur, l_prev, l_cur]
        self.trans_mlp = nn.Sequential(nn.Linear(state + 2 * emb + 2, mlp), nn.GELU(), nn.Linear(mlp, 1))
        nn.init.zeros_(self.trans_mlp[2].weight)
        nn.init.zeros_(self.trans_mlp[2].bias)
        self._partida = None
        self.register_buffer("escala_arbol", None if escala is None else
                             torch.tensor([float(x) for x in escala], dtype=torch.float32), persistent=False)

    def _w0(self):
        """trans_mlp[0] partida en estado / previo / siguiente: cada candidato se proyecta una vez en
        vez de armar las K x K concatenaciones."""
        if self._partida is not None and not self.training:
            return self._partida
        w, st, e = self.trans_mlp[0].weight, self.state, self.emb
        lp = st + 2 * e
        return (w[:, :st],
                torch.cat((w[:, st:st + e], w[:, lp:lp + 1]), 1),
                torch.cat((w[:, st + e:st + 2 * e], w[:, lp + 1:lp + 2]), 1))

    @torch.no_grad()
    def materializar(self) -> None:
        """Copias contiguas de la partida para servir (despues de cargar los pesos)."""
        self._partida = None
        self._partida = tuple(t.contiguous() for t in self._w0())

    def delta(self, emb_cand: torch.Tensor, logits: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """emb_cand [R, S, K, H], logits [R, S, K] (unarios del lm_head), hidden [R, S, H] -> [R, S-1, K, K]."""
        dt = self.cand_proj.weight.dtype
        e = self.emb
        base = F.layer_norm(self.cand_proj(emb_cand.to(dt)).float(), (e,))
        ctx = F.layer_norm(self.ctx_h_proj(hidden.to(dt)).float(), (e,))
        pe = self.cand_ctx_proj(torch.cat((base, ctx.unsqueeze(-2).expand_as(base)), -1).to(dt))
        pe = F.layer_norm(pe.float(), (e,), self.ctx_out_norm.weight.float(), self.ctx_out_norm.bias.float())
        feat = torch.cat((pe, logits.float().unsqueeze(-1)), -1).to(dt)          # [R, S, K, E+1]
        w_st, w_prev, w_cur = self._w0()
        prev = F.linear(feat[:, :-1], w_prev)                                     # [R, S-1, K, M]
        cur = F.linear(feat[:, 1:], w_cur)                                        # [R, S-1, K, M]
        st = self.pair_proj(torch.cat((hidden[:, :-1], hidden[:, 1:]), -1).to(dt))
        st = F.linear(st, w_st, self.trans_mlp[0].bias)                           # [R, S-1, M]
        h = self.trans_mlp[1]((prev + st.unsqueeze(-2)).unsqueeze(-2) + cur.unsqueeze(-3))
        return self.trans_mlp[2](h).squeeze(-1).float()                           # [R, S-1, K, K]

    def forward(self, scores, emb_cand, logits, hidden):
        """scores [R, S, K, K] del selector -> la misma tabla con la correccion en las filas 1.."""
        S = scores.shape[1]
        if S >= 2:
            d = self.delta(emb_cand, logits, hidden)
            scores = torch.cat((scores[:, :1], scores[:, 1:] + d.to(scores.dtype)), 1)
        if self.escala_arbol is not None:
            e = self.escala_arbol[:S]
            if e.shape[0] < S:
                e = torch.cat((e, e.new_ones(S - e.shape[0])))
            scores = scores * e.view(1, S, 1, 1).to(scores.dtype)
        return scores


def modo_env():
    """Con PN178 aplicado: GENESIS_PN178_CORRECTOR = auto (default: si el checkpoint trae la cabeza),
    1 (obligatoria) o 0 (apagada: la cabeza del checkpoint se descarta y el borrador corre como DFlash2)."""
    v = os.environ.get("GENESIS_PN178_CORRECTOR", "auto").strip().lower()
    return None if v in ("", "auto") else v in ("1", "true", "si", "on")


def crear(modelo: nn.Module, draft_config, hidden_size: int, modo=None) -> None:
    """Desde el __init__ del modelo del borrador. ``modo``: None = si el checkpoint trae la cabeza,
    True = obligatoria (error si no la trae), False = apagada aunque la traiga."""
    cfg = config_de(draft_config)
    estado = (draft_config or {}).get("dcoco_estado", "pares")
    if cfg is not None and estado != "pares" and modo is not False:
        raise ValueError(f"checkpoint DFlash2-DCOCO con dcoco_estado={estado!r}: PN178 solo sirve la cabeza de "
                         "pares (la recurrente es solo de entrenamiento). Apagalo con dcoco=false / "
                         "GENESIS_PN178_CORRECTOR=0")
    if modo is True and cfg is None:
        raise ValueError("DCOCO pedido pero el dflash_config del borrador no trae dcoco_emb_dim: no es un "
                         "checkpoint DFlash2-DCOCO")
    if modo is False or cfg is None:
        modelo.dcoco = None
        return
    modelo.dcoco = CabezaDCOCO(hidden_size, cfg["topk"], cfg["emb"], cfg["state"], cfg["mlp"], cfg["escala"])


def cargar(modelo: nn.Module, cargar_base, weights):
    """Envuelve el load_weights del borrador: sin cabeza, los pesos ``dcoco.*`` se descartan; con
    cabeza, pasan al cargador normal (que los manda a ``model.dcoco``) y se verifica que esten todos."""
    cab = getattr(modelo, "dcoco", None)
    vistos = set()

    def filtrar():
        for nombre, w in weights:
            corto = nombre[len("model."):] if nombre.startswith("model.") else nombre
            if corto.startswith(PREFIJO):
                if cab is None:
                    continue
                vistos.add(corto[len(PREFIJO):])
            yield nombre, w

    r = cargar_base(filtrar())
    if cab is not None:
        faltan = {n for n, _ in cab.named_parameters()} - vistos
        if faltan:
            raise ValueError(f"checkpoint DFlash2-DCOCO incompleto: faltan {sorted(faltan)}")
        cab.eval()
        cab.materializar()
    return r


def corregir(modelo: nn.Module, scores, candidate_ids, unary_logits, hidden_states):
    """Despues del CandidateSelector: la tabla corregida, o la misma si no hay cabeza.

    El embedding de los candidatos sale de ``embed_input_ids`` del borrador (con PN149, en la base
    original, la misma que ve el entrenamiento). Son operaciones de forma fija, aptas para grafo."""
    cab = getattr(modelo, "dcoco", None)
    if cab is None:
        return scores
    R, S, K = candidate_ids.shape
    emb = modelo.embed_input_ids(candidate_ids.reshape(-1)).view(R, S, K, -1)
    return cab(scores, emb, unary_logits.view(R, S, K), hidden_states.view(R, S, -1))
