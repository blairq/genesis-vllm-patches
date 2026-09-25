# SPDX-License-Identifier: Apache-2.0
"""Diagnostico: vuelca el flujo residual DESPUES de cada capa del Qwen3.5, para compararlo capa por
capa con otra implementacion (el forward de transformers de entrenamiento/cuant/).

Por que: con los mismos pesos, el forward de transformers da perplejidad 11,5 donde vLLM da 7,2
sobre las mismas ventanas, y la diferencia aparece recien despues de ~256 tokens. Descartado el
armado (identico al oficial), la GDN (chunk == recurrente), la atencion (sdpa == eager), el
hardware y el dtype. Falta ver EN QUE CAPA se separan.

Uso: GENESIS_DIAG_CAPAS_DUMP=<dir> y el servidor en --enforce-eager (sin torch.compile ni grafos:
si no, el gancho no corre). Vuelca mientras exista `<dir>/VOLCAR`, del rank 0, un archivo
`capas.npy` [capas, tokens, hidden] en fp16 con (hidden_states + residual) de cada capa, o sea el
flujo residual que recibe la capa siguiente.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("genesis.diag_capas")
DIR = os.environ.get("GENESIS_DIAG_CAPAS_DUMP", "").strip()
_buf: dict[int, list] = {}      # capa -> lista de chunks del prefill (se concatenan)
_rank0 = None


def _es_rank0():
    global _rank0
    if _rank0 is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            _rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            _rank0 = True
    return _rank0


def _envolver(cls):
    if "forward" not in cls.__dict__ or getattr(cls, "_genesis_diag_capas", False):
        return False
    f0 = cls.forward

    def forward(self, *a, **kw):
        out = f0(self, *a, **kw)
        try:
            if DIR and os.path.exists(os.path.join(DIR, "VOLCAR")) and _es_rank0():
                h, r = out
                s = (h.float() + r.float()) if r is not None else h.float()
                # el prefill llega en chunks: se acumulan y se reescribe el archivo al final de
                # cada chunk, asi despues del ultimo tiene todas las posiciones
                _buf.setdefault(int(self.layer_idx), []).append(s.half().cpu().numpy())
                n = int(os.environ.get("GENESIS_DIAG_CAPAS_N", "64"))
                if int(self.layer_idx) == n - 1 and len(_buf) == n:
                    arr = np.stack([np.concatenate(_buf[i]) for i in range(n)])
                    np.save(os.path.join(DIR, "capas.npy"), arr)
                    log.warning("[DIAG capas] volcado %s en %s", arr.shape, DIR)
        except Exception as e:  # noqa: BLE001
            log.error("[DIAG capas] %s: %s", type(e).__name__, e)
        return out

    cls.forward = forward
    cls._genesis_diag_capas = True
    return True


def instalar() -> None:
    if not DIR:
        return
    from vllm.model_executor.models import qwen3_5, qwen3_next
    hechos = [c.__name__ for c in (qwen3_5.Qwen3_5DecoderLayer, qwen3_next.Qwen3NextDecoderLayer) if _envolver(c)]
    log.warning("[DIAG capas] volcado del flujo residual por capa en %s (envueltas: %s)", DIR, hechos)
