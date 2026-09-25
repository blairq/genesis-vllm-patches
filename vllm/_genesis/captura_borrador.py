# SPDX-License-Identifier: Apache-2.0
"""Captura de datos para ajustar el borrador DFlash2 contra noon TAL COMO LO SERVIMOS.

Que se captura
--------------
El borrador no ve el checkpoint de noon: ve la salida de nuestro runtime (W4A8, lm_head int4, KV
int8, fp16). Lo que consume de cada token del target es ``combine_hidden_states(cat(aux))``: los
estados de las capas 5/19/33/47/61 ya proyectados por su ``fc`` a 5120 (10 KB por token en fp16).
Eso es lo que se guarda, junto con el id del token y su posicion.

Se captura SOLO en los pasos de PREFILL: el banco regenera la respuesta de noon y despues la
re-envia como prompt (prompt + respuesta, ``max_tokens=1``, ``cache_salt`` unico para que el
prefix-cache no saltee tokens). Asi sale la secuencia entera y contigua, sin el lio de las
posiciones rechazadas ni del arbol. Las etiquetas distribucionales salen del mismo pedido con
``prompt_logprobs`` (probado: vLLM los da con decodificacion especulativa).

Como se usa
-----------
``GENESIS_CAPTURA_BORRADOR=1`` instala el envoltorio. La captura solo corre mientras exista el
archivo ``<dir>/CAPTURAR`` (default ``/traces/captura``): el que maneja el banco lo crea antes del
re-prefill y lo borra despues, asi la regeneracion no se captura. Solo escribe el rank 0 de TP
(los estados son iguales en los dos ranks despues del all-reduce).

La bandera puede tener adentro una posicion minima: solo se guardan los tokens con ``pos >= pmin``.
El borrador mira hacia atras 2048 tokens (ventana deslizante), asi que alcanza con guardar desde
2048 antes del comienzo de la respuesta: ~40 MB por pedido en vez de ~300 MB con el contexto entero.

Un archivo ``<req_id>.<pos0>.npz`` por pedido y por chunk de prefill, con ``pos`` (int32), ``ids``
(int32) y ``feat`` (float16 [n, 5120]).
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("genesis.captura")

DIR = os.environ.get("GENESIS_CAPTURA_BORRADOR_DIR", "/traces/captura")
_BANDERA = os.path.join(DIR, "CAPTURAR")
_avisado = False
_rank0 = None


def _es_rank0() -> bool:
    global _rank0
    if _rank0 is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            _rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            _rank0 = True
    return _rank0


def _pmin() -> int:
    try:
        with open(_BANDERA) as f:
            t = f.read().strip()
        return int(t) if t else 0
    except (OSError, ValueError):
        return 0


def _volcar(spec, input_batch) -> None:
    R = int(input_batch.num_reqs)
    if R == 0:
        return
    pre = np.asarray(input_batch.is_prefilling_np[:R])
    if not pre.any():
        return
    qsl = np.asarray(input_batch.query_start_loc_np[: R + 1])
    N = int(qsl[R])
    feat = spec.hidden_states[:N].detach().to("cpu")
    ids = input_batch.input_ids[:N].detach().to("cpu").numpy().astype(np.int32)
    pos = input_batch.positions[:N].detach().to("cpu").numpy().astype(np.int32)
    feat = feat.half().numpy()
    os.makedirs(DIR, exist_ok=True)
    pmin = _pmin()
    for r in range(R):
        if not pre[r]:
            continue
        a, b = int(qsl[r]), int(qsl[r + 1])
        while a < b and pos[a] < pmin:
            a += 1
        if b <= a:
            continue
        rid = str(input_batch.req_ids[r]).replace("/", "_")
        np.savez(os.path.join(DIR, f"{rid}.{int(pos[a]):07d}.npz"),
                 pos=pos[a:b], ids=ids[a:b], feat=feat[a:b])


def instalar() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as m
    cls = m.DFlash2Speculator
    if getattr(cls, "_genesis_captura", False):
        return
    propose0 = cls.propose

    def propose(self, input_batch, *a, **kw):
        out = propose0(self, input_batch, *a, **kw)
        global _avisado
        if kw.get("dummy_run") or kw.get("is_profile"):
            return out
        try:
            if os.path.exists(_BANDERA) and _es_rank0():
                _volcar(self, input_batch)
        except Exception as e:  # noqa: BLE001
            if not _avisado:
                log.error("[CAPTURA] fallo el volcado: %s: %s", type(e).__name__, e)
                _avisado = True
        return out

    cls.propose = propose
    cls._genesis_captura = True
    log.warning("[CAPTURA] borrador DFlash2: volcado de features de prefill en %s (bandera %s)",
                DIR, _BANDERA)
