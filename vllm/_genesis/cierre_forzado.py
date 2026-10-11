# SPDX-License-Identifier: Apache-2.0
"""PN175 — cierre forzado del razonamiento.

El problema (medido el 10-10 con la suite del card): idiotSavant a veces termina el turno (EOS) sin haber
emitido ``</think>``. Todo lo que escribio queda como razonamiento y la respuesta llega vacia, aunque el
razonamiento haya llegado a la solucion. En HLE fue el 12% de las respuestas, y crece con el largo del
razonamiento (0% por debajo de 8k tokens, 25% entre 64k y 82k).

El arreglo: si un pedido con thinking termina por ``stop`` sin ``</think>`` (y sin ``<tool_call>``, que el
parser ya trata como fin del razonamiento), no se corta. Se agrega ``</think>\\n\\n`` a lo generado y se
lanza una continuacion con prompt = prompt original + lo generado + cierre: el modelo escribe la respuesta
condicionada por todo su razonamiento. El prefijo esta en la cache de prefijos, asi que solo cuestan los
tokens de la respuesta (~350 de mediana en HLE).

Medido sobre las 81 preguntas de HLE que no cerraron: el cierre forzado acierta 23, igual que juzgar el
final del razonamiento (24) y con una respuesta de verdad en ``content``.

Para el cliente es transparente: en streaming ve un delta "</think>\\n\\n" (que el parser convierte en el
paso de razonamiento a contenido) y despues la respuesta; sin streaming, un solo mensaje con las dos partes.

No se fuerza: con thinking apagado, si el razonamiento ya termino en el prompt, con beam search, n > 1,
logprobs, entradas multimodales (la continuacion necesitaria las imagenes otra vez) o si queda poco
presupuesto de tokens. ``GENESIS_PN175_MAX_TOKENS`` acota la respuesta forzada (4096 por defecto).
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

log = logging.getLogger("genesis.pn175")

_MAX = int(os.environ.get("GENESIS_PN175_MAX_TOKENS", "4096"))
_CIERRE = "</think>\n\n"
_cuenta = {"forzados": 0}


def _ids(tokenizer: Any) -> tuple[int | None, int | None, list[int]]:
    vocab = tokenizer.get_vocab()
    return (vocab.get("</think>"), vocab.get("<tool_call>"),
            list(tokenizer.encode(_CIERRE, add_special_tokens=False)))


def _aplica(engine_input: Any, sampling_params: Any, chat_template_kwargs: dict | None,
            include_reasoning: bool, reasoning_ended: bool | None) -> bool:
    from vllm.sampling_params import SamplingParams
    if not isinstance(sampling_params, SamplingParams):
        return False                                   # beam search
    if (sampling_params.n or 1) != 1 or sampling_params.logprobs is not None:
        return False
    if not include_reasoning or reasoning_ended:
        return False
    if (chat_template_kwargs or {}).get("enable_thinking", True) is False:
        return False
    try:
        if engine_input.get("mm_placeholders"):
            return False
    except AttributeError:
        return False
    return True


async def envolver(gen: AsyncGenerator, *, engine_client: Any, engine_input: Any, sampling_params: Any,
                   request_id: str, tokenizer: Any, chat_template_kwargs: dict | None,
                   include_reasoning: bool, reasoning_ended: bool | None, lora_request: Any = None,
                   priority: int = 0, data_parallel_rank: int | None = None) -> AsyncGenerator:
    """Pasa ``gen`` tal cual, salvo cuando termina sin cerrar el razonamiento: ahi lo cierra y continua."""
    if tokenizer is None or not _aplica(engine_input, sampling_params, chat_template_kwargs,
                                        include_reasoning, reasoning_ended):
        async for res in gen:
            yield res
        return

    from vllm.sampling_params import RequestOutputKind
    fin_id, tool_id, cierre_ids = _ids(tokenizer)
    if fin_id is None:
        async for res in gen:
            yield res
        return
    delta = sampling_params.output_kind == RequestOutputKind.DELTA
    prompt_ids: list[int] | None = None
    generados: list[int] = []
    texto = ""

    async for res in gen:
        if res.prompt_token_ids is not None and prompt_ids is None:
            prompt_ids = list(res.prompt_token_ids)
        out = res.outputs[0]
        if delta:
            generados += list(out.token_ids)
            texto += out.text
        else:
            generados, texto = list(out.token_ids), out.text
        sin_cerrar = (res.finished and out.finish_reason == "stop" and fin_id not in generados
                      and (tool_id is None or tool_id not in generados))
        restante = (sampling_params.max_tokens or 0) - len(generados) - len(cierre_ids)
        if not sin_cerrar or prompt_ids is None or restante < 64:
            yield res
            continue

        # ── forzar el cierre ──
        _cuenta["forzados"] += 1
        log.warning("[PN175] %s termino sin </think> tras %d tokens: cierre forzado (#%d)",
                    request_id, len(generados), _cuenta["forzados"])
        res.finished, out.finish_reason, out.stop_reason = False, None, None
        if delta:
            yield res
            out.text, out.token_ids = _CIERRE, list(cierre_ids)  # delta sintetico: el parser pasa a contenido
            yield res
        sp = sampling_params.clone()
        sp.max_tokens = min(restante, _MAX)
        sp.min_tokens = 0
        prompt2 = {"prompt_token_ids": prompt_ids + generados + list(cierre_ids)}
        async for r2 in engine_client.generate(prompt2, sp, f"{request_id}-pn175", lora_request=lora_request,
                                               priority=priority, data_parallel_rank=data_parallel_rank,
                                               reasoning_ended=True):
            o2 = r2.outputs[0]
            r2.request_id = res.request_id
            r2.prompt_token_ids = prompt_ids            # el uso reporta el prompt original
            r2.num_cached_tokens = res.num_cached_tokens
            if not delta:                                # respuesta acumulada: razonamiento + cierre + respuesta
                o2.text = texto + _CIERRE + o2.text
                o2.token_ids = generados + list(cierre_ids) + list(o2.token_ids)
            yield r2
        return
