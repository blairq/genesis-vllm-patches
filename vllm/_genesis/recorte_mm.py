# SPDX-License-Identifier: Apache-2.0
"""PN176 — imagenes por turno, no por conversacion.

El problema (16-10): vLLM valida ``--limit-mm-per-prompt`` contra TODAS las imagenes del pedido. Un cliente
de chat o de agente reenvia la conversacion entera en cada turno, asi que con un limite de 2, despues de la
tercera imagen cada turno vuelve con 400 y la conversacion queda trabada. Pero las imagenes de turnos
anteriores ya estan en la cache de prefijos: el scheduler solo corre el encoder sobre las imagenes cuyos
tokens caen en la parte del prompt que hay que calcular (``_try_schedule_encoder_inputs``), y el
procesador del API server las encuentra en su cache por hash. Lo caro es una imagen NUEVA.

El arreglo, sin estado y sin tocar nunca el prefijo:
- **Por turno:** un turno es lo que hay despues de un mensaje del asistente. En cada turno quedan como
  mucho ``GENESIS_PN176_POR_TURNO`` imagenes (2 por defecto), las mas nuevas; las otras se reemplazan por
  una nota de texto. La decision de cada turno depende solo de ese turno, asi que un turno viejo da siempre
  el mismo texto: el prompt de la conversacion solo crece por el final y la cache de prefijos sirve entera.
- **Total:** ``--limit-mm-per-prompt`` queda como techo de la conversacion (memoria del API server si hay que
  reprocesar todo, p. ej. tras un reinicio). Si se pasa, se omiten las mas viejas de a bloques de
  h = ceil(N/2): las d mas viejas con d = h * ceil((n - N) / h). d queda fijo durante h imagenes seguidas,
  asi que el prefijo cambia una vez cada h imagenes y no en cada una.

No toca nada si el limite de la modalidad es 0 (sigue dando el 400 de siempre) o si el modelo no es
multimodal. El limite total se lee de la config del modelo; el de por turno, del entorno.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.pn176")

_POR_TURNO = int(os.environ.get("GENESIS_PN176_POR_TURNO", "2"))

# tipo de parte -> modalidad (los de chat_utils.MM_PARSER_MAP)
_MODALIDAD = {
    "image_url": "image", "input_image": "image", "image_pil": "image", "image_embeds": "image",
    "video_url": "video", "video_embeds": "video",
    "audio_url": "audio", "input_audio": "audio", "audio_embeds": "audio",
}
_NOMBRE = {"image": "image", "video": "video", "audio": "audio clip"}


def _modalidad(parte: Any) -> str | None:
    if not isinstance(parte, dict):
        return None
    t = parte.get("type")
    if t is None:                       # partes sin "type": vLLM las reconoce por la clave
        t = next((k for k in _MODALIDAD if k in parte), None)
    return _MODALIDAD.get(t) if t else None


def _plural(m: str, n: int) -> str:
    return f"{n} {_NOMBRE[m]}{'s' if n != 1 else ''}"


def _omitir(request: Any, lugares: list[tuple[int, int]], nota: str) -> None:
    for i, j in lugares:
        msg = request.messages[i]
        contenido = list(msg["content"])            # copia: no mutar listas compartidas con el cliente
        contenido[j] = {"type": "text", "text": nota}
        msg["content"] = contenido


def recortar(request: Any, model_config: Any) -> None:
    """Como mucho ``_POR_TURNO`` items nuevos por turno y el limite de vLLM en total, sin cambiar el prefijo."""
    mm = getattr(model_config, "multimodal_config", None)
    if mm is None:
        return
    # (modalidad) -> lista de turnos, cada turno una lista de (mensaje, parte)
    turnos: dict[str, list[list[tuple[int, int]]]] = {}
    turno = 0
    for i, msg in enumerate(request.messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            turno += 1
            continue
        contenido = msg.get("content")
        if not isinstance(contenido, list):
            continue
        for j, parte in enumerate(contenido):
            m = _modalidad(parte)
            if m is not None:
                por_m = turnos.setdefault(m, [])
                while len(por_m) <= turno:
                    por_m.append([])
                por_m[turno].append((i, j))
    for m, por_turno in turnos.items():
        limite = mm.get_limit_per_prompt(m)
        if limite <= 0:
            continue
        x = min(_POR_TURNO, limite) if _POR_TURNO > 0 else limite
        quedan: list[tuple[int, int]] = []
        omitidas = 0
        for lugares in por_turno:
            if len(lugares) > x:
                _omitir(request, lugares[: len(lugares) - x],
                        f"[{_NOMBRE[m]} omitted: this server processes at most {_plural(m, x)} per turn]")
                omitidas += len(lugares) - x
                lugares = lugares[len(lugares) - x:]
            quedan += lugares
        if omitidas and por_turno and len(por_turno[-1]) > x:
            log.warning("[PN176] turno con %d %s nuevas, limite por turno %d: se omiten las %d mas viejas",
                        len(por_turno[-1]), m, x, len(por_turno[-1]) - x)
        if len(quedan) > limite:
            h = (limite + 1) // 2
            d = h * -(-(len(quedan) - limite) // h)     # multiplo de h: estable durante h items
            _omitir(request, quedan[:d],
                    f"[earlier {_NOMBRE[m]} omitted: this server keeps at most {_plural(m, limite)} per conversation]")
            log.warning("[PN176] %d %s en la conversacion, limite %d: se omiten las %d mas viejas",
                        len(quedan), m, limite, d)
