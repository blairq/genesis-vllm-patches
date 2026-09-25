# SPDX-License-Identifier: Apache-2.0
"""PN147 — saneo de pedidos para decodificacion especulativa, en el servidor.

Por que
-------
Hasta el 2026-09-24 todo el trafico entraba por el proxy j-space, que ademas de sus inyecciones
"cognitivas" hacia dos arreglos sin los que opencode se rompe con DFlash2:

1. Borraba ``min_p``. vLLM 0.29.0 RECHAZA el pedido entero si trae ``min_p > 0`` o
   ``logit_bias`` con decodificacion especulativa
   (``SamplingParams._validate_spec_decode``). En streaming el error llega ADENTRO del SSE con
   HTTP 200, asi que el cliente ve una respuesta vacia. opencode manda ``min_p`` (0,06-0,08) en
   casi todos sus proveedores.
2. Borraba ``thinking_budget_message``. Segun el proxy, con decodificacion especulativa el
   mensaje forzado al agotar el presupuesto de razonamiento dispara un bucle de tokens
   (rechazo en ciclo). El agente ``coder`` de opencode lo manda.

Sacar el proxy del camino (servir vLLM directo, como antes) exige que esos dos arreglos vivan
aca, para CUALQUIER cliente (opencode, Hermes, curl) y en todos los endpoints.

Que hace
--------
* ``min_p`` / ``logit_bias`` con decodificacion especulativa: en vez de rechazar, los anula y
  avisa una vez en el log. Es lo mismo que hacia el proxy; el resto del muestreo queda igual.
* ``thinking_budget_message`` del chat: se descarta al armar los SamplingParams.

Lo demas del proxy (tope de razonamiento a 4096, max_tokens, compresion de contexto, textos
inyectados) NO se replica: servir directo significa respetar lo que pide el cliente.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN147: saneo para spec decode]"

_SP_OLD = (
    "        # Some sampling parameters are not yet compatible with spec decoding.\n"
    "        if self.min_p > _SAMPLING_EPS or self.logit_bias:\n"
    "            raise VLLMValidationError(\n"
    "                \"The min_p and logit_bias sampling parameters \"\n"
    "                \"are not yet supported with speculative decoding.\"\n"
    "            )\n"
)
_SP_NEW = (
    "        # Some sampling parameters are not yet compatible with spec decoding.\n"
    "        # " + MARKER + " anularlos en vez de rechazar el pedido entero (lo que hacia\n"
    "        # el proxy j-space). En streaming el rechazo llegaba con HTTP 200 y cuerpo vacio.\n"
    "        if self.min_p > _SAMPLING_EPS or self.logit_bias:\n"
    "            global _GENESIS_PN147_AVISADO\n"
    "            try:\n"
    "                _GENESIS_PN147_AVISADO\n"
    "            except NameError:\n"
    "                _GENESIS_PN147_AVISADO = False\n"
    "            if not _GENESIS_PN147_AVISADO:\n"
    "                logger.warning('[Genesis PN147] min_p=%s / logit_bias no van con '\n"
    "                               'decodificacion especulativa: se anulan (aviso unico)',\n"
    "                               self.min_p)\n"
    "                _GENESIS_PN147_AVISADO = True\n"
    "            self.min_p = 0.0\n"
    "            self.logit_bias = None\n"
)

_CHAT_OLD = "            thinking_budget_message=self.thinking_budget_message,\n"
_CHAT_NEW = (
    "            # " + MARKER + " con spec decode el mensaje forzado cicla; se descarta.\n"
    "            thinking_budget_message=None,\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN147")
    log_decision("PN147", decision, reason)
    if not decision:
        return "skipped", reason

    sp = resolve_vllm_file("sampling_params.py")
    chat = resolve_vllm_file("entrypoints/openai/chat_completion/protocol.py")
    if sp is None:
        return "skipped", "sampling_params.py no esta en esta version de vLLM"

    p = TextPatcher(
        patch_name="PN147 saneo min_p/logit_bias", target_file=str(sp), marker=MARKER,
        sub_patches=[TextPatch(name="pn147_min_p", anchor=_SP_OLD, replacement=_SP_NEW, required=True)],
        upstream_drift_markers=["_GENESIS_PN147_AVISADO"])
    result, failure = p.apply()
    estado = result_to_wiring_status(
        result, failure, applied_message="min_p/logit_bias se anulan con spec decode",
        patch_name=p.patch_name)
    if estado[0] != "applied" or chat is None:
        return estado

    pc = TextPatcher(
        patch_name="PN147 saneo thinking_budget_message", target_file=str(chat), marker=MARKER,
        sub_patches=[TextPatch(name="pn147_tbm", anchor=_CHAT_OLD, replacement=_CHAT_NEW, required=True)],
        upstream_drift_markers=[])
    r2, f2 = pc.apply()
    e2 = result_to_wiring_status(
        r2, f2, applied_message="thinking_budget_message descartado", patch_name=pc.patch_name)
    if e2[0] != "applied":
        return e2
    return "applied", "min_p/logit_bias anulados con spec decode; thinking_budget_message descartado"
