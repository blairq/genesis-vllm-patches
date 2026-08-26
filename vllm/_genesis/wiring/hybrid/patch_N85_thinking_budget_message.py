# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N85 — mensaje inyectado al agotarse el presupuesto de thinking.

================================================================
QUE PROBLEMA RESUELVE
================================================================

vLLM ya trae `thinking_token_budget` (SamplingParams + API OpenAI): cuando el
razonamiento se pasa del presupuesto, fuerza los tokens de cierre y corta. Eso
arregla el caso patologico medido en este rig:

    sin budget        213 palabras de reasoning, content VACIO, finish=length
    budget=32          18 palabras, respuesta completa,          finish=stop

Pero corta EN SECO. El modelo pasa de estar a mitad de un analisis a tener que
responder, sin ninguna señal de que lo cortaron, y tipicamente reabre el
analisis desde cero en el content o arranca la respuesta a mitad de una frase.

Lo que falta es poder decirle ALGO en ese momento: "te cortaron, no rehagas el
analisis, usa lo que ya tenes". vLLM no tiene nada de eso — el unico texto que
se emite al forzar son los tokens de cierre.

================================================================
POR QUE NO ALCANZA CON --reasoning-config
================================================================

`ReasoningConfig` expone `reasoning_end_str` y es settable por CLI
(`--reasoning-config`), asi que la tentacion es poner ahi
`"...mensaje...</think>"`. Mecanicamente el forzador lo soportaria: recorre
`think_end_token_ids` token por token con un indice (`end_count`), asi que
acepta secuencias de cualquier largo.

PERO ESA MISMA LISTA CUMPLE DOS ROLES:

    forzar    _apply_forcing_to_logits:504   think_end_token_ids[end_count]
              _update_think_state:416,431
    DETECTAR  _init_state_entry:198          _find_last_sequence_index(prompt)
              _update_think_state:251        _find_last_sequence_index(output)
                                :333         "ya cerro por su cuenta?"

Si se contamina con un mensaje, la DETECCION deja de matchear: el modelo emite
`</think>` pelado, la busqueda pide `mensaje+</think>` y no lo encuentra nunca.
Consecuencia: en un turno que ya cerro bien el bloque de razonamiento, la
maquina sigue contando y termina forzando el cierre EN MEDIO DE LA RESPUESTA.
Silencioso y peor que el problema original.

================================================================
QUE HACE PN85
================================================================

Separa los dos roles. `think_end_token_ids` queda intacta para detectar, y se
agrega una secuencia de FORZADO por request:

    force_seq = tokens(mensaje) + think_end_token_ids

Cuando la request no trae mensaje, `force_seq` ES `think_end_token_ids` y el
comportamiento es identico al de upstream, byte por byte.

Y el mensaje viaja POR REQUEST, no por engine, que es el requisito real: se
configura desde el cliente (opencode lo manda en el extraBody del canal, al
lado de temperature/max_tokens) sin reiniciar el server ni afectar a los otros
canales.

    {"thinking_token_budget": 16384,
     "thinking_budget_message": "\\n\\n[Tu razonamiento fue cortado...]"}

El mensaje cae DENTRO del bloque <think>, justo antes del cierre. Asi el modelo
genera la respuesta condicionado a haberlo leido, que es exactamente lo que se
busca; y como queda del lado del razonamiento, el parser lo separa del content
igual que a todo lo demas.

================================================================
DONDE TOCA (4 archivos)
================================================================

 1. sampling_params.py
    Dos campos nuevos: `thinking_budget_message` (texto, lo que manda el
    cliente) y `thinking_budget_message_token_ids` (lo que consume el sampler).

 2. entrypoints/openai/chat_completion/protocol.py
    Acepta `thinking_budget_message` en el request y lo pasa a SamplingParams.

 3. v1/engine/input_processor.py
    Tokeniza. Es EL lugar: tiene tokenizer (`self.get_tokenizer()`) y ya es
    donde vLLM valida `thinking_token_budget`, asi que el mensaje se valida
    junto con el presupuesto del que depende. En el protocolo no hay tokenizer
    y en el sampler tampoco.

 4. v1/sample/thinking_budget_state.py
    La separacion en si: `force_seq` por request, usada SOLO en los 4 puntos
    de forzado. Los 3 de deteccion no se tocan.

================================================================
COMPATIBILIDAD
================================================================

- Sin `thinking_budget_message` en la request: comportamiento identico a
  upstream (force_seq colapsa a think_end_token_ids).
- Sin `thinking_token_budget`: el holder ni siquiera crea estado para la
  request, asi que PN85 no participa.
- El campo extra en el request es aditivo; los clientes que no lo mandan no se
  enteran.
- Default ON. Kill switch: GENESIS_DISABLE_PN85=1.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN85_MARKER = "[Genesis PN85 thinking budget message]"


# ---------------------------------------------------------------------------
# 1. sampling_params.py — los dos campos nuevos
# ---------------------------------------------------------------------------

SP_OLD = (
    "    thinking_token_budget: int | None = None\n"
    '    """Maximum number of tokens allowed for thinking operations."""\n'
)

SP_NEW = (
    "    thinking_token_budget: int | None = None\n"
    '    """Maximum number of tokens allowed for thinking operations."""\n'
    "    # " + GENESIS_PN85_MARKER + "\n"
    "    thinking_budget_message: str | None = None\n"
    '    """Texto a inyectar dentro del bloque de razonamiento cuando se agota\n'
    "    thinking_token_budget, justo antes de los tokens de cierre. Sin esto el\n"
    "    corte es en seco y el modelo suele rehacer el analisis en el content.\n"
    '    Lo tokeniza el input processor. Ver wiring/hybrid/patch_N85_*."""\n'
    "    thinking_budget_message_token_ids: list[int] | None = None\n"
    '    """Version tokenizada de thinking_budget_message. La llena el input\n'
    "    processor (es el unico punto del camino que tiene tokenizer Y ya valida\n"
    '    thinking_token_budget). No setear a mano."""\n'
)


# 1b. from_optional() — el camino que usa la API OpenAI (no el constructor)
SP_FO_SIG_OLD = (
    "        thinking_token_budget: int | None = None,\n"
    "        include_stop_str_in_output: bool = False,\n"
)

SP_FO_SIG_NEW = (
    "        thinking_token_budget: int | None = None,\n"
    "        # " + GENESIS_PN85_MARKER + "\n"
    "        thinking_budget_message: str | None = None,\n"
    "        include_stop_str_in_output: bool = False,\n"
)

SP_FO_PASS_OLD = (
    "            thinking_token_budget=thinking_token_budget,\n"
    "            include_stop_str_in_output=include_stop_str_in_output,\n"
)

SP_FO_PASS_NEW = (
    "            thinking_token_budget=thinking_token_budget,\n"
    "            # " + GENESIS_PN85_MARKER + "\n"
    "            thinking_budget_message=thinking_budget_message,\n"
    "            include_stop_str_in_output=include_stop_str_in_output,\n"
)



# ---------------------------------------------------------------------------
# 2. chat_completion/protocol.py — aceptar y reenviar
# ---------------------------------------------------------------------------

PROTO_FIELD_OLD = (
    "    thinking_token_budget: ThinkingTokenBudget = None\n"
    "    include_reasoning: bool = True\n"
)

PROTO_FIELD_NEW = (
    "    thinking_token_budget: ThinkingTokenBudget = None\n"
    "    # " + GENESIS_PN85_MARKER + "\n"
    "    thinking_budget_message: str | None = None\n"
    "    include_reasoning: bool = True\n"
)

PROTO_PASS_OLD = (
    "            thinking_token_budget=self.thinking_token_budget,\n"
    "            allowed_token_ids=self.allowed_token_ids,\n"
)

PROTO_PASS_NEW = (
    "            thinking_token_budget=self.thinking_token_budget,\n"
    "            # " + GENESIS_PN85_MARKER + "\n"
    "            thinking_budget_message=self.thinking_budget_message,\n"
    "            allowed_token_ids=self.allowed_token_ids,\n"
)


# ---------------------------------------------------------------------------
# 3. input_processor.py — tokenizar donde ya se valida el presupuesto
# ---------------------------------------------------------------------------

IP_OLD = (
    "                if self.use_v2_model_runner:\n"
    "                    raise VLLMValidationError(\n"
    '                        "thinking_token_budget is not yet supported by the V2 "\n'
    '                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "\n'
    '                        "to use thinking_token_budget."\n'
    "                    )\n"
)

IP_NEW = (
    "                if self.use_v2_model_runner:\n"
    "                    raise VLLMValidationError(\n"
    '                        "thinking_token_budget is not yet supported by the V2 "\n'
    '                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "\n'
    '                        "to use thinking_token_budget."\n'
    "                    )\n"
    "                # " + GENESIS_PN85_MARKER + "\n"
    "                # Aca y no en el protocolo: este es el unico punto del\n"
    "                # camino que tiene tokenizer y ademas ya valida el\n"
    "                # presupuesto del que el mensaje depende.\n"
    "                _g85_msg = getattr(params, 'thinking_budget_message', None)\n"
    "                if _g85_msg:\n"
    "                    params.thinking_budget_message_token_ids = (\n"
    "                        self.get_tokenizer().encode(\n"
    "                            _g85_msg, add_special_tokens=False\n"
    "                        )\n"
    "                    )\n"
)


# ---------------------------------------------------------------------------
# 4. thinking_budget_state.py — separar forzado de deteccion
# ---------------------------------------------------------------------------

# 4a. helper de modulo + secuencia de forzado por request
TB_HELPER_OLD = (
    "class ThinkingBudgetStateHolder:\n"
    '    """Tracks thinking sections and forces end tokens when budget is exceeded."""\n'
)

TB_HELPER_NEW = (
    "# " + GENESIS_PN85_MARKER + "\n"
    "def _g85_force_seq(holder, state):\n"
    '    """Secuencia que se FUERZA al agotarse el presupuesto.\n'
    "\n"
    "    Es mensaje+cierre si la request trajo mensaje, y el cierre pelado si no\n"
    "    (ahi colapsa a think_end_token_ids y el comportamiento es el de\n"
    "    upstream). OJO: esto es SOLO para forzar. La deteccion de 'el modelo ya\n"
    "    cerro por su cuenta' tiene que seguir usando holder.think_end_token_ids,\n"
    "    porque el modelo emite el cierre pelado; si se busca mensaje+cierre no\n"
    "    matchea nunca y se termina forzando un cierre en medio de la respuesta.\n"
    '    """\n'
    "    seq = state.get('g85_force_seq')\n"
    "    return seq if seq else holder.think_end_token_ids\n"
    "\n"
    "\n"
    "class ThinkingBudgetStateHolder:\n"
    '    """Tracks thinking sections and forces end tokens when budget is exceeded."""\n'
)

# 4b. armar force_seq al dar de alta la request
TB_SYNC_OLD = (
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
    '                self._state[index]["output_tok_ids"] = output_tok_ids\n'
)

TB_SYNC_NEW = (
    "                self._state[index] = self._init_state_entry(\n"
    "                    prompt_tok_ids, thinking_token_budget\n"
    "                )\n"
    "                # " + GENESIS_PN85_MARKER + "\n"
    "                _g85_ids = getattr(\n"
    "                    params, 'thinking_budget_message_token_ids', None\n"
    "                )\n"
    "                self._state[index]['g85_force_seq'] = (\n"
    "                    list(_g85_ids) + list(self.think_end_token_ids)\n"
    "                    if _g85_ids else None\n"
    "                )\n"
    '                self._state[index]["output_tok_ids"] = output_tok_ids\n'
)

# 4c. punto de forzado: "el rejection sampler tiro el token de cierre?"
TB_F1_OLD = (
    "            stopping_thinking = (\n"
    '                self.think_end_token_ids[state["end_count"]] in new_tokens\n'
    "            )\n"
)

TB_F1_NEW = (
    "            stopping_thinking = (\n"
    "                # " + GENESIS_PN85_MARKER + " (forzado)\n"
    '                _g85_force_seq(self, state)[state["end_count"]] in new_tokens\n'
    "            )\n"
)

# 4d. punto de forzado: avance de end_count contra los drafts de MTP
TB_F2_OLD = (
    '                    if state["end_count"] + 1 < len(self.think_end_token_ids):\n'
    '                        if token_id == self.think_end_token_ids[state["end_count"] + 1]:\n'
)

TB_F2_NEW = (
    "                    # " + GENESIS_PN85_MARKER + " (forzado)\n"
    '                    _g85_seq = _g85_force_seq(self, state)\n'
    '                    if state["end_count"] + 1 < len(_g85_seq):\n'
    '                        if token_id == _g85_seq[state["end_count"] + 1]:\n'
)

# 4d-bis. Con MENSAJE hay que emitir la secuencia EXACTA, y la rama de
# spec-decode no lo garantiza: acepta drafts que "matchean" y avanza el indice
# de a varios por paso. Con la secuencia corta de upstream (1-2 tokens de
# cierre) da igual; con un mensaje largo SALTEA TOKENS. Medido: se mando
# "fue cortado por limite ... No rehagas el analisis" y salio
# "fue interr limite ... No rehagas todo".
# Cuando hay mensaje se ignoran los drafts y se fuerza posicion 0 en cada paso,
# que es exactamente lo que hace la rama sin spec. Cuesta velocidad solo
# durante los pocos tokens del mensaje.
TB_F2B_OLD = (
    "        else:\n"
    '            state["force_index"] = []\n'
    '            if len(state["spec_token_ids"]) > 0:\n'
)

TB_F2B_NEW = (
    "        else:\n"
    '            state["force_index"] = []\n'
    "            # " + GENESIS_PN85_MARKER + " (exactitud con MTP)\n"
    "            if state.get('g85_force_seq'):\n"
    "                # NO se predice cuantos tokens sobreviven al rejection\n"
    "                # sampling: se MIDE cuanto de la secuencia ya salio,\n"
    "                # matcheando la cola de la salida real contra el prefijo\n"
    "                # de la secuencia. Se autocorrige pase lo que pase con los\n"
    "                # drafts de MTP. Predecirlo dio texto corrupto de tres\n"
    "                # formas distintas (tokens salteados, drafts intercalados,\n"
    "                # y contabilidad rota que ni respetaba el presupuesto).\n"
    "                _g85_s = state['g85_force_seq']\n"
    "                _g85_o = state.get('output_tok_ids') or []\n"
    "                _g85_k = 0\n"
    "                for _g85_i in range(min(len(_g85_s), len(_g85_o)), 0, -1):\n"
    "                    if list(_g85_o[-_g85_i:]) == list(_g85_s[:_g85_i]):\n"
    "                        _g85_k = _g85_i\n"
    "                        break\n"
    '                state["end_count"] = _g85_k\n'
    "                # Todas las posiciones del paso, no solo la 0: forzar\n"
    "                # una sola deja que MTP acepte sus drafts en las otras\n"
    "                # y el texto sale con palabras del modelo intercaladas.\n"
    "                state['force_index'] = list(\n"
    "                    range(len(state['spec_token_ids']) + 1)\n"
    "                )\n"
    "            elif len(state[\"spec_token_ids\"]) > 0:\n"
)

# 4e. punto de forzado: cuando termino de emitir la secuencia entera
TB_F3_OLD = (
    '            if state["end_count"] >= len(self.think_end_token_ids):\n'
)

TB_F3_NEW = (
    "            # " + GENESIS_PN85_MARKER + " (forzado)\n"
    '            if state["end_count"] >= len(_g85_force_seq(self, state)):\n'
)

# 4f. punto de forzado: el token concreto que se mete en los logits
TB_F4_OLD = (
    "                    for force_idx in force_index:\n"
    "                        if end_count < len(self.think_end_token_ids):\n"
    "                            mask_idx = self.cu_num_tokens[seq_idx] + force_idx\n"
    "                            if (\n"
    "                                mask_idx < self._mask_capacity\n"
    "                                and mask_idx < logits.shape[0]\n"
    "                            ):\n"
    "                                active_indices_cpu.append(mask_idx)\n"
    "                                force_tokens_cpu.append(\n"
    "                                    self.think_end_token_ids[end_count]\n"
    "                                )\n"
)

TB_F4_NEW = (
    "                    # " + GENESIS_PN85_MARKER + " (forzado)\n"
    "                    _g85_seq = _g85_force_seq(self, state)\n"
    "                    _g85_msg = bool(state.get('g85_force_seq'))\n"
    "                    for _g85_k, force_idx in enumerate(force_index):\n"
    "                        _g85_ec = end_count + _g85_k if _g85_msg else end_count\n"
    "                        if _g85_ec < len(_g85_seq):\n"
    "                            mask_idx = self.cu_num_tokens[seq_idx] + force_idx\n"
    "                            if (\n"
    "                                mask_idx < self._mask_capacity\n"
    "                                and mask_idx < logits.shape[0]\n"
    "                            ):\n"
    "                                active_indices_cpu.append(mask_idx)\n"
    "                                force_tokens_cpu.append(\n"
    "                                    _g85_seq[_g85_ec]\n"
    "                                )\n"
)


def _patchers() -> list[TextPatcher] | None:
    objetivos = {
        "sampling_params.py": [
            TextPatch(
                name="pn85_sampling_params_fields",
                anchor=SP_OLD,
                replacement=SP_NEW,
                required=True,
            ),
            # La API OpenAI construye con from_optional(), no con el
            # constructor: sin estos dos, un request con el campo muere en
            # "unexpected keyword argument".
            TextPatch(
                name="pn85_from_optional_signature",
                anchor=SP_FO_SIG_OLD,
                replacement=SP_FO_SIG_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_from_optional_passthrough",
                anchor=SP_FO_PASS_OLD,
                replacement=SP_FO_PASS_NEW,
                required=True,
            ),
        ],
        "entrypoints/openai/chat_completion/protocol.py": [
            TextPatch(
                name="pn85_request_field",
                anchor=PROTO_FIELD_OLD,
                replacement=PROTO_FIELD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_request_passthrough",
                anchor=PROTO_PASS_OLD,
                replacement=PROTO_PASS_NEW,
                required=True,
            ),
        ],
        "v1/engine/input_processor.py": [
            TextPatch(
                name="pn85_tokenize_message",
                anchor=IP_OLD,
                replacement=IP_NEW,
                required=True,
            ),
        ],
        "v1/sample/thinking_budget_state.py": [
            TextPatch(
                name="pn85_helper",
                anchor=TB_HELPER_OLD,
                replacement=TB_HELPER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_build_force_seq",
                anchor=TB_SYNC_OLD,
                replacement=TB_SYNC_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_force_point_rejection",
                anchor=TB_F1_OLD,
                replacement=TB_F1_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_force_point_spec_advance",
                anchor=TB_F2_OLD,
                replacement=TB_F2_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_exact_sequence_under_mtp",
                anchor=TB_F2B_OLD,
                replacement=TB_F2B_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_force_point_done",
                anchor=TB_F3_OLD,
                replacement=TB_F3_NEW,
                required=True,
            ),
            TextPatch(
                name="pn85_force_point_logits",
                anchor=TB_F4_OLD,
                replacement=TB_F4_NEW,
                required=True,
            ),
        ],
    }

    patchers: list[TextPatcher] = []
    for rel, sub in objetivos.items():
        target = resolve_vllm_file(rel)
        if target is None:
            return None
        patchers.append(
            TextPatcher(
                patch_name=f"PN85 thinking budget message ({rel})",
                target_file=str(target),
                marker=GENESIS_PN85_MARKER,
                sub_patches=sub,
                upstream_drift_markers=[
                    # señales de que upstream ya separo forzado de deteccion
                    "thinking_budget_message",
                    "force_seq",
                ],
            )
        )
    return patchers


def apply() -> tuple[str, str]:
    import os

    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN85")
    log_decision("PN85", decision, reason)
    if not decision:
        return "skipped", reason
    if os.environ.get("GENESIS_DISABLE_PN85") == "1":
        return "skipped", "GENESIS_DISABLE_PN85=1 (kill switch)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    ps = _patchers()
    if ps is None:
        return "skipped", "algun archivo objetivo de PN85 no existe"

    ultimo_result = None
    ultimo_failure = None
    for p in ps:
        result, failure = p.apply()
        ultimo_result, ultimo_failure = result, failure
        if failure:
            # Un fallo parcial deja el engine con la mitad del parche: mejor
            # reportarlo fuerte que seguir.
            return result_to_wiring_status(
                result,
                failure,
                applied_message="(no deberia llegar aca: hubo failure)",
                patch_name=p.patch_name,
            )

    return result_to_wiring_status(
        ultimo_result,
        ultimo_failure,
        applied_message=(
            "PN85 applied: thinking_budget_message por request. Cuando se agota "
            "thinking_token_budget, en vez de cortar el razonamiento en seco se "
            "inyecta el texto que mando el cliente justo antes del cierre, para "
            "que el modelo responda sabiendo que lo cortaron en vez de rehacer "
            "el analisis. Separa la secuencia de FORZADO de la de DETECCION: "
            "contaminar think_end_token_ids (que es lo que permitiria hacerlo "
            "con --reasoning-config solo) rompe la deteccion de cierre natural "
            "y termina forzando un </think> en medio de la respuesta. "
            "Sin mensaje en la request el comportamiento es identico a upstream. "
            "Kill switch: GENESIS_DISABLE_PN85=1."
        ),
        patch_name="PN85 thinking budget message",
    )


def is_applied() -> bool:
    """Reporter para verify_live_rebinds en apply_all.py."""
    if vllm_install_root() is None:
        return False
    ps = _patchers()
    if ps is None:
        return False
    for p in ps:
        try:
            with open(p.target_file, encoding="utf-8") as fh:
                if GENESIS_PN85_MARKER not in fh.read():
                    return False
        except OSError:
            return False
    return True
