# SPDX-License-Identifier: Apache-2.0
"""PN175 — cierre forzado del razonamiento: si el turno termina sin ``</think>``, se cierra y el modelo
escribe la respuesta como continuacion. Ver ``vllm._genesis.cierre_forzado``."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN175: cierre forzado]"
_OLD = (
    "        assert len(generators) == 1\n"
    "        (result_generator,) = generators\n"
)
_NEW = _OLD + (
    "        from vllm._genesis import cierre_forzado as _g175  # " + MARKER + "\n"
    "        if not isinstance(sampling_params, BeamSearchParams):\n"
    "            result_generator = _g175.envolver(\n"
    "                result_generator, engine_client=self.engine_client, engine_input=engine_input,\n"
    "                sampling_params=sampling_params, request_id=sub_request_id, tokenizer=tokenizer,\n"
    "                chat_template_kwargs=chat_template_kwargs, include_reasoning=request.include_reasoning,\n"
    "                reasoning_ended=locals().get(\"reasoning_ended\"), lora_request=lora_request,\n"
    "                priority=self._get_priority(request, raw_request), data_parallel_rank=data_parallel_rank)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN175")
    log_decision("PN175", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if f is None:
        return "skipped", "falta entrypoints/openai/chat_completion/serving.py"
    p = TextPatcher(patch_name="PN175 cierre forzado", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn175_envolver", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["_g175"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="razonamiento sin cerrar: cierre forzado y respuesta",
                                   patch_name=p.patch_name)
