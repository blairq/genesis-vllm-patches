# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N89 — Native KV offload requests endpoint and activity tracker."""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN89_MARKER = "[Genesis PN89 kv offload requests endpoint]"

_IMPORT = "from vllm._genesis import kv_offload_tracker as _g89"

# ─────────────────── chat_completion/api_router.py ───────────────────

A1_OLD = (
    "    generator = await handler.create_chat_completion(request, raw_request)\n"
    "\n"
    "    if isinstance(generator, ErrorResponse):\n"
    "        return JSONResponse(\n"
    "            content=generator.model_dump(), status_code=generator.error.code\n"
    "        )\n"
    "\n"
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        return JSONResponse(\n"
    "            content=generator.model_dump(),\n"
    "            headers=metrics_header(metrics_header_format),\n"
    "        )\n"
    "\n"
    "    return StreamingResponse(content=generator, media_type=\"text/event-stream\")\n"
)

A1_NEW = (
    "    # " + GENESIS_PN89_MARKER + "\n"
    "    " + _IMPORT + "\n"
    "    _tracker = _g89.create_tracker(request)\n"
    "    generator = await handler.create_chat_completion(request, raw_request)\n"
    "\n"
    "    if isinstance(generator, ErrorResponse):\n"
    "        _g89.finish_request(_tracker, None, status=generator.error.code)\n"
    "        return JSONResponse(\n"
    "            content=generator.model_dump(), status_code=generator.error.code\n"
    "        )\n"
    "\n"
    "    elif isinstance(generator, ChatCompletionResponse):\n"
    "        _g89.finish_request(_tracker, generator, status=200)\n"
    "        return JSONResponse(\n"
    "            content=generator.model_dump(),\n"
    "            headers=metrics_header(metrics_header_format),\n"
    "        )\n"
    "\n"
    "    return StreamingResponse(\n"
    "        content=_g89.wrap_streaming_generator(generator, _tracker),\n"
    "        media_type=\"text/event-stream\",\n"
    "    )\n"
)

B1_OLD = "def attach_router(app: FastAPI):\n    app.include_router(router)\n"
B1_NEW = (
    "def attach_router(app: FastAPI):\n"
    "    # " + GENESIS_PN89_MARKER + "\n"
    "    " + _IMPORT + "\n"
    "    app.include_router(_g89.router)\n"
    "    app.include_router(router)\n"
)


def apply() -> tuple[str, str]:
    target = resolve_vllm_file("entrypoints/openai/chat_completion/api_router.py")
    if target is None:
        return "skipped", "entrypoints/openai/chat_completion/api_router.py missing"

    patcher = TextPatcher(
        patch_name="PN89 KV offload requests endpoint",
        target_file=target,
        marker=GENESIS_PN89_MARKER,
        sub_patches=[
            TextPatch(
                name="pn89_create_chat_completion_hook",
                anchor=A1_OLD,
                replacement=A1_NEW,
                required=True,
            ),
            TextPatch(
                name="pn89_attach_router",
                anchor=B1_OLD,
                replacement=B1_NEW,
                required=True,
            ),
        ],
    )
    res, failure = patcher.apply()
    return result_to_wiring_status(
        res,
        failure,
        applied_message="PN89 applied: endpoint nativo /v1/kv-offload/requests registrado",
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    target = resolve_vllm_file("entrypoints/openai/chat_completion/api_router.py")
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN89_MARKER in f.read()
    except Exception:
        return False
