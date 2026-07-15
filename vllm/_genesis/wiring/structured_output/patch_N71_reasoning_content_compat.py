# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN71 — reasoning_content field compatibility for OpenWebUI.

vLLM 0.23.0 internally renamed the `reasoning_content` field to `reasoning`.
However, clients like OpenWebUI expect `reasoning_content` in the OpenAI spec.
This patch adds `reasoning_content` back to DeltaMessage and ChatMessage and
populates it automatically via a Pydantic model validator.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn71_reasoning_content_compat")

GENESIS_PN71_MARKER = (
    "Genesis PN71 reasoning_content compatibility v1.0"
)


def _is_enabled() -> bool:
    # Always enabled as a crucial quality-of-life fix for OpenWebUI
    return True


_OLD_ENGINE_PROTOCOL = (
    "class DeltaMessage(OpenAIBaseModel):\n"
    "    role: str | None = None\n"
    "    content: str | None = None\n"
    "    reasoning: str | None = None\n"
    "    tool_calls: list[DeltaToolCall] = Field(default_factory=list)"
)

_NEW_ENGINE_PROTOCOL = (
    "class DeltaMessage(OpenAIBaseModel):\n"
    "    role: str | None = None\n"
    "    content: str | None = None\n"
    "    reasoning: str | None = None\n"
    "    reasoning_content: str | None = None\n"
    "    tool_calls: list[DeltaToolCall] = Field(default_factory=list)\n"
    "\n"
    "    @model_validator(mode=\"before\")\n"
    "    @classmethod\n"
    "    def populate_reasoning_content(cls, data):\n"
    "        if isinstance(data, dict):\n"
    "            if data.get(\"reasoning\") is not None:\n"
    "                data[\"reasoning_content\"] = data[\"reasoning\"]\n"
    "            elif data.get(\"reasoning_content\") is not None:\n"
    "                data[\"reasoning\"] = data[\"reasoning_content\"]\n"
    "        return data"
)

_OLD_CHAT_PROTOCOL = (
    "    # vLLM-specific fields that are not in OpenAI spec\n"
    "    reasoning: str | None = None"
)

_NEW_CHAT_PROTOCOL = (
    "    # vLLM-specific fields that are not in OpenAI spec\n"
    "    reasoning: str | None = None\n"
    "    reasoning_content: str | None = None\n"
    "\n"
    "    @model_validator(mode=\"before\")\n"
    "    @classmethod\n"
    "    def populate_reasoning_content(cls, data):\n"
    "        if isinstance(data, dict):\n"
    "            if data.get(\"reasoning\") is not None:\n"
    "                data[\"reasoning_content\"] = data[\"reasoning\"]\n"
    "            elif data.get(\"reasoning_content\") is not None:\n"
    "                data[\"reasoning\"] = data[\"reasoning_content\"]\n"
    "        return data"
)


def apply() -> tuple[str, str]:
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # 1. Patch engine/protocol.py
    target_engine = resolve_vllm_file("entrypoints/openai/engine/protocol.py")
    if target_engine is None:
        return "skipped", "engine/protocol.py not found"

    patcher_engine = TextPatcher(
        patch_name="PN71 reasoning_content engine/protocol",
        target_file=str(target_engine),
        marker=GENESIS_PN71_MARKER,
        sub_patches=[
            TextPatch(
                name="pn71_delta_message_reasoning_content",
                anchor=_OLD_ENGINE_PROTOCOL,
                replacement=_NEW_ENGINE_PROTOCOL,
                required=True,
            ),
        ],
    )

    res_engine, err_engine = patcher_engine.apply()
    if res_engine == TextPatchResult.FAILED:
        return "failed", f"engine/protocol fail: {err_engine.reason if err_engine else 'unknown'}"

    # 2. Patch chat_completion/protocol.py
    target_chat = resolve_vllm_file("entrypoints/openai/chat_completion/protocol.py")
    if target_chat is None:
        # Chat completion protocol is required
        return "skipped", "chat_completion/protocol.py not found"

    patcher_chat = TextPatcher(
        patch_name="PN71 reasoning_content chat_completion/protocol",
        target_file=str(target_chat),
        marker=GENESIS_PN71_MARKER,
        sub_patches=[
            TextPatch(
                name="pn71_chat_message_reasoning_content",
                anchor=_OLD_CHAT_PROTOCOL,
                replacement=_NEW_CHAT_PROTOCOL,
                required=True,
            ),
        ],
    )

    res_chat, err_chat = patcher_chat.apply()
    if res_chat == TextPatchResult.FAILED:
        return "failed", f"chat_completion/protocol fail: {err_chat.reason if err_chat else 'unknown'}"

    return "applied", "PN71 reasoning_content compatibility applied successfully"
