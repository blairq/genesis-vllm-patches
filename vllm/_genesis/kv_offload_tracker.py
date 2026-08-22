# SPDX-License-Identifier: Apache-2.0
"""KV Offload Activity Tracker & REST Endpoints for vLLM (Genesis PN89/PN91).

Provides in-memory ring-buffer tracking of recent chat completion requests,
exposes GET /v1/kv-offload/requests, and provides safe zero-downtime cache
and metrics resetting via POST /v1/kv-offload/reset with graceful client
termination notifications.
"""
from __future__ import annotations

import asyncio
import collections
import glob
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("vllm._genesis.kv_offload_tracker")

_MAX_REQUESTS = 300
_RING_BUFFER = collections.deque(maxlen=_MAX_REQUESTS)
_LOCK = threading.Lock()

# Set of abort events for active streaming connections
_ACTIVE_STREAMS: set[asyncio.Event] = set()

router = APIRouter()


class RequestTrackerContext:
    __slots__ = ("t0", "timestamp", "agent", "id", "ttft", "usage", "status")

    def __init__(self, agent: Optional[str] = None):
        self.t0 = time.perf_counter()
        self.timestamp = time.time()
        self.agent = agent
        self.id: Optional[str] = None
        self.ttft: Optional[float] = None
        self.usage: Optional[dict[str, Any]] = None
        self.status = 200


def create_tracker(req: Any) -> RequestTrackerContext:
    agent = None
    try:
        if hasattr(req, "kv_transfer_params") and isinstance(req.kv_transfer_params, dict):
            agent = req.kv_transfer_params.get("genesis_agent") or req.kv_transfer_params.get("agent")
    except Exception:
        pass
    return RequestTrackerContext(agent=agent)


def observe_chunk(ctx: RequestTrackerContext, chunk_data: Any) -> None:
    try:
        if ctx.ttft is None:
            ctx.ttft = time.perf_counter() - ctx.t0
        if isinstance(chunk_data, (str, bytes)):
            raw = chunk_data if isinstance(chunk_data, str) else chunk_data.decode("utf-8", "replace")
            if '"id"' in raw and ctx.id is None:
                idx = raw.find('"id":"')
                if idx != -1:
                    end_idx = raw.find('"', idx + 6)
                    if end_idx != -1:
                        ctx.id = raw[idx + 6:end_idx]
            if '"usage"' in raw and '"usage":null' not in raw.replace(" ", ""):
                try:
                    for line in raw.split("\n"):
                        if line.startswith("data:") and line.strip() != "data: [DONE]":
                            obj = json.loads(line[5:].strip())
                            u = obj.get("usage")
                            if u:
                                ctx.usage = u
                                if obj.get("id"):
                                    ctx.id = obj.get("id")
                except Exception:
                    pass
    except Exception:
        pass


def finish_request(ctx: RequestTrackerContext, response_obj_or_dict: Any = None, status: int = 200) -> None:
    try:
        e2e_ms = round((time.perf_counter() - ctx.t0) * 1000.0, 1)
        ttft_ms = round(ctx.ttft * 1000.0, 1) if ctx.ttft is not None else None

        req_id = ctx.id
        prompt_tokens = None
        cached_tokens = None
        output_tokens = None

        if response_obj_or_dict is not None:
            if hasattr(response_obj_or_dict, "id"):
                req_id = response_obj_or_dict.id
            if hasattr(response_obj_or_dict, "usage") and response_obj_or_dict.usage:
                u = response_obj_or_dict.usage
                prompt_tokens = getattr(u, "prompt_tokens", None)
                output_tokens = getattr(u, "completion_tokens", None)
                det = getattr(u, "prompt_tokens_details", None)
                if det:
                    cached_tokens = getattr(det, "cached_tokens", None)
            elif isinstance(response_obj_or_dict, dict):
                req_id = response_obj_or_dict.get("id", req_id)
                u = response_obj_or_dict.get("usage") or {}
                prompt_tokens = u.get("prompt_tokens")
                output_tokens = u.get("completion_tokens")
                det = u.get("prompt_tokens_details") or {}
                cached_tokens = det.get("cached_tokens")
        elif ctx.usage:
            u = ctx.usage
            prompt_tokens = u.get("prompt_tokens")
            output_tokens = u.get("completion_tokens")
            det = u.get("prompt_tokens_details") or {}
            cached_tokens = det.get("cached_tokens")

        record = {
            "id": req_id,
            "timestamp": round(ctx.timestamp, 3),
            "agent": ctx.agent,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "status": status,
        }

        with _LOCK:
            _RING_BUFFER.append(record)
    except Exception as e:
        logger.debug("Error recording kv-offload request: %s", e)


async def wrap_streaming_generator(generator: AsyncIterator[str], ctx: RequestTrackerContext) -> AsyncIterator[str]:
    abort_event = asyncio.Event()
    with _LOCK:
        _ACTIVE_STREAMS.add(abort_event)

    try:
        async for chunk in generator:
            if abort_event.is_set():
                logger.info("Active stream for req %s received reset abort signal; closing cleanly", ctx.id)
                # Graceful termination message according to OpenAI SSE standard
                close_payload = {
                    "id": ctx.id or "chatcmpl-reset",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "qwen3.8",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n\n[KV Cache Reset: sesión finalizada ordenadamente por el servidor]"},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(close_payload)}\n\n"
                yield "data: [DONE]\n\n"
                finish_request(ctx, None, status=499)
                return

            observe_chunk(ctx, chunk)
            yield chunk
    except Exception as e:
        finish_request(ctx, None, status=500)
        raise
    else:
        finish_request(ctx, None, status=200)
    finally:
        with _LOCK:
            _ACTIVE_STREAMS.discard(abort_event)


def get_records() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_RING_BUFFER)


@router.get("/v1/kv-offload/requests")
@router.get("/kv-offload/requests")
async def get_kv_offload_requests():
    return JSONResponse(content=get_records())


@router.post("/v1/kv-offload/reset")
@router.post("/kv-offload/reset")
@router.post("/reset_prefix_cache")
async def reset_kv_cache_and_metrics(
    raw_request: Request,
    force: bool = Query(default=True, description="Force preemption and graceful notification of active requests"),
    notify_clients: bool = Query(default=True, description="Send [DONE] termination signal to active SSE clients"),
    clear_l1: bool = Query(default=True, description="Clear GPU VRAM L1 prefix cache"),
    clear_l2: bool = Query(default=True, description="Clear Host RAM L2 ARC offload cache"),
    clear_l3: bool = Query(default=True, description="Clear NVMe SSD L3 persistent cache"),
    clear_metrics: bool = Query(default=True, description="Reset Prometheus metrics to 0"),
    clear_history: bool = Query(default=True, description="Reset recent request activity ring buffer"),
):
    """Safely reset KV Cache across L1, L2, L3 tiers, notify open connections, and zero Prometheus metrics in-place."""
    logger.info(
        "KV Offload Reset requested (force=%s, notify_clients=%s, L1=%s, L2=%s, L3=%s, metrics=%s, history=%s)",
        force,
        notify_clients,
        clear_l1,
        clear_l2,
        clear_l3,
        clear_metrics,
        clear_history,
    )

    # 1. Notify and terminate all active SSE streaming client connections gracefully
    notified_clients_count = 0
    if notify_clients:
        with _LOCK:
            notified_clients_count = len(_ACTIVE_STREAMS)
            for ev in _ACTIVE_STREAMS:
                ev.set()

    # Small async yield to allow active generators to flush their [DONE] chunks
    if notified_clients_count > 0:
        await asyncio.sleep(0.05)

    report = {
        "active_clients_notified": notified_clients_count,
        "l1_vram_prefix_cache": False,
        "l2_ram_arc_cache": False,
        "l3_disk_nvme_cache": False,
        "prometheus_metrics": False,
        "request_history": False,
    }

    # 2. Reset L1 VRAM & L2 RAM through vLLM Engine Client
    try:
        engine_client = getattr(raw_request.app.state, "engine_client", None)
        if engine_client is None:
            chat_serving = getattr(raw_request.app.state, "openai_serving_chat", None)
            if chat_serving is not None:
                engine_client = getattr(chat_serving, "engine_client", None)

        if engine_client is not None and hasattr(engine_client, "reset_prefix_cache"):
            success = await engine_client.reset_prefix_cache(
                reset_running_requests=force,
                reset_connector=clear_l2,
            )
            report["l1_vram_prefix_cache"] = bool(success)
            report["l2_ram_arc_cache"] = bool(success and clear_l2)
        else:
            logger.warning("engine_client not found on app.state for L1/L2 reset")
    except Exception as e:
        logger.error("Error resetting L1/L2 cache: %s", e)
        if not force:
            return JSONResponse(
                status_code=409,
                content={"error": f"Failed to reset prefix cache: {e}. Active requests may be running. Try with ?force=true."},
            )

    # 3. Reset L3 NVMe Disk
    if clear_l3:
        try:
            disk_dirs = glob.glob("/kv-offload/*")
            for d in disk_dirs:
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                elif os.path.isfile(d):
                    os.remove(d)
            report["l3_disk_nvme_cache"] = True
        except Exception as e:
            logger.error("Error clearing L3 disk cache: %s", e)

    # 4. Reset Prometheus metrics in-place without destroying metric structures
    if clear_metrics:
        try:
            from prometheus_client import REGISTRY

            for collector in list(REGISTRY._collector_to_names):
                if hasattr(collector, "_metrics"):
                    try:
                        for child in list(collector._metrics.values()):
                            if hasattr(child, "_value"):
                                try:
                                    child._value.set(0.0)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                elif hasattr(collector, "_value"):
                    try:
                        collector._value.set(0.0)
                    except Exception:
                        pass

            try:
                from vllm._genesis import kv_tier_metrics as _g88
                _g88.sink().drain()
            except Exception:
                pass

            report["prometheus_metrics"] = True
        except Exception as e:
            logger.error("Error clearing prometheus metrics: %s", e)

    # 5. Reset Request Ring Buffer History
    if clear_history:
        try:
            with _LOCK:
                _RING_BUFFER.clear()
            report["request_history"] = True
        except Exception as e:
            logger.error("Error clearing ring buffer history: %s", e)

    return JSONResponse(
        content={
            "status": "success",
            "cleared": report,
            "timestamp": round(time.time(), 3),
        }
    )
