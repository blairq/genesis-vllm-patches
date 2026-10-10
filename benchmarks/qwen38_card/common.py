# SPDX-License-Identifier: Apache-2.0
"""Infraestructura comun de la suite del model card de Qwen3.8-27B.

Cliente OpenAI-compatible, presets de muestreo del card, corrida concurrente
con reanudacion (cada respuesta se guarda apenas llega) y juez LLM.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("QWEN38_CARD_CACHE", Path.home() / ".cache" / "qwen38_card"))
RESULTS = REPO / "benchmarks" / "results" / "qwen38_card"

# Model card, "Best Practices": muestreo recomendado por modo.
SAMPLING = {
    "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                     presence_penalty=0.0, repetition_penalty=1.0),
    "instruct": dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0,
                     presence_penalty=1.5, repetition_penalty=1.0),
}


def _key_from_compose_env() -> str:
    env = REPO / "compose" / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("VLLM_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    return ""


def default_endpoint() -> str:
    return os.environ.get("GENESIS_BENCH_ENDPOINT", "http://127.0.0.1:8360/v1")


def default_api_key() -> str:
    return os.environ.get("GENESIS_BENCH_API_KEY") or _key_from_compose_env()


def default_model() -> str:
    return os.environ.get("GENESIS_BENCH_MODEL", "qwen3.8")


# ═══════════════════════════════════════════════════════════════════════════
#                               CLIENTE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Reply:
    content: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    elapsed: float
    error: str = ""


class Client:
    def __init__(self, endpoint: str, api_key: str, model: str, timeout: float = 7200.0):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = httpx.Client(headers=headers, timeout=httpx.Timeout(timeout, connect=30.0))

    def ping(self) -> list[str]:
        r = self.http.get(f"{self.endpoint}/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]

    def chat(self, messages: list[dict], *, mode: str = "thinking", effort: str | None = "xhigh",
             max_tokens: int = 32768, retries: int = 3, **overrides) -> Reply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        body.update(SAMPLING[mode])
        ctk: dict[str, Any] = {"enable_thinking": mode == "thinking"}
        if mode == "thinking" and effort:
            # La plantilla del server lee reasoning_effort de chat_template_kwargs
            # (su default es medium; el card usa xhigh por defecto).
            ctk["reasoning_effort"] = effort
        body["chat_template_kwargs"] = ctk
        body.update(overrides)
        last = ""
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self.http.post(f"{self.endpoint}/chat/completions", json=body)
                if r.status_code >= 400:
                    last = f"HTTP {r.status_code}: {r.text[:500]}"
                    if r.status_code < 500:
                        break
                    time.sleep(5 * (attempt + 1))
                    continue
                d = r.json()
                ch = d["choices"][0]
                m = ch["message"]
                u = d.get("usage") or {}
                return Reply(
                    content=m.get("content") or "",
                    reasoning=m.get("reasoning") or m.get("reasoning_content") or "",
                    finish_reason=ch.get("finish_reason") or "",
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    elapsed=time.time() - t0,
                )
            except (httpx.HTTPError, ValueError, KeyError) as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1))
        return Reply("", "", "error", 0, 0, 0.0, error=last)


# ═══════════════════════════════════════════════════════════════════════════
#                          JUEZ (HLE, CharXiv, BabyVision)
# ═══════════════════════════════════════════════════════════════════════════

class Judge:
    """Juez LLM. Por defecto el mismo server en modo instruct a temperatura 0.

    El card usa GPT-4o (HLE) y la receta de CharXiv usa gpt-4o; para acercarse
    se puede apuntar a otro endpoint con --judge-endpoint/--judge-model/
    GENESIS_JUDGE_API_KEY. El juez queda registrado en el resultado.
    """

    def __init__(self, client: Client):
        self.client = client

    @property
    def name(self) -> str:
        return f"{self.client.model}@{self.client.endpoint}"

    def ask(self, prompt: str, max_tokens: int = 1024) -> str:
        body_extra = dict(temperature=0.0, top_p=1.0)
        rep = self.client.chat([{"role": "user", "content": prompt}], mode="instruct",
                               max_tokens=max_tokens, **body_extra)
        return rep.content if not rep.error else f"__error__ {rep.error}"


# ═══════════════════════════════════════════════════════════════════════════
#                           UTILIDADES DE TEXTO E IMAGEN
# ═══════════════════════════════════════════════════════════════════════════

def image_part(data: bytes | Any, fmt: str | None = None) -> dict:
    """Parte image_url con data URI. Acepta bytes o una PIL.Image."""
    if not isinstance(data, (bytes, bytearray)):
        buf = io.BytesIO()
        data.convert("RGB").save(buf, format="PNG")
        data, fmt = buf.getvalue(), "png"
    if fmt is None:
        fmt = "png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "jpeg"
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}


def user_msg(text: str, images: list | None = None) -> list[dict]:
    parts: list[dict] = [image_part(i) for i in (images or [])]
    if not parts:
        return [{"role": "user", "content": text}]
    parts.append({"type": "text", "text": text})
    return [{"role": "user", "content": parts}]


def last_boxed(text: str) -> str | None:
    """Contenido del ultimo \\boxed{...} (balanceando llaves)."""
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\%", "%").replace("\\,", "").replace("\\!", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("%", "").strip()
    m = re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", s, re.I)
    if m:
        return float(s)
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}|(-?\d+)/(-?\d+)", s)
    if m:
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return float(a) / float(b) if float(b) else None
    return None


def strip_think(text: str) -> str:
    """Por si el parser de razonamiento dejo el bloque en content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ═══════════════════════════════════════════════════════════════════════════
#                               CORRIDA
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Item:
    id: str
    messages: list[dict]
    meta: dict
    gen: dict  # kwargs extra para Client.chat (max_tokens, mode, ...)


def generate(client: Client, items: list[Item], out_jsonl: Path, *, concurrency: int,
             defaults: dict, log: Callable[[str], None] = print) -> dict[str, dict]:
    """Genera las respuestas que falten en out_jsonl (reanudable)."""
    done: dict[str, dict] = {}
    if out_jsonl.is_file():
        for line in out_jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("finish_reason") != "error":
                    done[r["id"]] = r
    todo = [it for it in items if it.id not in done]
    log(f"  {len(done)} ya generadas, {len(todo)} por generar (concurrencia {concurrency})")
    lock = threading.Lock()
    t0 = time.time()

    def work(it: Item) -> dict:
        kw = {**defaults, **it.gen}
        rep = client.chat(it.messages, **kw)
        return {"id": it.id, "content": rep.content, "reasoning": rep.reasoning,
                "finish_reason": rep.finish_reason, "prompt_tokens": rep.prompt_tokens,
                "completion_tokens": rep.completion_tokens, "elapsed": round(rep.elapsed, 2),
                "error": rep.error}

    n = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex, out_jsonl.open("a") as f:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if r["finish_reason"] != "error":
                    done[r["id"]] = r
                if n % 10 == 0 or n == len(todo) or r["error"]:
                    rate = n / max(time.time() - t0, 1e-9) * 3600
                    extra = f" ERROR {r['error'][:120]}" if r["error"] else ""
                    log(f"  [{n}/{len(todo)}] {rate:.0f}/h{extra}")
    return done


class DatasetUnavailable(RuntimeError):
    pass


def hf_file(repo: str, path: str) -> str:
    """Baja (o toma de la cache de HF) un archivo de un dataset."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        return hf_hub_download(repo, path, repo_type="dataset")
    except GatedRepoError as e:
        raise DatasetUnavailable(
            f"{repo} es gated: pedir acceso en https://huggingface.co/datasets/{repo} "
            f"con la cuenta del token de ~/.cache/huggingface/token") from e


def letter_answer(text: str, letters: str) -> str | None:
    """Letra elegida: 'ANSWER: X', \\boxed{X} o la ultima letra suelta."""
    t = strip_think(text)
    pats = [rf"(?i:answer)\s*[:：]\s*\**\s*\(?([{letters}])\)?\b",rf"\\boxed\{{\s*\(?([{letters}])\)?\s*\}}"]
    for p in pats:
        m = re.findall(p, t)
        if m:
            return m[-1].upper()
    m = re.findall(rf"\b([{letters}])\b", t[-200:])
    return m[-1] if m else None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
