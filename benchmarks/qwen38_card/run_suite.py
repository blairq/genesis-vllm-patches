# SPDX-License-Identifier: Apache-2.0
"""Corre la suite del model card de Qwen3.8-27B contra el vLLM local.

  python3 -m benchmarks.qwen38_card.run_suite --ping
  python3 -m benchmarks.qwen38_card.run_suite --bench all --limit 5        # humo
  python3 -m benchmarks.qwen38_card.run_suite --bench text                  # completa, texto
  python3 -m benchmarks.qwen38_card.run_suite --resume benchmarks/results/qwen38_card/<dir>

Resultados en benchmarks/results/qwen38_card/<fecha>/: respuestas crudas
(reanudables), calificacion por item, summary.json y summary.md con la
comparacion contra el card.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .common import (RESULTS, Client, DatasetUnavailable, Judge, default_api_key, default_endpoint,
                     default_model, generate, write_json)

TEXT = ["ifbench", "gpqa", "hle", "livecodebench"]
VISION = ["mathvision", "babyvision", "charxiv", "realworldqa", "erqa"]
ALL = TEXT + VISION

# Filas del card que no se pueden reproducir con un endpoint y un dataset:
# son internas, o piden un entorno de agente (VMs, emuladores, repos dockerizados).
NOT_REPRODUCED = [
    ("Agentic terminal coding", "Terminal Bench 2.1 (Terminus)", "73.0",
     "harness Terminus + ~90 entornos docker; ver README"),
    ("Agentic coding", "SWE-bench Pro", "61.7", "harness Claude Code, imagenes docker por repo, 256K"),
    ("Repo-level code generation", "NL2Repo-Bench", "42.3", "harness Claude Code"),
    ("Agentic coding", "DeepSWE 1.1", "42.2", "harness Claude Code"),
    ("Software engineering", "QwenSWEBench", "79.0", "interno de Qwen"),
    ("Long-horizon office work", "CoWorkBench", "70.7", "interno de Qwen"),
    ("Professional job tasks", "JobBench", "33.4", "entorno de agente"),
    ("Frontier agentic tasks", "Agents' Last Exam", "20.4 / 42.9", "entorno de agente"),
    ("Computer use", "OSWorld-Verified", "84.3", "VMs de escritorio"),
    ("Browser use", "WebArena-Verified", "64.8", "sitios web autohospedados"),
    ("Mobile use", "AndroidWorld", "81.9", "emulador Android"),
    ("Application recreation", "RecreationBench", "47.1", "interno de Qwen"),
    ("Multimodal tool use", "ClawEval-MM", "57.4 / 56.9", "entorno de agente"),
    ("Multimodal software engineering", "SWE-MM", "38.6", "harness Claude Code"),
    ("Visual web development", "Vision2Web", "62.9", "harness Claude Code + juez gpt-5.4"),
    ("Document intelligence", "OmniDocBench 1.5", "91.1",
     "pendiente: necesita el toolkit oficial (CDM para formulas, TEDS para tablas)"),
]


class Ctx:
    def __init__(self, args, judge):
        self.args, self.judge = args, judge

    def map(self, fn, items, workers=None):
        with ThreadPoolExecutor(max_workers=workers or self.args.judge_concurrency) as ex:
            return list(ex.map(fn, items))


def parse_args():
    p = argparse.ArgumentParser(prog="qwen38_card")
    p.add_argument("--endpoint", default=default_endpoint())
    p.add_argument("--api-key", default=None, help="por defecto GENESIS_BENCH_API_KEY o compose/.env")
    p.add_argument("--model", default=default_model())
    p.add_argument("--bench", default="all", help=f"lista separada por comas, 'text', 'vision' o 'all': {ALL}")
    p.add_argument("--limit", type=int, default=0, help="N items por benchmark (muestra fija con semilla)")
    p.add_argument("--limits", default="", help="limite por benchmark, pisa a --limit: 'hle=500,charxiv=300'")
    p.add_argument("--concurrency", type=int, default=6, help="pedidos simultaneos (el server tiene 11)")
    p.add_argument("--mode", choices=["thinking", "instruct"], default="thinking")
    p.add_argument("--effort", default="xhigh", help="reasoning_effort (card: xhigh por defecto)")
    p.add_argument("--max-tokens", type=int, default=81920,
                   help="tope de salida (razonamiento + respuesta); el card admite hasta 262k de razonamiento")
    p.add_argument("--resume", default=None, help="directorio de una corrida anterior para completarla")
    p.add_argument("--grade-only", action="store_true", help="recalificar sin generar")
    p.add_argument("--judge-endpoint", default=None, help="por defecto el mismo server")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--ping", action="store_true")
    mods = {b: importlib.import_module(f".benches.{b}", __package__) for b in ALL}
    for m in mods.values():
        if hasattr(m, "add_args"):
            m.add_args(p)
    return p.parse_args(), mods


def server_info(client: Client) -> dict:
    info = {"endpoint": client.endpoint, "model": client.model}
    try:
        info["served"] = client.ping()
    except Exception as e:  # noqa: BLE001
        info["served"] = f"error: {e}"
    try:
        info["genesis_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        pass
    return info


def main() -> int:
    args, mods = parse_args()
    key = args.api_key if args.api_key is not None else default_api_key()
    client = Client(args.endpoint, key, args.model)
    info = server_info(client)
    if args.ping:
        print(json.dumps(info, indent=2))
        return 0 if isinstance(info["served"], list) else 2
    if not isinstance(info["served"], list):
        print(f"server inaccesible: {info['served']}", file=sys.stderr)
        return 2

    jclient = client
    if args.judge_endpoint or args.judge_model:
        jclient = Client(args.judge_endpoint or args.endpoint,
                         os.environ.get("GENESIS_JUDGE_API_KEY", key), args.judge_model or args.model)
    ctx = Ctx(args, Judge(jclient))

    per_bench = {k: int(v) for k, v in (x.split("=") for x in args.limits.split(",") if x)}
    sel = {"all": ALL, "text": TEXT, "vision": VISION}.get(args.bench) or args.bench.split(",")
    run_dir = Path(args.resume) if args.resume else RESULTS / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in vars(args).items() if k != "api_key"}
    if not args.grade_only:  # recalificar no cambia con que se genero
        write_json(run_dir / "config.json", {"args": cfg, "server": info, "started": datetime.now().isoformat()})
    else:
        cfg = json.loads((run_dir / "config.json").read_text())["args"]
    log_f = (run_dir / "run.log").open("a")

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    defaults = {"mode": args.mode, "effort": args.effort, "max_tokens": args.max_tokens}
    summary_p = run_dir / "summary.json"
    summary = json.loads(summary_p.read_text()) if summary_p.is_file() else {}
    for b in sel:
        m = mods[b]
        log(f"== {m.NAME} (card: {m.CARD['score']})")
        t0 = time.time()
        try:
            items = m.load(args)
        except (DatasetUnavailable, RuntimeError) as e:
            log(f"  NO DISPONIBLE: {e}")
            summary[m.NAME] = {"card": m.CARD, "error": str(e)}
            write_json(summary_p, summary)
            continue
        lim = per_bench.get(b, args.limit)
        if lim and lim < len(items):
            items = random.Random(0).sample(items, lim)
        resp_p = run_dir / f"{m.NAME}.responses.jsonl"
        if args.grade_only:
            responses = {}
            for line in resp_p.read_text().splitlines() if resp_p.is_file() else []:
                r = json.loads(line)
                if r.get("finish_reason") != "error":
                    responses[r["id"]] = r
        else:
            responses = generate(client, items, resp_p, concurrency=args.concurrency,
                                 defaults=defaults, log=log)
        log(f"  calificando {len(items)} items")
        try:
            rows, extra = m.grade(items, responses, ctx)
        except Exception as e:  # noqa: BLE001 — que un calificador roto no tumbe el resto
            err = getattr(e, "stderr", "") or ""
            log(f"  CALIFICACION FALLIDA: {type(e).__name__}: {e} {err[-500:]}")
            summary[m.NAME] = {"card": m.CARD, "error": f"calificacion: {type(e).__name__}: {e}"}
            write_json(summary_p, summary)
            continue
        with (run_dir / f"{m.NAME}.graded.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        got = [responses[i.id] for i in items if i.id in responses]
        score = 100 * sum(r["score"] for r in rows) / max(len(rows), 1)
        # El modelo a veces termina (EOS) con la respuesta escrita dentro del razonamiento,
        # sin emitir </think>: el parser deja content vacio. El puntaje "rescatado" califica
        # el final del razonamiento en esos casos; el principal sigue siendo el estricto.
        sin_cierre = {i.id for i in items if i.id in responses and responses[i.id]["finish_reason"] == "stop"
                      and not responses[i.id]["content"].strip() and responses[i.id]["reasoning"].strip()}
        rescue = None
        if sin_cierre:
            sub = [i for i in items if i.id in sin_cierre]
            alt = {k: {**responses[k], "content": responses[k]["reasoning"][-6000:]} for k in sin_cierre}
            try:
                fixed = {r["id"]: r for r in m.grade(sub, alt, ctx)[0]}
                rescue = 100 * sum(fixed.get(r["id"], r)["score"] for r in rows) / max(len(rows), 1)
            except Exception as e:  # noqa: BLE001
                log(f"  rescate fallido: {e}")
        summary[m.NAME] = {
            "card": m.CARD, "score": round(score, 1), "n": len(items), "respondidas": len(got),
            "truncadas": sum(r["finish_reason"] == "length" for r in got),
            # termino sin contenido y sin tocar el tope: bucle cortado o fin de razonamiento sin respuesta
            "sin_respuesta": sum(r["finish_reason"] != "length" and not r["content"].strip() for r in got),
            "sin_cierre_think": len(sin_cierre),
            "score_rescatado": round(rescue, 1) if rescue is not None else None,
            "tokens_salida_media": round(sum(r["completion_tokens"] for r in got) / max(len(got), 1)),
            "minutos": round((time.time() - t0) / 60, 1), **extra,
        }
        write_json(summary_p, summary)
        log(f"  {m.NAME}: {score:.1f} (card {m.CARD['score']}) n={len(items)} "
            f"truncadas={summary[m.NAME]['truncadas']} sin_respuesta={summary[m.NAME]['sin_respuesta']} "
            f"(sin </think>: {len(sin_cierre)}, rescatado {summary[m.NAME]['score_rescatado']})")
    write_md(run_dir, summary, cfg, info)
    log(f"listo: {run_dir}/summary.md")
    return 0


def write_md(run_dir: Path, summary: dict, cfg: dict, info: dict) -> None:
    L = [f"# Suite del model card de Qwen3.8-27B — {run_dir.name}", "",
         f"- Modelo: `{info['model']}` en `{info['endpoint']}` (commit Genesis `{info.get('genesis_commit', '?')}`)",
         f"- Modo: {cfg['mode']}, reasoning_effort={cfg['effort']}, max_tokens={cfg['max_tokens']}, "
         f"limite por benchmark: {cfg['limit'] or 'ninguno (completo)'}"
         + (f"; por benchmark: {cfg['limits']}" if cfg.get("limits") else "")
         + (f"; MathVision split {cfg['mathvision_split']}" if cfg.get("mathvision_split") != "test" else ""), "",
         "| Capacidad | Benchmark | IdiotSavant | IdiotSavant (rescatado) | Qwen3.8-27b-bf16 | Δ | n | truncadas | sin respuesta (sin </think>) | notas |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        c = s["card"]
        if "error" in s:
            L.append(f"| {c['capability']} | {c['benchmark']} | — | | {c['score']} | | | | | no disponible: {s['error']} |")
            continue
        d = s["score"] - c["score"]
        note = c.get("note", "")
        if "juez" in s:
            note = (note + "; " if note else "") + f"juez {s['juez'].split('@')[0]}"
        resc = s.get("score_rescatado")
        L.append(f"| {c['capability']} | {c['benchmark']} | **{s['score']:.1f}** | {resc if resc is not None else '='} | "
                 f"{c['score']} | {d:+.1f} | {s['respondidas']}/{s['n']} | {s['truncadas']} | "
                 f"{s.get('sin_respuesta', 0)} ({s.get('sin_cierre_think', 0)}) | {note} |")
    L += ["", "Con `--limit` el error estandar es grande (n=20 → ±10 puntos): sirve para humo, no para comparar.",
          "", "## Filas del card no reproducidas", "", "| Capacidad | Benchmark | Card | Motivo |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in NOT_REPRODUCED]
    (run_dir / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
