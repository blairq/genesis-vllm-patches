# SPDX-License-Identifier: Apache-2.0
"""LiveCodeBench v6: generacion de codigo de concursos (AtCoder, LeetCode, Codeforces).

Ventana por defecto 2025-02-01 .. 2025-05-01, la que Qwen viene usando como
"v6" (175 problemas en test6.jsonl; 131 caen en la ventana). Prompt oficial de
LCB y pass@1 con los tests publicos + privados.

El codigo generado corre en un contenedor descartable sin red
(python:3.12-slim, --network none, 2 GB, 1 CPU), un contenedor por problema.
"""
from __future__ import annotations

import base64
import json
import pickle
import re
import subprocess
import tempfile
import zlib
from pathlib import Path

from ..common import Item, hf_file, strip_think, user_msg

NAME = "livecodebench_v6"
CARD = {"capability": "Competitive coding", "benchmark": "LiveCodeBench v6", "score": 90.3}

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) "
          "and will generate a correct Python program that matches the specification and passes all tests.")
FMT_STARTER = ("### Format: You will use the following starter code to write the solution to the problem "
               "and enclose your code within delimiters.\n```python\n{starter}\n```\n\n")
FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
             "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
             "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes "
             "output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")

IMAGE = "python:3.12-slim"
RUNNER = Path(__file__).with_name("_lcb_runner.py")


def add_args(p):
    p.add_argument("--lcb-start", default="2025-02-01", help="LCB: fecha minima de concurso")
    p.add_argument("--lcb-end", default="2025-05-01", help="LCB: fecha maxima (exclusiva)")
    p.add_argument("--lcb-timeout", type=float, default=6.0, help="LCB: segundos por test")


def _tests(row) -> list[dict]:
    pub = json.loads(row["public_test_cases"])
    priv = row["private_test_cases"]
    try:
        priv = json.loads(priv)
    except json.JSONDecodeError:
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(priv.encode()))))
    return pub + priv


def load(args) -> list[Item]:
    rows = []
    for f in ("test5.jsonl", "test6.jsonl"):
        rows += [json.loads(line) for line in open(hf_file("livecodebench/code_generation_lite", f))]
    items = []
    for row in rows:
        date = row["contest_date"][:10]
        if not (args.lcb_start <= date < args.lcb_end):
            continue
        starter = row["starter_code"] or ""
        prompt = f"### Question:\n{row['question_content']}\n\n"
        prompt += FMT_STARTER.format(starter=starter) if starter else FMT_STDIN
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        meta = json.loads(row["metadata"] or "{}")
        items.append(Item(id=row["question_id"],
                          messages=[{"role": "system", "content": SYSTEM}] + user_msg(prompt),
                          meta={"tests": _tests(row), "fn_name": meta.get("func_name"),
                                "difficulty": row["difficulty"], "platform": row["platform"],
                                "date": date}, gen={}))
    return items


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", strip_think(text), re.S)
    return blocks[-1] if blocks else None


def run_tests(code: str, tests: list[dict], fn_name: str | None, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="lcb_") as d:
        Path(d, "job.json").write_text(json.dumps({"code": code, "tests": tests, "fn_name": fn_name,
                                                   "timeout": timeout}))
        Path(d, "runner.py").write_text(RUNNER.read_text())
        budget = 60 + timeout * len(tests)
        cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "2g", "--cpus", "1",
               "--pids-limit", "256", "-v", f"{d}:/w:ro", IMAGE, "python", "/w/runner.py"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=budget)
            return json.loads(p.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return {"passed": False, "n_ok": 0, "error": "timeout global"}
        except (json.JSONDecodeError, IndexError):
            return {"passed": False, "n_ok": 0, "error": f"runner: {p.stderr[-300:]}"}


def grade(items, responses, ctx):
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        subprocess.run(["docker", "pull", "-q", IMAGE], check=True, capture_output=True)

    def one(it):
        r = responses.get(it.id)
        code = extract_code(r["content"]) if r else None
        base = {"id": it.id, "difficulty": it.meta["difficulty"], "n_tests": len(it.meta["tests"])}
        if not code:
            return {**base, "score": 0.0, "error": "sin bloque de codigo"}
        res = run_tests(code, it.meta["tests"], it.meta["fn_name"], ctx.args.lcb_timeout)
        return {**base, "score": float(res["passed"]), **res}
    rows = ctx.map(one, items, workers=8)
    by = {}
    for r in rows:
        by.setdefault(r["difficulty"], []).append(r["score"])
    return rows, {"ventana": f"{ctx.args.lcb_start}..{ctx.args.lcb_end}",
                  "por_dificultad": {k: f"{100 * sum(v) / len(v):.1f} ({len(v)})" for k, v in by.items()}}
