# SPDX-License-Identifier: Apache-2.0
"""CharXiv (RQ): 1000 preguntas de razonamiento sobre graficos de papers (split val).

Prompts de respuesta y de calificacion tomados del repo oficial
(github.com/princeton-nlp/CharXiv, src/constants.py, clonado por setup.sh).
La receta oficial califica con gpt-4o; aca con el juez configurado.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from ..common import CACHE, Item, hf_file, strip_think, user_msg

NAME = "charxiv_rq"
CARD = {"capability": "Scientific chart analysis", "benchmark": "CharXiv (RQ)", "score": 83.7,
        "note": "Without CI; With CI 90.2"}


def _constants():
    src = CACHE / "CharXiv" / "src"
    if not src.is_dir():
        raise RuntimeError(f"falta el clon de CharXiv en {src}; correr benchmarks/qwen38_card/setup.sh")
    sys.path.insert(0, str(src))
    import constants  # noqa: E402
    import reasoning_utils  # noqa: E402
    return constants, reasoning_utils


def load(args) -> list[Item]:
    c, ru = _constants()
    df = pd.read_parquet(hf_file("princeton-nlp/CharXiv", "val.parquet"))
    items = []
    for i, r in enumerate(df.itertuples()):
        cat = int(r.reasoning_a_type)
        if cat == 4:
            q = c.REASONING_RESP_INST[4].format(r.reasoning_q, ru.get_number_instruction(r.reasoning_a))
        else:
            q = c.REASONING_RESP_INST[cat].format(r.reasoning_q)
        items.append(Item(id=str(i), messages=user_msg(q, [r.image["bytes"]]),
                          meta={"question": r.reasoning_q, "answer": r.reasoning_a, "cat": cat,
                                "category": r.category}, gen={}))
    return items


def grade(items, responses, ctx):
    c, _ = _constants()

    def one(it):
        r = responses.get(it.id)
        resp = strip_think(r["content"]) if r else ""
        if not resp:
            return {"id": it.id, "score": 0.0, "extracted": "", "cat": it.meta["cat"]}
        q = (c.REASONING_GRADING_PREFIX + c.REASONING_GRADING_INST[it.meta["cat"]]
             .replace("<|question|>", it.meta["question"])
             .replace("<|ground_truth|>", it.meta["answer"]).replace("<|response|>", resp))
        out = ctx.judge.ask(q, max_tokens=512)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
            score = float(int(d.get("score", 0)) == 1)
            ext = d.get("extract_answer", d.get("extracted_answer", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            score, ext = 0.0, f"juez ilegible: {out[:200]}"
        return {"id": it.id, "score": score, "extracted": ext, "gold": it.meta["answer"],
                "cat": it.meta["cat"]}
    rows = ctx.map(one, items)
    names = {1: "text-in-chart", 2: "text-in-general", 3: "number-in-chart", 4: "number-in-general"}
    by = {}
    for r in rows:
        by.setdefault(names[r["cat"]], []).append(r["score"])
    return rows, {"juez": ctx.judge.name,
                  "por_tipo": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
