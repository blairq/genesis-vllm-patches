# SPDX-License-Identifier: Apache-2.0
"""BabyVision (UniPat): 388 tareas visuales "de bebe" (discriminacion fina,
percepcion espacial, seguimiento, patrones) que no se resuelven con lenguaje.

Opcion multiple -> letra exacta. Completar -> igualdad normalizada y, si no
coincide literalmente, el juez decide la equivalencia (como la receta oficial,
que califica con LLM). Se informa "Without CI".
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import Item, hf_file, last_boxed, norm_text, strip_think, user_msg

NAME = "babyvision"
CARD = {"capability": "General visual reasoning", "benchmark": "BabyVision", "score": 65.7,
        "note": "Without CI; With CI 85.6"}

SUFFIX = "\nThink about the question and put your final answer within \\boxed{}."
JUDGE = """Decide if the model's final answer is equivalent to the reference answer for this visual puzzle.
Formatting differences do not matter (spaces, brackets, order of words that do not change meaning), but any difference in the actual values does.

Question: {q}
Reference answer: {gold}
Model answer: {pred}

Reply with exactly one word: yes or no."""


def load(args) -> list[Item]:
    df = pd.read_parquet(hf_file("UnipatAI/BabyVision", "data/train-00000-of-00001.parquet"))
    items = []
    for r in df.itertuples():
        q = r.question
        if r.ansType == "choice":
            q += "\n" + "\n".join(f"{'ABCDEFGH'[i]}. {o}" for i, o in enumerate(r.options))
            gold = "ABCDEFGH"[int(r.choiceAns)]
        else:
            gold = str(r.blankAns)
        items.append(Item(id=str(r.taskId), messages=user_msg(q + SUFFIX, [r.image["bytes"]]),
                          meta={"question": r.question, "type": r.type, "ansType": r.ansType,
                                "answer": gold}, gen={}))
    return items


def grade(items, responses, ctx):
    def one(it):
        r = responses.get(it.id)
        t = strip_think(r["content"]) if r else ""
        pred = last_boxed(t) or (t.strip().splitlines()[-1] if t.strip() else "")
        gold = it.meta["answer"]
        if it.meta["ansType"] == "choice":
            m = re.search(r"[A-H]", pred.upper())
            ok = bool(m) and m.group(0) == gold
        else:
            squash = lambda s: re.sub(r"[\s()\[\]{}]", "", norm_text(s))  # noqa: E731
            ok = squash(pred) == squash(gold)
            if not ok and pred:
                v = ctx.judge.ask(JUDGE.format(q=it.meta["question"], gold=gold, pred=pred), max_tokens=8)
                ok = v.strip().lower().startswith("yes")
        return {"id": it.id, "gold": gold, "pred": pred, "type": it.meta["type"], "score": float(ok)}
    rows = ctx.map(one, items)
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r["score"])
    return rows, {"juez": ctx.judge.name,
                  "por_tipo": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
