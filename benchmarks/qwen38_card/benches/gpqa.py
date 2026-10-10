# SPDX-License-Identifier: Apache-2.0
"""GPQA Diamond: 198 preguntas de opcion multiple de ciencia (grado de doctorado).

Prompt y extraccion de simple-evals (OpenAI); las opciones se barajan con una
semilla fija por pregunta para que la respuesta correcta no sea siempre la A.
"""
from __future__ import annotations

import random

import pandas as pd

from ..common import Item, hf_file, letter_answer, user_msg

NAME = "gpqa_diamond"
CARD = {"capability": "Scientific reasoning", "benchmark": "GPQA Diamond", "score": 89.2}

TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of "
    "the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def load(args) -> list[Item]:
    df = pd.read_csv(hf_file("Idavidrein/gpqa", "gpqa_diamond.csv"))
    items = []
    for i, row in df.iterrows():
        opts = [row["Correct Answer"], row["Incorrect Answer 1"],
                row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        opts = [str(o).strip() for o in opts]
        order = list(range(4))
        random.Random(f"gpqa-{i}").shuffle(order)
        shuffled = [opts[k] for k in order]
        gold = "ABCD"[order.index(0)]
        prompt = TEMPLATE.format(q=str(row["Question"]).strip(), A=shuffled[0], B=shuffled[1],
                                 C=shuffled[2], D=shuffled[3])
        items.append(Item(id=str(i), messages=user_msg(prompt),
                          meta={"gold": gold, "domain": row.get("High-level domain", "")}, gen={}))
    return items


def grade(items, responses, ctx):
    rows = []
    for it in items:
        r = responses.get(it.id)
        pred = letter_answer(r["content"], "ABCD") if r else None
        rows.append({"id": it.id, "gold": it.meta["gold"], "pred": pred,
                     "domain": it.meta["domain"], "score": float(pred == it.meta["gold"])})
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r["score"])
    return rows, {"por_dominio": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
