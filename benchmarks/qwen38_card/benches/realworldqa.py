# SPDX-License-Identifier: Apache-2.0
"""RealWorldQA (xAI): 765 fotos del mundo real con pregunta y respuesta corta.

Cada pregunta ya trae su instruccion de formato ("answer directly with only
the letter..."), asi que se manda tal cual. Comparacion exacta normalizada.
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import Item, hf_file, letter_answer, norm_text, strip_think, user_msg

NAME = "realworldqa"
CARD = {"capability": "Real-world perception", "benchmark": "RealWorldQA", "score": 85.9}


def load(args) -> list[Item]:
    df = pd.concat([pd.read_parquet(hf_file("xai-org/RealworldQA", f"data/test-0000{k}-of-00002.parquet"))
                    for k in (0, 1)], ignore_index=True)
    return [Item(id=str(i), messages=user_msg(r.question, [r.image["bytes"]]),
                 meta={"answer": str(r.answer)}, gen={})
            for i, r in enumerate(df.itertuples())]


def is_correct(text: str, gold: str) -> tuple[bool, str]:
    t = strip_think(text)
    if re.fullmatch(r"[A-F]", gold):
        m = re.match(r"\s*\**\(?([A-F])\)?\**(?:[.):\s]|$)", t)
        pred = m.group(1) if m else letter_answer(t, "ABCDEF")
        return pred == gold, pred or ""
    pred = norm_text(t.splitlines()[-1] if t else "")
    g = norm_text(gold)
    return pred == g or re.fullmatch(rf"(the answer is )?{re.escape(g)}", pred) is not None, pred


def grade(items, responses, ctx):
    rows = []
    for it in items:
        r = responses.get(it.id)
        ok, pred = is_correct(r["content"], it.meta["answer"]) if r else (False, "")
        rows.append({"id": it.id, "gold": it.meta["answer"], "pred": pred, "score": float(ok)})
    return rows, {}
