# SPDX-License-Identifier: Apache-2.0
"""MathVision: problemas de matematica con figura (3040 en test, 304 en testmini).

Prompt fijo del card: "Please reason step by step, and put your final answer
within \\boxed{}." Se informa la configuracion "Without CI" (sin interprete de
codigo). El card corrigio algunas respuestas mal anotadas que no publico, asi
que el puntaje puede quedar ~1 punto abajo por eso solo.

Comparacion: letra para opcion multiple (o el texto de la opcion), y para
respuesta libre igualdad normalizada, numerica (1e-4 relativo) o simbolica
con sympy cuando la expresion lo permite.
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import Item, hf_file, last_boxed, norm_text, strip_think, to_float, user_msg

NAME = "mathvision"
CARD = {"capability": "Visual math problem solving", "benchmark": "MathVision", "score": 90.0,
        "note": "Without CI; With CI 94.6"}

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
FILES = {"test": "data/test-00000-of-00001-3532b8d3f1b4047a.parquet",
         "testmini": "data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet"}


def add_args(p):
    p.add_argument("--mathvision-split", choices=list(FILES), default="test")


def load(args) -> list[Item]:
    df = pd.read_parquet(hf_file("MathLLMs/MathVision", FILES[args.mathvision_split]))
    items = []
    for r in df.itertuples():
        q = re.sub(r"<image\d+>", "", r.question).strip()
        opts = list(r.options)
        if opts:
            q += "\nChoices:\n" + "\n".join(f"({'ABCDE'[i]}) {o}" for i, o in enumerate(opts))
        items.append(Item(id=str(r.id), messages=user_msg(f"{q}\n{SUFFIX}", [r.decoded_image["bytes"]]),
                          meta={"answer": str(r.answer), "options": opts, "subject": r.subject},
                          gen={}))
    return items


def _sym_equal(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                                standard_transformations)
        tr = standard_transformations + (implicit_multiplication_application,)

        def conv(s):
            s = s.replace("\\left", "").replace("\\right", "").replace("\\cdot", "*").replace("\\times", "*")
            s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
            s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s).replace("\\pi", "pi").replace("^", "**")
            s = s.replace("{", "(").replace("}", ")").replace("\\", "")
            # parse_expr hace eval: solo expresiones aritmeticas cortas, nada de atributos ni dunders
            if len(s) > 120 or "__" in s or not re.fullmatch(r"[0-9a-z+\-*/(). ]*", s):
                raise ValueError(s)
            return parse_expr(s, transformations=tr)
        return sympy.simplify(conv(a) - conv(b)) == 0
    except Exception:
        return False


def is_equal(pred: str, gold: str) -> bool:
    p, g = norm_text(pred), norm_text(gold)
    if p == g:
        return True
    fp, fg = to_float(p), to_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= 1e-4 * max(1.0, abs(fg))
    return _sym_equal(p, g)


def choice_letter(pred: str, options: list[str]) -> str | None:
    m = re.fullmatch(r"\(?([A-E])\)?[.:]?(\s.*)?", pred.strip())
    if m:
        return m.group(1)
    for i, o in enumerate(options):
        if is_equal(pred, o):
            return "ABCDE"[i]
    return None


def grade(items, responses, ctx):
    rows = []
    for it in items:
        r = responses.get(it.id)
        text = strip_think(r["content"]) if r else ""
        pred = last_boxed(text)
        if pred is None and text:
            pred = text.strip().splitlines()[-1]
        gold = it.meta["answer"]
        if not pred:
            ok = False
        elif it.meta["options"] and gold in "ABCDE":
            ok = choice_letter(pred, it.meta["options"]) == gold
        else:
            ok = is_equal(pred, gold)
        rows.append({"id": it.id, "gold": gold, "pred": pred, "subject": it.meta["subject"],
                     "score": float(ok)})
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r["score"])
    return rows, {"split": ctx.args.mathvision_split,
                  "por_tema": {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(by.items())}}
