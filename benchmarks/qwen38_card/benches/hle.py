# SPDX-License-Identifier: Apache-2.0
"""Humanity's Last Exam: 2500 preguntas (≈14% con imagen), sin herramientas.

Prompt de sistema y juez oficiales de github.com/centerforaisafety/hle. El
card juzga con GPT-4o; aca el juez es configurable (por defecto el propio
modelo en modo instruct, temperatura 0) y queda anotado en el resultado.
"""
from __future__ import annotations

import base64
import re

import pandas as pd

from ..common import Item, hf_file, image_part

NAME = "hle"
CARD = {"capability": "Multidisciplinary reasoning", "benchmark": "HLE", "score": 30.8,
        "note": "juez GPT-4o en el card"}

SYSTEM_EXACT = ("Your response should be in the following format:\nExplanation: {your explanation for "
                "your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your "
                "confidence score between 0% and 100% for your answer}")
SYSTEM_MC = ("Your response should be in the following format:\nExplanation: {your explanation for "
             "your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score "
             "between 0% and 100% for your answer}")

JUDGE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def add_args(p):
    p.add_argument("--hle-text-only", action="store_true", help="HLE: saltear preguntas con imagen")


def load(args) -> list[Item]:
    df = pd.read_parquet(hf_file("cais/hle", "data/test-00000-of-00001.parquet"))
    items = []
    for r in df.itertuples():
        has_img = isinstance(r.image, str) and r.image.startswith("data:")
        if has_img and getattr(args, "hle_text_only", False):
            continue
        content = [{"type": "text", "text": r.question}]
        if has_img:
            header, b64 = r.image.split(",", 1)
            fmt = re.search(r"image/(\w+)", header)
            content.append(image_part(base64.b64decode(b64), fmt.group(1) if fmt else None))
        system = SYSTEM_MC if r.answer_type == "multipleChoice" else SYSTEM_EXACT
        items.append(Item(id=r.id, messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": content}],
                          meta={"question": r.question, "answer": r.answer,
                                "answer_type": r.answer_type, "category": r.category,
                                "image": has_img}, gen={}))
    return items


def grade(items, responses, ctx):
    def one(it):
        r = responses.get(it.id)
        if not r or not r["content"].strip():
            return {"id": it.id, "score": 0.0, "judge": "sin respuesta", "category": it.meta["category"]}
        verdict = ctx.judge.ask(JUDGE.format(question=it.meta["question"], response=r["content"],
                                             correct_answer=it.meta["answer"]), max_tokens=2048)
        m = re.findall(r"(?i)correct\s*:\s*\**\s*(yes|no)", verdict)
        ok = bool(m) and m[-1].lower() == "yes"
        return {"id": it.id, "score": float(ok), "category": it.meta["category"],
                "image": it.meta["image"], "judge": verdict[-600:]}
    rows = ctx.map(one, items)
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r["score"])
    return rows, {"juez": ctx.judge.name,
                  "por_categoria": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
