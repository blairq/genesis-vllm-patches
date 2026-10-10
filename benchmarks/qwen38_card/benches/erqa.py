# SPDX-License-Identifier: Apache-2.0
"""ERQA (Google DeepMind): 400 preguntas de razonamiento corporizado, opcion multiple.

El server admite hasta 2 imagenes por pedido (--limit-mm-per-prompt). Los 53
items con mas imagenes se mandan como un mosaico rotulado ("Image 1..n") en
una sola imagen: es una desviacion del protocolo y el resultado la informa
aparte (acierto con y sin esos items).
"""
from __future__ import annotations

import io

import pandas as pd
from PIL import Image, ImageDraw

from ..common import Item, hf_file, letter_answer, strip_think, user_msg

NAME = "erqa"
CARD = {"capability": "Embodied intelligence", "benchmark": "ERQA", "score": 65.5}

INSTR = "\nAnswer with the letter of the correct option."


def add_args(p):
    p.add_argument("--max-images", type=int, default=2,
                   help="imagenes por pedido que acepta el server (mas que eso -> mosaico)")


def montage(blobs: list[bytes]) -> Image.Image:
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in blobs]
    cols = 2 if len(imgs) <= 4 else 4
    h = 448
    imgs = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in imgs]
    w = max(im.width for im in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + 28)), "white")
    d = ImageDraw.Draw(canvas)
    for k, im in enumerate(imgs):
        x, y = (k % cols) * w, (k // cols) * (h + 28)
        d.text((x + 6, y + 6), f"Image {k + 1}", fill="black")
        canvas.paste(im, (x, y + 28))
    return canvas


def load(args) -> list[Item]:
    df = pd.read_parquet(hf_file("FlagEval/ERQA", "data/test-00000-of-00001.parquet"))
    items = []
    for r in df.itertuples():
        blobs = [im["bytes"] for im in r.images]
        tiled = len(blobs) > args.max_images
        imgs = [montage(blobs)] if tiled else blobs
        q = r.question + (f"\n(The {len(blobs)} images are tiled in one picture, labeled Image 1..{len(blobs)}.)"
                          if tiled else "")
        items.append(Item(id=r.question_id, messages=user_msg(q + INSTR, imgs),
                          meta={"answer": r.answer.strip(), "type": r.question_type, "tiled": tiled}, gen={}))
    return items


def grade(items, responses, ctx):
    rows = []
    for it in items:
        r = responses.get(it.id)
        t = strip_think(r["content"]) if r else ""
        pred = letter_answer(t, "ABCDEFGH") or (t.strip()[:1].upper() if t.strip() else None)
        rows.append({"id": it.id, "gold": it.meta["answer"], "pred": pred, "type": it.meta["type"],
                     "tiled": it.meta["tiled"], "score": float(pred == it.meta["answer"])})
    sin = [r["score"] for r in rows if not r["tiled"]]
    con = [r["score"] for r in rows if r["tiled"]]
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r["score"])
    return rows, {"solo_items_sin_mosaico": round(100 * sum(sin) / max(len(sin), 1), 1),
                  "items_en_mosaico": f"{len(con)} ({100 * sum(con) / max(len(con), 1):.1f}%)",
                  "por_tipo": {k: round(100 * sum(v) / len(v), 1) for k, v in by.items()}}
