# SPDX-License-Identifier: Apache-2.0
"""IFBench (AllenAI): 300 prompts con 58 restricciones verificables nuevas.

La verificacion usa el codigo oficial (github.com/allenai/IFBench), instalado
en un venv propio en ~/.cache/qwen38_card/venv para no ensuciar el python del
sistema (necesita nltk, spacy, emoji, syllapy...). Ver setup.sh.

Metrica: prompt-level loose accuracy, la que reporta el paper. Se informa
tambien la strict y la instruction-level.
"""
from __future__ import annotations

import json
import subprocess

import pandas as pd

from ..common import CACHE, Item, hf_file, strip_think, user_msg

NAME = "ifbench"
CARD = {"capability": "Instruction following", "benchmark": "IFBench", "score": 79.5}

VENV_PY = CACHE / "venv" / "bin" / "python"
SCORER = __file__.replace("ifbench.py", "_ifbench_score.py")


def _clean_kwargs(kw):
    return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if v is not None}
            for d in kw]


def load(args) -> list[Item]:
    if not VENV_PY.exists():
        raise RuntimeError(f"falta el venv del verificador ({VENV_PY}); correr benchmarks/qwen38_card/setup.sh")
    df = pd.read_parquet(hf_file("allenai/IFBench_test", "data/train-00000-of-00001.parquet"))
    return [Item(id=str(r.key), messages=user_msg(r.prompt),
                 meta={"prompt": r.prompt, "instruction_id_list": list(r.instruction_id_list),
                       "kwargs": _clean_kwargs(list(r.kwargs))}, gen={})
            for r in df.itertuples()]


def grade(items, responses, ctx):
    payload = [{"id": it.id, "prompt": it.meta["prompt"],
                "instruction_id_list": it.meta["instruction_id_list"], "kwargs": it.meta["kwargs"],
                "response": strip_think(responses[it.id]["content"]) if it.id in responses else ""}
               for it in items]
    p = subprocess.run([str(VENV_PY), SCORER], input=json.dumps(payload), capture_output=True,
                       text=True, check=True)
    res = {r["id"]: r for r in json.loads(p.stdout)}
    rows, inst_strict, inst_loose, strict = [], [], [], []
    for it in items:
        r = res[it.id]
        rows.append({"id": it.id, "score": float(all(r["loose"])), "strict": all(r["strict"]),
                     "instructions": it.meta["instruction_id_list"], "loose_per_inst": r["loose"]})
        strict.append(all(r["strict"]))
        inst_strict += r["strict"]
        inst_loose += r["loose"]
    pct = lambda v: round(100 * sum(v) / max(len(v), 1), 1)  # noqa: E731
    return rows, {"prompt_strict": pct(strict), "inst_loose": pct(inst_loose),
                  "inst_strict": pct(inst_strict)}
