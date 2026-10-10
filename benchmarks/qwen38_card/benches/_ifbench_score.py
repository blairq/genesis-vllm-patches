# SPDX-License-Identifier: Apache-2.0
"""Verificador de IFBench. Corre DENTRO del venv de la cache (ver setup.sh).

stdin: [{id, prompt, instruction_id_list, kwargs, response}]
stdout: [{id, strict: [bool], loose: [bool]}]
"""
import json
import os
import sys

# El directorio de este script tiene un ifbench.py propio que taparia al paquete oficial.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import evaluation_lib  # noqa: E402  (del clon de IFBench; setup.sh lo agrega al venv con un .pth)

out = []
for d in json.load(sys.stdin):
    inp = evaluation_lib.InputExample(key=d["id"], instruction_id_list=d["instruction_id_list"],
                                      prompt=d["prompt"], kwargs=d["kwargs"])
    p2r = {d["prompt"].strip(): d["response"]}
    res = {}
    for fn, name in ((evaluation_lib.test_instruction_following_strict, "strict"),
                     (evaluation_lib.test_instruction_following_loose, "loose")):
        try:
            res[name] = list(fn(inp, p2r).follow_instruction_list)
        except Exception as e:  # un verificador que explota cuenta como no cumplido
            print(f"[ifbench] {d['id']} {name}: {e}", file=sys.stderr)
            res[name] = [False] * len(d["instruction_id_list"])
    out.append({"id": d["id"], **res})
json.dump(out, sys.stdout)
