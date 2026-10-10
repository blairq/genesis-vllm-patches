# SPDX-License-Identifier: Apache-2.0
"""Corre los tests de un problema de LiveCodeBench. Se ejecuta DENTRO del contenedor sin red.

Lee /w/job.json {code, tests, fn_name, timeout} e imprime {passed, n_ok, error}.
Comparacion como la oficial de LCB: stdin linea por linea con tolerancia
numerica; funcional por igualdad de JSON. Corta en el primer test que falla.
"""
import json
import os
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

# Importaciones que LCB antepone al codigo generado.
PREFIX = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""

CALL = """
import json as __json, sys as __sys
__args = [__json.loads(l) for l in __sys.stdin.read().split("\\n") if l.strip()]
__res = Solution().{fn}(*__args)
__sys.stdout.write(__json.dumps(__res))
"""


def same_stdout(got: str, exp: str) -> bool:
    g = [s.strip() for s in got.strip().split("\n")]
    e = [s.strip() for s in exp.strip().split("\n")]
    if len(g) != len(e):
        return False
    for a, b in zip(g, e):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        try:
            if any(Decimal(x) != Decimal(y) for x, y in zip(ta, tb)):
                return False
        except InvalidOperation:
            return False
    return True


def same_json(got: str, exp: str) -> bool:
    try:
        return json.loads(got) == json.loads(exp)
    except json.JSONDecodeError:
        return False


def main():
    job = json.load(open("/w/job.json"))
    fn = job["fn_name"]
    src = PREFIX + job["code"] + (CALL.format(fn=fn) if fn else "")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "sol.py")
    open(path, "w").write(src)
    n_ok = 0
    for i, t in enumerate(job["tests"]):
        try:
            p = subprocess.run([sys.executable, path], input=t["input"], capture_output=True,
                               text=True, timeout=job["timeout"], cwd=tmp)
        except subprocess.TimeoutExpired:
            print(json.dumps({"passed": False, "n_ok": n_ok, "error": f"timeout en test {i}"}))
            return
        if p.returncode != 0:
            print(json.dumps({"passed": False, "n_ok": n_ok,
                              "error": f"runtime en test {i}: {p.stderr.strip()[-300:]}"}))
            return
        ok = same_json(p.stdout, t["output"]) if fn else same_stdout(p.stdout, t["output"])
        if not ok:
            print(json.dumps({"passed": False, "n_ok": n_ok, "error": f"salida incorrecta en test {i}"}))
            return
        n_ok += 1
    print(json.dumps({"passed": True, "n_ok": n_ok, "error": ""}))


main()
