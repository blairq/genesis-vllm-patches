#!/usr/bin/env python3
"""Arma un banco de pedidos de CODIGO CON AGENTE para medir y entrenar el borrador.

Fuente: nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), trayectorias de OpenHands con
tool calls reales y el esquema de las herramientas. Cada pedido es una trayectoria cortada JUSTO
ANTES de un turno del asistente: el modelo tiene que producir ese turno (pensar, tool call o
codigo), que es lo que hace el agente de opencode.

Lee el parquet de a un grupo de filas: cargarlo entero con pandas se comio la RAM de la maquina.

Uso: armar_banco_codigo.py <trajectories.parquet> <salida.jsonl> [n] [min_chars] [max_chars] [semilla]

Variables: EXCLUIR=<banco.jsonl> saca los repos de ese banco (el de evaluacion no se entrena);
POR_REPO=<k> permite hasta k trayectorias por repo (default 1).
"""
import os
import json
import random
import sys

import pyarrow.parquet as pq

src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MIN_C = int(sys.argv[4]) if len(sys.argv) > 4 else 12_000      # ~3,5k tokens
MAX_C = int(sys.argv[5]) if len(sys.argv) > 5 else 140_000     # ~40k tokens
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 1234)

pf = pq.ParquetFile(src)
grupos = list(range(pf.metadata.num_row_groups))
rng.shuffle(grupos)
elegidos, repos = [], {}
POR_REPO = int(os.environ.get("POR_REPO", "1"))
EXCLUIDOS = set()
if os.environ.get("EXCLUIR"):
    EXCLUIDOS = {json.loads(l)["repo"] for l in open(os.environ["EXCLUIR"])}


def limpio(m):
    d = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        d["tool_calls"] = [{"id": t["id"], "type": "function",
                            "function": {"name": t["function"]["name"],
                                         "arguments": t["function"]["arguments"]}}
                           for t in m["tool_calls"]]
    if m["role"] == "tool":
        d["tool_call_id"] = m.get("tool_call_id") or ""
    return d


def lotes():
    """De a 32 filas: un grupo entero son ~4000 trayectorias y no entra en memoria como Python."""
    for g in grupos:
        it = pf.iter_batches(batch_size=32, row_groups=[g],
                             columns=["trajectory_id", "repo", "trajectory", "tools"])
        for b in it:
            if rng.random() < 0.25:                # salteo al azar: variedad sin leer todo
                yield b.to_pylist()


for t in lotes():
    if len(elegidos) >= N:
        break
    rng.shuffle(t)
    for fila in t:
        if len(elegidos) >= N:
            break
        if fila["repo"] in EXCLUIDOS or repos.get(fila["repo"], 0) >= POR_REPO:   # variedad
            continue
        msgs = fila["trajectory"]
        cortes, acum = [], 0
        for i, m in enumerate(msgs):
            if m["role"] == "assistant" and i > 1 and MIN_C <= acum <= MAX_C:
                cortes.append(i)
            acum += len(m.get("content") or "") + sum(len(tc["function"]["arguments"] or "")
                                                     for tc in (m.get("tool_calls") or []))
        if not cortes:
            continue
        i = rng.choice(cortes)
        tools = [{"type": "function", "function": {
            "name": x["function"]["name"], "description": x["function"]["description"],
            "parameters": json.loads(json.dumps(x["function"]["parameters"]),
                                     object_hook=lambda o: {k: v for k, v in o.items() if v is not None})}}
            for x in fila["tools"]]
        elegidos.append({"id": fila["trajectory_id"], "repo": fila["repo"], "corte": i,
                         "chars": sum(len(m.get("content") or "") for m in msgs[:i]),
                         "messages": [limpio(m) for m in msgs[:i]], "tools": tools})
        repos[fila["repo"]] = repos.get(fila["repo"], 0) + 1
    del t

with open(out, "w") as f:
    for e in elegidos:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
cs = sorted(e["chars"] for e in elegidos)
print(f"{len(elegidos)} pedidos, chars p10/p50/p90 = {cs[len(cs)//10]}/{cs[len(cs)//2]}/{cs[9*len(cs)//10]}")
