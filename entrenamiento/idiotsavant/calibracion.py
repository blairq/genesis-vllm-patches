#!/usr/bin/env python3
"""Arma el set de calibracion (.npy int32 [N, L]) para idiotsavant.py.

Tres formas de conseguirlo, de la mas exacta a la mas propia:

  1. La EXACTA de idiotSavant: viene en el repo de HuggingFace (reproducir/calib_256x4096.npy).
     Con esa, la reconstruccion da los mismos pesos (salvo el no-determinismo de la GPU).
  2. --desde-hf: la arma sola desde el dataset publico del que salio la original,
     nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0): trayectorias de agentes de codigo con
     tool calls reales. Baja el parquet (~2 GB, a la cache LOCAL del proyecto), toma una trayectoria
     por repo con semilla fija, la corta justo DESPUES de un turno del asistente (la ventana termina
     en una respuesta) y se queda con los ultimos L tokens.
     Diferencia con la original: alla las respuestas eran las del modelo servido (noon, perfil de
     agente de opencode); aca son las del agente que grabo el dataset. Da una calibracion equivalente
     en tipo de trafico, no identica.
  3. --conversaciones: un JSONL propio, una conversacion por linea en formato de chat de OpenAI
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}],
         "tools": [...]}                                   # "tools" es opcional
     Usar trafico REAL del uso que se le va a dar y el mismo chat template con el que se va a servir
     (--template si no es el del modelo). Ver DECISIONES.md, B4.

Uso:
    bash correr.sh calibracion.py --bf16 BF16 --desde-hf --salida calib.npy [--n 256] [--largo 4096]
    bash correr.sh calibracion.py --bf16 BF16 --conversaciones charlas.jsonl --salida calib.npy
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

DATASET = "nebius/SWE-rebench-openhands-trajectories"


def normalizar(mensajes):
    """Formato de OpenAI -> lo que espera el chat template de Qwen: los argumentos de las llamadas a
    tools vienen como TEXTO JSON y el template itera un diccionario (vLLM hace esta conversion al
    servir; aca hay que hacerla a mano para tokenizar igual)."""
    for m in mensajes:
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            if isinstance(f.get("arguments"), str):
                try:
                    f["arguments"] = json.loads(f["arguments"])
                except ValueError:
                    pass
    return mensajes


def tokenizar(tok, mensajes, tools):
    ids = tok.apply_chat_template(normalizar(mensajes), tools=tools, tokenize=True, add_generation_prompt=False)
    # transformers 5 devuelve un BatchEncoding (tipo diccionario, pero no dict)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def desde_jsonl(ruta):
    for linea in open(ruta):
        c = json.loads(linea)
        yield c["messages"], c.get("tools")


def desde_hf(largo, semilla):
    """Trayectorias del dataset publico, cortadas despues de un turno del asistente."""
    from huggingface_hub import constants, hf_hub_download
    constants.HF_HUB_OFFLINE = False             # entorno.sh lo apaga; aca SI hay que bajar
    import pyarrow.parquet as pq
    ruta = hf_hub_download(DATASET, "trajectories.parquet", repo_type="dataset")
    rng = random.Random(semilla)
    pf = pq.ParquetFile(ruta)
    grupos = list(range(pf.metadata.num_row_groups))
    rng.shuffle(grupos)
    vistos = set()
    min_c = 3 * largo                            # ~3,5 caracteres por token: asi suele alcanzar
    for g in grupos:
        # de a 32 filas: un grupo entero son miles de trayectorias y no entra en memoria como Python
        for b in pf.iter_batches(batch_size=32, row_groups=[g], columns=["repo", "trajectory", "tools"]):
            filas = b.to_pylist()
            rng.shuffle(filas)
            for f in filas:
                if f["repo"] in vistos:          # una por repo: variedad
                    continue
                msgs, acum, cortes = f["trajectory"], 0, []
                for i, m in enumerate(msgs):
                    acum += len(m.get("content") or "") + sum(len((tc.get("function") or {}).get("arguments") or "")
                                                              for tc in (m.get("tool_calls") or []))
                    if m["role"] == "assistant" and i > 1 and acum >= min_c:
                        cortes.append(i)
                if not cortes:
                    continue
                i = rng.choice(cortes)
                limpios = []
                for m in msgs[: i + 1]:          # INCLUYE el turno del asistente
                    d = {"role": m["role"], "content": m.get("content") or ""}
                    if m.get("tool_calls"):
                        d["tool_calls"] = [{"id": t["id"], "type": "function",
                                            "function": {"name": t["function"]["name"],
                                                         "arguments": t["function"]["arguments"]}}
                                           for t in m["tool_calls"]]
                    if m["role"] == "tool":
                        d["tool_call_id"] = m.get("tool_call_id") or ""
                    limpios.append(d)
                tools = [{"type": "function", "function": {
                    "name": x["function"]["name"], "description": x["function"]["description"],
                    "parameters": json.loads(json.dumps(x["function"]["parameters"]),
                                             object_hook=lambda o: {k: v for k, v in o.items() if v is not None})}}
                    for x in (f["tools"] or [])]
                vistos.add(f["repo"])
                yield limpios, tools


def main():
    ap = argparse.ArgumentParser(description="arma el .npy de calibracion para idiotsavant.py")
    ap.add_argument("--bf16", required=True, help="directorio del modelo (tokenizer y chat template)")
    fuente = ap.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--desde-hf", dest="desde_hf", action="store_true",
                        help=f"armarla desde {DATASET} (se baja ~2 GB a la cache local)")
    fuente.add_argument("--conversaciones", help="JSONL propio, una conversacion por linea")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--n", type=int, default=256, help="muestras (256 en idiotSavant)")
    ap.add_argument("--largo", type=int, default=4096, help="tokens por muestra (4096 en idiotSavant)")
    ap.add_argument("--semilla", type=int, default=1234)
    ap.add_argument("--template", default="", help="chat template .jinja si no es el del modelo")
    a = ap.parse_args()
    if a.desde_hf:                               # antes de importar transformers/huggingface_hub:
        os.environ["HF_HUB_OFFLINE"] = "0"       # leen la variable al importarse
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.bf16)
    if a.template:
        tok.chat_template = open(a.template).read()
    fuente_it = desde_hf(a.largo, a.semilla) if a.desde_hf else desde_jsonl(a.conversaciones)
    out, cortas, malas = [], 0, 0
    for mensajes, tools in fuente_it:
        if len(out) >= a.n:
            break
        try:
            ids = tokenizar(tok, mensajes, tools)
        except Exception as e:  # noqa: BLE001
            malas += 1
            print(f"conversacion {len(out) + cortas + malas}: se saltea ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        if len(ids) < a.largo:
            cortas += 1
            continue
        out.append(np.asarray(ids[-a.largo:], np.int32))
        if len(out) % 32 == 0:
            print(f"  {len(out)}/{a.n} muestras", flush=True)
    if not out:
        raise SystemExit(f"ninguna conversacion llega a {a.largo} tokens ({cortas} cortas, {malas} con error)")
    arr = np.stack(out)
    np.save(a.salida, arr)
    print(f"{len(out)} muestras de {a.largo} tokens -> {a.salida} ({arr.nbytes / 1e6:.0f} MB); "
          f"descartadas: {cortas} cortas, {malas} con error")
    if len(out) < a.n:
        print(f"AVISO: se pidieron {a.n} y hay {len(out)}. Con menos de ~128 la Hessiana de down_proj "
              f"(17408 entradas) queda mal condicionada: bajar --largo o sumar conversaciones.")


if __name__ == "__main__":
    main()
