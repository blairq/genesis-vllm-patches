# SPDX-License-Identifier: Apache-2.0
"""Corpus de razonamiento en ASD-STE100: traduce el thinking de trazas propias y verifica que siga sirviendo.

Por cada traza:
  1. traducir   el thinking a STE con el Qwen servido, SIN thinking y con la respuesta prefijada en
                prosa ("The task is"): con thinking 'low' sale igual de bien y 8,6x mas lento.
  2. completar  lo que el modelo no hace confiable con el prompt va por codigo: en tareas de formato
                se agrega textual el ultimo borrador cercado (```) de la traza original.
  3. verificar  que el STE solo alcance para llegar a la respuesta, sin la conclusion escrita:
                  gpqa  se borran las lineas de conclusion y se fuerza "ANSWER:" con 3 tokens;
                  lcb   se borran los bloques de codigo y el codigo que sale pasa los tests;
                  if    la respuesta que sale pasa el verificador oficial de IFBench.
  4. reintentar una vez, con mas temperatura, si falla. Si vuelve a fallar, la traza se rechaza.

El prompt del sistema es inmutable y largo a proposito (palabras aprobadas + rechazadas con su
reemplazo + reglas, ~8,4k tokens): en este modelo hibrido el prefijo solo se cachea si pasa ~4 bloques
de 880 tokens. Un pedido de calentamiento deja guardado su estado y el resto reusa ~6k tokens.

  python3 entrenamiento/ste_corpus/pipeline.py --fuente suite --n 50 --concurrencia 4
  python3 entrenamiento/ste_corpus/pipeline.py --salida <dir>      # reanuda lo que falte

OJO: --fuente suite usa trazas de los benchmarks de evaluacion: sirve para medir el pipeline, NUNCA
para entrenar (contaminaria la suite). Las fuentes de entrenamiento son otras (ver README).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from benchmarks.qwen38_card.common import default_api_key  # noqa: E402

AQUI = Path(__file__).parent
CACHE = Path.home() / ".cache" / "ste_corpus"
PREFIJO = "The task is"
SUITE = REPO / "benchmarks/results/qwen38_card/completa_20261009_180151"


# ── vocabulario ASD-STE100 (paquete biz-dfch-asdste100vocab, AGPL: se baja a la cache, no va al repo) ──

def vocabulario() -> dict:
    out = CACHE / "vocab.json"
    if out.is_file():
        return json.loads(out.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "biz-dfch-asdste100vocab==0.8.1", "--no-deps",
                    "-d", str(CACHE), "-q"], check=True)
    whl = next(CACHE.glob("biz_dfch_asdste100vocab-*.whl"))
    with zipfile.ZipFile(whl) as z:
        datos = z.read("biz/dfch/asdste100vocab/data/asdste100_issue9_base.jsonl").decode()
    aprob, rech = set(), {}
    for linea in datos.splitlines():
        w = json.loads(linea)
        if w["status"] == "approved":
            aprob.add(w["name"].lower())
        elif w["status"] == "rejected" and w["alternatives"]:
            rech[w["name"].lower()] = [a["name"].lower() for a in w["alternatives"]][:2]
    v = {"aprobadas": sorted(aprob), "rechazadas": rech}
    out.write_text(json.dumps(v))
    return v


# ── servidor ──

class Server:
    def __init__(self, endpoint: str):
        self.ep = endpoint.rstrip("/")
        self.http = httpx.Client(headers={"Authorization": f"Bearer {default_api_key()}"}, timeout=3600)

    def ntok(self, t: str) -> int:
        return self.http.post(f"{self.ep}/tokenize", json={"model": "qwen3.8", "prompt": t,
                                                         "add_special_tokens": False}).json()["count"]

    def traducir(self, sistema: str, traza: str, temperatura: float, usuario: str | None = None,
                 prefijo: str = PREFIJO) -> dict:
        t0 = time.time()
        r = self.http.post(f"{self.ep}/v1/chat/completions", json={
            "model": "qwen3.8", "max_tokens": 16000, "temperature": temperatura, "top_p": 0.9, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
            "continue_final_message": True, "add_generation_prompt": False,
            "messages": [{"role": "system", "content": sistema},
                         {"role": "user", "content": usuario or ("<reasoning>\n" + traza + "\n</reasoning>")},
                         {"role": "assistant", "content": prefijo}]}).json()
        u = r["usage"]
        return {"texto": (prefijo + (r["choices"][0]["message"]["content"] or "")).strip(),
                "s": round(time.time() - t0, 1), "salida": u["completion_tokens"], "entrada": u["prompt_tokens"],
                "cache": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                "fin": r["choices"][0]["finish_reason"]}

    def completar(self, prompt: str, max_tokens: int) -> str:
        r = self.http.post(f"{self.ep}/v1/completions", json={"model": "qwen3.8", "prompt": prompt,
                                                             "max_tokens": max_tokens, "temperature": 0.0}).json()
        return r["choices"][0]["text"]


def chatml(mensajes: list[tuple[str, str]], razonamiento: str) -> str:
    s = "".join(f"<|im_start|>{rol}\n{txt}<|im_end|>\n" for rol, txt in mensajes)
    return s + f"<|im_start|>assistant\n<think>\n{razonamiento.strip()}\n</think>\n\n"


# ── fuentes ──

def fuente_suite(n: int, semilla: int) -> list[dict]:
    """Trazas de la corrida completa de la suite que respondieron bien. Solo para medir el pipeline."""
    from benchmarks.qwen38_card.benches import gpqa, ifbench, livecodebench
    def graded_ok(b):
        filas = (json.loads(x) for x in open(SUITE / f"{b}.graded.jsonl"))
        return {r["id"] for r in filas if r["score"] == 1}
    def respuestas(b):
        return [json.loads(x) for x in open(SUITE / f"{b}.responses.jsonl")]
    casos = []
    gi = {i.id: i for i in gpqa.load(argparse.Namespace())}
    for r in respuestas("gpqa_diamond"):
        if r["id"] in graded_ok("gpqa_diamond") and r["finish_reason"] == "stop" and r["content"].strip():
            it = gi[r["id"]]
            casos.append({"tipo": "gpqa", "id": r["id"], "traza": r["reasoning"], "pregunta": it.messages[0]["content"],
                          "oro": it.meta["gold"]})
    li = {i.id: i for i in livecodebench.load(argparse.Namespace(lcb_start="2025-02-01", lcb_end="2025-05-01"))}
    for r in respuestas("livecodebench_v6"):
        if r["id"] in graded_ok("livecodebench_v6") and r["finish_reason"] == "stop" and r["completion_tokens"] <= 40000:
            casos.append({"tipo": "lcb", "id": r["id"], "traza": r["reasoning"], "item": li[r["id"]]})
    ii = {i.id: i for i in ifbench.load(None)}
    for r in respuestas("ifbench"):
        if r["id"] in graded_ok("ifbench") and r["finish_reason"] == "stop":
            casos.append({"tipo": "if", "id": r["id"], "traza": r["reasoning"], "item": ii[r["id"]],
                          "respuesta": r["content"]})
    rnd = random.Random(semilla)
    por_tipo = {t: [c for c in casos if c["tipo"] == t] for t in ("gpqa", "lcb", "if")}
    for v in por_tipo.values():
        rnd.shuffle(v)
    # mezcla pareja entre los tres tipos
    elegidos, k = [], 0
    while len(elegidos) < n and any(por_tipo.values()):
        t = ("gpqa", "lcb", "if")[k % 3]
        if por_tipo[t]:
            elegidos.append(por_tipo[t].pop())
        k += 1
    return elegidos


# ── verificacion ──

CONCLUSION = re.compile(r"(?im)^.*(answer\s*[:：]|correct (option|choice|answer) is|the answer is|final answer|"
                        r"option [A-D] is correct).*$")


def ultimo_borrador(traza: str, respuesta: str = "") -> str | None:
    b = re.findall(r"```[^\n]*\n.*?```", traza, re.S)
    if b:
        return b[-1]
    r = respuesta.strip()
    # sin cerco: si la respuesta final ya estaba escrita en el razonamiento, ese era el borrador verificado
    if r and len(r) <= 4000 and r[:120] in traza:
        return "```\n" + r + "\n```"
    return None


# Lo que el modelo no cumple confiable con el prompt se arregla aca.
ECO_PROMPT = re.compile(r"(?im)^\s*(use only approved words|keep the order of the steps|use the active voice|"
                        r"one idea in each sentence|do not (use tags|write an introduction)|write only the rewritten trace)\.?\s*$")
LINEA_ANSWER = re.compile(r"(?im)^\s*\**answer\s*[:：].*$")
BLOQUE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def limpiar(ste: str) -> str:
    ste = LINEA_ANSWER.sub("", ECO_PROMPT.sub("", ste))
    return re.sub(r"\n{3,}", "\n\n", ste).strip()


def cortado(t: str) -> bool:
    """Termina a mitad de oracion: letra o numero ASCII, coma o parentesis abierto al final."""
    return bool(re.search(r"[A-Za-z0-9,(\-]\s*$", t))


NUMERO = re.compile(r"\d{4}-\d{2}-\d{2}|\d+\.\d+|\d{3,}")


def numeros_inventados(ste: str, original: str) -> list[str]:
    """Numeros del STE (3+ digitos, decimales o fechas) que no estan en la traza original. El reescritor
    llego a cambiar 1812 por 2012: un numero corrupto con aspecto de correcto es lo peor que puede tener
    el corpus, y el filtro de la respuesta no lo ve si la conclusion viene textual del original."""
    return sorted({n for n in NUMERO.findall(ste) if n not in original})


def solo_ultimo_bloque(ste: str) -> str:
    """Los borradores de codigo son parte del trabajo, pero alcanza con la ultima version."""
    bloques = list(re.finditer(r"```[^\n]*\n.*?```", ste, re.S))
    for m in reversed(bloques[:-1]):
        ste = ste[:m.start()] + "[earlier code draft removed]" + ste[m.end():]
    return ste


def tramos(traza: str, srv: "Server", objetivo: int = 4000) -> list[str]:
    """Corta la traza en tramos de ~objetivo tokens en limites de parrafo."""
    parrafos, out, actual, n = re.split(r"(\n\s*\n)", traza), [], "", 0
    for p in parrafos:
        t = max(1, len(p) // 4)  # estimacion barata; el corte exacto no importa
        if n + t > objetivo and actual.strip():
            out.append(actual)
            actual, n = "", 0
        actual += p
        n += t
    if actual.strip():
        out.append(actual)
    return out


def traducir_traza(srv: "Server", sistema: str, traza: str, temp: float, umbral: int, tok: int) -> dict:
    if tok <= umbral:
        return srv.traducir(sistema, traza, temp)
    partes = tramos(traza, srv)
    textos, total_s, salida, cache = [], 0.0, 0, 0
    for i, parte in enumerate(partes, 1):
        usuario = (f"<reasoning part=\"{i} of {len(partes)}\">\n{parte}\n</reasoning>\n"
                   f"This is part {i} of {len(partes)} of one trace. Rewrite only this part. Keep the doubts, the "
                   f"approaches that the part tries, the evidence, and the state of the decision at the end of the part.")
        r = srv.traducir(sistema, parte, temp, usuario=usuario, prefijo=PREFIJO if i == 1 else "Next,")
        textos.append(r["texto"])
        total_s += r["s"]
        salida += r["salida"]
        cache += r["cache"]
    # Condensacion incremental: con toda la traza junta (~20k tokens) el modelo emite fin de mensaje a mitad
    # de oracion (mismo sintoma que el razonamiento largo de HLE). Se funde de a un tramo: la entrada queda chica.
    acum, fusiones, sin_fundir = textos[0], 0, 0
    for t in textos[1:]:
        usuario = (f"<reasoning_so_far>\n{acum}\n</reasoning_so_far>\n<next_part>\n{t}\n</next_part>\n"
                   "Both texts are already in STE. Merge the next part into the reasoning so far. Remove each repeated "
                   "check, recall or conclusion. Keep each doubt once, with the evidence for each side, and each "
                   "decision with its key evidence. Keep the calculations and counts that a decision uses. Keep the "
                   "order. Remove the planning of the final reply to the user (its format, its wording, \"the final "
                   "answer is\" restatements): the reply is written after the reasoning. Write the complete merged "
                   "reasoning, from the first step to the last step of the next part.")
        r = srv.traducir(sistema, t, temp, usuario=usuario)
        if cortado(r["texto"]):
            r = srv.traducir(sistema, t, min(temp + 0.3, 1.0), usuario=usuario)
        total_s += r["s"]
        salida += r["salida"]
        cache += r["cache"]
        if cortado(r["texto"]):
            acum = acum + "\n" + t   # este tramo queda sin fundir, pero no se pierde
            sin_fundir += 1
        else:
            acum = r["texto"]
            fusiones += 1
    return {"texto": acum, "s": round(total_s, 1), "salida": salida, "cache": cache,
            "fin": "tramos:%d+fusiones:%d+sin_fundir:%d" % (len(partes), fusiones, sin_fundir),
            "sin_fundir_texto": "\n".join(textos)}


VEREDICTO = re.compile(r"(?im)^.*(\b(option|statement|choice)\s+[A-D]\b.*\b(correct|incorrect|true|false|wrong|right|"
                       r"matches|fits|is the)\b|\bmatches option\b|\bthe correct (option|choice)\b).*$")


def verificar(srv: Server, caso: dict, ste: str) -> tuple[bool, str]:
    if caso["tipo"] == "gpqa":
        p = chatml([("user", caso["pregunta"])], VEREDICTO.sub("", CONCLUSION.sub("", ste))) + "ANSWER:"
        m = re.search(r"[A-D]", srv.completar(p, 3))
        letra = m.group(0) if m else None
        return letra == caso["oro"], f"letra {letra}"
    if caso["tipo"] == "lcb":
        from benchmarks.qwen38_card.benches import livecodebench as L
        it = caso["item"]
        sin_codigo = re.sub(r"```.*?```", "", ste, flags=re.S)
        texto = srv.completar(chatml([("system", it.messages[0]["content"]), ("user", it.messages[1]["content"])],
                                     sin_codigo), 8000)
        code = L.extract_code(texto)
        if not code:
            return False, "sin codigo"
        res = L.run_tests(code, it.meta["tests"], it.meta["fn_name"], 6.0)
        return res["passed"], "pasa" if res["passed"] else res.get("error", "")[:60]
    from benchmarks.qwen38_card.benches import ifbench as IB
    it = caso["item"]
    resp = srv.completar(chatml([("user", it.meta["prompt"])], ste), 6000)
    pl = [{"id": it.id, "prompt": it.meta["prompt"], "instruction_id_list": it.meta["instruction_id_list"],
           "kwargs": it.meta["kwargs"], "response": resp}]
    out = subprocess.run([str(IB.VENV_PY), IB.SCORER], input=json.dumps(pl), capture_output=True, text=True, check=True)
    loose = json.loads(out.stdout)[0]["loose"]
    return all(loose), f"{sum(loose)}/{len(loose)}"


def cumplimiento(t: str, aprob: set) -> dict:
    sin = re.sub(r"```.*?```|\$[^$]*\$", " ", t, flags=re.S)
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+|\n+", sin) if re.search(r"[A-Za-z]{2}", o)]
    largos = [len(re.findall(r"[A-Za-z][A-Za-z'-]*", o)) for o in oraciones]
    palabras = [w.lower() for w in re.findall(r"\b[A-Za-z][a-z]+\b", sin)]
    return {"pct_le20": round(100 * sum(x <= 20 for x in largos) / max(len(largos), 1)),
            "pct_aprobadas": round(100 * sum(w in aprob for w in palabras) / max(len(palabras), 1))}


# ── corrida ──

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8360")
    ap.add_argument("--fuente", default="suite", choices=["suite"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--concurrencia", type=int, default=8,
                    help="el server topea en ~600 tok/s con 6-8 hilos; en prosa rinden menos, por eso 8")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--prompt", default="p2", help="version en prompts/")
    ap.add_argument("--min-tokens", type=int, default=1500, help="trazas mas cortas no ganan nada: se saltean")
    ap.add_argument("--umbral-tramos", type=int, default=6000, help="trazas mas largas se traducen por tramos")
    a = ap.parse_args()

    v = vocabulario()
    aprob = set(v["aprobadas"])
    sistema = ((AQUI / "prompts" / f"{a.prompt}.txt").read_text().replace("{PALABRAS}", ", ".join(v["aprobadas"]))
               .replace("{RECHAZADAS}", "; ".join(f"{k} -> {' / '.join(x)}" for k, x in sorted(v["rechazadas"].items()))))
    salida = Path(a.salida) if a.salida else REPO / "entrenamiento/ste_corpus/corridas" / time.strftime("%Y%m%d_%H%M%S")
    salida.mkdir(parents=True, exist_ok=True)
    corpus_p = salida / "corpus.jsonl"
    hechos = {json.loads(x)["clave"] for x in corpus_p.read_text().splitlines()} if corpus_p.is_file() else set()

    srv = Server(a.endpoint)
    casos = [c for c in fuente_suite(a.n, a.semilla) if f"{c['tipo']}:{c['id']}" not in hechos]
    print(f"{len(hechos)} ya hechas, {len(casos)} por hacer, concurrencia {a.concurrencia}", flush=True)
    srv.traducir(sistema, "", 0.3)  # calentamiento del prefijo inmutable
    lock, t0 = threading.Lock(), time.time()

    def uno(c: dict) -> dict:
        orig_tok = srv.ntok(c["traza"])
        if orig_tok < a.min_tokens:
            return {"clave": f"{c['tipo']}:{c['id']}", "tipo": c["tipo"], "id": c["id"], "aceptada": False,
                    "salteada": "corta", "tok_orig": orig_tok, "tok_ste": 0, "intentos": [], "cumplimiento": {},
                    "ste": None, "ste_rechazado": None}
        intentos = []
        previo = None
        for temp in (0.3, 0.7):
            if previo and previo.get("sin_fundir_texto"):
                tr = {**previo, "texto": previo["sin_fundir_texto"], "fin": previo["fin"] + "->sin_fundir"}
            else:
                tr = traducir_traza(srv, sistema, c["traza"], temp, a.umbral_tramos, orig_tok)
            previo = tr
            ste = solo_ultimo_bloque(limpiar(tr["texto"]))
            if (inv := numeros_inventados(ste, c["traza"])):
                intentos.append({"temp": temp, "ok": False, "detalle": "numeros inventados: " + ", ".join(inv[:5]),
                                 "s": tr["s"], "cache": tr["cache"], "salida": tr["salida"], "fin": tr["fin"]})
                ok = False
                continue
            if cortado(ste):
                intentos.append({"temp": temp, "ok": False, "detalle": "texto cortado", "s": tr["s"],
                                 "cache": tr["cache"], "salida": tr["salida"], "fin": tr["fin"]})
                ok = False
                continue
            if c["tipo"] == "if" and (b := ultimo_borrador(c["traza"], c.get("respuesta", ""))):
                ste = ste.rstrip() + "\nFinal draft, checked:\n" + b
            ok, detalle = verificar(srv, c, ste)
            intentos.append({"temp": temp, "ok": ok, "detalle": detalle, "s": tr["s"], "cache": tr["cache"],
                             "salida": tr["salida"], "fin": tr["fin"]})
            if ok:
                break
        return {"clave": f"{c['tipo']}:{c['id']}", "tipo": c["tipo"], "id": c["id"], "aceptada": ok,
                "tok_orig": orig_tok, "tok_ste": srv.ntok(ste), "intentos": intentos,
                "cumplimiento": cumplimiento(ste, aprob), "ste": ste if ok else None,
                "ste_rechazado": None if ok else ste}

    filas = []
    with ThreadPoolExecutor(a.concurrencia) as ex, corpus_p.open("a") as f:
        for fut in as_completed([ex.submit(uno, c) for c in casos]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                filas.append(r)
                estado = "SALT" if r.get("salteada") else ("OK " if r["aceptada"] else "RECH")
                detalle = r["intentos"][-1]["detalle"] if r["intentos"] else r.get("salteada", "")
                print(f"[{len(filas)}/{len(casos)}] {r['clave']:<14} {estado} "
                      f"{r['tok_orig']:>6} -> {r['tok_ste']:>5} intentos {len(r['intentos'])} ({detalle})", flush=True)

    leidas = [json.loads(x) for x in corpus_p.read_text().splitlines()]
    todas = [r for r in leidas if not r.get("salteada")]
    ok = [r for r in todas if r["aceptada"]]
    resumen = {"salteadas_por_cortas": len(leidas) - len(todas), "trazas": len(todas), "aceptadas": len(ok), "tasa_aceptacion": round(100 * len(ok) / max(len(todas), 1), 1),
               "aceptadas_al_primer_intento": sum(r["aceptada"] and len(r["intentos"]) == 1 for r in todas),
               "por_tipo": {t: f"{sum(r['aceptada'] for r in todas if r['tipo'] == t)}/{sum(r['tipo'] == t for r in todas)}"
                            for t in ("gpqa", "lcb", "if")},
               "compresion_media_aceptadas_pct": round(100 * (1 - sum(r["tok_ste"] for r in ok) / max(sum(r["tok_orig"] for r in ok), 1)), 1),
               "segundos_esta_corrida": round(time.time() - t0), "concurrencia": a.concurrencia}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
