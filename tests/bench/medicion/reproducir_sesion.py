"""Reproduce una SESION capturada respetando los tiempos originales: cada pedido sale a su desfase desde el primero
(escalado), en su propio hilo, asi se superponen como en produccion. Por pedido: hilo, prompt, cacheados, TTFT.
Va ADENTRO del contenedor. Uso: python3 reproducir_sesion.py <salida.json> <max_tokens> <escala_tiempo> archivos..."""
import json, os, sys, threading, time, urllib.request
BASE = "http://127.0.0.1:8320"; KEY = os.environ.get("VLLM_API_KEY", "")
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
SALIDA, MAXT, ESC = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
regs = [(r, json.load(open(r))) for r in sys.argv[4:]]
regs.sort(key=lambda x: x[1]["t"])
t_ref = regs[0][1]["t"]
res = []

def hilo(reg):
    p = reg["pedido"]; n = len(p.get("tools") or [])
    return {12: "X", 17: "Y", 14: "Z"}.get(n, f"tools{n}")

def uno(ruta, reg, t0_global):
    body = dict(reg["pedido"]); body["max_tokens"] = MAXT; body.pop("max_completion_tokens", None)
    body["stream"] = True; body["stream_options"] = {"include_usage": True}
    espera = (reg["t"] - t_ref) * ESC - (time.time() - t0_global)
    if espera > 0: time.sleep(espera)
    hd = dict(H)
    if os.environ.get("PLUGIN") == "1":               # las cabeceras que pondria el plugin genesis-sesion de opencode
        import hashlib
        g = hilo(reg)
        if g == "Z":                                  # subagente de Y: una sesion por tarea (su primer mensaje de usuario)
            u = next((m for m in body["messages"] if m.get("role") == "user"), {})
            ses = "ses_Z_" + hashlib.sha1(json.dumps(u.get("content"))[:4000].encode()).hexdigest()[:10]
            hd.update({"X-Genesis-Sesion": ses, "X-Genesis-Padre": "ses_Y", "X-Genesis-Raiz": "ses_Y", "X-Genesis-Agente": "agi_coder"})
        else:
            hd.update({"X-Genesis-Sesion": "ses_" + g, "X-Genesis-Padre": "", "X-Genesis-Raiz": "ses_" + g, "X-Genesis-Agente": "build_low"})
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hd)
    t0 = time.time(); ttft = None; uso = None
    try:
        with urllib.request.urlopen(r, timeout=3600) as f:
            for raw in f:
                l = raw.decode().strip()
                if not l.startswith("data:") or l == "data: [DONE]": continue
                j = json.loads(l[5:])
                if j.get("usage"): uso = j["usage"]
                if ttft is None and any((c.get("delta") or {}).get(k) for c in j.get("choices", []) for k in ("content", "tool_calls", "reasoning_content")):
                    ttft = time.time() - t0
    except Exception as e:
        res.append({"archivo": os.path.basename(ruta), "error": str(e)}); return
    cache = (uso.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    x = {"archivo": os.path.basename(ruta), "hilo": hilo(reg), "salida_s": round(t0 - t0_global, 1), "prompt": uso["prompt_tokens"],
         "cache": cache, "ttft": round(ttft or 0, 1), "total": round(time.time() - t0, 1), "gen": uso["completion_tokens"]}
    res.append(x)
    print(f"{x['salida_s']:6.0f}s {x['hilo']} {x['archivo']}: prompt {x['prompt']} cacheados {cache} ({cache / x['prompt']:.0%}) TTFT {x['ttft']} s total {x['total']} s", flush=True)

t0g = time.time()
hs = [threading.Thread(target=uno, args=(ruta, reg, t0g)) for ruta, reg in regs]
for h in hs: h.start()
for h in hs: h.join()
json.dump(sorted(res, key=lambda x: x.get("salida_s", 0)), open(SALIDA, "w"), indent=1)
tot = [x for x in res if "prompt" in x]
print(f"TOTAL: prompt {sum(x['prompt'] for x in tot)} cacheados {sum(x['cache'] for x in tot)} "
      f"({sum(x['cache'] for x in tot) / max(1, sum(x['prompt'] for x in tot)):.0%}) | TTFT suma {sum(x['ttft'] for x in tot):.0f} s | duracion {time.time() - t0g:.0f} s")
