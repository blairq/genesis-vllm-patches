"""Reproduce pedidos capturados por PN166 (trazas/pedidos/*.json) TAL CUAL, en orden, contra el servidor; solo
cambia max_tokens (no entra en el hash del prefijo). Por pedido: prompt, cacheados, TTFT. Va ADENTRO del contenedor.
Uso: python3 reproducir_pedidos.py <max_tokens> archivo1.json archivo2.json ..."""
import json, os, sys, time, urllib.request
BASE = "http://127.0.0.1:8320"; KEY = os.environ.get("VLLM_API_KEY", "")
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY}
MAXT = int(sys.argv[1])
for ruta in sys.argv[2:]:
    reg = json.load(open(ruta)); body = dict(reg["pedido"])
    body["max_tokens"] = MAXT; body.pop("max_completion_tokens", None)
    body["stream"] = True; body["stream_options"] = {"include_usage": True}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); ttft = None; uso = None
    with urllib.request.urlopen(r, timeout=1800) as f:
        for raw in f:
            l = raw.decode().strip()
            if not l.startswith("data:") or l == "data: [DONE]": continue
            j = json.loads(l[5:])
            if j.get("usage"): uso = j["usage"]
            if ttft is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("tool_calls") or (c.get("delta") or {}).get("reasoning_content") for c in j.get("choices", [])):
                ttft = time.time() - t0
    cache = (uso.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    print(f"{os.path.basename(ruta)}: prompt {uso['prompt_tokens']} cacheados {cache} ({cache / uso['prompt_tokens']:.0%}) "
          f"TTFT {ttft or 0:.1f} s total {time.time() - t0:.1f} s gen {uso['completion_tokens']}", flush=True)
