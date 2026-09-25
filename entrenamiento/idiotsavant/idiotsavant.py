#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  qwen3.8_27b_idiotSavant_sm_86 — reconstruccion completa desde el BF16, en UN script
# ══════════════════════════════════════════════════════════════════════════════════════════════
#
#  Entrada : orcarouter/Qwen3.8-27B-Uncensored (BF16, 55 GB, 64 capas: 48 GDN + 16 atencion)
#            + un set de calibracion de tokens [N, L] int32 (.npy)
#  Salida  : checkpoint compressed-tensors (pack-quantized, W4 simetrico g128) que vLLM carga tal
#            cual, CON EL RESIDUO ROTADO. Se sirve como W4A8 (VLLM_MARLIN_INPUT_DTYPE=int8) y
#            necesita GENESIS_ENABLE_PN148_ROT_DOWN=1 (Hadamard antes de down_proj).
#
#  Uso (ver README.md; todo corre en el virtualenv local del proyecto, preparado con preparar.sh):
#    ./correr.sh idiotsavant.py todo --dry-run  --bf16 BF16 --calib CALIB --trabajo DIR --salida DIR
#    ./correr.sh idiotsavant.py todo            --bf16 BF16 --calib CALIB --trabajo DIR --salida DIR
#    ./correr.sh idiotsavant.py estado [--json] --trabajo DIR --salida DIR      (en cualquier momento)
#    ./correr.sh tui.py modelo --trabajo DIR --salida DIR                        (visor opcional)
#  Las decisiones y las mediciones que las respaldan estan en DECISIONES.md.
#
#  El borrador DFlash2 (qwen3.8_27b_idiotSavant_sm_86_dflash2) NO sale de aca: se entrena contra
#  este modelo SERVIDO; ver dflash2.sh en esta misma carpeta.
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  EL OBJETIVO Y LA HISTORIA CORTA
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  El rig es 2x RTX 3090 (GA102, sm_86). En SM86 el tensor core rinde fp16 1x, int8 4x, int4 8x:
#  la idea del proyecto es calcular en int8 en TODO el decode. Los pesos van en int4 (el decode esta
#  en el techo de ancho de banda: menos bytes = mas rapido) y las ACTIVACIONES en int8 por token
#  (Marlin W4A8). Eso deja dos fuentes de error, y las dos se atacan aca:
#
#   1. Pesos int4. GPTQ (Frantar 2022) con Hessianas de trafico real de agente de codigo. Contra
#      el AutoRound de noon (el cuant publico que usabamos) da ~18% menos error local por capa.
#
#   2. Activaciones int8 por token. ESTE era el problema grande y no se veia: la escala int8 de
#      cada token la fija su maximo, y las entradas que salen del residuo (qkv, in_proj del GDN,
#      gate/up) tienen "cresta" max/rms ~56: un canal con una activacion masiva aplasta al resto.
#      Medido por lineal (error relativo de la salida, int8 por token contra fp16):
#          k_proj/v_proj  10-14% (37-48% en las capas 3, 7, 11, 15)     cresta ~56
#          in_proj_qkv    12%    (24% en la capa 6)                     cresta ~58
#          gate/up        5-7%                                          cresta ~32
#          down_proj      6%     (peores: 54-61)                        cresta ~27
#          o_proj/out     2-3%                                          cresta 12-21
#      En el modelo entero, W4A8 duplicaba el error de W4A16 (KL sobre respuestas 0,012 -> 0,029).
#
#  La solucion es ROTAR: una transformacion ortogonal R reparte el pico entre todos los canales
#  (la cresta baja de ~56 a ~4) sin cambiar lo que calcula el modelo en aritmetica exacta
#  (QuaRot, arXiv 2404.00456; SpinQuant 2405.16406; para Mamba/SSM, MambaQuant 2501.13484).
#  Con Hadamard por bloques el error de A8 queda <1% en todas las lineales (o_proj/out 1,2%), y
#  ademas GPTQ cuantiza MEJOR los pesos rotados: el error local int4 baja en las 64 capas
#  (mediana -8%, capa 0 -43%).
#
#  Resultado (KL(BF16 || modelo) sobre las posiciones de RESPUESTA de trafico real, top-20):
#                           torch W4A16   torch W4A8   vLLM W4A8
#      noon (AutoRound)        0,0155       0,0300       0,0385
#      GPTQ sin rotar          0,0122       0,0289       0,0365
#      idiotSavant (este)      0,0104       0,0116       0,0194      <- la mitad que noon
#  y la velocidad igual (prefill/decode en paridad; SK-23 fusiona la Hadamard de down con el int8).
#
#  OJO al medir: el KL sobre texto de PROMPT (sistema/usuario) es basura — el modelo de chat ahi
#  predice <|im_end|> al 99% y cualquier perturbacion lo da vuelta (KL ~1,2 con cualquier cuant).
#  Solo vale sobre lo que genera el asistente.
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  QUE SE HACE EN CADA CAPA Y POR QUE (convencion de filas de torch: y = x @ W.T)
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  R = rotacion global del residuo:  Rt = diag(signos) @ Hb,  Hb = Hadamard por bloques de 1024
#  (5120 = 5 x 1024), signos +-1 al azar con semilla fija. Todo el residuo del modelo servido vive
#  en la base h~ = h @ Rt. Por que Hadamard por BLOQUES y no densa: con bloques de 1024 el error
#  de A8 ya es <1%, y es la misma forma que usan PN148/PN149 en linea (sin leer pesos).
#
#  Por que eso es gratis: RMSNorm(h Rt) = RMSNorm(h) Rt, porque la norma euclidea no cambia con una
#  rotacion. Lo que NO conmuta es el peso elementwise de la norma (GemmaRMSNorm: x * (1 + w)), asi
#  que se PLIEGA en la lineal siguiente y la norma queda en w = 0.
#
#  Por lineal (n = entrada normalizada SIN el peso de la norma, g = 1 + w):
#
#   * Entradas del residuo: self_attn.{q,k,v}_proj / linear_attn.in_proj_{qkv,z} (tras
#     input_layernorm) y mlp.{gate,up}_proj (tras post_attention_layernorm).
#         se cuantiza  A = W diag(g) Rt        con la Hessiana de n~ = n Rt:  H~ = R H_n Rt
#     -> es donde vive el pico: aca esta TODO el beneficio de A8.
#     (Se acumula H_n directo desde la entrada de la norma. La capa 7 tiene un canal con g = 0
#      EXACTO — el 3994, el de la activacion masiva, rms 70 contra 0,15 de mediana — que la norma
#      "apaga". Rotado, esa energia entra a gate/up y SIN su fila en la Hessiana el error de la
#      MLP se multiplicaba x7 y el KL servido saltaba a 0,17. Con H_n completa GPTQ lo compensa.)
#
#   * Escriben al residuo: self_attn.o_proj / linear_attn.out_proj.
#         se cuantiza  A = R W                 con su Hessiana tal cual
#     -> su SALIDA tiene que quedar en la base rotada. Rotar su entrada casi no ayuda (cresta
#        12-21, A8 1,2% con o sin rotacion), asi que no se toca.
#
#   * down_proj: escribe al residuo Y ademas su entrada tiene cresta ~27.
#         se cuantiza  A = R W Hd              con H~ = Hd H Hd
#     Hd = Hadamard por bloques de 512 sobre la dimension intermedia, aplicada EN LINEA en vLLM
#     (PN148/SK-23, fusionada con SiluAndMul y el int8: sin costo medible). 512 porque con TP=2
#     cada GPU tiene la mitad de 17408 = 8704 = 17 x 512: los bloques no cruzan la particion.
#     Sin esta Hadamard el KL W4A8 sube 15% (0,0116 -> 0,0134).
#
#   * linear_attn.in_proj_a / in_proj_b (compuertas del GDN): quedan en BF16 (son chicas y el
#     GDN es sensible a sus compuertas), pero leen el residuo: se pliegan igual, W diag(g) Rt.
#
#   * Todo lo demas de la capa (conv1d, A_log, dt_bias, normas internas q/k y la gated norm del
#     GDN) opera DENTRO de la cabeza, despues de la proyeccion: no ve la base del residuo.
#
#  Fuera de las capas:
#   * embed_tokens:  E Rt   (el residuo nace rotado).
#   * norma final:   w = 0, y su g se pliega en lm_head:  (W diag(g)) Rt.  Se guarda g en el config
#     (genesis_rotacion.g_final): el borrador (PN149) y su entrenador la necesitan para volver a la
#     base original. (vLLM cuantiza el lm_head a int4 al cargar con PN139; rotado da el mismo error.)
#   * vision: la torre queda igual; solo el merger ESCRIBE al residuo: linear_fc2 -> R W, bias b Rt.
#   * mtp.*: fuera. No se usa (el especulativo es DFlash2) y quedaria invalido en la base rotada.
#
#  Precision y formato (lo que Marlin W4A8 + PN130 esperan): int4 simetrico por grupo de 128,
#  escala fp16 = max|w|/7,5 fijada al entrar al grupo sobre los pesos YA corregidos por GPTQ,
#  damp 0,01, SIN reordenamiento (sin g_idx: Marlin lee los grupos contiguos). Empaquetado
#  compressed-tensors: q + 8 en nibbles, el valor i del int32 en los bits 4i..4i+3.
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  SENSIBILIDAD POR CAPA (para leer el informe; KL de respuesta con UNA capa cuantizada)
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Perfil suave: bajo en 0-18, meseta en 23-51, baja hacia 62; la 63 es la mas sensible (2x).
#  Una capa de atencion completa pesa ~1,5x una GDN. Las 8 peores suman solo el 22%: no hay
#  "capas culpables", por eso no hay precision mixta — lo que paga es mejorar todas por igual.
#  Correlacion entre error local y KL: 0,6 (el error local solo orienta).
#
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  CALIBRACION
# ──────────────────────────────────────────────────────────────────────────────────────────────
#  256 muestras x 4096 tokens de trafico REAL del servidor (agente de codigo de opencode con tools):
#  cada muestra es la ventana FINAL de prompt + respuesta de noon, tokenizada con el mismo chat
#  template y el hook P69 que usa produccion (tests/bench/medicion/trazas/banco/armar_calibracion.py).
#  Las dos primeras muestras quedan APARTADAS (no entran a las Hessianas) para medir el error
#  local de cada capa sin favorecer a GPTQ. La calibracion propaga la salida BF16 de cada capa
#  (no la cuantizada): asi cada capa se calibra contra la entrada "verdadera".
#
#  Diferencia con el armado original (etapas A/B/C cacheadas en cuant-cache/): alla la Hessiana se
#  guardaba sobre x = g n y la fila del canal muerto de la capa 7 se estimaba con la diagonal desde
#  la muestra apartada; aca se acumula H_n exacta. Resultado identico en todas las capas salvo la 7,
#  donde esto es lo correcto.
# ══════════════════════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

G = 128                  # grupo de las escalas int4
SEMILLA = 20260924       # signos de la rotacion (los mismos que el borrador y PN149)
BLOQUE = 1024            # Hadamard del residuo
BLOQUE_DOWN = 512        # Hadamard en linea antes de down_proj (PN148)
DAMP = 0.01

COPIAR = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json",
          "special_tokens_map.json", "LICENSE")
# El BF16 de orcarouter no trae processor_config.json y vLLM lo pide (sin el intenta bajarlo del Hub).
PROCESSOR_CONFIG = {
    "image_processor": {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
                        "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessor",
                        "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16, "resample": 3,
                        "rescale_factor": 0.00392156862745098,
                        "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2},
    "processor_class": "Qwen3VLProcessor",
    "video_processor": {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
                        "do_sample_frames": True, "fps": 2, "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5], "max_frames": 768, "merge_size": 2, "min_frames": 4,
                        "patch_size": 16, "resample": 3, "rescale_factor": 0.00392156862745098,
                        "return_metadata": False, "size": {"longest_edge": 25165824, "shortest_edge": 4096},
                        "temporal_patch_size": 2, "video_processor_type": "Qwen3VLVideoProcessor"}}

# lineal -> (entrada cuya Hessiana usa, clase de rotacion)
#   "entrada": lee el residuo tras una norma (se pliega g y se rota la entrada)
#   "escribe": escribe al residuo (se rota la salida)
#   "bajada" : escribe al residuo y ademas lleva la Hadamard en linea en la entrada
LINEALES = {
    "full_attention": {"self_attn.q_proj": ("attn_in", "entrada"), "self_attn.k_proj": ("attn_in", "entrada"),
                       "self_attn.v_proj": ("attn_in", "entrada"), "self_attn.o_proj": ("attn_out", "escribe"),
                       "mlp.gate_proj": ("mlp_in", "entrada"), "mlp.up_proj": ("mlp_in", "entrada"),
                       "mlp.down_proj": ("mlp_down", "bajada")},
    "linear_attention": {"linear_attn.in_proj_qkv": ("attn_in", "entrada"),
                         "linear_attn.in_proj_z": ("attn_in", "entrada"),
                         "linear_attn.out_proj": ("attn_out", "escribe"),
                         "mlp.gate_proj": ("mlp_in", "entrada"), "mlp.up_proj": ("mlp_in", "entrada"),
                         "mlp.down_proj": ("mlp_down", "bajada")},
}
# BF16 que leen el residuo (compuertas del GDN): se pliegan y rotan pero no se cuantizan
BF16_ENTRADA = ("linear_attn.in_proj_a", "linear_attn.in_proj_b")
NORMA_DE = {"attn_in": "input_layernorm", "mlp_in": "post_attention_layernorm"}


# ─── lectura perezosa del BF16 (55 GB en 32 GB de RAM: tensor por tensor) ────────────────────────
class Pesos:
    def __init__(self, d):
        self.d = d
        self.mapa = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]

    def get(self, k, dev="cpu", dtype=None):
        with safe_open(os.path.join(self.d, self.mapa[k]), "pt", device="cpu") as f:
            t = f.get_tensor(k)
        return t.to(dev) if dtype is None else t.to(dev, dtype)

    def claves(self, pref):
        return [k for k in self.mapa if k.startswith(pref)]


# ─── rotaciones ─────────────────────────────────────────────────────────────────────────────────
def hadamard(n, dev, dt=torch.float32):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / n ** 0.5).to(dev, dt)


def signos(n):
    g = torch.Generator().manual_seed(SEMILLA)
    return (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(torch.int8)


def bloques(X, b):
    """X @ Hd con Hd Hadamard por bloques de b sobre la ultima dim (simetrica, Hd = Hd^-1)."""
    return (X.reshape(*X.shape[:-1], -1, b) @ hadamard(b, X.device, X.dtype)).reshape(X.shape)


def Rt_de(D, dev):
    return signos(D).to(dev, torch.float32)[:, None] * torch.block_diag(*[hadamard(BLOQUE, dev)] * (D // BLOQUE))


# ─── GPTQ W4 simetrico g128 (el mismo que valido la etapa B) ─────────────────────────────────────
@torch.no_grad()
def gptq(W, H, damp=DAMP, bloque=128):
    """W [out, in], H [in, in] fp32. Devuelve (q int8 en [-8, 7], escala fp16 [out, in/G])."""
    W = W.float().clone()
    out, inn = W.shape
    H = H.float().clone()
    muertos = torch.diag(H) == 0              # entradas que nunca se activan: peso irrelevante
    H[muertos, muertos] = 1
    W[:, muertos] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(inn, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(out, inn, dtype=torch.int8, device=W.device)
    S = torch.zeros(out, inn // G, dtype=torch.float16, device=W.device)
    for i1 in range(0, inn, bloque):
        i2 = min(i1 + bloque, inn)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hi1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % G == 0:
                # escala del grupo sobre los pesos YA corregidos (como el observer minmax de llm-compressor)
                ini = col - i1
                s = (torch.cat([W1[:, ini:], W[:, i2:col + G]], 1)[:, :G].abs().amax(1) / 7.5).clamp_min(1e-10).half()
                S[:, col // G] = s
            sc = S[:, col // G].float()
            w = W1[:, i]
            q = (w / sc).round().clamp(-8, 7)
            Q[:, col] = q.to(torch.int8)
            err = (w - q * sc) / Hi1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, S


def empaquetar(Q, S):
    """compressed-tensors pack-quantized: q + 8 en nibbles, valor i del int32 en los bits 4i..4i+3."""
    out, inn = Q.shape
    q = (Q.to(torch.int32) + 8).view(out, inn // 8, 8).cpu()
    p = (q << torch.arange(0, 32, 4, dtype=torch.int32)).sum(-1, dtype=torch.int64)
    p = torch.where(p >= 2 ** 31, p - 2 ** 32, p).to(torch.int32)
    return p.contiguous(), S.cpu().contiguous(), torch.tensor([out, inn], dtype=torch.int64)


def dequant(Q, S):
    out, inn = Q.shape
    return (Q.float().view(out, inn // G, G) * S.float().unsqueeze(-1)).view(out, inn)


# ══════════════════════════════════════════════════════════════════════════════════════════════
#  EJECUCION: etapas separadas, trabajo cacheado, recursos controlados
# ══════════════════════════════════════════════════════════════════════════════════════════════
#  El BF16 pesa 55 GB, la maquina tiene 32 GB de RAM y dos GPUs de 24 GB, y una corrida entera
#  lleva hora y media. Por eso:
#
#  * ETAPAS en procesos separados, que se comunican por disco (--trabajo):
#      calibrar   pasada BF16 capa por capa; escribe las Hessianas de cada capa + la muestra apartada
#      cuantizar  consume esas Hessianas: GPTQ rotado, escribe capa_NN.safetensors, mide la capa
#      armar      embedding/lm_head/vision/config cuando estan las 64 capas
#      todo       chequea recursos y corre calibrar y cuantizar EN PARALELO (una GPU cada uno si hay
#                 dos; si hay una, la comparten: ~13 + ~8 GB), despues armar.
#    Si uno de los dos muere, lo hecho queda; se relanza y sigue.
#
#  * CACHE: cada capa deja una marca (CALIBRADA / CUANTIZADA) y todo se escribe atomicamente
#    (archivo temporal + rename): relanzar saltea lo terminado. La calibracion guarda el estado
#    oculto cada --cada capas (10,7 GB, solo si el disco da) para no repasar desde el embedding;
#    sin ese estado igual repasa rapido (solo el forward) las capas ya cuantizadas.
#    Las Hessianas se BORRAN al cuantizar la capa, salvo --conservar_hessianas (~0,8 GB por capa, ~50
#    GB en total: permiten rehacer variantes sin recalibrar).
#
#  * RECURSOS: antes de empezar se chequean RAM (la del host y el limite del cgroup del contenedor),
#    memoria libre de GPU y disco. Durante la corrida, antes de cada capa, si falta RAM o disco se
#    ESPERA (con aviso) en vez de reventar; y la calibracion no se adelanta mas de --adelanto capas a
#    la cuantizacion, asi las Hessianas pendientes nunca llenan el disco.
#
#  * INTERFAZ: linea de comandos pura (texto plano, un evento por linea con el prefijo de la etapa,
#    codigos de salida: 0 ok, 1 error, 2 recursos insuficientes / dry-run fallido). `estado --json`
#    da el avance en forma de maquina (pensado para que lo consulte una LLM o un script). La TUI
#    (tui.py) es solo un visor que lee lo mismo: nunca es necesaria.
#
#  * DRY-RUN (--dry-run): valida TODO sin escribir capas: el BF16 (todas las claves que cada capa
#    necesita), la calibracion, los recursos, la matematica (rotaciones, GPTQ, empaquetado) y una
#    capa GDN y una de atencion de punta a punta en memoria, con su error local. ~2 minutos.
# ══════════════════════════════════════════════════════════════════════════════════════════════
GB = 1024 ** 3


# ─── recursos ───────────────────────────────────────────────────────────────────────────────────
def ram_libre_gb():
    """MemAvailable del host, acotada por el limite del cgroup (adentro de docker manda ese)."""
    libre = None
    for linea in open("/proc/meminfo"):
        if linea.startswith("MemAvailable:"):
            libre = int(linea.split()[1]) * 1024
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        usado = int(open("/sys/fs/cgroup/memory.current").read())
        if lim != "max":
            libre = min(libre, int(lim) - usado)
    except OSError:
        pass
    return libre / GB


def disco_libre_gb(ruta):
    return shutil.disk_usage(ruta).free / GB


def gpu_libre_gb(dev):
    if not dev.startswith("cuda"):
        return float("inf")
    libre, _ = torch.cuda.mem_get_info(torch.device(dev))
    return libre / GB


def esperar(cond, que, cada=30):
    """Espera a que cond() sea verdad, avisando cada 5 minutos."""
    t0, aviso = time.time(), 0.0
    while not cond():
        if time.time() - aviso > 300:
            print(f"  esperando: {que} ({(time.time() - t0) / 60:.0f} min)", flush=True)
            aviso = time.time()
        time.sleep(cada)


def esperar_recursos(ram_gb, disco_gb, ruta, que):
    esperar(lambda: ram_libre_gb() >= ram_gb and disco_libre_gb(ruta) >= disco_gb,
            f"{que}: RAM {ram_libre_gb():.1f}/{ram_gb} GB, disco {disco_libre_gb(ruta):.1f}/{disco_gb} GB")


def guardar_atomico(obj, ruta, fn=torch.save):
    tmp = ruta + ".tmp"
    fn(obj, tmp)
    os.replace(tmp, ruta)


def marca(dirc, nombre, texto="ok"):
    with open(os.path.join(dirc, nombre + ".tmp"), "w") as f:
        f.write(texto)
    os.replace(os.path.join(dirc, nombre + ".tmp"), os.path.join(dirc, nombre))


def hay(dirc, nombre):
    return os.path.exists(os.path.join(dirc, nombre))


# ─── Hessianas ──────────────────────────────────────────────────────────────────────────────────
class Hess:
    def __init__(self, n, dev):
        self.H = torch.zeros(n, n, dtype=torch.float32, device=dev)
        self.cresta = []            # max/rms por token, antes y despues de rotar (el porque de rotar)

    @torch.no_grad()
    def sumar(self, x, Rt=None):
        x = x.reshape(-1, x.shape[-1])
        for i in range(0, x.shape[0], 4096):
            xf = x[i:i + 4096].float()
            self.H.addmm_(xf.T, xf)
            if Rt is not None and len(self.cresta) < 8:
                c = lambda z: float((z.abs().amax(-1) / z.pow(2).mean(-1).sqrt().clamp_min(1e-12)).median())  # noqa: E731
                self.cresta.append((c(xf), c(xf @ Rt)))


def h_guardar(H, ruta):
    """Solo el triangulo superior en fp32 (la de down_proj: 0,6 GB en vez de 1,2)."""
    msk = torch.ones_like(H, dtype=torch.bool).triu_()
    guardar_atomico({"n": H.shape[0], "triu": H.masked_select(msk).cpu()}, ruta)


def h_cargar(ruta, dev):
    d = torch.load(ruta)
    n = d["n"]
    msk = torch.ones(n, n, dtype=torch.bool, device=dev).triu_()
    H = torch.zeros(n, n, dtype=torch.float32, device=dev)
    H.masked_scatter_(msk, d["triu"].to(dev))
    return H + H.triu(1).T


def normalizada(r, eps):
    """n = r / rms(r): la entrada de la lineal SIN el peso de la norma (el que se pliega)."""
    rf = r.float()
    return rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps)


def enganchar(capa, lin, D, eps, Rt, dev, sumar):
    """Acumuladores + ganchos de las 4 entradas de una capa. sumar = {"si": bool} (se consulta en cada
    llamada: las muestras apartadas pasan por la capa pero no suman)."""
    n_out = [k for k, v in lin.items() if v[0] == "attn_out"][0]
    acc = {"attn_in": Hess(D, dev), "mlp_in": Hess(D, dev),
           "attn_out": Hess(capa.get_submodule(n_out).in_features, dev),
           "mlp_down": Hess(capa.mlp.down_proj.in_features, dev)}
    ganchos = [
        capa.input_layernorm.register_forward_pre_hook(
            lambda mo, ar: acc["attn_in"].sumar(normalizada(ar[0], eps), Rt) if sumar["si"] else None),
        capa.post_attention_layernorm.register_forward_pre_hook(
            lambda mo, ar: acc["mlp_in"].sumar(normalizada(ar[0], eps), Rt) if sumar["si"] else None),
        capa.get_submodule(n_out).register_forward_pre_hook(
            lambda mo, ar: acc["attn_out"].sumar(ar[0]) if sumar["si"] else None),
        capa.mlp.down_proj.register_forward_pre_hook(
            lambda mo, ar: acc["mlp_down"].sumar(ar[0]) if sumar["si"] else None)]
    return acc, ganchos


def cuantizar_capa(i, tipo, sd, hessiana, dev, Rt, R):
    """La capa i en su forma SERVIBLE. hessiana(entrada) -> H [n, n] fp32 (del disco o en memoria).
    Devuelve (tensores con nombres del checkpoint, pesos efectivos para medir, error de pesos, muertos)."""
    pref = f"model.language_model.layers.{i}."
    lin = LINEALES[tipo]
    g = {e: 1 + sd[nm + ".weight"].float() for e, nm in NORMA_DE.items()}
    muertos = {e: (g[e] == 0).nonzero().flatten().tolist() for e in g}
    tens, efect, err_w, Ht = {}, {}, {}, {}
    for base, (ent, clase) in lin.items():
        W = sd[base + ".weight"].float()
        if ent not in Ht:
            Ht = {}                                   # una viva por vez (la de down es 1,2 GB)
            H = hessiana(ent)
            if clase == "entrada":
                H = R @ H @ Rt                        # Hessiana de n~ = n Rt
            elif clase == "bajada":
                H = bloques(bloques(H, BLOQUE_DOWN).T.contiguous(), BLOQUE_DOWN)   # Hd H Hd
            Ht = {ent: H}
            del H
        if clase == "entrada":
            A = (W * g[ent][None, :]) @ Rt            # norma plegada + entrada rotada
        elif clase == "escribe":
            A = R @ W                                 # salida rotada
        else:
            A = bloques(R @ W, BLOQUE_DOWN)           # salida rotada + Hadamard en linea en la entrada
        Q, S = gptq(A, Ht[ent])
        p, s2, shp = empaquetar(Q, S)
        tens[pref + base + ".weight_packed"] = p
        tens[pref + base + ".weight_scale"] = s2
        tens[pref + base + ".weight_shape"] = shp
        Ad = dequant(Q, S)
        err_w[base] = float((Ad - A).norm() / A.norm())
        efect[base + ".weight"] = Ad.cpu()
        del A, Ad, Q, S, W
    del Ht
    for k, v in sd.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else k
        if base in lin:
            continue
        if base in ("input_layernorm", "post_attention_layernorm"):
            v = torch.zeros_like(v)                   # g plegada en las lineales: la norma queda en 1
        elif base in BF16_ENTRADA:
            v = ((v.float() * g["attn_in"][None, :]) @ Rt).to(torch.bfloat16)   # compuertas GDN
        tens[pref + k] = v.cpu() if k.endswith("A_log") else v.to(torch.bfloat16).cpu()
        efect[k] = tens[pref + k]
    return tens, efect, err_w, muertos


def error_local(m, tc, i, efect, x, y, dev, Rt, rot_emb):
    """Error de la capa SERVIBLE: entra x Rt, tiene que salir y Rt; relativo a lo que la capa agrega.
    Incluye la fuga de un canal muerto si la hubiera (la simulacion en la base original no la ve)."""
    with torch.device("meta"):
        cq = m.Qwen3_5DecoderLayer(tc, i)
    cq.load_state_dict({k: (v.float() if not k.endswith("A_log") else v).to(dev) for k, v in efect.items()},
                       assign=True, strict=True)
    cq.mlp.down_proj.register_forward_pre_hook(lambda mo, ar: (bloques(ar[0], BLOQUE_DOWN),))
    xr = x.to(dev).float() @ Rt
    ref = y.to(dev).float() @ Rt
    n_ap, L = xr.shape[:2]
    pos = torch.arange(L, device=dev)[None].expand(n_ap, -1)
    cos, sin = rot_emb(xr, pos)
    o = cq(xr, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
    o = o[0] if isinstance(o, tuple) else o
    return float((o - ref).norm() / (ref - xr).norm())


def modelo_hf(bf16):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = json.load(open(os.path.join(bf16, "config.json")))
    tc = Qwen3_5TextConfig(**cfg["text_config"])
    tc._attn_implementation = "sdpa"
    return m, tc, cfg


def cargar_capa(m, tc, P, i, dev, dtype=torch.bfloat16):
    pref = f"model.language_model.layers.{i}."
    with torch.device("meta"):
        capa = m.Qwen3_5DecoderLayer(tc, i)
    sd = {k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else dtype) for k in P.claves(pref)}
    capa.load_state_dict(sd, assign=True, strict=True)
    return capa.eval(), sd


def dir_capa(a, i):
    d = os.path.join(a.trabajo, f"capa_{i:02d}")
    os.makedirs(d, exist_ok=True)
    return d


def listo_cuantizada(a, i):
    return hay(dir_capa(a, i), "CUANTIZADA") and os.path.exists(os.path.join(a.salida, f"capa_{i:02d}.safetensors"))


# ─── etapa: calibrar ──────────────────────────────────────────────────────────────────────────────
def calibrar(a):
    """Pasada BF16 capa por capa. Por capa escribe H de sus 4 entradas y la muestra apartada.

    attn_in / mlp_in se toman en la entrada de la NORMA y se normalizan sin su peso (n): es lo que ve
    la lineal en el modelo rotado (peso plegado), incluido el canal muerto de la capa 7.
    La calibracion propaga la salida BF16 (no la cuantizada)."""
    dev = a.dispositivo
    torch.set_grad_enabled(False)
    m, tc, _ = modelo_hf(a.bf16)
    D, eps = tc.hidden_size, tc.rms_norm_eps
    n_capas = a.capas or tc.num_hidden_layers
    P = Pesos(a.bf16)
    Rt = Rt_de(D, dev)
    ids = np.load(a.calib)
    if a.muestras:
        ids = ids[: a.muestras]
    if a.largo:
        ids = ids[:, -a.largo:]
    N, L = ids.shape
    necesita = N * L * D * 2 / GB + 3.5            # estado + capa + acumuladores (+ down 1,2 GB)
    if gpu_libre_gb(dev) < necesita:
        raise SystemExit(f"calibrar: la GPU tiene {gpu_libre_gb(dev):.1f} GB libres y hacen falta ~{necesita:.1f}")
    print(f"[calibrar] {N} x {L} tokens, {n_capas} capas, {dev}", flush=True)

    # estado oculto: el ultimo guardado o los embeddings
    ruta_est = os.path.join(a.trabajo, "estado.npy")
    ini = 0
    if os.path.exists(ruta_est) and os.path.exists(ruta_est + ".json"):
        ini = json.load(open(ruta_est + ".json"))["capa"]
        mm = np.load(ruta_est, mmap_mode="r")
        h = torch.empty(N, L, D, dtype=torch.bfloat16, device=dev)
        for s in range(N):
            h[s] = torch.from_numpy(np.array(mm[s])).view(torch.bfloat16).to(dev)
        del mm
        print(f"[calibrar] retomando desde la capa {ini} (estado guardado)", flush=True)
    else:
        E = P.get("model.language_model.embed_tokens.weight", dev, torch.bfloat16)
        h = torch.empty(N, L, D, dtype=torch.bfloat16, device=dev)
        for s in range(N):
            h[s] = E[torch.from_numpy(ids[s].astype(np.int64)).to(dev)]
        del E
    rot_emb = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None].expand(a.micro, -1)
    cos, sin = rot_emb(h[: a.micro], pos)
    h_capa_gb = {t: (2 * D * D + 2 * (tc.linear_num_value_heads * tc.linear_value_head_dim if t == "linear_attention"
                                     else tc.num_attention_heads * tc.head_dim) ** 2
                     + tc.intermediate_size ** 2) * 2 / GB for t in set(tc.layer_types)}

    for i in range(ini, n_capas):
        if all(hay(dir_capa(a, j), "CALIBRADA") or listo_cuantizada(a, j) for j in range(i, n_capas)):
            print(f"[calibrar] capas {i}..{n_capas - 1} ya calibradas: nada mas que hacer", flush=True)
            break
        t0 = time.time()
        dirc = dir_capa(a, i)
        ya = hay(dirc, "CALIBRADA") or listo_cuantizada(a, i)
        tipo = tc.layer_types[i]
        if not ya:
            # no adelantarse demasiado a la cuantizacion: las Hessianas pendientes ocupan disco
            esperar(lambda: sum(1 for j in range(i) if not listo_cuantizada(a, j)) <= a.adelanto,
                    f"capa {i}: la cuantizacion va {a.adelanto}+ capas atras")
            esperar_recursos(4, h_capa_gb[tipo] + a.margen_disco, a.trabajo, f"calibrar capa {i}")
        capa, sd = cargar_capa(m, tc, P, i, dev)
        del sd
        acc, ganchos, sumar = {}, [], {"si": not ya}
        if not ya:
            acc, ganchos = enganchar(capa, LINEALES[tipo], D, eps, Rt, dev, sumar)
            x_ap = h[: a.apartar].cpu().clone()
        for s in range(0, N, a.micro):
            # las primeras --apartar muestras NO entran a las Hessianas: sirven para medir la capa
            sumar["si"] = (not ya) and s >= a.apartar
            k = min(a.micro, N - s)
            o = capa(h[s:s + k], position_embeddings=(cos[:k], sin[:k]), attention_mask=None, position_ids=pos[:k])
            h[s:s + k] = o[0] if isinstance(o, tuple) else o
        for g_ in ganchos:
            g_.remove()
        del capa
        if not torch.isfinite(h[: a.micro]).all():
            raise FloatingPointError(f"capa {i}: NaN/inf en el estado oculto")
        if not ya:
            for e, ac in acc.items():
                h_guardar(ac.H, os.path.join(dirc, f"H_{e}.pt"))
            guardar_atomico({"x": x_ap, "y": h[: a.apartar].cpu().clone()}, os.path.join(dirc, "apartada.pt"))
            cresta = {e: [round(float(np.median([c[0] for c in acc[e].cresta])), 1),
                          round(float(np.median([c[1] for c in acc[e].cresta])), 1)] for e in ("attn_in", "mlp_in")}
            json.dump(cresta, open(os.path.join(dirc, "cresta.json"), "w"))
            marca(dirc, "CALIBRADA", f"{time.time() - t0:.0f}s")
            del acc
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
        # estado oculto cada --cada capas, si el disco da (10,7 GB): retomar sin repasar desde el embedding
        if (i + 1) % a.cada == 0 and i + 1 < n_capas and disco_libre_gb(a.trabajo) > N * L * D * 2 / GB + 2 * a.margen_disco:
            mm = np.lib.format.open_memmap(ruta_est + ".tmp.npy", mode="w+", dtype=np.int16, shape=(N, L, D))
            for s in range(N):
                mm[s] = h[s].view(torch.int16).cpu().numpy()
            mm.flush()
            del mm
            os.replace(ruta_est + ".tmp.npy", ruta_est)
            json.dump({"capa": i + 1}, open(ruta_est + ".json", "w"))
        print(f"[calibrar] capa {i:02d} {tipo:16s} {'(ya estaba: solo forward) ' if ya else ''}"
              f"{time.time() - t0:5.0f}s  RAM libre {ram_libre_gb():.1f} GB  disco {disco_libre_gb(a.trabajo):.0f} GB",
              flush=True)
    marca(a.trabajo, "CALIBRACION_TERMINADA")


# ─── etapa: cuantizar ─────────────────────────────────────────────────────────────────────────────
def cuantizar(a):
    """Consume las Hessianas de calibrar, en orden, a medida que aparecen."""
    dev = a.dispositivo
    torch.set_grad_enabled(False)
    m, tc, _ = modelo_hf(a.bf16)
    D = tc.hidden_size
    n_capas = a.capas or tc.num_hidden_layers
    P = Pesos(a.bf16)
    Rt = Rt_de(D, dev)
    R = Rt.T.contiguous()
    os.makedirs(a.salida, exist_ok=True)
    rot_emb = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    print(f"[cuantizar] {n_capas} capas, {dev}", flush=True)
    for i in range(n_capas):
        dirc = dir_capa(a, i)
        if listo_cuantizada(a, i):
            continue
        esperar(lambda: hay(dirc, "CALIBRADA") or hay(a.trabajo, "ERROR_CALIBRAR"), f"que se calibre la capa {i}")
        if hay(a.trabajo, "ERROR_CALIBRAR") and not hay(dirc, "CALIBRADA"):
            raise SystemExit("[cuantizar] la calibracion fallo")
        esperar_recursos(6, 1.5 + a.margen_disco, a.salida, f"cuantizar capa {i}")
        t0 = time.time()
        tipo = tc.layer_types[i]
        pref = f"model.language_model.layers.{i}."
        sd = {k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else torch.bfloat16)
              for k in P.claves(pref)}
        tens, efect, err_w, muertos = cuantizar_capa(
            i, tipo, sd, lambda e: h_cargar(os.path.join(dirc, f"H_{e}.pt"), dev), dev, Rt, R)
        del sd
        guardar_atomico({k: v.contiguous() for k, v in tens.items()},
                        os.path.join(a.salida, f"capa_{i:02d}.safetensors"), fn=save_file)
        ap = torch.load(os.path.join(dirc, "apartada.pt"))
        e_capa = error_local(m, tc, i, efect, ap["x"], ap["y"], dev, Rt, rot_emb)
        cresta = json.load(open(os.path.join(dirc, "cresta.json")))
        info = {"tipo": tipo, "error_capa": e_capa, "error_pesos": err_w, "canales_muertos": muertos,
                "cresta_sin_y_con_rotacion": cresta}
        json.dump(info, open(os.path.join(dirc, "informe.json"), "w"), indent=1)
        del efect, tens, ap
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
        if not a.conservar_hessianas:
            for f in os.listdir(dirc):
                if f.startswith("H_"):
                    os.remove(os.path.join(dirc, f))
        marca(dirc, "CUANTIZADA")
        nota = f"  canales con g=0: {muertos}" if any(muertos.values()) else ""
        print(f"[cuantizar] capa {i:02d} {tipo:16s} error local {e_capa:.4f}  pesos {min(err_w.values()):.3f}-"
              f"{max(err_w.values()):.3f}  cresta attn_in {cresta['attn_in'][0]}->{cresta['attn_in'][1]} "
              f"mlp_in {cresta['mlp_in'][0]}->{cresta['mlp_in'][1]}  {time.time() - t0:4.0f}s{nota}", flush=True)
    marca(a.trabajo, "CUANTIZACION_TERMINADA")


# ─── etapa: armar ────────────────────────────────────────────────────────────────────────────────
def armar(a):
    """Lo que esta fuera de las capas, el indice y el config. Ver el encabezado."""
    dev = a.dispositivo
    torch.set_grad_enabled(False)
    m, tc, cfg = modelo_hf(a.bf16)
    D = tc.hidden_size
    n = tc.num_hidden_layers
    faltan = [i for i in range(n) if not listo_cuantizada(a, i)]
    if faltan:
        raise SystemExit(f"[armar] faltan capas cuantizadas: {faltan[:10]}")
    esperar_recursos(8, 8 + a.margen_disco, a.salida, "armar")
    P = Pesos(a.bf16)
    Rt = Rt_de(D, dev)
    R = Rt.T.contiguous()
    g_final = 1 + P.get("model.language_model.norm.weight", dev, torch.float32)
    ruta_resto = os.path.join(a.salida, "resto.safetensors")
    if not os.path.exists(ruta_resto):
        resto = {}
        for k in P.mapa:
            if k.startswith("model.language_model.layers.") or k.startswith("mtp."):
                continue                                    # mtp: no se usa y quedaria invalido
            v = P.get(k)
            if k == "model.language_model.embed_tokens.weight":
                v = torch.cat([(c.to(dev, torch.float32) @ Rt).to(v.dtype).cpu() for c in v.split(16384)])
            elif k == "lm_head.weight":
                v = torch.cat([((c.to(dev, torch.float32) * g_final[None, :]) @ Rt).to(v.dtype).cpu()
                               for c in v.split(16384)])
            elif k == "model.language_model.norm.weight":
                v = torch.zeros_like(v)
            elif k == "model.visual.merger.linear_fc2.weight":   # el merger ESCRIBE al residuo
                v = (R @ v.to(dev, torch.float32)).to(v.dtype).cpu()
            elif k == "model.visual.merger.linear_fc2.bias":
                v = (v.to(dev, torch.float32) @ Rt).to(v.dtype).cpu()
            resto[k] = v
        guardar_atomico(resto, ruta_resto, fn=save_file)
        del resto
    mapa, total = {}, 0
    for f in [f"capa_{i:02d}.safetensors" for i in range(n)] + ["resto.safetensors"]:
        with safe_open(os.path.join(a.salida, f), "pt") as fh:
            for k in fh.keys():
                mapa[k] = f
        total += os.path.getsize(os.path.join(a.salida, f))
    json.dump({"metadata": {"total_size": total}, "weight_map": mapa},
              open(os.path.join(a.salida, "model.safetensors.index.json"), "w"), indent=1)

    ignore = []
    for b in range(cfg["vision_config"]["depth"]):
        ignore += [f"model.visual.blocks.{b}.attn.qkv", f"model.visual.blocks.{b}.attn.proj",
                   f"model.visual.blocks.{b}.mlp.linear_fc1", f"model.visual.blocks.{b}.mlp.linear_fc2"]
    ignore += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
    for i, t in enumerate(tc.layer_types):
        if t == "linear_attention":
            p_ = f"model.language_model.layers.{i}.linear_attn"
            ignore += [p_, p_ + ".norm", p_ + ".in_proj_b", p_ + ".in_proj_a"]
    ignore += ["lm_head"]
    qcfg = {"config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 4, "type": "int", "symmetric": True, "group_size": G, "strategy": "group",
                    "block_structure": None, "dynamic": False, "actorder": None, "scale_dtype": None,
                    "zp_dtype": None, "observer": "memoryless_minmax", "observer_kwargs": {}},
        "input_activations": None, "output_activations": None, "format": "pack-quantized"}},
        "quant_method": "compressed-tensors", "kv_cache_scheme": None, "format": "pack-quantized",
        "quantization_status": "compressed", "global_compression_ratio": None, "ignore": ignore}
    cfg["quantization_config"] = qcfg
    cfg["genesis_rotacion"] = {"semilla": SEMILLA, "bloque": BLOQUE, "bloque_down": BLOQUE_DOWN,
                               "requiere": "GENESIS_ENABLE_PN148_ROT_DOWN=1", "g_final": g_final.cpu().tolist()}
    json.dump(cfg, open(os.path.join(a.salida, "config.json"), "w"), indent=2)
    json.dump(qcfg, open(os.path.join(a.salida, "quantization_config.json"), "w"), indent=2)
    json.dump(PROCESSOR_CONFIG, open(os.path.join(a.salida, "processor_config.json"), "w"), indent=2)
    for f in COPIAR:
        if os.path.exists(os.path.join(a.bf16, f)):
            shutil.copy2(os.path.join(a.bf16, f), os.path.join(a.salida, f))
    informe = {i: json.load(open(os.path.join(dir_capa(a, i), "informe.json"))) for i in range(n)}
    json.dump(informe, open(os.path.join(a.salida, "informe_idiotsavant.json"), "w"), indent=1)
    e = [v["error_capa"] for v in informe.values()]
    print(f"[armar] LISTO {a.salida}: {len(mapa)} tensores, {total / 1e9:.1f} GB; error local mediano "
          f"{np.median(e):.4f}", flush=True)


# ─── estado (para humanos, scripts y LLMs) ─────────────────────────────────────────────────────────
def leer_estado(trabajo, salida, n_capas=64):
    """Todo lo que se sabe de una corrida, leyendo SOLO el disco (no molesta al proceso que corre)."""
    capas, t_cuant = [], []
    for i in range(n_capas):
        d = os.path.join(trabajo, f"capa_{i:02d}")
        cal = os.path.exists(os.path.join(d, "CALIBRADA"))
        cua = os.path.exists(os.path.join(d, "CUANTIZADA")) and os.path.exists(os.path.join(salida, f"capa_{i:02d}.safetensors"))
        inf = None
        if os.path.exists(os.path.join(d, "informe.json")):
            try:
                inf = json.load(open(os.path.join(d, "informe.json")))
            except ValueError:
                pass
        if cua:
            t_cuant.append(os.path.getmtime(os.path.join(d, "CUANTIZADA")))
        capas.append({"capa": i, "calibrada": cal or cua, "cuantizada": cua,
                      "tipo": inf["tipo"] if inf else None,
                      "error_capa": inf["error_capa"] if inf else None,
                      "canales_muertos": inf["canales_muertos"] if inf else None})
    vivos = {}
    try:
        for etapa, pid in json.load(open(os.path.join(trabajo, "pids.json"))).items():
            try:      # el PID tiene que ser ESTE script (en otro contenedor el numero se reusa)
                vivos[etapa] = "idiotsavant" in open(f"/proc/{pid}/cmdline").read()
            except OSError:
                vivos[etapa] = False
    except (OSError, ValueError):
        pass
    t_cuant.sort()
    ritmo = (t_cuant[-1] - t_cuant[max(0, len(t_cuant) - 8)]) / max(1, min(7, len(t_cuant) - 1)) if len(t_cuant) > 1 else None
    faltan = sum(1 for c in capas if not c["cuantizada"])
    e = [c["error_capa"] for c in capas if c["error_capa"] is not None]
    fin = {f: os.path.exists(os.path.join(trabajo, f)) for f in
           ("CALIBRACION_TERMINADA", "CUANTIZACION_TERMINADA", "ERROR_CALIBRAR")}
    fin["ARMADO"] = os.path.exists(os.path.join(salida, "model.safetensors.index.json")) and fin["CUANTIZACION_TERMINADA"]
    return {"capas_total": n_capas, "calibradas": sum(c["calibrada"] for c in capas),
            "cuantizadas": n_capas - faltan, "segundos_por_capa": ritmo,
            "eta_minutos": round(faltan * ritmo / 60, 1) if ritmo else None,
            "error_local_mediano": float(np.median(e)) if e else None,
            "procesos_vivos": vivos, "marcas": fin,
            "recursos": {"ram_libre_gb": round(ram_libre_gb(), 1),
                         "disco_trabajo_gb": round(disco_libre_gb(trabajo), 1),
                         "disco_salida_gb": round(disco_libre_gb(salida), 1)},
            "capas": capas}


def estado(a):
    n = a.capas or modelo_hf(a.bf16)[1].num_hidden_layers if a.bf16 else (a.capas or 64)
    e = leer_estado(a.trabajo, a.salida, n)
    if a.json:
        print(json.dumps(e, indent=1))
        return
    print(f"capas calibradas {e['calibradas']}/{n}, cuantizadas {e['cuantizadas']}/{n}; "
          f"{'%.0f s/capa, faltan ~%.0f min' % (e['segundos_por_capa'], e['eta_minutos']) if e['segundos_por_capa'] else 'sin ritmo aun'}; "
          f"error local mediano {e['error_local_mediano'] if e['error_local_mediano'] is None else round(e['error_local_mediano'], 4)}")
    print(f"procesos: {e['procesos_vivos'] or 'ninguno registrado'}; marcas: {[k for k, v in e['marcas'].items() if v]}")
    print(f"recursos: {e['recursos']}")
    for c in e["capas"]:
        if c["cuantizada"]:
            print(f"  capa {c['capa']:02d} {c['tipo']:16s} error {c['error_capa']:.4f}"
                  + (f"  canales con g=0 {c['canales_muertos']}" if any((c['canales_muertos'] or {}).values()) else ""))


# ─── dry-run ─────────────────────────────────────────────────────────────────────────────────────
def ensayo(a):
    """Valida todo lo que la corrida va a necesitar, SIN escribir capas. Codigo 2 si algo falla."""
    dev = a.dispositivo if torch.cuda.is_available() or not a.dispositivo.startswith("cuda") else "cpu"
    torch.set_grad_enabled(False)
    fallas = []

    def chequeo(nombre, fn):
        t0 = time.time()
        try:
            det = fn()
            print(f"[dry-run] OK    {nombre}{': ' + det if det else ''}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            fallas.append(nombre)
            print(f"[dry-run] FALLA {nombre}: {type(ex).__name__}: {ex}", flush=True)

    m, tc, cfg = modelo_hf(a.bf16)
    P = Pesos(a.bf16)
    D = tc.hidden_size

    def c_bf16():
        faltan = []
        for i, t in enumerate(tc.layer_types):
            with torch.device("meta"):
                capa = m.Qwen3_5DecoderLayer(tc, i)
            pref = f"model.language_model.layers.{i}."
            tiene = {k[len(pref):] for k in P.claves(pref)}
            faltan += [pref + k for k in capa.state_dict() if k not in tiene]
            faltan += [pref + b for b in LINEALES[t] if b + ".weight" not in tiene]
        for k in ("model.language_model.embed_tokens.weight", "lm_head.weight", "model.language_model.norm.weight",
                  "model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"):
            if k not in P.mapa:
                faltan.append(k)
        for f in set(P.mapa.values()):
            if not os.path.exists(os.path.join(a.bf16, f)):
                faltan.append(f)
        if faltan:
            raise KeyError(f"{len(faltan)} faltantes, p. ej. {faltan[:4]}")
        return f"{tc.num_hidden_layers} capas ({tc.layer_types.count('full_attention')} de atencion), {len(P.mapa)} tensores"
    chequeo("checkpoint BF16 completo", c_bf16)

    def c_calib():
        ids = np.load(a.calib, mmap_mode="r")
        if ids.ndim != 2 or ids.dtype.kind not in "iu":
            raise ValueError(f"se espera int [N, L], hay {ids.dtype} {ids.shape}")
        mx, mn = int(ids.max()), int(ids.min())
        if mn < 0 or mx >= tc.vocab_size:
            raise ValueError(f"tokens fuera del vocabulario: [{mn}, {mx}] vs {tc.vocab_size}")
        if ids.shape[0] <= a.apartar:
            raise ValueError("no quedan muestras para las Hessianas")
        return f"{ids.shape[0]} x {ids.shape[1]} tokens en [{mn}, {mx}]"
    chequeo("calibracion", c_calib)

    def c_recursos():
        N, L = np.load(a.calib, mmap_mode="r").shape
        N, L = (a.muestras or N), (a.largo or L)
        est = N * L * D * 2 / GB
        det = [f"RAM libre {ram_libre_gb():.1f} GB", f"disco salida {disco_libre_gb(a.salida):.0f} GB",
               f"disco trabajo {disco_libre_gb(a.trabajo):.0f} GB"]
        prob = []
        if ram_libre_gb() < 12:
            prob.append("RAM < 12 GB")
        if disco_libre_gb(a.salida) < 19 + a.margen_disco:
            prob.append(f"disco de salida < {19 + a.margen_disco:.0f} GB (el checkpoint son ~19)")
        ng = torch.cuda.device_count()
        for d in range(ng):
            libre = torch.cuda.mem_get_info(d)[0] / GB
            det.append(f"GPU{d} {libre:.1f} GB libres")
        if ng == 0:
            prob.append("no hay GPU")
        elif torch.cuda.mem_get_info(0)[0] / GB < est + 3.5:
            prob.append(f"GPU0 necesita ~{est + 3.5:.1f} GB para calibrar (estado oculto {est:.1f} GB)")
        if prob:
            raise RuntimeError("; ".join(prob) + " | " + ", ".join(det))
        return ", ".join(det) + f"; {'dos procesos en GPUs separadas' if ng > 1 else 'una GPU compartida'}"
    chequeo("recursos", c_recursos)

    def c_mate():
        Rt = Rt_de(D, dev)
        err_o = float((Rt @ Rt.T - torch.eye(D, device=dev)).abs().max())
        x = torch.randn(3, 8704, device=dev)
        err_b = float((bloques(bloques(x, BLOQUE_DOWN), BLOQUE_DOWN) - x).abs().max())
        W = torch.randn(256, 1024, device=dev)
        X = torch.randn(4096, 1024, device=dev)
        Q, S = gptq(W, X.T @ X)
        p, s2, shp = empaquetar(Q, S)
        sh = torch.arange(0, 32, 4, dtype=torch.int32)
        q2 = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(256, -1).to(torch.int8) - 8
        if not torch.equal(q2, Q.cpu()):
            raise ValueError("el empaquetado no vuelve")
        e = float((dequant(Q, S) - W).norm() / W.norm())
        if err_o > 1e-4 or err_b > 1e-4 or not (0.05 < e < 0.2):
            raise ValueError(f"ortogonalidad {err_o:.1e}, involucion {err_b:.1e}, error int4 {e:.3f}")
        return f"R ortogonal ({err_o:.0e}), Hadamard involutiva, GPTQ int4 {e:.3f}, empaquetado exacto"
    chequeo("matematica (rotaciones, GPTQ, empaquetado)", c_mate)

    def c_capa(i):
        def f():
            tipo = tc.layer_types[i]
            ids = np.load(a.calib, mmap_mode="r")
            L = min(256, ids.shape[1])
            toks = torch.from_numpy(np.array(ids[: 2, -L:]).astype(np.int64)).to(dev)
            with safe_open(os.path.join(a.bf16, P.mapa["model.language_model.embed_tokens.weight"]), "pt") as fh:
                E = fh.get_slice("model.language_model.embed_tokens.weight")
                x = torch.stack([torch.stack([E[int(t):int(t) + 1][0] for t in fila]) for fila in toks.cpu()])
            x = x.to(dev, torch.bfloat16)
            capa, sd = cargar_capa(m, tc, P, i, dev)
            Rt = Rt_de(D, dev)
            acc, ganchos = enganchar(capa, LINEALES[tipo], D, tc.rms_norm_eps, Rt, dev, {"si": True})
            rot_emb = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
            pos = torch.arange(L, device=dev)[None].expand(2, -1)
            cos, sin = rot_emb(x, pos)
            y = capa(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
            y = y[0] if isinstance(y, tuple) else y
            for g_ in ganchos:
                g_.remove()
            tens, efect, err_w, muertos = cuantizar_capa(i, tipo, sd, lambda e: acc[e].H, dev, Rt, Rt.T.contiguous())
            e = error_local(m, tc, i, efect, x, y, dev, Rt, rot_emb)
            if not (e < 0.5):
                raise ValueError(f"error local {e:.3f}: algo esta mal armado")
            return (f"{tipo}, {len(tens)} tensores, pesos int4 {min(err_w.values()):.3f}-{max(err_w.values()):.3f}, "
                    f"error local {e:.4f} (embeddings crudos, 2 x {L} tokens: solo verifica que cierre)")
        return f
    chequeo("capa 0 (GDN) de punta a punta, en memoria", c_capa(0))
    i_att = tc.layer_types.index("full_attention")
    chequeo(f"capa {i_att} (atencion) de punta a punta, en memoria", c_capa(i_att))

    def c_salida():
        for d in (a.salida, a.trabajo):
            prueba = os.path.join(d, ".escritura_ok")
            open(prueba, "w").write("x")
            os.remove(prueba)
        faltan = [f for f in COPIAR if not os.path.exists(os.path.join(a.bf16, f))]
        ya = leer_estado(a.trabajo, a.salida, tc.num_hidden_layers)
        return (f"escribible; archivos del BF16 que no estan (se omiten): {faltan or 'ninguno'}; "
                f"ya hechas: {ya['cuantizadas']} capas (se saltean)")
    chequeo("salida y trabajo", c_salida)
    if fallas:
        print(f"[dry-run] {len(fallas)} FALLAS: {fallas}", flush=True)
        raise SystemExit(2)
    print("[dry-run] TODO OK: la corrida real deberia terminar (~85 min con dos GPUs, ~19 GB de salida)", flush=True)


# ─── etapa: todo ─────────────────────────────────────────────────────────────────────────────────
def todo(a):
    """Chequea recursos y corre calibrar y cuantizar en paralelo (procesos separados), despues armar."""
    import subprocess
    import sys
    m, tc, _ = modelo_hf(a.bf16)
    N, L = np.load(a.calib, mmap_mode="r").shape
    N, L = (a.muestras or N), (a.largo or L)
    est_gb = N * L * tc.hidden_size * 2 / GB
    ngpu = torch.cuda.device_count()
    salida_gb = 19 if not a.capas else 0.3 * a.capas
    print(f"[todo] GPUs {ngpu}; RAM libre {ram_libre_gb():.1f} GB; disco trabajo {disco_libre_gb(a.trabajo):.0f} GB, "
          f"salida {disco_libre_gb(a.salida):.0f} GB; estado oculto {est_gb:.1f} GB", flush=True)
    problemas = []
    if ngpu == 0:
        problemas.append("no hay GPU")
    if ram_libre_gb() < 12:
        problemas.append(f"RAM libre {ram_libre_gb():.1f} GB (< 12: cuantizar ~6 + calibrar ~4 + margen)")
    if disco_libre_gb(a.salida) < salida_gb + a.margen_disco:
        problemas.append(f"disco de salida {disco_libre_gb(a.salida):.0f} GB (< {salida_gb + a.margen_disco:.0f})")
    for d in range(ngpu):
        libre = torch.cuda.mem_get_info(d)[0] / GB
        if libre < 8:
            problemas.append(f"GPU {d} con {libre:.1f} GB libres (hay otro proceso? el servidor vLLM?)")
    if problemas:
        print("[todo] no arranco:\n  " + "\n  ".join(problemas), flush=True)
        raise SystemExit(2)
    base = [sys.executable, os.path.abspath(__file__)]
    comunes = ["--bf16", a.bf16, "--calib", a.calib, "--trabajo", a.trabajo, "--salida", a.salida,
               "--apartar", str(a.apartar), "--micro", str(a.micro), "--capas", str(a.capas),
               "--muestras", str(a.muestras), "--largo", str(a.largo), "--adelanto", str(a.adelanto),
               "--cada", str(a.cada), "--margen_disco", str(a.margen_disco)]
    if a.conservar_hessianas:
        comunes.append("--conservar_hessianas")
    gpu_q = "1" if ngpu > 1 else "0"
    for f in ("ERROR_CALIBRAR",):
        if hay(a.trabajo, f):
            os.remove(os.path.join(a.trabajo, f))
    env_c = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env_q = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_q)
    import threading
    os.makedirs(os.path.join(a.trabajo, "logs"), exist_ok=True)

    def lanzar(etapa, env):
        """Proceso separado; su salida va a la consola Y a trabajo/logs/<etapa>.log."""
        pr = subprocess.Popen(base + [etapa] + comunes, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)
        log = open(os.path.join(a.trabajo, "logs", f"{etapa}.log"), "a")

        def copiar():
            for linea in pr.stdout:
                if "Warning" in linea or "warn(" in linea:
                    continue
                sys.stdout.write(linea)
                sys.stdout.flush()
                log.write(linea)
                log.flush()
        th = threading.Thread(target=copiar, daemon=True)
        th.start()
        return pr, th

    pc, tc_ = lanzar("calibrar", env_c)
    pq, tq_ = lanzar("cuantizar", env_q)
    json.dump({"todo": os.getpid(), "calibrar": pc.pid, "cuantizar": pq.pid},
              open(os.path.join(a.trabajo, "pids.json"), "w"))
    rc = pc.wait()
    if rc != 0:
        marca(a.trabajo, "ERROR_CALIBRAR", str(rc))
    rq = pq.wait()
    tc_.join(5)
    tq_.join(5)
    if rc or rq:
        raise SystemExit(f"[todo] calibrar={rc} cuantizar={rq}: relanzar retoma desde lo hecho")
    if a.capas:
        print("[todo] capas recortadas (prueba): no se arma el checkpoint", flush=True)
        return
    pa, ta_ = lanzar("armar", env_c)
    rc = pa.wait()
    ta_.join(5)
    if rc:
        raise SystemExit(f"[todo] armar={rc}")


def main():
    ap = argparse.ArgumentParser(description="reconstruye qwen3.8_27b_idiotSavant_sm_86 desde el BF16")
    ap.add_argument("etapa", choices=["todo", "calibrar", "cuantizar", "armar", "estado"])
    ap.add_argument("--bf16", help="orcarouter/Qwen3.8-27B-Uncensored (BF16)")
    ap.add_argument("--calib", help=".npy int [N, L] de tokens de trafico real")
    ap.add_argument("--trabajo", required=True, help="cache de trabajo (Hessianas, marcas, estado)")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--apartar", type=int, default=2, help="muestras fuera de las Hessianas (error local)")
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--adelanto", type=int, default=3, help="capas que calibrar puede adelantarse a cuantizar")
    ap.add_argument("--cada", type=int, default=8, help="guardar el estado oculto cada N capas (si el disco da)")
    ap.add_argument("--margen_disco", type=float, default=5.0, help="GB libres que nunca se tocan")
    ap.add_argument("--conservar_hessianas", action="store_true", help="no borrarlas (~50 GB; variantes sin recalibrar)")
    ap.add_argument("--capas", type=int, default=0, help="solo las primeras N capas (prueba)")
    ap.add_argument("--muestras", type=int, default=0, help="solo las primeras N muestras (prueba)")
    ap.add_argument("--largo", type=int, default=0, help="recortar cada muestra (prueba)")
    ap.add_argument("--dispositivo", default="cuda", help="cpu solo para pruebas chicas")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="validar todo (BF16, calibracion, recursos, matematica, dos capas en memoria) sin escribir capas")
    ap.add_argument("--json", action="store_true", help="estado: salida en JSON")
    a = ap.parse_args()
    os.makedirs(a.trabajo, exist_ok=True)
    os.makedirs(a.salida, exist_ok=True)
    if a.etapa == "estado":
        return estado(a)
    if not a.bf16 or not a.calib:
        ap.error("--bf16 y --calib son obligatorios salvo en 'estado'")
    if a.dry_run:
        return ensayo(a)
    {"todo": todo, "calibrar": calibrar, "cuantizar": cuantizar, "armar": armar}[a.etapa](a)


if __name__ == "__main__":
    main()
