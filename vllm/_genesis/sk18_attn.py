# SPDX-License-Identifier: Apache-2.0
"""PN131 — decode de atencion ENTERO (SK-18h, PTX) sobre la KV int8_per_token_head.

Que hace
--------
Con ``--kv-cache-dtype int8_per_token_head`` y el backend TRITON_ATTN, reemplaza:

* ``do_kv_cache_update``: escribe la KV con un layout PROPIO dentro de la misma
  reserva de vLLM (520 B por token-cabeza, se usan 516):
      K int8 [BS][NH][256] | V int8 [NH][256][BS] | escalas int16 [BS][NH][2]
  V va por dimension porque el mma de w.v la lee asi sin transponer.
* ``forward``: los pasos solo-decode (<= 5 tokens por pedido: MTP K=3 son 4) van por
  SK-18h (Q.K int8, softmax entero en streaming, w.v int8) + union, todo PTX entero.
  Los pasos con prefill decuantizan los bloques que tocan a fp16 y llaman al kernel
  Triton de vLLM sin cuantizar.

CUDA graph
----------
El camino de decode uniforme (lo unico que vLLM corre en FULL graph) no tiene ni una
sincronizacion ni una decision que dependa de datos: buffers estaticos compartidos
por todas las capas (una sola reserva del tamano maximo, vistas por lote), grilla con
la cantidad MAXIMA de paginas (las paginas sin tokens salen en el acto), referencias de
escala como tensores de GPU. Los lanzamientos PTX son cuLaunchKernel, igual que Triton,
y quedan grabados en el grafo.

Escalas
-------
skf/svf son int16 Q15 relativos a una referencia por capa (potencia de 2, 64x el
maximo de la primera escritura real, fijada en GPU sin sincronizar). K entra al kernel
con multiplicacion de 64 bits (sin perdida) y V con desplazamiento VSH=11 (svf <= 2048:
hasta 4x el maximo inicial; por encima satura).

Frontera flotante: cuantizar q/k/v de entrada (fp16 -> int8) y la division final
O * escala / S -> fp16. Todo lo del medio es entero.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch

log = logging.getLogger("genesis.pn131")

QD = 256
# Filas por (secuencia, cabeza KV): multiplo de 32 (BQ del kernel) que entre los L*G queries
# del decode. Con MTP K=3 y 6 cabezas Q por KV son 24; K=4 -> 30; K=5 -> 36, y ahi hace falta 64
# (dos bloques por secuencia-cabeza, la grilla sale sola de R = gridDim.y * BQ).
def _max_tok_decode() -> int:
    """Tokens por pedido que el camino de decode entero se banca: K+1 del MTP.

    Sale de la config de vLLM, NO de una constante. Cuando estaba clavado en 5, subir el MTP a
    K>=5 hacia que el decode se pasara del tope y cayera al camino de PREFILL, que planifica
    FlashInfer con tensores de CPU => "Cannot copy between CPU and CUDA tensors during CUDA graph
    capture" y el server no arrancaba. El sintoma no se parecia en nada a la causa.
    """
    global _MTD
    if _MTD is not None:
        return _MTD
    v = os.environ.get("GENESIS_PN131_MAXTOK")
    if v:
        _MTD = int(v)
        return _MTD
    try:
        from vllm.config import get_current_vllm_config
        spec = get_current_vllm_config().speculative_config
        if spec is not None and spec.num_speculative_tokens:
            _MTD = int(spec.num_speculative_tokens) + 1
            log.info("PN131: tope del decode entero = %d tokens por pedido (MTP K=%d)",
                     _MTD, _MTD - 1)
            return _MTD
    except Exception:
        pass
    # La config global puede no tener todavia el speculative_config segun cuando se pregunte, asi
    # que la segunda fuente es la linea de comandos, que siempre esta.
    try:
        import re as _r
        import sys as _s
        linea = " ".join(_s.argv)
        m = _r.search(r'"num_speculative_tokens"\s*:\s*(\d+)', linea)
        if m:
            _MTD = int(m.group(1)) + 1
            log.info("PN131: tope del decode entero = %d tokens por pedido (MTP K=%d, de argv)",
                     _MTD, _MTD - 1)
            return _MTD
    except Exception:
        pass
    return 5          # sin cachear: puede ser que la config todavia no exista


_MTD = None
MB = 32
ZSH = 13
ZSH4 = int(os.environ.get("GENESIS_PN131_ZSH4", 16))   # int4: z = (sum_g acc_g*rq_g*rk_g*kmax) >> ZSH4
WCAP = int(os.environ.get("GENESIS_PN131_WCAP", 1911))  # tope de wp: 3 planos de nibbles
QPLANOS = int(os.environ.get("GENESIS_PN131_QPLANOS", 2))  # 2 = q int8 en dos planos de nibbles
PLANOS_A = int(os.environ.get("GENESIS_PN131_PLANOSA", 1 if QPLANOS == 2 else QPLANOS))
ESPERA4 = int(os.environ.get("GENESIS_PN131_ESPERA", 1))
# Ventana reciente en int8 (camino hibrido): PAGS paginas por secuencia espejadas en un pool
# aparte. La atencion se concentra ahi (39-49% de la masa en las ultimas 832 posiciones), y
# dejarla exacta baja el error de 6,5% a 4,0% (capa 35) y de 5,6% a 0,9% (capa 3).
VENT = int(os.environ.get("GENESIS_PN131_VENTANA", 2))     # paginas espejadas (0 = sin espejo)
SCH = int(os.environ.get("GENESIS_PN131_SCH", 8))           # trozos por pagina espejada
ESC8 = 3               # el espejo usa refs 2^ESC8 mas chicas (mas precision de escala)   # grupos cp.async pendientes (0 = esperar todo)  # planos en la pasada del maximo
NIV4 = int(os.environ.get("GENESIS_PN131_NIV", 119 if QPLANOS == 2 else 7))
# escala del logit int4: q = mx/NIV4 (una escala por fila), k = KM*r/16 * 2^ek/32767, 16 = sqrt(256)
MQ4 = round(2 ** 56 / (NIV4 * 32767 * 16 * 16 * math.log(2)))
CLIP4 = int(os.environ.get("GENESIS_PN131_CLIP", 243))   # recorte de la escala, sobre 256
MARGEN_K4 = float(os.environ.get("GENESIS_PN131_MARGENK4", 8))   # int4: KM de 12 bits
MARGEN4 = float(os.environ.get("GENESIS_PN131_MARGEN4", 16))    # int4: svf <= 2^VSH
VSH = 11
MARGEN = 64.0          # V: svf <= 2^VSH -> hasta 4x el maximo inicial
MARGEN_K = 16.0        # K: skf con ~11 bits; satura recien a 16x (k rotada/normalizada no crece tanto)
CPG = int(os.environ.get("GENESIS_PN131_CPG", 16))
LN2 = math.log(2)
SH_H = 32 * (256 + 128) + 2 * 64 * 256 + 2 * 256 * 64 + 2 * 64 * 2 * 4
# Warps por bloque de batch2 (int8): con 8, un bloque toma las 64 filas de una (secuencia, cabeza KV)
# (9 tokens del arbol x 6 cabezas Q = 54) y carga K/V de la pagina UNA vez para los 8 warps; con 4
# eran 2 bloques que cargaban lo mismo. Bit a bit igual (misma cuenta por fila). 28-09.
NW8 = int(os.environ.get("GENESIS_SK18H_NW", "8"))


def sh_h(nw: int) -> int:
    return 8 * nw * (256 + 128) + 2 * 64 * 256 + 2 * 256 * 64 + 2 * 64 * 2 * 4
SH_I = 32 * QPLANOS * 128 + 2 * 128 * 128 + 2 * 256 * 64 + 2 * 32 * 128 + 2 * 128 * 2 * 8
_QA_QB = None
_k = {}
# Rotacion Hadamard de q/k: "ptx" = entera dentro de prep/escribir (reemplaza PN126 en capas PN131)
ROT_PTX = os.environ.get("GENESIS_PN131_ROT", "ptx") == "ptx"
# Verificacion de un ARBOL de borrador (ver vllm._genesis.arbol_borrador): la query ve el contexto
# y, entre los tokens nuevos, solo a sus ancestros. Solo el camino int8 con rotacion PTX. Apagado
# el texto de los kernels queda identico al de siempre.
ARBOL = os.environ.get("GENESIS_ENABLE_ARBOL", "0") == "1"
_signos = {}


def _signos_dev(dev):
    s = _signos.get(dev.index)
    if s is None:
        g = torch.Generator(device=dev).manual_seed(126)            # mismos signos que PN126
        s = (torch.randint(0, 2, (QD,), generator=g, device=dev).to(torch.int32) * 2 - 1).contiguous()
        _signos[dev.index] = s
    return s


def _rotar_q_prefill(query):
    """Prefill (eager): rotacion Hadamard ENTERA (FWHT), igual que el decode en PTX."""
    from vllm._genesis import rot_qk
    q_rot = query.clone()
    rot_qk.rotar_tensor(q_rot, QD)
    return q_rot


_H = {}


def _hadamard(dev, dtype):
    k = (dev.index, dtype)
    if k not in _H:
        H = torch.ones(1, 1, device=dev, dtype=torch.float32)
        while H.shape[0] < QD:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H[k] = H.to(dtype)
    return _H[k]


def rot_ptx_activa() -> bool:
    return _habilitado() and ROT_PTX


# kv-cache-dtype -> camino PN131 (se elige con --kv-cache-dtype, nada mas)
_MODOS = {"int8_per_token_head": "int8", "int4_per_token_head": "int4"}


def _habilitado() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN131_SK18", "0") == "1"


def _int4_listo() -> bool:
    return os.environ.get("GENESIS_PN131_INT4", "1") == "1" and _INT4_IMPL


_INT4_IMPL = True    # camino int4 completo (SK-18i)


def modo(impl) -> str:
    """"int8" o "int4" segun --kv-cache-dtype (KVQuantMode del impl)."""
    from vllm.v1.kv_cache_interface import KVQuantMode
    return "int4" if getattr(impl, "_kv_quant_mode", None) == KVQuantMode.INT4_PER_TOKEN_HEAD else "int8"


def dtype_activo(cache_dtype: str) -> bool:
    m = _MODOS.get(str(cache_dtype))
    return _habilitado() and (m == "int8" or (m == "int4" and _int4_listo()))


def activo(impl, layer=None) -> bool:
    if not _habilitado():
        return False
    excl = os.environ.get("GENESIS_PN131_EXCLUIR", "")
    if excl and layer is not None and any(x and x in getattr(layer, "layer_name", "") for x in excl.split(",")):
        return False
    from vllm.v1.kv_cache_interface import KVQuantMode
    modo = getattr(impl, "_kv_quant_mode", None)
    ok_modo = modo == KVQuantMode.INT8_PER_TOKEN_HEAD or (modo == KVQuantMode.INT4_PER_TOKEN_HEAD and _int4_listo())
    return (ok_modo
            and impl.head_size == QD and impl.alibi_slopes is None and impl.sinks is None
            and tuple(impl.sliding_window) in ((-1, -1), (None, None)) and not impl.logits_soft_cap)


def _coef():
    global _QA_QB
    if _QA_QB is None:
        x = np.linspace(0, 1, 2000)
        bs = np.linspace(0.1, 0.25, 3001)
        err = [np.abs((1 - (0.5 + b) * x + b * x * x) / 2 ** (-x) - 1).max() for b in bs]
        b = bs[int(np.argmin(err))]
        _QA_QB = (round((0.5 + b) * 32768), round(b * 32768))
    return _QA_QB


def _kernels(md="int8"):
    dev = (torch.cuda.current_device(), md)
    if dev not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        qa, qb = _coef()
        defs = [f"-DQA={qa}", f"-DQB={qb}"]
        if md == "int4":
            d4 = [f"-DWCAP={WCAP}", f"-DQPLANOS={QPLANOS}", f"-DPLANOS_A={PLANOS_A}", f"-DESPERA={ESPERA4}", f"-DSCH={SCH}", f"-DDIAG={int(os.environ.get('GENESIS_PN131_DIAG4', 0))}"]
            ks = dict(main=Kernel("sk18i_batch.cu", "sk18i_batch", defs=defs + d4, warps=4),
                      escribir=Kernel("sk18i_escribir.cu", "sk18i_escribir", defs=[f"-DCLIP={CLIP4}"], warps=1),
                      prep=Kernel("sk18i_prep.cu", "sk18i_prep", defs=[f"-DMQ4={MQ4}", f"-DQPLANOS={QPLANOS}", f"-DNIV={NIV4}"], warps=1),
                      union=Kernel("sk18h_union4.cu", "sk18h_union4", defs=defs + ["-DO32=1"], warps=1),
                      decuant=Kernel("sk18i_decuant.cu", "sk18i_decuant", warps=1),
                      salida=Kernel("sk18i_salida.cu", "sk18i_salida", warps=1))
            if VENT > 0:     # espejo int8 de la ventana reciente
                ks["lado"] = Kernel("sk18i_lado.cu", "sk18i_lado", warps=4)
                ks["escribir8"] = Kernel("sk18h_escribir2.cu", "sk18h_escribir2", defs=["-DROTV=1"], warps=1)
                ks["decuant8"] = Kernel("sk18h_decuant.cu", "sk18h_decuant", defs=["-DROTV=1"], warps=1)
                ks["prep8"] = Kernel("sk18h_prep2.cu", "sk18h_prep2", warps=1)
                ks["espejo"] = Kernel("sk18h_batch2.cu", "sk18h_batch2",
                                      defs=defs + [f"-DHIB=1", f"-DESC8={ESC8}", f"-DWCAPW={WCAP}", f"-DSCH={SCH}"], warps=4)
            for x in ks.values():
                x.cargar()
            _k[dev] = ks
            return ks
        arb = ["-DARBOL=1"] if (ARBOL and ROT_PTX) else []
        ks = dict(main=Kernel("sk18h_batch2.cu", "sk18h_batch2", defs=defs + arb, warps=4),
                  main8=Kernel("sk18h_batch2.cu", "sk18h_batch2", defs=defs + arb + [f"-DNWARPS={NW8}"], warps=NW8),
                  escribir=Kernel("sk18h_escribir2.cu" if ROT_PTX else "sk18h_escribir.cu",
                                  "sk18h_escribir2" if ROT_PTX else "sk18h_escribir", warps=1),
                  prep=Kernel("sk18h_prep2.cu" if ROT_PTX else "sk18h_prep.cu",
                              "sk18h_prep2" if ROT_PTX else "sk18h_prep", defs=arb, warps=1),
                  union=Kernel("sk18h_union4.cu", "sk18h_union4", defs=defs, warps=1),
                  decuant=Kernel("sk18h_decuant.cu", "sk18h_decuant", warps=1),
                  salida=Kernel("sk18h_salida.cu", "sk18h_salida", warps=1))
        for x in ks.values():
            x.cargar()
        _k[dev] = ks
    return _k[dev]


# ─────────────────────────── estado por capa (GPU) ─────────────────────────
class _Capa:
    """refs = [ek, ev] int32 en GPU: referencias 2^ek / 2^ev (las leen los kernels).
    Se fijan en la primera escritura eager con slots validos (siempre un prefill)."""
    __slots__ = ("refs", "refs8", "fija", "lado")

    def __init__(self, dev):
        self.refs = torch.zeros(2, dtype=torch.int32, device=dev)
        self.refs8 = torch.zeros(2, dtype=torch.int32, device=dev)   # espejo int8 (refs - ESC8)
        self.fija = False
        self.lado = None          # pool espejo int8 de la ventana reciente, PROPIO de esta capa


_capas: dict[int, _Capa] = {}


def _capa(impl, dev) -> _Capa:
    c = _capas.get(id(impl))
    if c is None:
        c = _capas[id(impl)] = _Capa(dev)
    return c


def _lado(c, dev, bmax, nh, bs):
    """Pool espejo de la capa (una sola reserva; 1,7 MB por secuencia y pagina).
    OJO: reservar esto DENTRO de la captura de un CUDA graph da memoria del pool privado de la
    captura, que no vale en el replay. Se reserva siempre en modo eager."""
    if c.lado is None and torch.cuda.is_current_stream_capturing():
        return None
    if c.lado is None:
        c.lado = torch.zeros(bmax * VENT * bs * nh * 520, dtype=torch.int8, device=dev)
        log.warning("PN131 espejo int8: %d paginas de %d tokens (%.1f MiB por capa)",
                    bmax * VENT, bs, c.lado.numel() / 2**20)
    return c.lado


# ─────────────────────── buffers estaticos compartidos ─────────────────────
class _Bufs:
    def __init__(self, dev, bmax, nchmax, nh, bs=832, G=6):
        self.bmax, self.nchmax, self.nh, self.bs = bmax, nchmax, nh, bs
        self.MB = ((_max_tok_decode() * G + 31) // 32) * 32
        MB = self.MB
        R = bmax * nh * MB
        NG = (nchmax + CPG - 1) // CPG
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=dev)
        self.Q = z(bmax * nh * MB * QD, dt=torch.int8)
        self.rq = z(R * 4, dt=torch.int32)
        self.sq = z(R * 2 * 4, dt=torch.int32)        # sumas de q por grupo (int4, hasta 2 planos)
        self.oc = z((nchmax + (VENT * SCH if VENT > 0 else 0)) * R, dt=torch.int32)       # correccion del cero de V (int4)
        if VENT > 0:                                  # espejo int8 de la ventana reciente
            self.dueno = torch.full((bmax * VENT,), -1, dtype=torch.int32, device=dev)
            self.slot2 = z(16384, dt=torch.int64)
            self.Q8 = z(R * QD, dt=torch.int8)
            self.mqb8 = z(R, dt=torch.int32)
            self.dcap8 = z(R, dt=torch.int32)
            self.lim8 = z(R, dt=torch.int32)
        self.lim = z(R, dt=torch.int32)
        if ARBOL:
            self.abase = z(R, dt=torch.int32)
            self.amask = z(R, dt=torch.int32)
            # mascara de la CADENA de siempre, por L: el token j ve a los j-1 anteriores y a si mismo
            self.anc_cadena = {}
        self.mqb = z(R, dt=torch.int32)
        self.dcap = z(R, dt=torch.int32)
        self.nx = VENT * SCH if VENT > 0 else 0       # ranuras extra para los trozos del espejo
        nchx = nchmax + self.nx
        self.oh = z(nchx * R * QD, dt=torch.int32)
        self.ol = z(nchx * R * QD, dt=torch.int32)
        self.om = z(nchx * R, dt=torch.int32)
        self.os = z(nchx * R, dt=torch.int32)
        self.Og = z(NG * R * QD, dt=torch.int64)
        self.Sg = z(NG * R, dt=torch.int64)
        self.ar = torch.arange(_max_tok_decode(), device=dev, dtype=torch.int32)
        mb = (self.oh.numel() + self.ol.numel()) * 4 / 2**20
        log.info("PN131 buffers: bmax=%d nchmax=%d MB=%d (%.0f MiB de acumuladores)", bmax, nchmax, MB, mb)


_bufs: dict[int, _Bufs] = {}


def _get_bufs(dev, nh, bs, G=6):
    b = _bufs.get(dev.index)
    if b is None:
        try:
            from vllm.config import get_current_vllm_config
            cfg = get_current_vllm_config()
            maxlen = int(cfg.model_config.max_model_len)
            bmax = int(cfg.scheduler_config.max_num_seqs)
        except Exception as e:
            # El contexto de configuracion de vLLM no siempre esta activo cuando se arman los
            # buffers (esto corre en la primera inferencia del worker). Caer a un 10 en SILENCIO
            # hacia que el servidor se muriera recien al llegar el lote 11:
            #   RuntimeError: PN131: lote 12 > GENESIS_PN131_BMAX 10
            # Con --max-num-seqs 24 eso pasa apenas hay carga. Ahora avisa y se puede fijar.
            maxlen, bmax = 262144, 10
            log.warning(
                "PN131: no se pudo leer la config de vLLM (%s); usando bmax=%d. Si "
                "--max-num-seqs es mayor, fijar GENESIS_PN131_BMAX o el lote lo va a tirar.",
                type(e).__name__, bmax)
        bmax = int(os.environ.get("GENESIS_PN131_BMAX", bmax))
        nchmax = (maxlen + bs - 1) // bs
        b = _bufs[dev.index] = _Bufs(dev, bmax, nchmax, nh, bs, G)
        if ARBOL:
            # La mascara tiene que existir ANTES de capturar (un arange adentro del grafo la
            # repondria a cadena en cada replay): se crea ya, para el largo del decode spec.
            try:
                from vllm._genesis import arbol_runner
                if arbol_runner.E.listo:          # lo fija el borrador al construirse, antes del KV
                    mascara_arbol(dev, arbol_runner.E.T, b)
                else:
                    log.warning("PN131/ARBOL: el borrador todavia no fijo K; la mascara se crea "
                                "en el primer decode (tiene que ser antes de capturar)")
            except Exception as e:
                log.warning("PN131/ARBOL: no pude precrear la mascara (%s)", type(e).__name__)
    return b


def mascara_arbol(dev, L, bf=None, capturando=False):
    """Bits de ancestros por fila del decode uniforme, ``[bmax * L]`` int32, UN buffer estatico
    por largo de query (los grafos CUDA hornean la direccion: por eso no viaja en el metadata).

    En reposo tiene la mascara de cadena, ``(1 << t) - 1``: la causalidad de siempre, bit a
    bit. ``arbol_runner`` escribe el arbol de cada pedido antes del forward del target y la
    repone despues, porque el borrador comparte estos buffers y el no verifica ningun arbol."""
    if bf is None:
        bf = _bufs.get(dev.index)
        if bf is None:
            return None
    anc = bf.anc_cadena.get(L)
    if anc is None:
        if capturando:
            raise RuntimeError("PN131/ARBOL: la mascara tiene que existir antes de capturar")
        anc = ((1 << torch.arange(L, device=dev, dtype=torch.int32)) - 1).repeat(bf.bmax).contiguous()
        bf.anc_cadena[L] = anc
    return anc


def geom_bloque(kv_cache):
    """(NH, BS, BLK) del KV de PN131, para los kernels que mueven tokens entre slots. El bloque es
    ``K [BS][NH][256] | V [NH][256][BS] | escalas int16 [BS][NH][2]``: V esta TRASPUESTA, asi que
    un token no es contiguo y copiar ``kv[bloque, :, token]`` moveria bytes de otros tokens."""
    _nb, nh, bs, blk, _raw = _geom(kv_cache)
    return nh, bs, blk


_geom_avisado = False


def _geom(kv_cache):
    # La forma esperada es (bloques, cabezas, tokens_por_bloque, contenido), que es lo que
    # daba `TritonAttentionBackend.get_kv_cache_shape` — "K y V empaquetados en el eje de
    # contenido: logico (B, H, N, 2*hs)". En vLLM v0.29.0 ese metodo ya no existe: la forma
    # la decide un layout central (LBNHC / LBHNC / BLHNC / ...), y si el orden de ejes o el
    # rango cambian, desempaquetar a ciegas escribe el KV en offsets equivocados y el modelo
    # genera basura SIN fallar. Por eso se anuncia una vez y se verifica el rango.
    global _geom_avisado
    if not _geom_avisado:
        _geom_avisado = True
        log.info("[PN131] forma del KV: %s  strides=%s  dtype=%s",
                 tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype)
    if kv_cache.dim() != 4:
        raise RuntimeError(
            f"[PN131] el KV vino con {kv_cache.dim()} ejes {tuple(kv_cache.shape)}; se "
            "esperaban 4 (bloques, cabezas, tokens, contenido). El layout de KV cambio: "
            "hay que revisar `escribir()` y el indexado del kernel antes de seguir."
        )
    nb, nh, bs, cont = kv_cache.shape
    blk = kv_cache.stride(0) * kv_cache.element_size()
    raw = torch.as_strided(kv_cache, (nb, blk), (blk, 1))
    return nb, nh, bs, blk, raw


def _filas(x):
    """[T, H, 256] con cada fila contigua (strides (s0, 256, 1)): se usa sin copiar."""
    if x.dim() == 3 and x.stride(2) == 1 and x.stride(1) == QD:
        return x
    return x.contiguous()


def _cuant(x):
    """fp [..., 256] -> int8, escala float32 [...]."""
    s = x.abs().amax(-1).float().clamp_min(1e-8) / 127.0
    q = torch.round(x.float() / s[..., None]).clamp_(-127, 127).to(torch.int8)
    return q, s


# ─────────────────────────────── escritura ────────────────────────────────
def escribir(impl, layer, key, value, kv_cache, slot_mapping):
    """Un lanzamiento PTX entero (sk18h_escribir). Apta para CUDA graph."""
    if kv_cache.numel() == 0:
        return
    n = slot_mapping.shape[0]
    if n == 0:
        return
    dev = key.device
    nb, nh, bs, blk, raw = _geom(kv_cache)
    md = modo(impl)
    niv = 7.0 if md == "int4" else 127.0
    c = _capa(impl, dev)
    if not c.fija and not torch.cuda.is_current_stream_capturing():
        ok = slot_mapping >= 0
        if bool(ok.any()):
            kk_ = key[:n].float()
            vv_ = value[:n].float()
            if ROT_PTX:   # la referencia de K es la de k ROTADA (el max baja ~2x con Hadamard)
                kk_ = _rotar_q_prefill(kk_.view(n, -1, QD).to(torch.float32))
            if md == "int4":   # en int4 tambien la V va rotada
                vv_ = _rotar_q_prefill(vv_.view(n, -1, QD).to(torch.float32))
            sk = float(kk_.abs().amax(-1)[ok].max()) / niv
            sv = float(vv_.abs().amax(-1)[ok].max()) / niv
            # Las pasadas de calentamiento escriben ceros en slots "validos": no fijar la
            # referencia con eso (la MTP quedaba en 2^-20 y toda su atencion saturaba).
            if sk > 1e-3 and sv > 1e-3:
                ek = math.ceil(math.log2(sk * (MARGEN_K4 if md == "int4" else MARGEN_K)))
                ev = math.ceil(math.log2(sv * (MARGEN4 if md == "int4" else MARGEN)))
                c.refs.copy_(torch.tensor([ek, ev], dtype=torch.int32))
                c.refs8.copy_(torch.tensor([ek - ESC8, ev - ESC8], dtype=torch.int32))
                c.fija = True
                log.warning("PN131 %s: ek=%d ev=%d", getattr(layer, "layer_name", "?"), ek, ev)
    ks = _kernels(md)
    k16 = _filas(key[:n]).view(torch.int16)
    v16 = _filas(value[:n]).view(torch.int16)
    slot = slot_mapping if slot_mapping.dtype == torch.int64 else slot_mapping.to(torch.int64)
    if md == "int4":
        # dos tokens vecinos comparten byte en V: un lanzamiento por paridad de slot
        for par in (0, 1):
            ks["escribir"].lanzar((n, nh), [k16, v16, slot, raw, c.refs, _signos_dev(dev),
                                            nh, bs, blk, VSH, k16.stride(0), v16.stride(0), par])
        if VENT > 0:
            _espejar(impl, layer, ks, k16, v16, slot, n, dev, nh, bs, c)
    elif ROT_PTX:
        ks["escribir"].lanzar((n, nh), [k16, v16, slot, raw, c.refs, _signos_dev(dev), nh, bs, blk, VSH, k16.stride(0), v16.stride(0)])
    else:
        ks["escribir"].lanzar((n, nh), [k16, v16, slot, raw, c.refs, nh, bs, blk, VSH, k16.stride(0), v16.stride(0)])


_md_forzada = None      # solo para los tests offline, que llaman escribir() sin contexto


def _md_actual(impl, layer=None):
    """Metadata del paso desde el contexto de forward (do_kv_cache_update no la recibe)."""
    if _md_forzada is not None:
        return _md_forzada
    try:
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
    except Exception:
        return None
    if isinstance(md, dict):
        md = md.get(getattr(layer, "layer_name", None)) or next(iter(md.values()), None)
    return md if md is not None and getattr(md, "seq_lens", None) is not None else None


def _espejar(impl, layer, ks, k16, v16, slot, n, dev, nh, bs, c):
    """Copia los tokens del paso al pool int8 de la ventana reciente (ranura b*VENT + p%VENT)."""
    md = _md_actual(impl, layer)
    if md is None:
        return
    lado = _lado(c, dev, _get_bufs(dev, nh, bs, max(1, impl.num_heads // nh)).bmax, nh, bs)
    if lado is None:
        return
    bf = _get_bufs(dev, nh, bs, max(1, impl.num_heads // nh))
    B = int(md.query_start_loc.shape[0]) - 1
    if B > bf.bmax:
        return
    slot2 = bf.slot2[:n]
    blk8 = bs * nh * 520
    ks["lado"].lanzar(((n + 127) // 128, 1), [slot, md.query_start_loc, md.seq_lens, slot2,
                                              bf.dueno, n, B, bs, VENT])
    ks["escribir8"].lanzar((n, nh), [k16, v16, slot2, lado, c.refs8, _signos_dev(dev),
                                     nh, bs, blk8, VSH, k16.stride(0), v16.stride(0)])


# ─────────────────────────────── forward ──────────────────────────────────
_aviso_decode = [False, False]


def forward(impl, layer, query, kv_cache, md, output):
    # Un aviso por camino, la primera vez. Sirve para no tener que deducir de la velocidad si
    # SK-18 esta corriendo de verdad: con el enganche por anclas o por subclase registrada, lo
    # que importa es que ESTA linea aparezca.
    qsl = getattr(md, "genesis_qsl_cpu", None)
    if not _aviso_decode[0]:
        _aviso_decode[0] = True
        log.warning("[PN131] decode entero ACTIVO (qsl_cpu %s)",
                    "presente" if qsl is not None else "AUSENTE — cae al camino generico")
    if qsl is not None:
        qsl = qsl.numpy() if hasattr(qsl, "numpy") else np.asarray(qsl)
        nreq = len(qsl) - 1
        qlen = np.diff(qsl)[:nreq]
        if nreq > 0 and qlen.max() <= _max_tok_decode() and qlen.min() >= 1:
            L = int(qlen.max())
            if bool((qlen == L).all()):
                capturando = torch.cuda.is_current_stream_capturing()
                _decode_uniforme(impl, query, kv_cache, md, output, nreq, L, capturando)
                _diag(impl, layer, query, kv_cache, md, output, nreq, L, capturando)
                return output
        if _MIXTO and nreq > 1 and _mixto(impl, layer, query, kv_cache, md, output, qsl, qlen, nreq):
            return output
    return _prefill_decuant(impl, layer, query, kv_cache, md, output)


# ─────────────────────── pasos mixtos: decode por SK-18h, prefill aparte ───────────────────────
# En un paso con prefill, el lote entero iba por _prefill_decuant: descuantizar a fp16 las paginas de
# TODOS los pedidos (tambien de los que solo decodifican, con todo su contexto) + FlashInfer para todos.
# Medido en is25_decode4 (27-09): sk18h_decuant 0,6-1,3 ms por capa de atencion, 10-20 ms por paso mixto.
# vLLM ordena el lote con los decodes primero: el prefijo uniforme (mismos L tokens por pedido) va por
# el decode entero SK-18h, sin descuantizar, y solo el resto por descuantizar + FlashInfer (con SUS paginas).
_MIXTO = os.environ.get("GENESIS_PN131_MIXTO", "0").strip().lower() in ("1", "true", "yes", "on")
_MIXTO_VERIF = int(os.environ.get("GENESIS_PN131_MIXTO_VERIFICAR", "0") or 0)
_mixto_n = {"n": 0, "aviso": False}


class _Sub:
    """Vista de la metadata de atencion para un tramo de pedidos; lo demas pasa al original."""

    def __init__(self, md, **kw):
        self._md = md
        self.__dict__.update(kw)

    def __getattr__(self, n):
        return getattr(self._md, n)


def _mixto(impl, layer, query, kv_cache, md, output, qsl, qlen, nreq) -> bool:
    L = int(qlen[0])
    if L < 1 or L > _max_tok_decode():
        return False
    nd = 0
    while nd < nreq and int(qlen[nd]) == L:
        nd += 1
    if nd == 0 or nd == nreq:
        return False
    nt = nd * L
    q0 = int(qsl[nd])
    if q0 != nt:
        return False
    nact = int(md.num_actual_tokens)
    mdd = _Sub(md, seq_lens=md.seq_lens[:nd], block_table=md.block_table[:nd])
    qp = qsl[nd:nreq + 1] - q0
    mdp = _Sub(md, seq_lens=md.seq_lens[nd:nreq], block_table=md.block_table[nd:nreq],
               query_start_loc=md.query_start_loc[nd:nreq + 1] - q0,
               genesis_qsl_cpu=torch.from_numpy(np.ascontiguousarray(qp)),
               num_actual_tokens=nact - nt, genesis_clave=(id(md), "mixto", nd))
    _decode_uniforme(impl, query, kv_cache, mdd, output, nd, L, False)
    _prefill_decuant(impl, layer, query[nt:], kv_cache, mdp, output[nt:])
    if not _mixto_n["aviso"]:
        _mixto_n["aviso"] = True
        log.warning("[PN131 mixto] paso mixto partido: %d pedidos por SK-18h (L=%d) + %d por prefill", nd, L, nreq - nd)
    # solo pasos reales (el calentamiento de vLLM trae lotes de relleno que dan diferencia 0 exacta)
    if _MIXTO_VERIF and _mixto_n["n"] < _MIXTO_VERIF and int(md.max_seq_len) >= 870:
        _mixto_n["n"] += 1
        ref = torch.zeros_like(output)
        _prefill_decuant(impl, layer, query, kv_cache, md, ref)
        def rel(a, b):
            a, b = a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1)
            return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).max().item()
        log.warning("[PN131 mixto] verif %d %s: decode (%d pedidos x %d) dif_rel_max=%.2e | prefill (%d tok) dif_rel_max=%.2e",
                    _mixto_n["n"], getattr(layer, "layer_name", "?"), nd, L, rel(output[:nt], ref[:nt]),
                    nact - nt, rel(output[nt:nact], ref[nt:nact]))
    return True


_DIAG = os.environ.get("GENESIS_PN131_DIAG", "layers.3.self_attn.attn")
_diag_n = {}


def _diag(impl, layer, query, kv_cache, md, output, B, L, capturando):
    """Compara el decode SK-18h contra decuantizado + Triton en capas que matcheen."""
    nombre = getattr(layer, "layer_name", "")
    if not _DIAG or capturando or _DIAG not in nombre:
        return
    c_ = _capa(impl, query.device)
    if not c_.fija or int(md.seq_lens[:B].max()) < 870:   # nada de pasos de calentamiento
        return
    k = _diag_n.get(nombre, 0)
    if k >= 40:
        return
    _diag_n[nombre] = k + 1
    nt = B * L
    ref = torch.zeros_like(output)
    _prefill_decuant(impl, layer, query, kv_cache, md, ref)
    a = output[:nt].float().view(nt, -1)
    r = ref[:nt].float().view(nt, -1)
    dif = ((a - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-6))
    log.warning("PN131 DIAG %s B=%d L=%d seq=%s dif_rel=%s refs=%s", nombre, B, L,
                md.seq_lens[:B].tolist(), [round(x, 4) for x in dif.tolist()[:8]], _capa(impl, query.device).refs.tolist())


def _decode_uniforme(impl, query, kv_cache, md, output, B, L, capturando):
    mo = modo(impl)
    ks = _kernels(mo)
    dev = query.device
    nb, nh, bs, blk, raw = _geom(kv_cache)
    G = impl.num_heads // nh
    bf = _get_bufs(dev, nh, bs, G)
    MB = bf.MB
    if B > bf.bmax:
        raise RuntimeError(f"PN131: lote {B} > GENESIS_PN131_BMAX {bf.bmax}")
    c = _capa(impl, dev)
    nt = B * L
    R = B * nh * MB
    # En CUDA graph la grilla usa la cantidad MAXIMA de paginas; en eager, la justa.
    NCH = bf.nchmax if capturando else min(bf.nchmax, (int(md.max_seq_len) + bs - 1) // bs)
    NG = (NCH + CPG - 1) // CPG
    Qb = bf.Q[: R * QD]
    lim = bf.lim[:R]
    mqb = bf.mqb[:R]
    dcap = bf.dcap[:R]
    q16 = _filas(query[:nt].view(nt, nh * G, QD)).view(torch.int16)
    seq = md.seq_lens
    if mo == "int4":
        Qb = bf.Q[: R * QPLANOS * (QD // 2)].view(torch.uint8)
        rq = bf.rq[: R * 4]
        sqb = bf.sq[: R * QPLANOS * 4]
        ks["prep"].lanzar((nt, nh * G), [q16, seq, c.refs, _signos_dev(dev), Qb, rq, sqb, lim, mqb, dcap,
                                         L, nh, G, MB, ZSH4, q16.stride(0)])
        if VENT > 0:   # q en int8 y mqb/dcap propios para el espejo de la ventana
            ks["prep8"].lanzar((nt, nh * G), [q16, seq, c.refs8, _signos_dev(dev), bf.Q8[: R * QD],
                                              bf.lim8[:R], bf.mqb8[:R], bf.dcap8[:R],
                                              L, nh, G, MB, ZSH, q16.stride(0)])
    elif ROT_PTX and ARBOL:
        anc = mascara_arbol(dev, L, bf, capturando)
        ks["prep"].lanzar((nt, nh * G), [q16, seq, c.refs, _signos_dev(dev), Qb, lim, mqb, dcap, anc,
                                         bf.abase[:R], bf.amask[:R], L, nh, G, MB, ZSH, q16.stride(0)])
    elif ROT_PTX:
        ks["prep"].lanzar((nt, nh * G), [q16, seq, c.refs, _signos_dev(dev), Qb, lim, mqb, dcap, L, nh, G, MB, ZSH, q16.stride(0)])
    else:
        ks["prep"].lanzar((nt, nh * G), [q16, seq, c.refs, Qb, lim, mqb, dcap, L, nh, G, MB, ZSH, q16.stride(0)])
    bt = md.block_table
    NX = bf.nx if (mo == "int4" and VENT > 0) else 0      # ranuras extra del espejo
    oh = bf.oh[: (NCH + NX) * R * QD]
    ol = bf.ol[: (NCH + NX) * R * QD]
    om = bf.om[: (NCH + NX) * R]
    os_ = bf.os[: (NCH + NX) * R]
    if mo == "int4":
        oc = bf.oc[: (NCH + NX) * R]
        ks["main"].lanzar((NCH, B * nh * (MB // 32)), [Qb, rq, sqb, raw, bt, seq, lim, mqb, dcap, oh, ol, om, os_, oc,
                                          blk, bt.stride(0), bs, NCH, nh, MB // 32, ZSH4, VSH,
                                          bf.dueno if VENT > 0 else oc, VENT], shared=SH_I)
        if VENT > 0 and c.lado is not None:
            blk8 = bs * nh * 520
            ks["espejo"].lanzar((VENT * SCH, B * nh * (MB // 32)), [bf.Q8[: R * QD], _lado(c, dev, bf.bmax, nh, bs), bt, seq, bf.lim8[:R],
                                                bf.mqb8[:R], bf.dcap8[:R], oh, ol, om, os_,
                                                blk8, bt.stride(0), bs, NCH, nh, MB // 32, ZSH, VSH,
                                                bf.dueno, mqb, oc, VENT], shared=SH_H)
    else:
        bq = 8 * NW8 if (NW8 != 4 and MB % (8 * NW8) == 0) else 32
        km, sh = (ks["main8"], sh_h(NW8)) if bq != 32 else (ks["main"], SH_H)
        if ARBOL and ROT_PTX:
            km.lanzar((NCH, B * nh * (MB // bq)), [Qb, raw, bt, seq, lim, mqb, dcap, bf.abase[:R], bf.amask[:R],
                                                   oh, ol, om, os_,
                                                   blk, bt.stride(0), bs, NCH, nh, MB // bq, ZSH, VSH], shared=sh)
        else:
            km.lanzar((NCH, B * nh * (MB // bq)), [Qb, raw, bt, seq, lim, mqb, dcap, oh, ol, om, os_,
                                                   blk, bt.stride(0), bs, NCH, nh, MB // bq, ZSH, VSH], shared=sh)
    Og = bf.Og[: NG * R * QD]
    Sg = bf.Sg[: NG * R]
    if mo == "int4":
        ks["union"].lanzar((R, NG), [oh, ol, om, os_, oc, mqb, dcap, seq, Og, Sg, R, NCH, CPG, nh * MB, bs, NX])
    else:
        ks["union"].lanzar((R, NG), [oh, ol, om, os_, mqb, dcap, seq, Og, Sg, R, NCH, CPG, nh * MB, bs, 0])
    o16 = output[:nt].view(torch.int16)
    if mo == "int4":
        ks["salida"].lanzar((nt, nh * G), [Og, Sg, c.refs, _signos_dev(dev), o16, NG, R, L, nh, G, MB, VSH])
    else:
        ks["salida"].lanzar((nt, nh * G), [Og, Sg, c.refs, o16, NG, R, L, nh, G, MB, VSH])


# ─────────────────────────── prefill (decuantizado) ───────────────────────
def decuantizar_bloques(impl, kv_cache, ids):
    """Paginas -> (k, v) fp16 [n, BS, NH, 256] con el kernel entero sk18h_decuant (vistas de
    una sola reserva [n, 2, BS, NH, 256], que es lo que usa FlashInfer)."""
    kv = decuantizar_kv(impl, kv_cache, ids)
    return kv[:, 0], kv[:, 1]


def decuantizar_kv(impl, kv_cache, ids):
    """Paginas -> fp16 para el prefill. En int4, las paginas que tienen espejo int8 (la ventana
    reciente, donde vive el prompt que se esta procesando) se decuantizan DESDE el espejo: si no,
    el prefill ve la version int4 y la calidad de lo estructurado (tool calls) se resiente."""
    nb, nh, bs, blk, raw = _geom(kv_cache)
    dev = kv_cache.device
    c = _capa(impl, dev)
    ids = ids.to(torch.int64).contiguous()
    n = ids.shape[0]
    kv = torch.empty((n, 2, bs, nh, QD), dtype=torch.float16, device=dev)
    mo = modo(impl)
    ks = _kernels(mo)
    if mo != "int4":
        ks["decuant"].lanzar((n, bs), [raw, ids, c.refs, _signos_dev(dev), kv.view(torch.int16), blk, bs, nh])
        return kv
    ids4 = ids
    if VENT > 0 and c.lado is not None:
        dueno = _get_bufs(dev, nh, bs, max(1, impl.num_heads // nh)).dueno
        igual = ids[:, None] == dueno.to(torch.int64)[None, :]        # [n, ranuras]
        esp = torch.where(igual.any(1), igual.float().argmax(1).to(torch.int64), torch.full_like(ids, -1))
        ids4 = torch.where(esp >= 0, torch.full_like(ids, -1), ids)
        if bool((esp >= 0).any()):
            blk8 = bs * nh * 520
            ks["decuant8"].lanzar((n, bs), [c.lado, esp, c.refs8, _signos_dev(dev),
                                            kv.view(torch.int16), blk8, bs, nh])
    ks["decuant"].lanzar((n, bs), [raw, ids4, c.refs, _signos_dev(dev), kv.view(torch.int16), blk, bs, nh])
    return kv


_fi = {}


def _prefill_flashinfer(impl, layer, query, kv_cache, md, output):
    """Prefill: decuantiza a fp16 SOLO las paginas que tocan los pedidos del paso y usa el
    prefill paginado de FlashInfer (page 832) sobre ese cache temporal. El plan se arma una
    vez por paso (misma metadata para las 16 capas) y se reusa."""
    import flashinfer
    dev = query.device
    nact = md.num_actual_tokens
    nb, nh, bs, blk, raw = _geom(kv_cache)
    G = impl.num_heads // nh
    B = md.query_start_loc.shape[0] - 1
    est = _fi.get(dev.index)
    if est is None:
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        est = _fi[dev.index] = dict(w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD"), clave=None)
    clave = (getattr(md, "genesis_clave", None) or id(md), md.num_actual_tokens, md.max_seq_len)
    if est["clave"] != clave:
        seq = md.seq_lens[:B].cpu().to(torch.int64)
        qsl = getattr(md, "genesis_qsl_cpu", None)
        qsl = (qsl if qsl is not None else md.query_start_loc.cpu()).to(torch.int32)
        npag = (seq + bs - 1) // bs
        maxp = int(npag.max())
        bt = md.block_table[:B, :maxp].to(torch.int64)
        valid = (torch.arange(maxp)[None, :] < npag[:, None]).to(dev)
        ids, inv = torch.unique(bt[valid], return_inverse=True)
        kvi = torch.zeros(B + 1, dtype=torch.int32)
        kvi[1:] = torch.cumsum(npag, 0).to(torch.int32)
        last = (seq - (npag - 1) * bs).to(torch.int32)
        est["w"].plan(qsl, kvi, inv.to(torch.int32), last, impl.num_heads, nh, QD, bs, causal=True,
                      pos_encoding_mode="NONE", sm_scale=impl.scale,
                      q_data_type=torch.float16, kv_data_type=torch.float16)
        est.update(clave=clave, ids=ids)
    kv = decuantizar_kv(impl, kv_cache, est["ids"])
    o = est["w"].run(query[:nact].to(torch.float16), kv)
    output[:nact].view(nact, impl.num_heads, QD).copy_(o.view(nact, impl.num_heads, QD))
    return output


def _prefill_decuant(impl, layer, query, kv_cache, md, output):
    if ROT_PTX:
        nact = md.num_actual_tokens
        query = _rotar_q_prefill(query[:nact].view(nact, impl.num_heads, QD))
    if os.environ.get("GENESIS_PN131_PREFILL", "flashinfer") == "flashinfer":
        try:
            return _prefill_flashinfer(impl, layer, query, kv_cache, md, output)
        except ImportError:
            pass
    return _prefill_triton(impl, layer, query, kv_cache, md, output)


def _prefill_triton(impl, layer, query, kv_cache, md, output):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode
    nact = md.num_actual_tokens
    nb, nh, bs, blk, raw = _geom(kv_cache)
    B = md.query_start_loc.shape[0] - 1
    npag = (md.seq_lens[:B] + bs - 1) // bs
    maxp = int((int(md.max_seq_len) + bs - 1) // bs)
    bt = md.block_table[:B, :maxp].to(torch.int64)
    valid = torch.arange(maxp, device=bt.device)[None, :] < npag[:, None]
    ids, inv = torch.unique(bt[valid], return_inverse=True)
    kd, vd = decuantizar_bloques(impl, kv_cache, ids)
    bt2 = torch.zeros_like(bt, dtype=torch.int32)
    bt2[valid] = inv.to(torch.int32)
    unified_attention(
        q=query[:nact], k=kd, v=vd, out=output[:nact],
        cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
        seqused_k=md.seq_lens, max_seqlen_k=md.max_seq_len, softmax_scale=impl.scale,
        causal=md.causal, alibi_slopes=None, use_alibi_sqrt=False, window_size=impl.sliding_window,
        block_table=bt2, softcap=impl.logits_soft_cap, q_descale=None,
        k_descale=layer._k_scale.expand((B, nh)), v_descale=layer._v_scale.expand((B, nh)),
        seq_threshold_3D=md.seq_threshold_3D, num_par_softmax_segments=md.num_par_softmax_segments,
        softmax_segm_output=md.softmax_segm_output, softmax_segm_max=md.softmax_segm_max,
        softmax_segm_expsum=md.softmax_segm_expsum, sinks=None, output_scale=None,
        mm_prefix_range=md.mm_prefix_range_tensor, rswa_prefix_lens=md.rswa_prefix_lens,
        rswa_window=md.rswa_window, kv_quant_mode=KVQuantMode.NONE, k_scale_cache=None,
        v_scale_cache=None, chunk_lookback=impl.chunk_lookback, use_td=impl.use_td,
        mm_prefix_clamp_sliding_window=getattr(layer, "mm_prefix_clamp_sliding_window", False),
    )
    return output
