# SPDX-License-Identifier: Apache-2.0
"""Genesis PN122: rollback del MTP en GDN con CINTA en vez de K copias del estado.

Por qué
-------
Medido con opencode (status de PN115, ``bloques_por_request``): con MTP K=3 cada
request ocupa 5-7 bloques en CADA uno de los 3 grupos GDN, ~18 bloques = ~15k
tokens de KV. La causa es ``MambaSpec.num_speculative_blocks = K``: el kernel de
decode spec escribe el estado completo (786 KB por capa por rank) tras cada uno
de los K+1 tokens, y el pool reserva un bloque por cada una de esas copias.

Qué hace
--------
La actualización de GDN es de rango 1::

    S <- S * exp(g) ;  d = (v - S k) * beta ;  S <- S + d k^T

así que para rehacer un token alcanza con guardar ``(k normalizada, v, g, beta)``,
~8 KB por token por capa, en vez del estado. Entonces:

* forward spec: lee el estado del slot de la columna 0, **reproduce
  num_accepted-1 filas de la cinta** del paso anterior, procesa los K+1 tokens,
  escribe el estado tras el token 0 (siempre aceptado) y la cinta de 1..K;
* las copias de ``align`` (``mamba_utils``) que upstream hace leyendo
  ``estado[columna + bias]`` se reemplazan por ``estado[columna]`` + ``bias``
  filas de cinta. La contabilidad de upstream (``num_accepted``, columnas,
  bordes cacheados) queda intacta: el estado que se materializa es el mismo.

La cinta NO entra en el relleno de la página (con spec decode el estado conv es
de kernel-1+K columnas y el relleno queda en 4.096 B), así que vive en un buffer
por capa indexado por un SLOT por request: ``[slots, K, fila]`` en fp16, ~13 MB
por rank con max-num-seqs 10.

Validado aislado (tests/proto/gdn_cinta.py): error relativo 1,9e-4 contra
upstream en 40 pasos con aceptaciones al azar (redondeo fp16).
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn122")

_TRUTHY = ("1", "true", "yes", "on")
_ACTIVO = os.environ.get("GENESIS_ENABLE_PN122_GDN_CINTA", "0").strip().lower() in _TRUTHY


def activo() -> bool:
    return _ACTIVO


_DEBUG = os.environ.get("GENESIS_PN122_DEBUG", "0").strip().lower() in _TRUTHY

# Arbol de borrador (apagado por defecto). Con el arbol, los tokens 1..K del paso no son una
# cadena: cada uno cuelga de un ancestro. Cambian dos cosas y nada mas:
#   * el forward spec arma el estado de cada token desde el de SU padre (``_k_spec_arbol``);
#   * lo que se reproduce al paso siguiente no son las filas 0..r-1 de la cinta sino las del
#     camino aceptado: ``camino[slot, j]`` = fila de cinta del j-esimo token aceptado.
# La cinta se escribe igual que siempre (fila t-1 = token t del paso).
_ARBOL = os.environ.get("GENESIS_ENABLE_ARBOL", "0").strip().lower() in _TRUTHY
# Forma cerrada del arbol (Bole arXiv 2608.01651 / TreeWY arXiv 2608.20961): un sistema triangular
# en vez de rehacer los ancestros con actualizaciones de rango 1. APAGADA: es correcta (2,2e-4
# contra fp32, igual que el secuencial) y su costo NO depende de la topologia del arbol, que era
# el objetivo, pero medida con grafos CUDA sale 37% mas cara (48 capas, 6 pedidos):
#
#   secuencial  cadena 77,0 us | tipico 84,6 | estrella 79,4 | peor 77,1
#   cerrada     116 us para TODAS las topologias  (prep 11,3 + main 68,3 + escribir ~36)
#
# El main es lo caro: 68,3 us contra ~48,6 del secuencial. La teoria dice que deberia hacer la
# MITAD del trabajo (no toca el estado: solo S_0 k_t y S_0 q_t por token, contra decay + S k +
# rango 1 + S q del secuencial), pero Triton no indexa registros dinamicamente y el sistema
# triangular termina simulado con `tl.sum(tl.where(at == t, ...))`, que son reducciones completas
# sobre [BT, BV] — y hay tres lazos sobre T en vez de uno.
#
# Para que gane hay que hacerlo como Bole: serie de Neumann con `tl.dot` sobre matrices [BT, BT]
# (Gb es nilpotente, asi que la serie es finita) y todo en layout traspuesto [BV, BT], con k y q
# ya normalizados en un buffer del prep. El riesgo es el derrame de registros: b_h [BV, BK] ya son
# 4096 floats por programa. Ver [[arbol-gdn-forma-cerrada-bole-treewy]].
_CERRADA = os.environ.get("GENESIS_ARBOL_CERRADA", "0").strip().lower() in _TRUTHY
# Los productos pesados (S_0 k y S_0 q) en int8 sobre tensor cores. Solo con la forma cerrada:
# es la unica que deja S_0 FIJO durante el arbol, asi que se cuantiza una vez y se usa 2T veces.
_INT8 = os.environ.get("GENESIS_ARBOL_INT8", "1").strip().lower() in _TRUTHY
_camino_gpu: torch.Tensor | None = None     # [slots, TM] int32; la cadena es arange(TM)
_cerr: dict = {}                            # buffers del sistema triangular, por device
_anc_gpu: torch.Tensor | None = None        # [tokens del lote] int32, bits de ancestros por token
_anc3_gpu: torch.Tensor | None = None       # [tokens del lote, 3] int32, los 3 ancestros mas cercanos


def arbol() -> bool:
    return _ARBOL


def camino_gpu() -> torch.Tensor | None:
    """``[slots, TM]``: el runner escribe aca, despues de aceptar, las filas de cinta del
    camino aceptado (nodo - 1). Persistente: lo leen el forward y las copias align."""
    return _camino_gpu


def fijar_ancestros(anc: torch.Tensor | None) -> None:
    """Bits de ancestros por token del lote (formato de ``arbol_borrador.bits_ancestros``),
    en un buffer ESTATICO (grafos CUDA). ``None`` = todos los pedidos van en cadena."""
    global _anc_gpu
    _anc_gpu = anc


def ancestros_gpu() -> torch.Tensor | None:
    """El buffer estatico de ancestros (lo crea ``enlazar``; en reposo, mascara de cadena)."""
    return _anc_gpu


def ancestros3_gpu() -> torch.Tensor | None:
    """Los 3 ancestros mas cercanos por token, que es lo unico que mira la conv causal."""
    return _anc3_gpu


def tokens_arbol() -> int | None:
    """Tokens por pedido del paso en arbol (K+1). OJO: ``ancestros3_gpu()`` es un buffer fijo de
    ``n_slots * (K+1)`` filas para los grafos, asi que su largo NO dice cuantos tokens tiene el lote."""
    if _anc3_gpu is None or not _n_slots:
        return None
    return _anc3_gpu.shape[0] // _n_slots


# Por paso. Los grafos FULL (decode uniforme) hornean el kernel de arbol, y ahi lo unico que
# decide es el CONTENIDO del buffer (con la mascara de cadena da bit a bit lo de siempre). En
# eager/piecewise (lotes mezclados) Python corre cada vez y esta bandera elige el kernel. Por
# eso arranca prendida: es lo que tiene que ver la captura.
_paso_arbol = True


def paso_en_arbol(si: bool) -> None:
    global _paso_arbol
    _paso_arbol = bool(si)


def paso_arbol_activo() -> bool:
    if not _ARBOL or _anc_gpu is None:
        return False
    # Capturando un grafo se hornea SIEMPRE el kernel de arbol, valga lo que valga la bandera
    # (una corrida dummy anterior la puede haber dejado apagada, y el grafo quedaba con la conv
    # y el GDN de cadena para siempre: medido, el nodo tras un salto de rama fallaba el 75%).
    return _paso_arbol or torch.cuda.is_current_stream_capturing()
_debug_n = 0


_SYNC_FILE = "/dev/shm/pn122_sync"


_sync_cache = [0, 0.0]


def sync_bits() -> int:
    """Bitmask de sincronizaciones de diagnostico, leido en caliente de un archivo.

    Se relee como mucho una vez por segundo: leerlo en cada llamada frenaba la
    CPU lo suficiente como para ESCONDER la carrera que se esta buscando.
    """
    if not _DEBUG:
        return 0
    import time
    now = time.monotonic()
    if now - _sync_cache[1] > 1.0:
        _sync_cache[1] = now
        try:
            with open(_SYNC_FILE) as f:
                _sync_cache[0] = int(f.read().strip() or 0)
        except (OSError, ValueError):
            _sync_cache[0] = 0
    return _sync_cache[0]


def debug_builder(num_spec_decodes, nacc, slots, sidx, cu, seq_lens) -> None:
    """Diagnostico (sincroniza con la GPU: solo con GENESIS_PN122_DEBUG=1)."""
    global _debug_n
    bits = sync_bits()
    if bits & 2:
        torch.cuda.synchronize()
    if not bits & 8 or num_spec_decodes == 0 or _debug_n > 400:
        return
    _debug_n += 1
    n = num_spec_decodes
    log.warning("[PN122 dbg] n=%d acc=%s slot=%s sidx=%s cu=%s seq=%s", n,
                nacc[:n].tolist() if nacc is not None else None,
                slots[:n].tolist() if slots is not None else None,
                sidx[:n, 0].tolist() if sidx is not None else None,
                cu[: n + 1].tolist() if cu is not None else None,
                seq_lens[:n].tolist() if seq_lens is not None else None)


def num_speculative_blocks(vllm_config) -> int:
    """Lo que upstream pone en ``MambaSpec.num_speculative_blocks``.

    Hay que reproducir la cuenta de cada version, porque con la cinta apagada este valor
    TIENE que ser exactamente el de upstream:

    * v0.27.1: ``speculative_config.num_speculative_tokens if speculative_config else 0``
    * v0.29.0: ``0 if cache_config.use_kda_recoverssm else num_speculative_tokens`` — el
      atajo subio a ``vllm_config`` y aparecio la rama de RecoverSSM, que verifica la
      ventana entera contra un solo checkpoint y por eso nunca escribe los slots por
      token de draft.
    """
    k = getattr(vllm_config, "num_speculative_tokens", None)
    if k is None:                                   # v0.27.1 y anteriores
        sc = vllm_config.speculative_config
        k = sc.num_speculative_tokens if sc else 0
    if getattr(vllm_config.cache_config, "use_kda_recoverssm", False):
        k = 0                                       # v0.29.0: RecoverSSM no usa esos slots
    # Diagnostico: GENESIS_PN122_SIN_LIBERAR=1 conserva los bloques especulativos
    # (cinta y kernel activos, sin ahorro) para aislar el efecto de sacarlos.
    if os.environ.get("GENESIS_PN122_SIN_LIBERAR", "0") == "1":
        return k
    return 0 if _ACTIVO else k


# ─────────────────────────────── slots ────────────────────────────────

_slot_de: dict[str, int] = {}
_libres: list[int] = []
_slots_cpu: torch.Tensor | None = None
_slots_gpu: torch.Tensor | None = None
_n_slots = 0
_ultimos_vals: list = []


def _init_slots(max_reqs: int, device) -> None:
    global _slots_cpu, _slots_gpu, _n_slots, _libres
    if _slots_gpu is not None:
        return
    # Slot 0 es de relleno (filas de padding de CUDA graph). El resto, holgado:
    # un slot se libera recien cuando el request termina o es preemptado.
    _n_slots = 2 * max_reqs + 2
    _libres = list(range(_n_slots - 1, 0, -1))
    # Filas: el builder indexa con la mascara spec del batch, que puede venir
    # rellenada para CUDA graph mas alla de max_reqs.
    filas = 4 * max_reqs + 8
    _slots_cpu = torch.zeros(filas, dtype=torch.int32, pin_memory=True)
    _slots_gpu = torch.zeros(filas, dtype=torch.int32, device=device)


def n_slots() -> int:
    return _n_slots


def slots_gpu() -> torch.Tensor:
    assert _slots_gpu is not None
    return _slots_gpu


def actualizar_slots(scheduler_output, req_ids: list[str], max_reqs: int, device,
                     vivos) -> None:
    """Corre en ``preprocess_mamba`` (antes de la metadata y del forward).

    Deja en ``slots_gpu()[i]`` el slot de cinta del request de la fila ``i`` del
    batch, que es el orden que usan el builder de GDN y los kernels de copia.

    ``vivos`` es ``GPUModelRunner.requests``. NO alcanza con
    ``finished_req_ids``: un request que termina en un paso sin tokens agendados
    nunca pasa por aca (``execute_model`` sale antes) y su slot se fugaba —
    medido: el engine murio con 21 slots ocupados tras 21 requests seriales.
    """
    _init_slots(max_reqs, device)
    pre = scheduler_output.preempted_req_ids or set()
    res = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    muertos = [rid for rid in _slot_de if rid not in vivos or rid in pre or rid in res]
    for rid in muertos:
        _libres.append(_slot_de.pop(rid))
    n = len(req_ids)
    vals = []
    for rid in req_ids:
        s = _slot_de.get(rid)
        if s is None:
            if not _libres:
                raise RuntimeError(
                    "[PN122] sin slots de cinta: %d requests vivos con %d slots"
                    % (len(_slot_de), _n_slots - 1))
            s = _libres.pop()
            _slot_de[rid] = s
            _sombra_nuevos.add(s)
        vals.append(s)
    global _ultimos_vals
    if n and vals != _ultimos_vals:
        _ultimos_vals = vals
        # Desde memoria NO pinned a proposito: con un buffer pinned reutilizado
        # y non_blocking, el paso siguiente podia pisar los valores antes de
        # que la GPU (atrasada, async scheduling) ejecutara la copia, y el
        # kernel reproducia la cinta de OTRO request.
        _slots_gpu[:n].copy_(torch.tensor(vals, dtype=torch.int32), non_blocking=True)
    if sync_bits() & 1:
        torch.cuda.synchronize()


# ─────────────────────────────── cinta por capa ────────────────────────────────

def dims(layer) -> tuple[int, int, int, int]:
    tp = layer.tp_size
    return (layer.num_k_heads // tp, layer.num_v_heads // tp,
            layer.head_k_dim, layer.head_v_dim)


def fila(layer) -> int:
    H, HV, K, V = dims(layer)
    return H * K + HV * V + 2 * HV


def enlazar(layer, device) -> None:
    """Crea la cinta de la capa GDN. Se llama desde ``bind_kv_cache``."""
    if not hasattr(layer, "num_v_heads") or getattr(layer, "num_spec", 0) <= 0:
        return
    if getattr(layer, "_g122_cinta", None) is not None:
        return
    try:
        from vllm.config import get_current_vllm_config
        max_reqs = int(get_current_vllm_config().scheduler_config.max_num_seqs)
    except Exception:
        max_reqs = 32
    _init_slots(max_reqs, device)
    log.warning("[PN122] cinta de %s: %d slots x %d tokens x %d", getattr(layer, "prefix", "?"),
                _n_slots, layer.num_spec, fila(layer))
    layer._g122_cinta = torch.zeros(
        (_n_slots, layer.num_spec, fila(layer)), dtype=torch.float32, device=device)
    global _camino_gpu, _anc_gpu, _anc3_gpu
    if _ARBOL and _camino_gpu is None:
        _camino_gpu = torch.arange(layer.num_spec, dtype=torch.int32, device=device)[None] \
            .repeat(_n_slots, 1).contiguous()
    if _ARBOL and _anc_gpu is None:
        T = layer.num_spec + 1
        _anc_gpu = ((1 << torch.arange(T, device=device, dtype=torch.int32)) - 1) \
            .repeat(_n_slots).contiguous()
        ar = torch.arange(T, device=device, dtype=torch.int32)
        # [slots*T, 3]: en reposo, la cadena (t-1, t-2, t-3 con el convenio -1/-2/-3)
        _anc3_gpu = torch.stack([ar - 1, ar - 2, ar - 3], dim=1).repeat(_n_slots, 1).contiguous()


# ─────────────────────────────── kernels ────────────────────────────────

from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit(do_not_specialize=["N"])
def _k_spec(A_log, a, b, dt_bias, beta_sp, threshold, q, k, v, o, h, stride_h, cu, sidx,
            nacc, slots, cinta, scale, N,
            H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, ROW: tl.constexpr,
            IS_L2: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K
    mv = o_v < V
    mh = mv[:, None] & mk[None, :]
    Al = tl.load(A_log + i_hv).to(tl.float32)
    db = tl.load(dt_bias + i_hv).to(tl.float32)
    p_h = h + s * stride_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)

    # 1) reproducir las filas aceptadas del paso anterior
    r = tl.load(nacc + i_n).to(tl.int64) - 1
    slot = tl.load(slots + i_n).to(tl.int64)
    for j in range(0, r):
        row = cinta + (slot * TM + j) * ROW
        rk = tl.load(row + i_h * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(row + H * K + i_hv * V + o_v, mask=mv, other=0).to(tl.float32)
        rg = tl.load(row + H * K + HV * V + i_hv).to(tl.float32)
        rb = tl.load(row + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h *= tl.exp(rg)
        b_d = (rv - tl.sum(b_h * rk[None, :], 1)) * rb
        b_h += b_d[:, None] * rk[None, :]

    # 2) tokens del paso actual, igual que upstream
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    for t in range(0, T):
        b_q = tl.load(p_q, mask=mk, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mv, other=0).to(tl.float32)
        x = tl.load(p_a).to(tl.float32) + db
        sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        b_g = -tl.exp(Al) * sp
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))
        if IS_L2:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_d = (b_v - tl.sum(b_h * b_k[None, :], 1)) * b_beta
        b_h += b_d[:, None] * b_k[None, :]
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mv)
        if t == 0:  # la unica escritura del estado completo
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_a += HV
        p_b += HV
        p_o += HV * V


@triton.jit
def _regla_delta(b_h, b_k, b_v, b_g, b_beta):
    b_h = b_h * tl.exp(b_g)
    b_d = (b_v - tl.sum(b_h * b_k[None, :], 1)) * b_beta
    return b_h + b_d[:, None] * b_k[None, :]


@triton.jit(do_not_specialize=["N"])
def _k_spec_arbol(A_log, a, b, dt_bias, beta_sp, threshold, q, k, v, o, h, stride_h, cu, sidx,
                  nacc, slots, cinta, camino, anc, scale, N,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, ROW: tl.constexpr,
                  IS_L2: tl.constexpr):
    """``_k_spec`` para un paso en ARBOL. Aparte a proposito: el kernel de produccion no se toca.

    Los tokens vienen en orden topologico, y mejor en preorden (``arbol_borrador.orden_dfs``):
    mientras el padre de un token sea el token anterior se sigue con el estado corriente, igual
    que la cadena; en un salto de rama se vuelve al estado tras el token 0 y se rehacen los
    ancestros (actualizaciones de rango 1 sobre un bloque que ya esta en registros: no hay
    trafico de estado, que es lo que cuesta aca). Guardar un estado por nodo seria peor: el
    estado local pasaria de 4k a 64k elementos por programa.

    ``anc[bos + t]``: bit j-1 = el token j es ancestro de t, o es t. Con la mascara de cadena,
    ``(1 << t) - 1``, nunca hay salto y la salida es identica bit a bit a la de ``_k_spec``.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K
    mv = o_v < V
    mh = mv[:, None] & mk[None, :]
    # invariantes del lazo, calculadas UNA vez: exp(A_log) se re-evaluaba por token y por cada
    # ancestro rehecho, y beta_sp/threshold son constantes de la llamada (1.0 y 20.0).
    nexpAl = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    db = tl.load(dt_bias + i_hv).to(tl.float32)
    inv_b = 1.0 / beta_sp
    p_h = h + s * stride_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)

    # 1) reproducir el CAMINO aceptado del paso anterior
    r = tl.load(nacc + i_n).to(tl.int64) - 1
    slot = tl.load(slots + i_n).to(tl.int64)
    for j in range(0, r):
        jj = tl.load(camino + slot * TM + j).to(tl.int64)
        row = cinta + (slot * TM + jj) * ROW
        rk = tl.load(row + i_h * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(row + H * K + i_hv * V + o_v, mask=mv, other=0).to(tl.float32)
        rg = tl.load(row + H * K + HV * V + i_hv).to(tl.float32)
        rb = tl.load(row + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _regla_delta(b_h, rk, rv, rg, rb)

    # 2) tokens del paso actual, cada uno sobre el estado de su padre
    b_raiz = b_h
    m_prev = tl.zeros((), tl.int32)
    for t in range(0, T):
        m_t = tl.load(anc + bos + t).to(tl.int32)
        if t >= 2:
            if m_t != (m_prev | (1 << (t - 1))):
                b_h = b_raiz                       # salto de rama: rehacer los ancestros
                for j in range(1, t):
                    if ((m_t >> (j - 1)) & 1) != 0:
                        src = bos + j
                        jk = tl.load(k + (src * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
                        jv = tl.load(v + (src * HV + i_hv) * V + o_v, mask=mv, other=0).to(tl.float32)
                        jx = tl.load(a + src * HV + i_hv).to(tl.float32) + db
                        jsp = tl.where(beta_sp * jx <= threshold,
                                       inv_b * tl.log(1 + tl.exp(beta_sp * jx)), jx)
                        jb = tl.sigmoid(tl.load(b + src * HV + i_hv).to(tl.float32))
                        if IS_L2:
                            jk = jk * tl.rsqrt(tl.sum(jk * jk) + 1e-6)
                        b_h = _regla_delta(b_h, jk, jv, nexpAl * jsp, jb)
        m_prev = m_t
        src = bos + t
        b_q = tl.load(q + (src * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
        b_k = tl.load(k + (src * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
        b_v = tl.load(v + (src * HV + i_hv) * V + o_v, mask=mv, other=0).to(tl.float32)
        x = tl.load(a + src * HV + i_hv).to(tl.float32) + db
        sp = tl.where(beta_sp * x <= threshold, inv_b * tl.log(1 + tl.exp(beta_sp * x)), x)
        b_beta = tl.sigmoid(tl.load(b + src * HV + i_hv).to(tl.float32))
        if IS_L2:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_h = _regla_delta(b_h, b_k, b_v, nexpAl * sp, b_beta)
        p_o = o + (src * HV + i_hv) * V + o_v
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mv)
        if t == 0:  # la unica escritura del estado completo
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)
            b_raiz = b_h


@triton.jit
def _redondear(x):
    """Al entero mas cercano (empate al par), que es lo que hace torch.round."""
    return tl.extra.cuda.libdevice.rint(x)


@triton.jit
def _k_prep_cerrada(A_log, a, b, dt_bias, beta_sp, threshold, k, q, cu, sidx, anc,
                    Gb_out, C_out, aux_out, k8_out, q8_out, esc_out, scale, s_g, s_a, s_k8, s_e,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                    BK: tl.constexpr, BT: tl.constexpr, T: tl.constexpr, IS_L2: tl.constexpr,
                    INT8: tl.constexpr):
    """Prepara, UNA vez por (pedido, cabeza v), las matrices del sistema triangular del arbol.

    Sale de la forma cerrada del gated delta rule (Bole arXiv 2608.01651, TreeWY arXiv 2608.20961;
    derivacion y validacion en tests/proto/gdn_forma_cerrada.py). Con P_t = suma de g sobre el
    camino raiz->t, e "i < t" = i es ancestro ESTRICTO de t:

        (I + diag(beta) G) D = R,   G[t,i] = (P_t/P_i) (k_i . k_t)
        o_t = P_t (S_0 q_t) + sum_{i <= t} C[t,i] D_i,   C[t,i] = (P_t/P_i) (k_i . q_t)

    Es decir: la salida de TODOS los nodos sale de un sistema de T x T, sin tocar el estado ni
    rehacer ancestros. Aca se calculan ``Gb = diag(beta) G``, ``C`` y ``(P, beta)``; el estado
    S_0 no entra, asi que esto no depende del bloque de V y se hace una sola vez por cabeza.
    """
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    if tl.load(cu + i_n + 1).to(tl.int64) - bos != T:
        return
    if tl.load(sidx + i_n).to(tl.int64) <= 0:
        return
    at = tl.arange(0, BT)
    mt = at < T
    ok = tl.arange(0, BK)
    mk = ok < K
    Al = tl.load(A_log + i_hv).to(tl.float32)
    db = tl.load(dt_bias + i_hv).to(tl.float32)
    x = tl.load(a + (bos + at) * HV + i_hv, mask=mt, other=0.0).to(tl.float32) + db
    sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
    g = -tl.exp(Al) * sp
    bet = tl.sigmoid(tl.load(b + (bos + at) * HV + i_hv, mask=mt, other=0.0).to(tl.float32))
    # mascara de ancestros desde los bits: el nodo 0 (el ancla) es ancestro de todos
    m = tl.load(anc + bos + at, mask=mt, other=0).to(tl.int32)
    j = at[None, :]
    bit = (m[:, None] >> tl.maximum(j - 1, 0)) & 1     # j = 0 lo cubre la rama de abajo
    inc = ((j == 0) | ((j >= 1) & (bit != 0))) & (j <= at[:, None]) & mt[:, None] & mt[None, :]
    est = inc & (j != at[:, None])
    P = tl.sum(tl.where(inc, g[None, :], 0.0), 1)                      # log del decay acumulado
    rel = tl.exp(P[:, None] - P[None, :])
    idx = (bos + at)[:, None] * (H * K) + i_h * K + ok[None, :]
    mkk = mt[:, None] & mk[None, :]
    k_all = tl.load(k + idx, mask=mkk, other=0.0).to(tl.float32)
    q_all = tl.load(q + idx, mask=mkk, other=0.0).to(tl.float32)
    if IS_L2:
        k_all = k_all * tl.rsqrt(tl.sum(k_all * k_all, 1) + 1e-6)[:, None]
        q_all = q_all * tl.rsqrt(tl.sum(q_all * q_all, 1) + 1e-6)[:, None]
    q_all = q_all * scale
    # ieee y no tf32: tf32 deja 10 bits de mantisa y el error entra en el sistema triangular
    KK = tl.dot(k_all, tl.trans(k_all), input_precision="ieee")
    QK = tl.dot(q_all, tl.trans(k_all), input_precision="ieee")
    # (I + diag(beta) G) es triangular inferior UNITARIA: su inversa sale de una serie de Neumann
    # FINITA (Gb es nilpotente, orden = profundidad del arbol). Se calcula aca, sobre matrices de
    # T x T, y se fusiona con C: asi el kernel pesado hace UN solo dot para el sistema y la salida
    # juntos, sin lazos ni indexado de registros (que en Triton se simula con reducciones caras).
    Gb = tl.where(est, bet[:, None] * rel * KK, 0.0)
    C = tl.where(inc, rel * QK, 0.0)
    eye = tl.where(at[:, None] == at[None, :], 1.0, 0.0)
    M = eye
    term = eye
    for _ in tl.static_range(1, BT):
        term = -tl.dot(term, Gb, input_precision="ieee")
        M += term
    base = (i_n * HV + i_hv) * s_g
    tl.store(Gb_out + base + at[:, None] * BT + at[None, :], M, mask=mt[:, None] & mt[None, :])
    tl.store(C_out + base + at[:, None] * BT + at[None, :],
             tl.dot(C, M, input_precision="ieee"), mask=mt[:, None] & mt[None, :])
    ab = (i_n * HV + i_hv) * s_a
    tl.store(aux_out + ab + at * 2, P, mask=mt)
    tl.store(aux_out + ab + at * 2 + 1, bet, mask=mt)
    if INT8:
        # k y q ya normalizados, a int8 por token: es el formato que el main necesita para el
        # mma.s8. Lo hace el primer programa de cada cabeza k (los demas escribirian lo mismo).
        if (i_hv % (HV // H)) == 0:
            ek = tl.max(tl.abs(k_all), 1) / 127.0
            eq = tl.max(tl.abs(q_all), 1) / 127.0
            bk8 = (i_n * H + i_h) * s_k8
            tl.store(k8_out + bk8 + at[:, None] * K + ok[None, :],
                     _redondear(k_all / ek[:, None]).to(tl.int8), mask=mkk)
            tl.store(q8_out + bk8 + at[:, None] * K + ok[None, :],
                     _redondear(q_all / eq[:, None]).to(tl.int8), mask=mkk)
            be = (i_n * H + i_h) * s_e
            tl.store(esc_out + be + at * 2, ek, mask=mt)
            tl.store(esc_out + be + at * 2 + 1, eq, mask=mt)


@triton.jit(do_not_specialize=["N"])
def _k_spec_cerrada(q, k, v, o, h, stride_h, cu, sidx, nacc, slots, cinta, camino,
                    Gb, C, aux, s_g, s_a, scale, N,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, T: tl.constexpr,
                    TM: tl.constexpr, ROW: tl.constexpr, IS_L2: tl.constexpr):
    """Forma cerrada: resuelve el sistema triangular y NO toca el estado durante el arbol.

    El secuencial (``_k_spec_arbol``) toca S (V x K) cuatro veces por token — decay, S k, rango 1,
    S q — y otra vez entera por cada ancestro que rehace en un salto de rama. Aca S_0 queda fijo en
    registros: solo dos productos ``S_0 k_t`` y ``S_0 q_t`` por token, mas un sistema de T x T.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    if tl.load(cu + i_n + 1).to(tl.int64) - bos != T:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    at = tl.arange(0, BT)
    mk = o_k < K
    mv = o_v < V
    mt = at < T
    mh = mv[:, None] & mk[None, :]
    p_h = h + s * stride_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)

    # 1) reproducir el CAMINO aceptado del paso anterior (rango 1, como siempre: es corto y no
    #    tiene saltos, y sus entradas ya estan preparadas en la cinta)
    r = tl.load(nacc + i_n).to(tl.int64) - 1
    slot = tl.load(slots + i_n).to(tl.int64)
    for jj in range(0, r):
        row = cinta + (slot * TM + tl.load(camino + slot * TM + jj).to(tl.int64)) * ROW
        rk = tl.load(row + i_h * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(row + H * K + i_hv * V + o_v, mask=mv, other=0).to(tl.float32)
        rg = tl.load(row + H * K + HV * V + i_hv).to(tl.float32)
        rb = tl.load(row + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _regla_delta(b_h, rk, rv, rg, rb)

    # 2) el arbol, de una: R[t] = beta_t (v_t - P_t S_0 k_t) y Sq[t] = S_0 q_t
    ab = (i_n * HV + i_hv) * s_a
    P = tl.load(aux + ab + at * 2, mask=mt, other=0.0)
    bet = tl.load(aux + ab + at * 2 + 1, mask=mt, other=0.0)
    expP = tl.exp(P)
    D = tl.zeros((BT, BV), tl.float32)      # R, y despues la solucion in situ
    Sq = tl.zeros((BT, BV), tl.float32)
    for t in range(0, T):
        src = bos + t
        b_k = tl.load(k + (src * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
        b_q = tl.load(q + (src * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
        b_v = tl.load(v + (src * HV + i_hv) * V + o_v, mask=mv, other=0).to(tl.float32)
        if IS_L2:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale          # el prep ya lo aplica a su copia de q; aca tambien hace falta
        sel = at == t
        ep = tl.sum(tl.where(sel, expP, 0.0), 0)
        bt_ = tl.sum(tl.where(sel, bet, 0.0), 0)
        rt = bt_ * (b_v - ep * tl.sum(b_h * b_k[None, :], 1))
        D = tl.where(sel[:, None], rt[None, :], D)
        Sq = tl.where(sel[:, None], tl.sum(b_h * b_q[None, :], 1)[None, :], Sq)

    # 3) sustitucion hacia adelante: D[t] -= sum_{i<t} Gb[t,i] D[i]. En preorden el padre siempre
    #    viene antes, asi que Gb es estrictamente triangular inferior y una pasada alcanza.
    gb = tl.load(Gb + (i_n * HV + i_hv) * s_g + at[:, None] * BT + at[None, :],
                 mask=mt[:, None] & mt[None, :], other=0.0)
    cc = tl.load(C + (i_n * HV + i_hv) * s_g + at[:, None] * BT + at[None, :],
                 mask=mt[:, None] & mt[None, :], other=0.0)
    for t in range(1, T):
        sel = at == t
        gt = tl.sum(tl.where(sel[:, None], gb, 0.0), 0)             # fila t de Gb
        corr = tl.sum(gt[:, None] * D, 0)                           # [BV]
        D = tl.where(sel[:, None], (tl.sum(tl.where(sel[:, None], D, 0.0), 0) - corr)[None, :], D)

    # 4) salida: o_t = P_t (S_0 q_t) + sum_{i<=t} C[t,i] D_i
    for t in range(0, T):
        sel = at == t
        ct = tl.sum(tl.where(sel[:, None], cc, 0.0), 0)
        ot = tl.sum(tl.where(sel[:, None], Sq, 0.0), 0) * tl.sum(tl.where(sel, expP, 0.0), 0) \
            + tl.sum(ct[:, None] * D, 0)
        p_o = o + ((bos + t) * HV + i_hv) * V + o_v
        tl.store(p_o, ot.to(p_o.dtype.element_ty), mask=mv)

    # 5) el estado tras el token 0 (el ancla, siempre aceptado), que es lo que el paso siguiente
    #    toma como punto de partida:  S_0' = P_0 S_0 + D_0 k_0^T
    b_k0 = tl.load(k + (bos * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
    if IS_L2:
        b_k0 = b_k0 * tl.rsqrt(tl.sum(b_k0 * b_k0) + 1e-6)
    sel0 = at == 0
    d0 = tl.sum(tl.where(sel0[:, None], D, 0.0), 0)
    b_h = b_h * tl.sum(tl.where(sel0, expP, 0.0), 0) + d0[:, None] * b_k0[None, :]
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)


@triton.jit(do_not_specialize=["N"])
def _k_spec_cerrada8(q, k, v, o, h, stride_h, cu, sidx, nacc, slots, cinta, camino,
                     CM, aux, k8_p, q8_p, esc_p, s_g, s_a, s_k8, s_e, scale, N,
                     H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, T: tl.constexpr,
                     TM: tl.constexpr, ROW: tl.constexpr, IS_L2: tl.constexpr):
    """Forma cerrada con los productos pesados en INT8, sobre tensor cores.

    El estado S_0 queda FIJO durante todo el arbol (esa es la gracia de la forma cerrada), asi que
    se cuantiza UNA vez y se usa en 2T productos -> ``mma.s8``. Y como el estado que se escribe al
    slot se reconstruye desde el S_0 original, el error no se acumula entre pasos.

    Tres dots y ningun lazo sobre los nodos:
        Sk = S8 · k8^T      Sq = S8 · q8^T      (int8, tensor cores)
        O  = P (S_0 q) + R · (C M)^T            (M = (I + diag(beta) G)^-1, ya fusionada en el prep)

    El token 0 (el ancla) va en fp: es el UNICO cuyo d_t entra en el estado que queda escrito, y
    ademas la fila 0 de M es e_0, o sea d_0 = R_0 sin resolver nada. Asi el estado propagado queda
    EXACTO (3e-17 medido en tests/proto/gdn_forma_cerrada.py) y el int8 solo toca las salidas de
    los nodos de draft, que es donde ya se vive con 0,5-0,65% (SK-18).
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    if tl.load(cu + i_n + 1).to(tl.int64) - bos != T:
        return
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0:
        return
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    at = tl.arange(0, BT)
    mk = o_k < K
    mv = o_v < V
    mt = at < T
    mh = mv[:, None] & mk[None, :]
    p_h = h + s * stride_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h, mask=mh, other=0).to(tl.float32)
    r = tl.load(nacc + i_n).to(tl.int64) - 1
    slot = tl.load(slots + i_n).to(tl.int64)
    for jj in range(0, r):
        row = cinta + (slot * TM + tl.load(camino + slot * TM + jj).to(tl.int64)) * ROW
        rk = tl.load(row + i_h * K + o_k, mask=mk, other=0).to(tl.float32)
        rv = tl.load(row + H * K + i_hv * V + o_v, mask=mv, other=0).to(tl.float32)
        rg = tl.load(row + H * K + HV * V + i_hv).to(tl.float32)
        rb = tl.load(row + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _regla_delta(b_h, rk, rv, rg, rb)

    ab = (i_n * HV + i_hv) * s_a
    P = tl.load(aux + ab + at * 2, mask=mt, other=0.0)
    bet = tl.load(aux + ab + at * 2 + 1, mask=mt, other=0.0)
    expP = tl.exp(P)
    # S_0 a int8 por fila: una sola vez, para 2T productos
    ss = tl.max(tl.abs(b_h), 1) / 127.0
    S8 = _redondear(b_h / ss[:, None]).to(tl.int8)
    bk8 = (i_n * H + i_h) * s_k8
    idx8 = at[:, None] * K + o_k[None, :]
    m8 = mt[:, None] & mk[None, :]
    k8 = tl.load(k8_p + bk8 + idx8, mask=m8, other=0)
    q8 = tl.load(q8_p + bk8 + idx8, mask=m8, other=0)
    be = (i_n * H + i_h) * s_e
    ek = tl.load(esc_p + be + at * 2, mask=mt, other=0.0)
    eq = tl.load(esc_p + be + at * 2 + 1, mask=mt, other=0.0)
    Sk = tl.dot(S8, tl.trans(k8), out_dtype=tl.int32).to(tl.float32) * ss[:, None] * ek[None, :]
    Sq = tl.dot(S8, tl.trans(q8), out_dtype=tl.int32).to(tl.float32) * ss[:, None] * eq[None, :]
    # el ancla, en fp
    b_k0 = tl.load(k + (bos * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
    b_q0 = tl.load(q + (bos * H + i_h) * K + o_k, mask=mk, other=0).to(tl.float32)
    if IS_L2:
        b_k0 = b_k0 * tl.rsqrt(tl.sum(b_k0 * b_k0) + 1e-6)
        b_q0 = b_q0 * tl.rsqrt(tl.sum(b_q0 * b_q0) + 1e-6)
    b_q0 = b_q0 * scale
    es0 = at[None, :] == 0
    Sk = tl.where(es0, tl.sum(b_h * b_k0[None, :], 1)[:, None], Sk)
    Sq = tl.where(es0, tl.sum(b_h * b_q0[None, :], 1)[:, None], Sq)

    p_v = v + ((bos + at[None, :]) * HV + i_hv) * V + o_v[:, None]
    mvt = mv[:, None] & mt[None, :]
    vT = tl.load(p_v, mask=mvt, other=0).to(tl.float32)
    R = bet[None, :] * (vT - expP[None, :] * Sk)                         # [BV, BT]
    cm = tl.load(CM + (i_n * HV + i_hv) * s_g + at[:, None] * BT + at[None, :],
                 mask=mt[:, None] & mt[None, :], other=0.0)
    O = expP[None, :] * Sq + tl.dot(R, tl.trans(cm), input_precision="ieee")
    p_o = o + ((bos + at[None, :]) * HV + i_hv) * V + o_v[:, None]
    tl.store(p_o, O.to(p_o.dtype.element_ty), mask=mvt)
    # estado tras el ancla: la fila 0 de M es e_0, asi que d_0 = R_0 y no hay que resolver nada
    d0 = tl.sum(tl.where(es0, R, 0.0), 1)
    b_h = b_h * tl.sum(tl.where(at == 0, expP, 0.0), 0) + d0[:, None] * b_k0[None, :]
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mh)


@triton.jit(do_not_specialize=["N"])
def _k_escribir(A_log, a, b, dt_bias, beta_sp, threshold, k, v, cu, sidx, slots, cinta, N,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                BK: tl.constexpr, BHV: tl.constexpr, BH: tl.constexpr,
                TM: tl.constexpr, ROW: tl.constexpr, IS_L2: tl.constexpr):
    """Filas de cinta de los tokens 1..T-1. Lanzamiento APARTE de ``_k_spec``:
    adentro, los programas que comparten cabeza k (y los escalares por cabeza v)
    pisaban lo que otro programa todavia estaba reproduciendo."""
    i_n = tl.program_id(0)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    T = eos - bos
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0 or T <= 1:
        return
    slot = tl.load(slots + i_n).to(tl.int64)
    ok = tl.arange(0, BK)
    mk = ok < K
    ohv = tl.arange(0, BHV)
    mhv = ohv < HV * V
    oh = tl.arange(0, BH)
    mh = oh < HV
    Al = tl.load(A_log + oh, mask=mh, other=0).to(tl.float32)
    db = tl.load(dt_bias + oh, mask=mh, other=0).to(tl.float32)
    for t in range(1, T):
        src = bos + t
        row = cinta + (slot * TM + t - 1) * ROW
        for hh in range(0, H):
            kk = tl.load(k + (src * H + hh) * K + ok, mask=mk, other=0).to(tl.float32)
            if IS_L2:
                kk = kk * tl.rsqrt(tl.sum(kk * kk) + 1e-6)
            tl.store(row + hh * K + ok, kk.to(tl.float32), mask=mk)
        tl.store(row + H * K + ohv,
                 tl.load(v + src * HV * V + ohv, mask=mhv, other=0).to(tl.float32), mask=mhv)
        x = tl.load(a + src * HV + oh, mask=mh, other=0).to(tl.float32) + db
        sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        tl.store(row + H * K + HV * V + oh, (-tl.exp(Al) * sp).to(tl.float32), mask=mh)
        bb = tl.sigmoid(tl.load(b + src * HV + oh, mask=mh, other=0).to(tl.float32))
        tl.store(row + H * K + HV * V + HV + oh, bb.to(tl.float32), mask=mh)


@triton.jit(do_not_specialize=["N"])
def _k_escribir_par(A_log, a, b, dt_bias, beta_sp, threshold, k, v, cu, sidx, slots, cinta, N,
                    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr,
                    TM: tl.constexpr, ROW: tl.constexpr, IS_L2: tl.constexpr):
    """Lo mismo que ``_k_escribir``, repartido en una grilla (N, TM, H + HV).

    ``_k_escribir`` corre UN programa por pedido que recorre en serie los T-1 tokens y las H
    cabezas k: con un pedido son 34 us de pura latencia en 1 bloque para 82 SM, 48 veces por
    paso (7% del paso de decode). Cada fila de la cinta y cada tramo de la fila son
    independientes, asi que acá cada programa escribe un tramo de una fila:

      programa (n, t-1, p)   p <  H : la cabeza k ``p`` normalizada (l2), K floats
                             p >= H : la cabeza v ``p-H`` (V floats) y sus dos escalares g y beta

    Mismas cuentas y en el mismo orden que ``_k_escribir`` (la norma l2 reduce un vector de BK
    con el mismo num_warps), asi que la cinta sale identica bit a bit.
    """
    i_n = tl.program_id(0)
    t = tl.program_id(1) + 1
    p = tl.program_id(2)
    bos = tl.load(cu + i_n).to(tl.int64)
    eos = tl.load(cu + i_n + 1).to(tl.int64)
    s = tl.load(sidx + i_n).to(tl.int64)
    if s <= 0 or t >= eos - bos:
        return
    slot = tl.load(slots + i_n).to(tl.int64)
    src = bos + t
    row = cinta + (slot * TM + t - 1) * ROW
    if p < H:
        ok = tl.arange(0, BK)
        mk = ok < K
        kk = tl.load(k + (src * H + p) * K + ok, mask=mk, other=0).to(tl.float32)
        if IS_L2:
            kk = kk * tl.rsqrt(tl.sum(kk * kk) + 1e-6)
        tl.store(row + p * K + ok, kk.to(tl.float32), mask=mk)
    else:
        hv = p - H
        ov = tl.arange(0, BV)
        mv = ov < V
        tl.store(row + H * K + hv * V + ov,
                 tl.load(v + (src * HV + hv) * V + ov, mask=mv, other=0).to(tl.float32), mask=mv)
        Al = tl.load(A_log + hv).to(tl.float32)
        x = tl.load(a + src * HV + hv).to(tl.float32) + tl.load(dt_bias + hv).to(tl.float32)
        sp = tl.where(beta_sp * x <= threshold, (1 / beta_sp) * tl.log(1 + tl.exp(beta_sp * x)), x)
        tl.store(row + H * K + HV * V + hv, (-tl.exp(Al) * sp).to(tl.float32))
        bb = tl.sigmoid(tl.load(b + src * HV + hv).to(tl.float32))
        tl.store(row + H * K + HV * V + HV + hv, bb.to(tl.float32))


_ESCRIBIR_MODO = os.environ.get("GENESIS_PN122_ESCRIBIR_PAR", "0").strip().lower()
_ESCRIBIR_MODO = "par" if _ESCRIBIR_MODO in _TRUTHY else ("ptx" if _ESCRIBIR_MODO == "ptx" else "serie")
_k_cinta_ptx: dict = {}
_f32_cache: dict = {}
_avisos_cinta: set = set()


def _f32(t):
    if t.dtype == torch.float32:
        return t
    c = _f32_cache.get(t.data_ptr())
    if c is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        c = _f32_cache[t.data_ptr()] = t.float().contiguous()
    return c


def _avisar_cinta(motivo):
    if motivo not in _avisos_cinta:
        _avisos_cinta.add(motivo)
        log.warning("[PN122 cinta] PTX no aplica, uso Triton: %s", motivo)


def escribir_cinta(A_log, a, b, dt_bias, k, v, cu_seqlens, sidx, slots, cinta, N, H, HV, K, V,
                   par=None):
    """Escribe las filas 1..T-1 de la cinta. Tres caminos, mismo resultado bit a bit:

    * ``serie`` (default): ``_k_escribir``, un programa por pedido (34 us por capa con un pedido);
    * ``par`` (GENESIS_PN122_ESCRIBIR_PAR=1): ``_k_escribir_par``, Triton en grilla (2,1 us);
    * ``ptx`` (GENESIS_PN122_ESCRIBIR_PAR=ptx): ``kernels/cuda/pn122_cinta.cu``, un bloque por fila
      con cargas vectoriales y la reduccion sin shared (ver el encabezado del .cu).

    ``par`` fuerza el modo (``False``/``True``/``"ptx"``), para los tests."""
    TM, ROW = cinta.shape[1], cinta.shape[2]
    BK = triton.next_power_of_2(K)
    modo = _ESCRIBIR_MODO if par is None else ("ptx" if par == "ptx" else ("par" if par else "serie"))
    if modo == "ptx":
        # A_log y dt_bias: el PTX los lee en fp32. Si vienen en otro tipo, copia fp32 (exacta: Triton
        # hacia la misma conversion), cacheada por puntero y creada FUERA de la captura de un grafo.
        A32, db32 = _f32(A_log), _f32(dt_bias)
        motivo = None
        if not (K == 128 and V == 128 and H <= 8 and HV <= 32):
            motivo = f"formas K={K} V={V} H={H} HV={HV}"
        elif A32 is None or db32 is None:
            motivo = "A_log/dt_bias sin copia fp32 (primera vez dentro de una captura)"
        elif not (k.dtype == v.dtype == a.dtype == b.dtype == torch.float16 and cinta.dtype == torch.float32):
            motivo = f"dtypes k={k.dtype} v={v.dtype} a={a.dtype} b={b.dtype} cinta={cinta.dtype}"
        elif not (stride_ab(a, b) is not None and k.stride(-1) == 1 and v.stride(-1) == 1
                  and k.stride(-2) == K and v.stride(-2) == V and k.stride(-3) % 4 == 0 and v.stride(-3) % 4 == 0):
            motivo = f"layout k {tuple(k.stride())} v {tuple(v.stride())} (el PTX lee k/v con stride de token)"
        if motivo is not None:
            _avisar_cinta(motivo)
            modo = "par"
        else:
            A_log, dt_bias = A32, db32
    if modo == "ptx":
        sab = stride_ab(a, b)
        clave = (H, HV, TM, sab)
        kern = _k_cinta_ptx.get(clave)
        if kern is None:
            from vllm._genesis.kernels.ptx_lab import Kernel
            kern = _k_cinta_ptx[clave] = Kernel("pn122_cinta.cu", "pn122_cinta",
                                                defs=[f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}", f"-DSAB={sab}"],
                                                warps=8)
        import ctypes
        kern.lanzar((N, TM, 1 + (HV * V // 4 + 255) // 256),
                    [A_log, a, b, dt_bias, 1.0, 20.0, k, v, ctypes.c_longlong(k.stride(-3)),
                     ctypes.c_longlong(v.stride(-3)), cu_seqlens, sidx, slots, cinta, N])
    elif modo == "par":
        k, v, a, b = k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous()
        _k_escribir_par[(N, TM, H + HV)](
            A_log, a, b, dt_bias, 1.0, 20.0, k, v, cu_seqlens, sidx, slots, cinta, N,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=triton.next_power_of_2(V), TM=TM, ROW=ROW,
            IS_L2=True, num_warps=4, num_stages=1)
    else:
        k, v, a, b = k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous()
        _k_escribir[(N,)](
            A_log, a, b, dt_bias, 1.0, 20.0, k, v, cu_seqlens, sidx, slots, cinta, N,
            H=H, HV=HV, K=K, V=V, BK=BK, BHV=triton.next_power_of_2(HV * V),
            BH=triton.next_power_of_2(HV), TM=TM, ROW=ROW, IS_L2=True,
            num_warps=4, num_stages=3)


_DIAG_CRESTA = os.environ.get("GENESIS_DIAG_GDN_CRESTA", "0").strip().lower() in _TRUTHY
_cresta_n = 0


def _diag_cresta(layer, ssm_state, sidx) -> None:
    """Cresta (max/mediana de |x|) del estado GDN REAL, que es lo que decide si rotar antes de
    cuantizar a int8 sirve o no (ver [[estrategia-cuantizacion-por-capa]]). Mide por FILA (eje K,
    que es como se cuantiza para el dot) y por COLUMNA (un canal con outliers estira todas las
    filas), y ademas la cresta del estado ROTADO con Hadamard, que es la alternativa.

    Sincroniza con la GPU: solo con GENESIS_DIAG_GDN_CRESTA=1 y cada 200 pasos.
    """
    global _cresta_n
    _cresta_n += 1
    if _cresta_n % 200 != 0 or _cresta_n > 1400:
        return
    idx = sidx[sidx > 0].long()
    if idx.numel() == 0:
        return
    S = ssm_state[idx[:2]].float()                      # [n, HV, V, K]
    K = S.shape[-1]
    H = torch.ones(1, 1, device=S.device)
    while H.shape[0] < K:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / K ** 0.5
    def cr(X, eje):
        A = X.abs().transpose(-1, -2) if eje == "col" else X.abs()
        return (A.amax(-1) / A.median(-1).values.clamp(min=1e-30)).median().item()
    SR = S @ H
    log.warning("[GDN cresta] %s paso~%d  fila %.1f  columna %.1f  | rotado: fila %.1f columna %.1f"
                "  | max|S| %.3g  mediana|S| %.3g",
                getattr(layer, "prefix", "?"), _cresta_n, cr(S, "fila"), cr(S, "col"),
                cr(SR, "fila"), cr(SR, "col"), S.abs().max().item(), S.abs().median().item())


_ARBOL_PTX = os.environ.get("GENESIS_PN122_ARBOL_PTX", "0").strip().lower() in _TRUTHY
_ARBOL_RPW = 4                      # filas por warp (barrido 27-09: 4 gana a 8 y a 2)


def qkv_con_stride() -> bool:
    """PN54 pregunta esto en ``rearrange_mixed_qkv``: con el PTX del arbol, q/k/v se pasan como vistas con
    stride de token (sin copias). Si despues el paso no va por el PTX, ``spec_update`` hace las copias."""
    return _ACTIVO and _ARBOL_PTX
_k_arbol_ptx: dict = {}


def _arbol_ptx_motivo(layer, A_log, dt_bias, q, k, v, a, b, h, H, HV, K, V):
    """None si gdn_arbol.cu sirve para estos tensores; si no, el motivo (va al log una vez)."""
    T1 = layer.num_spec + 1
    m = None
    if not (K == 128 and V == 128 and HV % H == 0 and T1 <= 32 and V % (4 * _ARBOL_RPW) == 0):
        m = f"formas K={K} V={V} H={H} HV={HV} T={T1}"
    elif not (q.dtype == k.dtype == v.dtype == a.dtype == b.dtype == torch.float16):
        m = f"dtypes q={q.dtype} a={a.dtype}"
    elif h.dtype not in (torch.float16, torch.float32) or tuple(h.stride()[1:]) != (V * K, K, 1):
        m = f"estado {h.dtype} {tuple(h.stride())}"
    elif not all(t.stride(-1) == 1 and t.stride(-2) == K and t.stride(-3) % 4 == 0 and t.data_ptr() % 8 == 0
                 for t in (q, k, v)):
        m = f"layout q {tuple(q.stride())} k {tuple(k.stride())} v {tuple(v.stride())}"
    elif _f32(A_log) is None or _f32(dt_bias) is None:
        m = "A_log/dt_bias sin copia fp32 (primera vez dentro de una captura)"
    if m is not None and m not in _avisos_cinta:
        _avisos_cinta.add(m)
        log.warning("[PN122 arbol] PTX no aplica, uso Triton: %s", m)
    return m


def stride_ab(a, b):
    """Stride de fila comun de a y b [T, HV] si los PTX pueden leerlas asi (ultima dim contigua), si no None."""
    if a.dim() != 2 or b.dim() != 2 or a.stride(-1) != 1 or b.stride(-1) != 1 or a.stride(0) != b.stride(0):
        return None
    return int(a.stride(0)) if a.shape[0] > 1 else int(a.shape[1])


def _lanzar_arbol_ptx(layer, A_log, a, b, dt_bias, q, k, v, o, h, cu, sidx, nacc, slots, cinta, N,
                      H, HV, K, V, TM):
    import ctypes
    T1 = layer.num_spec + 1
    hdt = 0 if h.dtype == torch.float16 else 1
    sab = stride_ab(a, b) or HV
    clave = (H, HV, TM, T1, hdt, sab)
    kern = _k_arbol_ptx.get(clave)
    if kern is None:
        from vllm._genesis.kernels.ptx_lab import Kernel
        kern = _k_arbol_ptx[clave] = Kernel(
            "gdn_arbol.cu", "gdn_arbol", warps=4,
            defs=[f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}", f"-DTMAX={T1}", f"-DHDT={hdt}", f"-DRPW={_ARBOL_RPW}",
                  f"-DSAB={sab}"])
    L = ctypes.c_longlong
    kern.lanzar((V // (4 * _ARBOL_RPW), N * HV),
                [_f32(A_log), a, b, _f32(dt_bias), q, k, v, L(q.stride(-3)), L(k.stride(-3)), L(v.stride(-3)),
                 o, h, L(h.stride(0)), cu, sidx, nacc, slots, cinta, _camino_gpu, _anc_gpu, K ** -0.5, N])


def spec_update(layer, A_log, a, b, dt_bias, q, k, v, ssm_state, cu_seqlens,
                spec_state_indices, num_accepted_tokens, slots, o_dest=None):
    """Reemplazo de ``fused_sigmoid_gating_delta_rule_update`` en el camino spec.

    Devuelve lo mismo que upstream: ``(o [1, T, HV, V], ssm_state)``. Con ``o_dest`` (PN163) la salida se
    escribe ahi (una vista de core_attn_out) y vLLM saltea la copia.
    """
    if _DIAG_CRESTA:
        _diag_cresta(layer, ssm_state, spec_state_indices[:, 0]
                     if spec_state_indices.ndim == 2 else spec_state_indices)
    _, Ttot, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.shape[0] - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    cinta = layer._g122_cinta
    TM, ROW = cinta.shape[1], cinta.shape[2]
    sidx = spec_state_indices[:, 0].contiguous() if spec_state_indices.ndim == 2 \
        else spec_state_indices
    nacc = num_accepted_tokens
    # PTX del arbol (gdn_arbol.cu): lee q/k/v con su stride de token, asi que no hacen falta las tres
    # copias .contiguous() (6,6 us por capa en el perfil del 25-09).
    arbol_ptx = (_ARBOL_PTX and paso_arbol_activo() and _camino_gpu is not None and not _CERRADA
                 and _arbol_ptx_motivo(layer, A_log, dt_bias, q, k, v, a, b, ssm_state, H, HV, K, V) is None)
    # a y b: con PN164 llegan como vistas de la salida de in_proj (stride de fila > HV). Los dos PTX (arbol y
    # cinta) las leen con ese stride; cualquier otro camino las quiere contiguas.
    if not (arbol_ptx and _ESCRIBIR_MODO == "ptx" and stride_ab(a, b) is not None):
        a, b = a.contiguous(), b.contiguous()
    if not arbol_ptx:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    _bits = sync_bits()
    if _bits & 64:
        torch.cuda.synchronize()
    _pref = getattr(layer, "prefix", "")
    sombra = (_bits & 32) and any(f".layers.{i}." in _pref for i in (0, 1, 2, 44))
    if sombra:
        _sombra_antes(layer, ssm_state, sidx, nacc, slots, N)
    if (o_dest is not None and tuple(o_dest.shape) == (1, Ttot, HV, V) and o_dest.dtype == q.dtype
            and o_dest.is_contiguous()):
        o = o_dest
    else:
        o = q.new_empty(1, Ttot, HV, V)
    if paso_arbol_activo() and _camino_gpu is not None and _CERRADA:
        # Contrato: solo se llega aca si el runner ya verifico que el lote es UNIFORME de T = K+1
        # tokens (lo exige la mascara de arbol de PN131). El kernel igual lo comprueba y se saltea
        # el pedido que no cumpla, antes que escribir basura.
        T1 = layer.num_spec + 1
        BT = triton.next_power_of_2(T1)
        c = _cerr.get(q.device.index)
        if c is None or c[0].shape[0] < N or c[0].shape[1] != HV or c[0].shape[2] != BT * BT:
            nmax = max(N, _n_slots or N)
            c = (torch.empty(nmax, HV, BT * BT, dtype=torch.float32, device=q.device),
                 torch.empty(nmax, HV, BT * BT, dtype=torch.float32, device=q.device),
                 torch.empty(nmax, HV, BT * 2, dtype=torch.float32, device=q.device),
                 torch.empty(nmax, H, BT * K, dtype=torch.int8, device=q.device),
                 torch.empty(nmax, H, BT * K, dtype=torch.int8, device=q.device),
                 torch.empty(nmax, H, BT * 2, dtype=torch.float32, device=q.device))
            _cerr[q.device.index] = c
        Gb, Cm, aux, k8, q8, esc = c
        _k_prep_cerrada[(N, HV)](
            A_log, a, b, dt_bias, 1.0, 20.0, k, q, cu_seqlens, sidx, _anc_gpu,
            Gb, Cm, aux, k8, q8, esc, K ** -0.5, Gb.stride(1), aux.stride(1), k8.stride(1),
            esc.stride(1), H=H, HV=HV, K=K, BK=BK, BT=BT, T=T1, IS_L2=True, INT8=_INT8,
            num_warps=4, num_stages=2)
        if _INT8:
            _k_spec_cerrada8[(triton.cdiv(V, BV), N * HV)](
                q, k, v, o, ssm_state, ssm_state.stride(0), cu_seqlens, sidx, nacc, slots, cinta,
                _camino_gpu, Cm, aux, k8, q8, esc, Gb.stride(1), aux.stride(1), k8.stride(1),
                esc.stride(1), K ** -0.5, N,
                H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, BT=BT, T=T1, TM=TM, ROW=ROW, IS_L2=True,
                num_warps=4, num_stages=3)
        else:
            _k_spec_cerrada[(triton.cdiv(V, BV), N * HV)](
                q, k, v, o, ssm_state, ssm_state.stride(0), cu_seqlens, sidx, nacc, slots, cinta,
                _camino_gpu, Gb, Cm, aux, Gb.stride(1), aux.stride(1), K ** -0.5, N,
                H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, BT=BT, T=T1, TM=TM, ROW=ROW, IS_L2=True,
                num_warps=4, num_stages=3)
    elif arbol_ptx:
        _lanzar_arbol_ptx(layer, A_log, a, b, dt_bias, q, k, v, o, ssm_state, cu_seqlens, sidx, nacc, slots,
                          cinta, N, H, HV, K, V, TM)
    elif paso_arbol_activo() and _camino_gpu is not None:
        _k_spec_arbol[(triton.cdiv(V, BV), N * HV)](
            A_log, a, b, dt_bias, 1.0, 20.0, q, k, v, o, ssm_state, ssm_state.stride(0),
            cu_seqlens, sidx, nacc, slots, cinta, _camino_gpu, _anc_gpu, K ** -0.5, N,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TM=TM, ROW=ROW, IS_L2=True,
            num_warps=4, num_stages=3)
    else:
        _k_spec[(triton.cdiv(V, BV), N * HV)](
            A_log, a, b, dt_bias, 1.0, 20.0, q, k, v, o, ssm_state, ssm_state.stride(0),
            cu_seqlens, sidx, nacc, slots, cinta, K ** -0.5, N,
            H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TM=TM, ROW=ROW, IS_L2=True,
            num_warps=4, num_stages=3)
    if _bits & 128:
        torch.cuda.synchronize()
    escribir_cinta(A_log, a, b, dt_bias, k, v, cu_seqlens, sidx, slots, cinta, N, H, HV, K, V)
    if _bits & 256:
        torch.cuda.synchronize()
    if sombra:
        _sombra_despues(layer, A_log, a, b, dt_bias, q, k, v, cu_seqlens, sidx, nacc, slots,
                        N, o, ssm_state)
    return o, ssm_state


# ─────────────── verificacion en sombra (diagnostico, capa 0, eager) ───────────────
_sombra: dict = {}
_sombra_nuevos: set = set()


def _sombra_antes(layer, ssm_state, sidx, nacc, slots, N):
    """Estado upstream: K+1 columnas por slot. La columna 0 de la sombra se toma
    del estado real SOLO en el primer paso spec del request (estado de prefill);
    despues la sombra evoluciona sola con el kernel de upstream y se compara.

    OJO CON LEER ``estado(antes)``: me hizo sacar una conclusion falsa (2026-09-19).
    Compara ``h[1 + slot*K1 + acc - 1]`` — el estado de UPSTREAM despues de acc-1 tokens —
    contra ``ssm_state[s]`` — el de PN122 despues del token 0. Esos dos solo coinciden cuando
    ``acc == 1``: para acc>1 TIENEN que diferir, porque PN122 recupera la diferencia
    reproduciendo la cinta, que es exactamente lo que el parche hace. Las muestras con acc>1
    no se pueden leer como error.

    Y aun con acc==1 el numero mezcla el error del paso con la DERIVA ACUMULADA: la sombra
    evoluciona sola desde que se siembra, asi que cualquier paso que el hook no vea (prefill,
    decode sin spec) la deja atras sin que la cinta tenga la culpa.

    Para decidir si PN122 esta bien NO alcanza ni esto ni comparar texto greedy: un error de
    1e-4 alcanza para dar vuelta un argmax y hacer divergir el texto sin que haya nada roto.
    Lo que si discrimina es la TASA DE ACEPTACION del spec decode, que se derrumbaria si el
    estado estuviera corrupto. Medido 2026-09-19 con DFlash2 K=8: 5,37 a 1k y 5,17 a 50k con
    PN122, contra 5,15-5,69 y 4,64-5,46 sin el. Dentro del rango: no hay corrupcion.
    """
    K1 = layer.num_spec + 1
    key = layer.prefix
    if key not in _sombra:
        _sombra[key] = torch.zeros((_n_slots * K1 + 1, *ssm_state.shape[1:]),
                                   dtype=ssm_state.dtype, device=ssm_state.device)
        _sombra[key + "nuevos"] = set(range(_n_slots))
    h = _sombra[key]
    nuevos = _sombra[key + "nuevos"]
    torch.cuda.synchronize()
    difs = []
    for i in range(N):
        s = int(sidx[i]); sl = int(slots[i]); acc = int(nacc[i])
        if s <= 0:
            continue
        if sl in _sombra_nuevos:
            for kk in list(_sombra):
                if kk.endswith("nuevos"):
                    _sombra[kk].add(sl)
            _sombra_nuevos.discard(sl)
        if sl in nuevos:
            nuevos.discard(sl)
            h[1 + sl * K1] = ssm_state[s]
            difs.append("init")
        else:
            esperado = h[1 + sl * K1 + acc - 1].float()
            real0 = ssm_state[s].float()
            difs.append(round(((esperado - real0).norm() / (esperado.norm() + 1e-9)).item(), 5))
    _sombra[key + "difs"] = difs


def _sombra_despues(layer, A_log, a, b, dt_bias, q, k, v, cu, sidx, nacc, slots, N, o, ssm_state):
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update as ref)
    K1 = layer.num_spec + 1
    h = _sombra[layer.prefix]
    cols = torch.zeros((N, K1), dtype=torch.int32, device=q.device)
    for i in range(N):
        sl = int(slots[i])
        cols[i] = torch.arange(1 + sl * K1, 1 + sl * K1 + K1, dtype=torch.int32)
        if int(sidx[i]) <= 0:
            cols[i] = 0
    o_ref, _ = ref(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=h,
                   inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=cols,
                   num_accepted_tokens=nacc[:N].contiguous(), use_qk_l2norm_in_kernel=True)
    torch.cuda.synchronize()
    T = int(cu[N])
    err = ((o_ref[:, :T].float() - o[:, :T].float()).norm() / (o_ref[:, :T].float().norm() + 1e-9)).item()
    log.warning("[PN122 sombra] %s N=%d acc=%s err_salida=%.2e estado(antes)=%s",
                layer.prefix.split("model.")[-1], N, nacc[:N].tolist(), err,
                _sombra.get(layer.prefix + "difs"))


# ───────────────────── materializacion para las copias align ─────────────────────

@triton.jit(do_not_specialize=["num_reqs"])
def _k_materializar(MODO_POST: tl.constexpr, nacc, state_idx, nsched, ncomp, ndraft,
                    src_col_p, bias_p, bt_ptrs, bt_stride: tl.int64, ssm_addrs, ssm_strides,
                    grupos, cinta_addrs, slots, idx_map, camino, num_reqs,
                    block_size: tl.constexpr, IDX: tl.constexpr, ARBOL: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    BK: tl.constexpr, BV: tl.constexpr, TM: tl.constexpr, ROW: tl.constexpr):
    """Escribe en ``dst`` el estado que upstream leeria de ``columna src + bias``:
    ``estado[src]`` + ``bias`` filas de cinta. Solo casos con bias > 0; el resto
    (conv, bias 0) lo sigue copiando el kernel de upstream.

    ``IDX`` (model runner v2): los arreglos por request viven indexados por slot de
    req-state (``idx_map[fila]``), la block table sigue por fila del batch, ``ncomp`` ya
    trae el computado DESPUES del paso, y el slot de cinta es ``slot de req-state + 1``
    (lo mismo que ``v2_pre`` deja en ``slots_gpu()`` para el builder de GDN)."""
    req = tl.program_id(0)
    lay = tl.program_id(1)
    if req >= num_reqs:
        return
    if IDX:
        ri = tl.load(idx_map + req)
        if ri < 0:
            return
    else:
        ri = req
    if MODO_POST:
        acc = tl.load(nacc + ri)
        src_col = tl.load(state_idx + ri)
        if IDX:
            nuevo = tl.load(ncomp + ri)
            running = nuevo - acc + 1
        else:
            running = tl.load(ncomp + ri) + tl.load(nsched + ri) - tl.load(ndraft + ri)
            nuevo = running + acc - 1
        alineado = (nuevo // block_size) * block_size
        if alineado < running:
            return
        bias = alineado - running
        dst_col = alineado // block_size - 1
    else:
        src_col = tl.load(src_col_p + ri)
        dst_col = tl.load(state_idx + ri)
        if src_col < 0 or src_col == dst_col:
            return
        bias = tl.load(bias_p + ri)
    if bias <= 0 or src_col < 0 or dst_col < 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    bt = tl.load(bt_ptrs + g).to(tl.pointer_type(tl.int32)) + req * bt_stride
    sb = tl.load(bt + src_col).to(tl.int64)
    dbk = tl.load(bt + dst_col).to(tl.int64)
    if sb <= 0 or dbk <= 0:   # 0 es el bloque nulo de vLLM: ni leerlo ni escribirlo
        return
    base = tl.load(ssm_addrs + lay)
    stride = tl.load(ssm_strides + lay)
    src = (base + sb * stride).to(tl.pointer_type(tl.float16))
    dst = (base + dbk * stride).to(tl.pointer_type(tl.float16))
    if IDX:
        slot = (ri + 1).to(tl.int64)
    else:
        slot = tl.load(slots + req).to(tl.int64)
    tape = tl.load(cinta_addrs + lay).to(tl.pointer_type(tl.float32)) + slot * TM * ROW
    o_k = tl.arange(0, BK)
    mk = o_k < K
    for hv in range(0, HV):
        ih = hv // (HV // H)
        for c in range(0, V, BV):
            o_v = c + tl.arange(0, BV)
            mv = o_v < V
            mh = mv[:, None] & mk[None, :]
            b_h = tl.load(src + hv * V * K + o_v[:, None] * K + o_k[None, :], mask=mh,
                          other=0).to(tl.float32)
            for j in range(0, bias):
                if ARBOL:   # la j-esima fila aceptada, no la fila j
                    row = tape + tl.load(camino + slot * TM + j).to(tl.int64) * ROW
                else:
                    row = tape + j * ROW
                rk = tl.load(row + ih * K + o_k, mask=mk, other=0).to(tl.float32)
                rv = tl.load(row + H * K + hv * V + o_v, mask=mv, other=0).to(tl.float32)
                rg = tl.load(row + H * K + HV * V + hv).to(tl.float32)
                rb = tl.load(row + H * K + HV * V + HV + hv).to(tl.float32)
                b_h *= tl.exp(rg)
                b_d = (rv - tl.sum(b_h * rk[None, :], 1)) * rb
                b_h += b_d[:, None] * rk[None, :]
            tl.store(dst + hv * V * K + o_v[:, None] * K + o_k[None, :],
                     b_h.to(tl.float16), mask=mh)


class _Meta:
    capas: list = []
    ssm_addrs = None
    ssm_strides = None
    grupos = None
    cinta_addrs = None
    dims = None


_meta = _Meta()


def _init_meta(ctx, kv_cache_config, forward_context) -> bool:
    if _meta.ssm_addrs is not None:
        return True
    addrs, strides, grupos, cintas, capas = [], [], [], [], []
    for g_local, gid in enumerate(ctx.mamba_group_ids):
        for name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context[name]
            cinta = getattr(layer, "_g122_cinta", None)
            if cinta is None:
                return False
            ssm = layer.kv_cache[1]
            assert ssm.dtype == torch.float16, "PN122 asume ssm fp16"
            addrs.append(ssm.data_ptr())
            strides.append(ssm.stride(0) * ssm.element_size())
            grupos.append(g_local)
            cintas.append(cinta.data_ptr())
            capas.append(layer)
    dev = capas[0]._g122_cinta.device
    _meta.ssm_addrs = torch.tensor(addrs, dtype=torch.int64, device=dev)
    _meta.ssm_strides = torch.tensor(strides, dtype=torch.int64, device=dev)
    _meta.grupos = torch.tensor(grupos, dtype=torch.int32, device=dev)
    _meta.cinta_addrs = torch.tensor(cintas, dtype=torch.int64, device=dev)
    _meta.capas = capas
    l0 = capas[0]
    _meta.dims = (*dims(l0), l0.num_spec, fila(l0))
    return True


def capas_gdn() -> list:
    """Capas GDN en el orden de ``_meta.grupos`` (vacio hasta la primera migracion de estado)."""
    return _meta.capas if _meta.ssm_addrs is not None else []


def grupos_gdn():
    return _meta.grupos


def _lanzar(modo_post, ctx, num_reqs, nacc, state_idx, nsched, ncomp, ndraft, src_col, bias,
            idx_map=None):
    H, HV, K, V, TM, ROW = _meta.dims
    _k_materializar[(num_reqs, len(_meta.capas))](
        modo_post, nacc, state_idx, nsched, ncomp, ndraft, src_col, bias,
        ctx.block_table_ptrs, ctx.block_table_stride_req, _meta.ssm_addrs,
        _meta.ssm_strides, _meta.grupos, _meta.cinta_addrs, slots_gpu(),
        idx_map if idx_map is not None else slots_gpu(),
        _camino_gpu if _camino_gpu is not None else slots_gpu(), num_reqs,
        block_size=ctx.block_size, IDX=idx_map is not None,
        ARBOL=_ARBOL and _camino_gpu is not None, H=H, HV=HV, K=K, V=V,
        BK=triton.next_power_of_2(K), BV=min(triton.next_power_of_2(V), 32),
        TM=TM, ROW=ROW, num_warps=4, num_stages=3)


def materializar_post(ctx, kv_cache_config, forward_context, num_reqs, nacc, state_idx_buf,
                      nsched_buf, ncomp_buf, ndraft_buf) -> None:
    """Recibe los CpuGpuBuffer de upstream: la decision de si ALGUN request puede
    cruzar un borde con bias > 0 se toma en CPU con los mismos valores que se
    subieron a la GPU, y si ninguno puede, no se lanza el kernel (ahorra un
    lanzamiento de Triton por paso de decode)."""
    bits = sync_bits()
    if bits & 4:  # diagnostico: saltear la materializacion post
        return
    if num_reqs == 0:
        return
    import numpy as np
    comp = ncomp_buf.np[:num_reqs].astype(np.int64)
    sched = nsched_buf.np[:num_reqs].astype(np.int64)
    draft = ndraft_buf.np[:num_reqs].astype(np.int64)
    running = comp + sched - draft
    bs = ctx.block_size
    # bias = alineado - running > 0 exige un borde en (running, running + draft].
    if not ((((running + draft) // bs) * bs) > running).any():
        return
    log.warning("[PN122] materializar_post DISPARADO: running=%s draft=%s bs=%d",
                running[:num_reqs].tolist(), draft[:num_reqs].tolist(), bs)
    if not _init_meta(ctx, kv_cache_config, forward_context):
        return
    _lanzar(True, ctx, num_reqs, nacc, state_idx_buf.gpu, nsched_buf.gpu, ncomp_buf.gpu,
            ndraft_buf.gpu, nacc, nacc)


def materializar_pre(ctx, kv_cache_config, forward_context, num_reqs, state_idx_buf,
                     src_col_buf, bias_buf) -> None:
    """Idem: solo se lanza si algun request migra de bloque con bias > 0."""
    if num_reqs == 0:
        return
    src = src_col_buf.np[:num_reqs]
    bias = bias_buf.np[:num_reqs]
    if not ((src >= 0) & (bias > 0)).any():
        return
    log.warning("[PN122] materializar_pre DISPARADO: src=%s bias=%s",
                src[:num_reqs].tolist(), bias[:num_reqs].tolist())
    if not _init_meta(ctx, kv_cache_config, forward_context):
        return
    g = state_idx_buf.gpu
    _lanzar(False, ctx, num_reqs, g, g, g, g, g, src_col_buf.gpu, bias_buf.gpu)


# ───────────────────────── model runner v2 (mamba_hybrid.py) ─────────────────────────
# DFlash2 obliga al runner v2, que no pasa por preprocess_mamba/postprocess_mamba_all: migra
# el estado desde MambaHybridModelState.{pre,post}process_state, todo en GPU e indexado por
# slot de req-state. Sin estos dos ganchos el salteo de bias>0 en _copy_mamba_state_block
# dejaba el estado viejo en cada borde de bloque y la salida degeneraba (2026-09-20).
# No hay decision en CPU: bajo async scheduling los espejos numpy son optimistas, asi que el
# kernel se lanza siempre y sale solo por request (mismo criterio que upstream en v2).

def v2_pre(ctx, kv_cache_config, forward_context, num_reqs, idx_mapping, state_idx, src_col,
           src_off) -> None:
    """Antes de ``ctx.run_fused_precopy``. Fija el slot de cinta de cada fila del batch
    (= slot de req-state + 1; el 0 es relleno) y materializa las migraciones con bias>0."""
    if not _ACTIVO or _slots_gpu is None or num_reqs == 0:
        return
    n = min(num_reqs, _slots_gpu.shape[0])
    _slots_gpu[:n].copy_(idx_mapping[:n])
    _slots_gpu[:n].add_(1)
    if not _init_meta(ctx, kv_cache_config, forward_context):
        return
    _lanzar(False, ctx, num_reqs, state_idx, state_idx, state_idx, state_idx, state_idx,
            src_col, src_off, idx_map=idx_mapping)


def v2_post(ctx, num_reqs, nacc, state_idx, ncomp_nuevo, idx_mapping) -> None:
    """Antes de ``ctx.run_fused_postprocess_align`` (que pisa ``nacc`` con 1)."""
    if not _ACTIVO or num_reqs == 0 or _meta.ssm_addrs is None:
        return
    if sync_bits() & 4:
        return
    _lanzar(True, ctx, num_reqs, nacc, state_idx, ncomp_nuevo, ncomp_nuevo, ncomp_nuevo,
            nacc, nacc, idx_map=idx_mapping)


__all__ = ["v2_pre", "v2_post", "activo", "num_speculative_blocks", "actualizar_slots", "enlazar",
           "spec_update", "materializar_post", "materializar_pre", "slots_gpu", "arbol", "camino_gpu",
           "fijar_ancestros", "ancestros_gpu", "ancestros3_gpu", "paso_en_arbol", "paso_arbol_activo", "capas_gdn"]
