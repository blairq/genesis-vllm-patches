# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N80 — GDN `h` budget probe + headroom projection.

================================================================
QUÉ RESUELVE
================================================================

Dimensionar `--max-num-seqs` y `--max-num-batched-tokens` en rigs de
24 GB era a ciegas: el operador sube un número, corre carga, y el
engine OOMea o no. Los logs de vLLM no ayudan porque muestran
"Available KV cache memory" — que NO es la memoria que falta cuando
revienta la atención lineal.

PN80 emite, desde el punto exacto de la asignación, el cálculo real y
la proyección de cuántos tokens más entran con la VRAM que queda.

================================================================
LA FÓRMULA (medida, no estimada)
================================================================

En `chunk_delta_h.py::chunk_gated_delta_rule_fwd_h`:

    h = k.new_empty(B, NT, H, V, K)

con NT = ceil(T / FLA_CHUNK_SIZE). Por tanto:

    h_bytes = B * ceil(T/64) * H * V * K * itemsize

Verificado con 200 llamadas reales sobre Qwen3.8-27B W8A16 en 2× RTX
3090 (TP=2, H=24 tras el split de los 48 heads GDN, V=K=128, fp16):

    T=1650  nseq=2  h=19.5 MiB   -> 12.1 KiB/token
    T=3120  nseq=1  h=36.8 MiB   -> 12.1 KiB/token
    T=3200  nseq=1  h=37.5 MiB   -> 12.0 KiB/token
    T=3760  nseq=2  h=44.2 MiB   -> 12.0 KiB/token

Dos invariantes confirmados y que el log hace explícitos:

1. `h` se asigna por BATCH COMPLETO, no por secuencia. T=1650 con
   nseq=2 produce UN solo h. El log imprime `nseq` justamente para que
   esto se vea y nadie vuelva a asumir lo contrario.
2. El coeficiente KiB/token es constante para una config dada, así que
   sirve para proyectar linealmente.

================================================================
QUÉ IMPRIME
================================================================

    [PN80] ===== presupuesto del tensor `h` de GDN (chunk_gated_delta_rule_fwd_h) =====
    [PN80]  ENTRADA   T=6820 tokens en ESTE forward (suma de las 3 secuencias del batch; B=1)
    [PN80]  PASO 1    chunks: NT = ceil(T / FLA_CHUNK_SIZE) = ceil(6820 / 64) = 107
    [PN80]  PASO 2    shape:  h = (B, NT, H, V, K) = (1, 107, 24, 128, 128)
    [PN80]            donde H=24 heads GDN por GPU (ya dividido por tensor-parallel),
                      V=128 y K=128 dims de value/key
    [PN80]  PASO 3    elementos: 1 x 107 x 24 x 128 x 128 = 42,074,112
    [PN80]  PASO 4    bytes: 42,074,112 elem x 2 B (float16) = 84,148,224 B = 80.2 MiB POR GPU
    [PN80]  RATIO     80.2 MiB / 6820 tok = 12.05 KiB por token (constante para esta config)
    [PN80]  VRAM      libre AHORA en esta GPU: 67.0 MiB (de 24576 MiB totales)
    [PN80]  PROYECC.  67.0 MiB libres / 12.05 KiB por token = 5,693 tokens mas por forward
    [PN80]  MARGEN    5,693 tok proyectados / 6820 tok actuales = 0.8x
    [PN80]  *** AVISO: quedarian 12 MiB libres, objetivo 100 (GENESIS_PN80_TARGET_FREE_MIB). ***
    [PN80]  OJO       esta cuenta acota SOLO `h`. En el mismo forward tambien se piden
                      v_new (~6.03 KiB/tok) y los intermedios del FFN. Tomar como COTA SUPERIOR.

Se emite línea por línea a propósito: cada paso del cálculo queda
legible y auditable en el log del contenedor, en vez de una sola línea
larga donde hay que ir contando campos. Quien lo lea puede rehacer la
cuenta a mano y verificarla.

El campo PROYECC. es el que buscábamos: dice cuánto margen queda EN LAS
UNIDADES EN LAS QUE EL OPERADOR PIENSA (tokens por forward), en vez de
obligar a inferirlo de un OOM.

⚠️ La proyección acota SOLO el tensor `h`. En el mismo forward hay más
buffers transitorios (`v_new` ~6 KiB/tok, y los intermedios del FFN,
que en este rig son un sitio de OOM DISTINTO — ver
DIAGNOSTICO-OOM-qwen38-27b.md §4). Por eso el log dice "antes de agotar
[por h]" y el operador debe tomarlo como COTA SUPERIOR, no como techo
seguro. Se emite un aviso explícito cuando el margen baja de 1.5x.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default OFF (`GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE=1`).
- Rate-limited: por defecto 1 log cada 200 llamadas
  (`GENESIS_PN80_EVERY`), más un log SIEMPRE que el margen caiga bajo
  el umbral de aviso. Sin esto el log se inunda: hay una llamada por
  capa GDN por forward (48 capas en Qwen3.8-27B).
- Ignora los forwards de warmup/captura (T <= 256 por defecto,
  `GENESIS_PN80_MIN_T`) — si no, las 40 primeras líneas son todas
  T=64 del profiling de arranque y no dicen nada.
- `torch.cuda.mem_get_info()` se consulta solo cuando se va a emitir.
- Todo el cuerpo va dentro de try/except: cualquier fallo de la sonda
  degrada a no-op silencioso. NUNCA debe tumbar un forward.
- No cambia ninguna asignación ni ningún resultado numérico: es
  observación pura, insertada ANTES de la línea original que queda
  intacta.

================================================================
COMPOSICIÓN
================================================================

Ortogonal a todo. Conviene tenerlo ON mientras se evalúan PN12/PN25
(pools de FFN) o PN32/PN59/P103 (chunking de GDN), porque da el número
antes/después en la misma unidad.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN80_MARKER = "[Genesis PN80 gdn h budget probe]"


ANCHOR_OLD = "    h = k.new_empty(B, NT, H, V, K)\n"


ANCHOR_NEW = (
    "    # " + GENESIS_PN80_MARKER + "\n"
    "    # Observación pura: calcula el tamaño de `h` y proyecta cuántos\n"
    "    # tokens/forward más entran con la VRAM libre. No altera nada.\n"
    "    # Ver wiring/hybrid/patch_N80_gdn_h_budget_probe.py.\n"
    "    try:\n"
    "        import os as _g80_os\n"
    "        if _g80_os.environ.get('GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE') == '1':\n"
    "            global _GENESIS_PN80_CALLS\n"
    "            try:\n"
    "                _GENESIS_PN80_CALLS += 1\n"
    "            except NameError:\n"
    "                _GENESIS_PN80_CALLS = 1\n"
    "            _g80_min_t = int(_g80_os.environ.get('GENESIS_PN80_MIN_T', '256'))\n"
    "            if T > _g80_min_t:\n"
    "                _g80_every = max(int(_g80_os.environ.get('GENESIS_PN80_EVERY', '200')), 1)\n"
    "                # El objetivo es MiB libres DESPUES de servir el transitorio de\n"
    "                # este paso, no un ratio: 'margen 1.5x' no dice cuanta VRAM\n"
    "                # sobra, y lo que se tunea es exactamente eso. El default de 100\n"
    "                # MiB es el minimo con el que este rig paso 5 prompts de 30k en\n"
    "                # paralelo sin OOMear (medido en los 3 engines qwen38).\n"
    "                _g80_obj = float(_g80_os.environ.get('GENESIS_PN80_TARGET_FREE_MIB', '100'))\n"
    "                _g80_warn_at = float(_g80_os.environ.get('GENESIS_PN80_WARN_RATIO', '0'))\n"
    "                _g80_item = k.element_size()\n"
    "                _g80_bytes = B * NT * H * V * K * _g80_item\n"
    "                _g80_per_tok = _g80_bytes / float(T)\n"
    "                _g80_free, _g80_total = torch.cuda.mem_get_info()\n"
    "                # ── ancho del FFN (el OTRO sitio de OOM) ──────────────────\n"
    "                # Los OOM reales de este rig no fueron por `h` sino por el\n"
    "                # buffer gate_up del FFN: (T, 2*intermediate/TP) fp16. Si el\n"
    "                # veredicto mira solo `h`, dice OK justo antes de reventar.\n"
    "                # Se descubre una vez y se cachea en un global.\n"
    "                global _GENESIS_PN80_FFNW\n"
    "                try:\n"
    "                    _GENESIS_PN80_FFNW\n"
    "                except NameError:\n"
    "                    _GENESIS_PN80_FFNW = 0\n"
    "                    # Override manual, para rigs donde el config no se puede leer:\n"
    "                    #   GENESIS_PN80_FFN_WIDTH = 2 * intermediate_size / TP\n"
    "                    _g80_ov = _g80_os.environ.get('GENESIS_PN80_FFN_WIDTH')\n"
    "                    if _g80_ov:\n"
    "                        try:\n"
    "                            _GENESIS_PN80_FFNW = int(_g80_ov)\n"
    "                        except Exception:\n"
    "                            _GENESIS_PN80_FFNW = 0\n"
    "                    else:\n"
    "                        try:\n"
    "                            # get_current_vllm_config() LANZA en el forward: vLLM\n"
    "                            # solo lo deja seteado durante la construccion del\n"
    "                            # modelo (su propio error lo dice: '...or at model\n"
    "                            # forward time when config is not set'). Por eso este\n"
    "                            # bloque caia SIEMPRE al except y el FFN quedaba en 0,\n"
    "                            # subestimando el transitorio ~2.6x. El sub-patch\n"
    "                            # pn80_config_capture guarda una referencia que\n"
    "                            # sobrevive al contexto.\n"
    "                            import vllm._genesis as _g80_ns\n"
    "                            _g80_c = getattr(_g80_ns, 'VLLM_CONFIG_CAPTURADO', None)\n"
    "                            if _g80_c is None:\n"
    "                                from vllm.config import get_current_vllm_config as _g80_cf\n"
    "                                _g80_c = _g80_cf()\n"
    "                            _g80_mc = _g80_c.model_config\n"
    "                            _g80_i = 0\n"
    "                            # el config puede estar plano, anidado en text_config, o\n"
    "                            # expuesto via hf_text_config segun modelo y version\n"
    "                            for _g80_src in (getattr(_g80_mc, 'hf_text_config', None),\n"
    "                                             getattr(getattr(_g80_mc, 'hf_config', None),\n"
    "                                                     'text_config', None),\n"
    "                                             getattr(_g80_mc, 'hf_config', None)):\n"
    "                                _g80_i = int(getattr(_g80_src, 'intermediate_size', 0) or 0)\n"
    "                                if _g80_i:\n"
    "                                    break\n"
    "                            _g80_tp = int(_g80_c.parallel_config.tensor_parallel_size) or 1\n"
    "                            _GENESIS_PN80_FFNW = (2 * _g80_i) // _g80_tp\n"
    "                        except Exception:\n"
    "                            _GENESIS_PN80_FFNW = 0\n"
    "                _g80_ffn = T * _GENESIS_PN80_FFNW * _g80_item\n"
    "                # v_new tiene la mitad del tamano de h (misma forma sin la dim NT\n"
    "                # de chunks, medido ~6 KiB/tok contra los ~12 de h).\n"
    "                _g80_vnew = _g80_bytes // 2\n"
    "                _g80_total_tr = _g80_bytes + _g80_vnew + _g80_ffn\n"
    "                _g80_tr_tok = _g80_total_tr / float(T)\n"
    "                _g80_proj = int(_g80_free / _g80_tr_tok) if _g80_tr_tok > 0 else -1\n"
    "                _g80_ratio = (_g80_proj / float(T)) if T > 0 else 0.0\n"
    "                _g80_sobra_mib = (_g80_free - _g80_total_tr) / 1048576.0\n"
    "                _g80_tight = _g80_sobra_mib < _g80_obj\n"
    "                # Solo el rank 0 imprime: con tensor-parallel cada worker\n"
    "                # reporta SU GPU y el bloque salia duplicado. Las GPUs de un\n"
    "                # mismo TP group llevan la misma carga, asi que una alcanza.\n"
    "                # GENESIS_PN80_ALL_RANKS=1 fuerza que impriman todas.\n"
    "                _g80_rank0 = True\n"
    "                if _g80_os.environ.get('GENESIS_PN80_ALL_RANKS') != '1':\n"
    "                    try:\n"
    "                        import torch.distributed as _g80_dist\n"
    "                        if _g80_dist.is_available() and _g80_dist.is_initialized():\n"
    "                            _g80_rank0 = _g80_dist.get_rank() == 0\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                # El camino 'tight' imprimia en CADA llamada, y hay 48 capas\n"
    "                # GDN por forward: bajo presion de memoria eso inunda el log con\n"
    "                # cientos de cajas identicas por segundo, justo cuando hay que\n"
    "                # leerlo. Se le pone un cooldown por tiempo.\n"
    "                global _GENESIS_PN80_LAST_EMIT\n"
    "                try:\n"
    "                    _GENESIS_PN80_LAST_EMIT\n"
    "                except NameError:\n"
    "                    _GENESIS_PN80_LAST_EMIT = 0.0\n"
    "                _g80_cool = float(_g80_os.environ.get('GENESIS_PN80_TIGHT_COOLDOWN', '30'))\n"
    "                import time as _g80_time\n"
    "                _g80_now = _g80_time.monotonic()\n"
    "                _g80_emitir = (_g80_tight and (_g80_now - _GENESIS_PN80_LAST_EMIT) >= _g80_cool) \\\n"
    "                              or (_GENESIS_PN80_CALLS % _g80_every) == 0\n"
    "                if _g80_rank0 and _g80_emitir:\n"
    "                    _GENESIS_PN80_LAST_EMIT = _g80_now\n"
    "                    _g80_nseq = (len(cu_seqlens) - 1) if cu_seqlens is not None else B\n"
    "                    _g80_dt = str(k.dtype).replace('torch.', '')\n"
    "                    _g80_elems = B * NT * H * V * K\n"
    "                    _g80_mib = _g80_bytes / 1048576.0\n"
    "                    _g80_fmib = _g80_free / 1048576.0\n"
    "                    _g80_tmib = _g80_total / 1048576.0\n"
    "                    _g80_ktok = _g80_per_tok / 1024.0\n"
    "                    # Cuanta VRAM haria falta para llegar al margen sano, y a\n"
    "                    # cuantos puntos de --gpu-memory-utilization equivale.\n"
    "                    _g80_need = _g80_total_tr + _g80_obj * 1048576.0\n"
    "                    _g80_falta = max(_g80_need - _g80_free, 0.0)\n"
    "                    _g80_pts = _g80_falta / float(_g80_total)\n"
    "                    _g80_util = None\n"
    "                    try:\n"
    "                        from vllm.config import get_current_vllm_config as _g80_cfg\n"
    "                        _g80_util = _g80_cfg().cache_config.gpu_memory_utilization\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                    # print() y no logging: el logger a nivel INFO no se\n"
    "                    # propaga desde los procesos worker de vLLM (solo salian\n"
    "                    # los WARNING), y esto es un informe para leer a ojo en\n"
    "                    # `docker logs`.\n"
    "                    def _g80_e(_f, *_a):\n"
    "                        print(('[PN80] ' + _f) % _a if _a else ('[PN80] ' + _f),\n"
    "                              flush=True)\n"
    "                    _g80_ver = 'RIESGO DE OOM' if _g80_tight else 'OK'\n"
    "                    _g80_e('')\n"
    "                    _g80_e('+-- PN80 - alcanza la VRAM para este lote? ---- VEREDICTO: %s',\n"
    "                           _g80_ver)\n"
    "                    _g80_e('|')\n"
    "                    _g80_e('|  QUE ESTA PASANDO')\n"
    "                    _g80_e('|    Se estan procesando %s tokens de una sola vez '\n"
    "                           '(%d secuencias juntas en el lote).',\n"
    "                           format(T, ','), _g80_nseq)\n"
    "                    _g80_e('|    La atencion lineal (GDN) necesita reservar %.0f MiB '\n"
    "                           'en CADA GPU para eso.', _g80_mib)\n"
    "                    _g80_e('|')\n"
    "                    _g80_e('|  DE DONDE SALE ESE NUMERO')\n"
    "                    _g80_e('|    %s tokens / %d por chunk           = %d chunks',\n"
    "                           format(T, ','), BT, NT)\n"
    "                    _g80_e('|    %d chunks x %d heads x %d x %d      = %s valores',\n"
    "                           NT, H, V, K, format(_g80_elems, ','))\n"
    "                    _g80_e('|    %s valores x %d bytes (%s)  = %.0f MiB',\n"
    "                           format(_g80_elems, ','), _g80_item, _g80_dt, _g80_mib)\n"
    "                    _g80_e('|    O sea: %.1f KiB de VRAM por cada token del lote.',\n"
    "                           _g80_ktok)\n"
    "                    _g80_e('|')\n"
    "                    _g80_e('|  Y ESO NO ES TODO LO QUE SE PIDE EN ESTE PASO')\n"
    "                    _g80_e('|    tensor h (atencion lineal GDN) : %7.0f MiB', _g80_mib)\n"
    "                    _g80_e('|    v_new (mismo camino GDN)       : %7.0f MiB',\n"
    "                           _g80_vnew / 1048576.0)\n"
    "                    if _GENESIS_PN80_FFNW:\n"
    "                        _g80_e('|    buffer gate_up del FFN        : %7.0f MiB   '\n"
    "                               '(%d tok x %d x %d bytes)',\n"
    "                               _g80_ffn / 1048576.0, T, _GENESIS_PN80_FFNW, _g80_item)\n"
    "                    else:\n"
    "                        _g80_e('|    buffer gate_up del FFN        :       ? '\n"
    "                               '(no se pudo leer intermediate_size del config)')\n"
    "                    _g80_e('|    ------------------------------------------')\n"
    "                    _g80_e('|    TOTAL transitorio                : %7.0f MiB '\n"
    "                           '= %.1f KiB/token',\n"
    "                           _g80_total_tr / 1048576.0, _g80_tr_tok / 1024.0)\n"
    "                    _g80_e('|')\n"
    "                    _g80_e('|  COMO ESTA LA GPU')\n"
    "                    _g80_e('|    Libres ahora: %.0f MiB de %.0f MiB totales.',\n"
    "                           _g80_fmib, _g80_tmib)\n"
    "                    _g80_e('|    Con eso entrarian %s tokens mas (contando TODOS los '\n"
    "                           'buffers de arriba); el lote actual pide %s.',\n"
    "                           format(_g80_proj, ','), format(T, ','))\n"
    "                    _g80_e('|    Despues de servir este lote quedarian %.0f MiB '\n"
    "                           'libres (objetivo: %.0f).',\n"
    "                           _g80_sobra_mib, _g80_obj)\n"
    "                    _g80_e('|')\n"
    "                    if not _g80_tight:\n"
    "                        _g80_e('|  >>> QUE HACER: NADA. La VRAM alcanza, no hace falta '\n"
    "                               'bajar --gpu-memory-utilization.')\n"
    "                    else:\n"
    "                        _g80_e('|  >>> QUE HACER: SI, HAY QUE BAJAR LA VRAM RESERVADA.')\n"
    "                        _g80_e('|      Faltan ~%.0f MiB por GPU para el objetivo de '\n"
    "                               '%.0f MiB libres.', _g80_falta / 1048576.0, _g80_obj)\n"
    "                        if _g80_util is not None:\n"
    "                            _g80_e('|      BAJA --gpu-memory-utilization de %.3f a %.3f '\n"
    "                                   '(son %.1f puntos porcentuales).',\n"
    "                                   _g80_util, max(_g80_util - _g80_pts, 0.50),\n"
    "                                   _g80_pts * 100.0)\n"
    "                        else:\n"
    "                            _g80_e('|      BAJA --gpu-memory-utilization en %.1f puntos '\n"
    "                                   'porcentuales.', _g80_pts * 100.0)\n"
    "                        _g80_e('|      Alternativa sin tocar VRAM: bajar --max-num-seqs '\n"
    "                               'para que entren menos secuencias por lote.')\n"
    "                    _g80_e('|')\n"
    "                    if _GENESIS_PN80_FFNW:\n"
    "                        _g80_e('|  El veredicto cuenta h + v_new + FFN. Puede haber otros '\n"
    "                               'buffers menores (norms, residuales), asi que tomalo como '\n"
    "                               'una cota AJUSTADA, no exacta.')\n"
    "                    else:\n"
    "                        _g80_e('|  OJO: no se pudo medir el FFN, asi que el veredicto cuenta '\n"
    "                               'solo h + v_new y el margen real es MENOR.')\n"
    "                    _g80_e('+------------------------------------------------------------')\n"
    "    except Exception as _g80_err:\n"
    "        # La sonda NUNCA debe tumbar un forward, pero tampoco debe fallar en\n"
    "        # silencio: un error a mitad del bloque truncaba el log y parecia que\n"
    "        # el reporte salia incompleto por otra razon. Se avisa UNA sola vez.\n"
    "        try:\n"
    "            global _GENESIS_PN80_ERR\n"
    "            try:\n"
    "                _GENESIS_PN80_ERR\n"
    "            except NameError:\n"
    "                _GENESIS_PN80_ERR = True\n"
    "                import logging as _g80_elog\n"
    "                _g80_elog.getLogger('genesis.pn80').warning(\n"
    "                    'PN80: la sonda fallo y se desactiva este reporte '\n"
    "                    '(el forward sigue normal): %r', _g80_err)\n"
    "        except Exception:\n"
    "            pass\n"
) + ANCHOR_OLD


# ── captura del VllmConfig ──────────────────────────────────────────────────
# `get_current_vllm_config()` LANZA fuera del contexto de construccion del
# modelo; vLLM lo dice en su propio mensaje de error: "...or a CustomOp was
# instantiated at module import time or model forward time when config is not
# set". Cualquier sonda que corra en el forward —como esta— no puede leerlo.
#
# Este sub-patch guarda una referencia mientras el config SI esta seteado, para
# que sobreviva al contexto. Sin esto PN80 no podia leer intermediate_size y el
# buffer gate_up del FFN quedaba en 0: con lotes de 4096 tokens son 142 MiB por
# GPU sin contar, o sea que el transitorio real es ~52 KiB/token y PN80
# reportaba ~18. Todos sus veredictos venian subestimados ~2.6x.
ANCHOR_CAPTURE_OLD = """        _current_vllm_config = vllm_config
        _current_prefix = prefix
"""

ANCHOR_CAPTURE_NEW = """        _current_vllm_config = vllm_config
        _current_prefix = prefix
        # _GENESIS_PN80_CONFIG_CAPTURE
        try:
            import vllm._genesis as _g80_ns

            _g80_ns.VLLM_CONFIG_CAPTURADO = vllm_config
        except Exception:
            pass
"""


def _patcher_capture() -> TextPatcher | None:
    # OJO: set_current_vllm_config se importa desde vllm.config pero se DEFINE
    # en config/vllm.py. Ademas es un @contextmanager, asi que
    # inspect.getsourcefile() devuelve contextlib.py y hay que seguir __wrapped__.
    target = resolve_vllm_file("config/vllm.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN80 config capture",
        target_file=str(target),
        marker="_GENESIS_PN80_CONFIG_CAPTURE",
        sub_patches=[
            TextPatch(
                name="pn80_config_capture",
                anchor=ANCHOR_CAPTURE_OLD,
                replacement=ANCHOR_CAPTURE_NEW,
                required=True,
            ),
        ],
    )


def _is_enabled() -> bool:
    import os

    return os.environ.get("GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE") == "1"


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "third_party/flash_linear_attention/ops/chunk_delta_h.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN80 GDN h budget probe",
        target_file=str(target),
        marker=GENESIS_PN80_MARKER,
        sub_patches=[
            TextPatch(
                name="pn80_h_alloc_probe",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Si upstream cambia la forma de asignar h (pool, buffer
            # reusado, allocator propio), el cálculo de esta sonda deja
            # de representar la realidad -> mejor SKIP que mentir.
            "h_buffer_pool",
            "_get_h_workspace",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN80")
    log_decision("PN80", decision, reason)
    if not decision:
        return "skipped", reason
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE not set; default OFF. "
            "Sonda de observacion: loguea el tamano de h y proyecta cuantos "
            "tokens/forward entran con la VRAM libre. Sin costo en el hot "
            "path cuando esta OFF (la rama env se evalua una vez por "
            "llamada y sale)."
        )
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "third_party/flash_linear_attention/ops/chunk_delta_h.py not found"
    # captura del config: sin esto la sonda no puede leer intermediate_size y
    # el FFN queda sin contar. Best-effort: si el ancla derivo, el resto de PN80
    # sigue sirviendo (avisa en el reporte que el margen real es menor).
    cp = _patcher_capture()
    if cp is not None:
        try:
            cp.apply()
        except Exception:
            pass
    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN80 applied: sonda de presupuesto de h activa. Formula "
            "h_bytes = B*ceil(T/64)*H*V*K*itemsize, verificada sobre 200 "
            "llamadas reales (12.05 KiB/tok en Qwen3.8-27B W8A16 TP=2). "
            "Tunear con GENESIS_PN80_EVERY / _MIN_T / _TARGET_FREE_MIB / _TIGHT_COOLDOWN."
        ),
        patch_name="PN80 GDN h budget probe",
    )


def is_applied() -> bool:
    """Reporter para verify_live_rebinds en apply_all.py."""
    if vllm_install_root() is None:
        return False
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN80_MARKER in f.read()
    except Exception:
        return False
