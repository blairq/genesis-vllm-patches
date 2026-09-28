# SPDX-License-Identifier: Apache-2.0
"""Genesis PN120: all-reduce de TP comprimido a INT8.

Por qué
-------
El perfil con nsys del prompt processing (42k, TP=2 en dos 3090) dio que el
``ncclDevKernel_AllReduce`` se lleva el **24,4% del prefill**: 6.538 llamadas de
6,3 ms, moviendo 76,7 MB cada una.

Y no hay nada que ganar del lado del kernel. Medido:

    copia P2P pura de 79,7 MB   6,06 ms  = 13,1 GB/s
    all-reduce de NCCL          6,30 ms  <- 96% del limite fisico

O sea que NCCL ya está al techo del PCIe 4.0 x8. Lo único que queda es **mandar
menos bytes**. Se descartaron antes, todos medidos:

  * async TP + sequence parallelism: 3-4% PEOR (el GEMM que alimenta al
    all-reduce dura 2,4 ms contra 6,3 de comunicacion, el techo era bajo);
  * fusion allreduce+RMSNorm de FlashInfer: sin cambio;
  * pipeline parallel: ``NotImplementedError`` para este modelo;
  * DBO (micro-batch overlap): sólo soporta backends all2all de MoE, y este
    modelo es denso.

Qué hace
--------
Para TP=2 un all-reduce equivale a intercambiar los parciales y sumar. Si los
parciales viajan en int8 con escala por grupo, son la mitad de bytes::

    v = x.view(M, H/G, G)                # G = GENESIS_PN120_GRUPO (default 64)
    s = amax(|v|, dim=-1) / 127          # una escala POR GRUPO
    q = round(v / s)                     # int8
    all_gather(q), all_gather(s)         # la MITAD de bytes que el fp16
    out = sum_i q_i * s_i                # se suma en fp16, NO en int8

**No se puede** ``all_reduce(int8, SUM)``: dos parciales de ±127 desbordan.

Medido
------
::

    all-reduce fp16 (hoy)            6,49 ms
    int8 con torch suelto            5,61 ms   1,16x
    int8 con torch.compile           3,77 ms   1,72x   <- inductor fusiona
    solo la comunicacion (piso)      3,42 ms   1,90x

La fusion de inductor sobre el (des)cuantizado es lo que lo lleva de 1,16x a
1,72x, asi que hace falta que ese trabajo este compilado.

Pero el gate por talla NO puede vivir en el forward trazado
-------------------------------------------------------------
Medido: con ``M_MIN=512`` el decode (M=4..40) se ralentizaba 12% aunque nunca
deberia comprimir, y con ``M_MIN=2048`` el prefill (M=7488) dejaba de ganar
aunque si deberia. La causa es que **dynamo traza el forward UNA vez y hornea
la rama para todo**, y la config de vLLM lo garantiza: ``dynamic_shapes_config``
trae ``evaluate_guards: False``, o sea que los guards de shape ni se evaluan.

Por eso el gate vive en un **custom op opaco** (corre de verdad en cada forward)
que por dentro llama a helpers **pre-compilados** con ``torch.compile``. Asi se
tiene la decision en runtime Y la fusion.

Por que la escala va POR GRUPO y no por token
---------------------------------------------
Una sola escala para los 5120 valores de un token es muy mala con activaciones
reales: **un solo outlier fuerza una escala grande y todos los valores chicos
quedan groseros**. Medido con outliers dispersos (0,1% de los valores x12), que
es el regimen realista::

    escala          error     velocidad
    por token       0,0419      1,64x     <- inaceptable
    grupo 128       0,0107      1,61x
    grupo 64        0,0084      1,59x     <- elegido
    grupo 32        0,0067      1,54x
    grupo 16        0,0054      1,46x
    grupo 8         0,0043      1,32x

Grupo 64 da **5x menos error** que por token por 3% menos de aceleracion. Las
escalas extra casi no pesan: 80 escalas fp16 por token contra 5120 bytes de
datos, o sea +3% de bytes.

Se probo tambien **rotacion de Hadamard** antes de cuantizar (QuaRot/SpinQuant),
que aprovecha que H es lineal y conmuta con la suma: se rota, se suma en el
dominio rotado y se des-rota UNA vez. Baja el error 1,7x (0,01065 -> 0,00627 con
grupo 128) pero cuesta 1,61x -> 1,47x por el matmul de la rotacion.
**Grupo 32 sin Hadamard da el mismo error y es mas rapido**, asi que la rotacion
queda descartada.

El error **no se acumula** a traves de las capas: simulando la cadena real de 64
capas con residual y RMSNorm (128 all-reduce), el error final es apenas mayor
que el de un solo all-reduce. Las conexiones residuales dominan — el error vive
en la correccion ``h``, que es chica contra ``x``.
"""

from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn120")

_TRUTHY = ("1", "true", "yes", "on")


def _flag(n: str, d: str = "0") -> bool:
    return os.environ.get(n, d).strip().lower() in _TRUTHY


# Se leen al IMPORTAR: esto corre dentro de la region que traza dynamo y leer
# os.environ ahi mete guards y ruido.
_ACTIVO = _flag("GENESIS_ENABLE_PN120_AR_INT8")
try:
    _M_MIN = int(os.environ.get("GENESIS_PN120_M_MIN", "512"))
except ValueError:
    _M_MIN = 512
try:
    _GRUPO = int(os.environ.get("GENESIS_PN120_GRUPO", "64"))
except ValueError:
    _GRUPO = 64


def activo() -> bool:
    return _ACTIVO


def grupo() -> int:
    """Elementos por escala. Ver el docstring del modulo para la tabla."""
    return _GRUPO


def m_min() -> int:
    """Piso de tokens para comprimir.

    En decode el all-reduce es de ~400 KB y esta limitado por LATENCIA, no por
    ancho de banda: partir los bytes al medio no compra nada y los kernels de
    (des)cuantizacion se pagan igual. Comprimir solo tiene sentido con los
    chunks grandes del prefill.
    """
    return _M_MIN


def _cuantizar(x: torch.Tensor, g: int):
    """[m, h] -> (int8 [m, h], escalas [m, h/g]). Se compila."""
    m, h = x.shape
    v = x.view(m, h // g, g)
    s = v.abs().amax(dim=-1).clamp_min(1e-6) / 127.0
    q = (v / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8).view(m, h)
    return q, s


def _sumar(qg: torch.Tensor, sg: torch.Tensor, m: int, g: int, w: int,
           dt: torch.dtype) -> torch.Tensor:
    """Descuantiza los w parciales y los suma en `dt`. Se compila."""
    h = qg.shape[-1]
    ng = h // g
    out = qg[:m].view(m, ng, g).to(dt) * sg[:m].unsqueeze(-1).to(dt)
    for i in range(1, w):
        lo, hi = i * m, (i + 1) * m
        out = out + (qg[lo:hi].view(m, ng, g).to(dt)
                     * sg[lo:hi].unsqueeze(-1).to(dt))
    return out.view(m, h)


_cuantizar_c = None
_sumar_c = None


def _compilados():
    """Compila los helpers una vez.

    `dynamic=False` a proposito: el scheduler produce pocas tallas distintas de
    chunk (medido: 1189, 4160, 7488, 8192), asi que recompilar por talla cuesta
    unas pocas compilaciones al arranque y a cambio inductor fusiona de verdad.
    Con `dynamic=True` el grafo generico fusiona peor y se pierde la ganancia.
    """
    global _cuantizar_c, _sumar_c
    if _cuantizar_c is None:
        _cuantizar_c = torch.compile(_cuantizar, dynamic=False)
        _sumar_c = torch.compile(_sumar, dynamic=False)
    return _cuantizar_c, _sumar_c


def _impl(x: torch.Tensor) -> torch.Tensor:
    """Cuerpo real del custom op. Corre por forward, no se traza."""
    from vllm.distributed.communication_op import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )

    w = get_tensor_model_parallel_world_size()
    if w < 2:
        return x

    plano = x.reshape(-1, x.shape[-1])
    m, h = plano.shape
    # PN152: con M chico (decode), all-reduce P2P directo (SK-24) en vez de NCCL. Se inicializa en la
    # primera llamada en EAGER de cualquier tamano (la corrida de perfilado de memoria de vLLM), nunca
    # dentro de una captura; si no aplica, sigue por NCCL. Mismo resultado: con 2 rangos es una suma fp16.
    from vllm._genesis import ar_p2p as _g152
    _g152.asegurar_inicializado()
    # ESTE if corre de verdad, porque el op es opaco a dynamo.
    if m < _M_MIN or h % _GRUPO != 0:
        if _g152.activo() and w == 2 and _g152.sirve(x):
            return _g152.all_reduce(x)
        return tensor_model_parallel_all_reduce(x)

    cuant, sumar = _compilados()
    q, s = cuant(plano, _GRUPO)
    qg = tensor_model_parallel_all_gather(q, dim=0)
    sg = tensor_model_parallel_all_gather(s, dim=0)
    return sumar(qg, sg, m, _GRUPO, w, x.dtype).reshape(x.shape)


def _fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


_registrado = False


def registrar() -> bool:
    """Registra `vllm::genesis_ar_int8`. Idempotente."""
    global _registrado
    if _registrado:
        return True
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="genesis_ar_int8",
        op_func=_impl,
        mutates_args=[],
        fake_impl=_fake,
    )
    _registrado = True
    return True


def all_reduce_int8(x: torch.Tensor) -> torch.Tensor:
    """All-reduce equivalente con los parciales en int8.

    Va por el custom op para que el gate por talla se evalue en RUNTIME.
    """
    return torch.ops.vllm.genesis_ar_int8(x)


if _ACTIVO:
    try:
        registrar()
    except Exception as _e:   # pragma: no cover
        log.error("[PN120] no se pudo registrar el custom op: %s", _e)


__all__ = ["activo", "m_min", "grupo", "registrar", "all_reduce_int8"]
