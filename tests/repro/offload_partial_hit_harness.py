#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Banco de pruebas del scheduler + OffloadingConnector, SIN GPU y SIN modelo.

Reproduce el crash de `genesis-27b-qwen38-fp8` del 2026-08-16 20:03:34:

    File ".../kv_connector/v1/offloading/scheduler.py", line 612,
      in update_state_after_alloc
        num_locally_computed_tokens
    AssertionError

El engine murio entero (EngineDeadError) tras 14 horas y ~60 requests, con 4
trabajos en paralelo, 5 en cola y el KV al 65%.

POR QUE UN BANCO Y NO TIRARLE CARGA AL ENGINE DE VERDAD
-------------------------------------------------------
El sitio del assert es logica pura de Python del scheduler: no toca la GPU,
ni los pesos, ni un solo kernel. Reproducirlo levantando el 27B cuesta ~6
minutos de arranque por intento y depende de que la loteria de desalojos
caiga justo. Aca corren miles de escenarios por segundo y, cuando uno falla,
queda grabado con su semilla para repetirlo exacto.

Se usa el Scheduler REAL de vLLM, el KVCacheManager REAL y el
OffloadingConnector REAL, con la config REAL del modelo (sale del config.json
cacheado; no se descarga ni se carga ningun peso). Lo unico simulado es:

  - el forward del modelo -> ModelRunnerOutput con tokens inventados
  - el worker del connector -> completa los jobs de transferencia a mano

Uso (dentro de la imagen del engine, sin GPU real):

    docker run --rm --gpus all -e HF_HUB_OFFLINE=1 --shm-size=2gb \
      -v /home/usuario/Proyectos/models-cache:/root/.cache/huggingface \
      -v $PWD/tests:/tests --entrypoint python3 vllm/vllm-openai:v0.27.1 \
      /tests/repro/offload_partial_hit_harness.py <modo> [...]

Modos:
    matriz --iters 60      las 4 celdas hibrido x spec-decode, agrupando
                           los fallos por firma (archivo:linea). Es el modo
                           que aisla la causa.
    fuzz --iters 400       barrido de semillas con la config del engine real
    seed <N> --dump        repite UNA semilla y vuelca el estado exacto que
                           mira el assert, grupo por grupo

Flags: --no-hibrido, --spec si|no, --sin-async

Para probar CON los parches Genesis aplicados, montar vllm/_genesis y correr
antes `python3 -m vllm._genesis.patches.apply_all`.

Resultado medido (60 semillas por celda):

                 spec=SI          spec=NO
    hibrido=SI   3 x :612         0
    hibrido=NO   0                0

    con PN84:    0 x :612  (y 6x mas eventos del camino ejercitados)

Los prompts se topean a max_model_len-1: el engine real rechaza con 400 los
mas largos antes del scheduler, y sin ese tope el banco disparaba un assert
en scheduler.py:771 que en produccion no puede pasar (ver KV-OFFLOADING §10.4).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import traceback
from dataclasses import dataclass

import torch

MODELO = os.environ.get(
    "REPRO_MODEL", "orcarouter/Qwen3.8-27B-Uncensored-FP8"
)
BLOCK_SIZE = 1600  # el que elige vLLM para este modelo (ver logs de arranque)


# ── construccion del entorno ────────────────────────────────────────────────


def construir_vllm_config(
    *,
    spec_decode: bool,
    cpu_bytes: int,
    fs_dir: str | None,
    max_model_len: int,
    async_sched: bool = True,
):
    from vllm.engine.arg_utils import EngineArgs

    extra: dict = {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": cpu_bytes,
        "eviction_policy": "arc",
    }
    if fs_dir:
        extra["secondary_tiers"] = [{"type": "fs", "root_dir": fs_dir}]

    kwargs = dict(
        model=MODELO,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=4096,
        long_prefill_token_threshold=4096,
        max_num_seqs=4,
        async_scheduling=async_sched,
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        },
    )
    if spec_decode:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": 3,
        }
    return EngineArgs(**kwargs).create_engine_config()


def construir_kv_cache_config(vllm_config, *, num_blocks: int, hibrido: bool):
    """Reproduce el layout que el model runner arma para Qwen3.8.

    Segun los logs del engine:
      "Setting attention block size to 1600 tokens to ensure that attention
       page size is >= mamba page size"
      "Padding mamba page size ... exactly equal"
    o sea DOS grupos con el mismo block_size y el mismo page size.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
    )

    attn = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
    )
    grupos = [KVCacheGroupSpec(["attn.0"], attn)]
    page = attn.page_size_bytes

    if hibrido:
        # una sola "shape" de estado por capa GDN, dimensionada para que la
        # pagina quede igual a la de atencion (que es lo que hace vLLM).
        elems = page // 2  # float16
        mamba = MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((elems,),),
            dtypes=(torch.float16,),
            page_size_padded=page,
            mamba_type="linear_attention",
            mamba_cache_mode="align",
            num_speculative_blocks=0,
        )
        grupos.append(KVCacheGroupSpec(["gdn.0"], mamba))

    tensores = [
        KVCacheTensor(size=page * num_blocks, shared_by=[g.layer_names[0]])
        for g in grupos
    ]
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensores, kv_cache_groups=grupos
    )


def construir_scheduler(vllm_config, kv_cache_config):
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    # mismos dos tamaños que calcula EngineCore (no siempre coinciden entre si)
    sched_bs, hash_bs = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=sched_bs,
        hash_block_size=hash_bs,
        log_stats=False,
    ), hash_bs


# ── simulacion ──────────────────────────────────────────────────────────────


def construir_hasher(vllm_config, hash_block_size):
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
    init_none_hash(fn)
    return get_request_block_hasher(hash_block_size, fn)


def nueva_request(rid: str, tokens: list[int], hasher, max_tokens: int):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        pooling_params=None,
        block_hasher=hasher,
    )


def completar_jobs(sched, rng, *, prob: float = 1.0):
    """Hace de worker: marca como terminados los jobs de transferencia.

    `prob < 1` deja jobs colgando un paso mas, que es lo que pasa de verdad
    cuando el disco tarda.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingWorkerMetadata,
    )
    from vllm.v1.outputs import KVConnectorOutput

    cs = sched.connector.connector_scheduler
    pendientes = {
        jid: st.pending_count
        for jid, st in cs._jobs.items()
        if prob >= 1.0 or rng.random() < prob
    }
    if not pendientes:
        return None
    return KVConnectorOutput(
        kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs=pendientes)
    )


def salida_falsa(sched, salida_sched, kv_out):
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(salida_sched.num_scheduled_tokens.keys())
    muestreados = []
    for rid in req_ids:
        req = sched.requests.get(rid)
        n = salida_sched.num_scheduled_tokens[rid]
        # solo emite token cuando termino el prefill
        if req is not None and req.num_computed_tokens + n >= req.num_tokens:
            muestreados.append([1000 + len(muestreados)])
        else:
            muestreados.append([])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=muestreados,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=kv_out,
    )


class Contadores:
    """Sin esto no se sabe si un escenario ejercito el camino del bug.

    El assert solo puede fallar cuando hay hit LOCAL (GPU) y ADEMAS hit
    EXTERNO (tier de RAM/disco) en la misma request. Un fuzz que nunca
    produce esa combinacion "pasa" sin haber probado nada.
    """

    def __init__(self) -> None:
        self.alloc = 0
        self.con_externo = 0
        self.local_y_externo = 0
        self.max_local = 0

    def __str__(self) -> str:
        return (
            f"alloc={self.alloc} con_hit_externo={self.con_externo} "
            f"LOCAL+EXTERNO={self.local_y_externo} max_local_tok={self.max_local}"
        )


def instrumentar(sched, cont: Contadores, dump: bool = False):
    from vllm.utils.math_utils import cdiv

    cs = sched.connector.connector_scheduler
    orig = cs.update_state_after_alloc

    # Que devolvio get_computed_blocks para esta request, ANTES de allocate_slots.
    # Es la unica forma de distinguir "el hit local nunca tuvo hash" de "lo
    # perdio en el medio".
    visto: dict[str, list] = {}
    kvm = sched.kv_cache_manager
    orig_gcb = kvm.get_computed_blocks

    pool = kvm.block_pool

    def gcb(request):
        res = orig_gcb(request)
        bloques, ntok = res
        # ¿el grupo de atencion TIENE realmente cacheado el primer bloque del
        # prefijo? Si el hit dice 1600 tokens pero el grupo 0 no tiene bloque,
        # el hit es inconsistente y el KV de esos tokens no existe.
        en_cache = []
        for gid in range(len(bloques.blocks)):
            try:
                b = pool.get_cached_block(request.block_hashes[0], [gid])
            except Exception:
                b = "?"
            if isinstance(b, list):
                b = [x.block_id for x in b]
            elif hasattr(b, "block_id"):
                b = b.block_id
            en_cache.append(b)
        visto[request.request_id] = [
            (ntok, [[(b.block_id, b.block_hash is not None) for b in g]
                    for g in bloques.blocks], f"bloque0_en_cache_por_grupo={en_cache}")
        ]
        return res

    kvm.get_computed_blocks = gcb

    # Traza de allocate_slots: que le pidieron y cuantos bloques quedo teniendo.
    # Sin esto no se puede distinguir "el connector rastrea mal" de "el
    # scheduler asigno menos bloques de los que los tokens necesitan".
    alocaciones: dict[str, list] = {}
    orig_alloc = kvm.allocate_slots

    def alloc(request, num_new_tokens, *a, **kw):
        res = orig_alloc(request, num_new_tokens, *a, **kw)
        try:
            total = [len(g) for g in kvm.get_blocks(request.request_id).blocks]
        except Exception:
            total = "?"
        alocaciones.setdefault(request.request_id, []).append(
            f"nuevos={num_new_tokens} computados_nuevos={kw.get('num_new_computed_tokens')} "
            f"externos={kw.get('num_external_computed_tokens')} "
            f"lookahead={kw.get('num_lookahead_tokens')} "
            f"req.computed={request.num_computed_tokens} -> bloques={total} "
            f"{'RECHAZADA' if res is None else ''}"
        )
        return res

    kvm.allocate_slots = alloc
    instrumentar.alocaciones = alocaciones

    def envuelto(request, blocks, num_external_tokens):
        cont.alloc += 1
        if num_external_tokens:
            cont.con_externo += 1
            st = cs._req_status.get(request.request_id)
            loc = getattr(st, "num_locally_computed_tokens", 0) if st else 0
            cont.max_local = max(cont.max_local, loc)
            if loc:
                cont.local_y_externo += 1
                if dump:
                    _dump_estado(
                        cs, request, blocks, num_external_tokens, loc, cdiv,
                        visto.get(request.request_id),
                    )
        return orig(request, blocks, num_external_tokens)

    cs.update_state_after_alloc = envuelto

    # Segundo assert pelado del mismo archivo (linea 771, _build_store_jobs):
    #   assert len(offload_keys) == len(offload_block_ids)
    # Tambien mata el EngineCore. Se envuelve para volcar el estado del grupo.
    orig_store = cs._build_store_jobs

    def store_envuelto(scheduler_output):
        try:
            return orig_store(scheduler_output)
        except AssertionError:
            if dump:
                bsf = cs.config.block_size_factor
                print("\n### assert en _build_store_jobs ###")
                for rid in scheduler_output.num_scheduled_tokens:
                    st = cs._req_status.get(rid)
                    if st is None:
                        continue
                    req = st.req
                    print(
                        f"  {rid}: computed={req.num_computed_tokens} "
                        f"sched={scheduler_output.num_scheduled_tokens[rid]} "
                        f"num_tokens={req.num_tokens} "
                        f"prompt={req.num_prompt_tokens} "
                        f"preempciones={req.num_preemptions} bsf={bsf}"
                    )
                    for gi, (gc, gs) in enumerate(
                        zip(cs.config.kv_group_configs, st.group_states)
                    ):
                        # cuantos bloques cree el KVCacheManager que tiene la
                        # request, contra cuantos rastrea el connector
                        try:
                            reales = len(
                                kvm.coordinator.single_type_managers[gi]
                                .req_to_blocks[rid]
                            )
                        except Exception:
                            reales = "?"
                        print(
                            f"    g{gc.group_idx} obs={gc.offloaded_block_size} "
                            f"keys={len(gs.offload_keys)} "
                            f"block_ids_connector={len(gs.block_ids)} "
                            f"bloques_reales={reales} "
                            f"next_stored={gs.next_stored_block_idx}"
                        )
                    for ln in alocaciones.get(rid, [])[-3:]:
                        print(f"    allocate_slots: {ln}")
                    print(
                        f"    scheduled_new={[r.req_id for r in scheduler_output.scheduled_new_reqs]} "
                        f"cached={list(scheduler_output.scheduled_cached_reqs.req_ids)} "
                        f"resumed={scheduler_output.scheduled_cached_reqs.resumed_req_ids} "
                        f"preempted={scheduler_output.preempted_req_ids}"
                    )
            raise

    cs._build_store_jobs = store_envuelto


def _dump_estado(cs, request, blocks, num_external, loc, cdiv, antes=None):
    """Imprime exactamente lo que mira el assert, grupo por grupo."""
    print(f"\n--- {request.request_id}: L={loc} E={num_external} ---")
    print(f"  get_computed_blocks devolvio: {antes}")
    num_cached = loc + num_external
    for gc, gb in zip(cs.config.kv_group_configs, blocks.blocks):
        bs = gc.gpu_block_size
        n = cdiv(num_cached, bs)
        boundary = n
        for i, b in enumerate(gb[:n]):
            if not b.is_null and b.block_hash is None:
                boundary = i
                break
        patron = "".join(
            "." if b.is_null else ("H" if b.block_hash is not None else "n")
            for b in gb[:n]
        )
        ids = [b.block_id for b in gb[:n]]
        marca = "  <<< ROMPE" if loc > boundary * bs else ""
        print(
            f"  g{gc.group_idx} bs={bs} sw={gc.sliding_window_size_in_blocks} "
            f"nblk={n} borde={boundary} patron={patron or '-'} ids={ids} "
            f"len(blocks)={len(gb)}{marca}"
        )


@dataclass
class Escenario:
    semilla: int
    hibrido: bool
    spec_decode: bool
    num_blocks: int
    cpu_bytes: int
    con_disco: bool
    n_requests: int
    max_model_len: int
    async_sched: bool = True


def correr(esc: Escenario, cont: Contadores | None = None, dump: bool = False) -> str | None:
    """Devuelve None si paso, o el traceback si reventó."""
    cont = cont if cont is not None else Contadores()
    rng = random.Random(esc.semilla)
    tmp = tempfile.mkdtemp(prefix="repro_kv_") if esc.con_disco else None

    vc = construir_vllm_config(
        spec_decode=esc.spec_decode,
        cpu_bytes=esc.cpu_bytes,
        fs_dir=tmp,
        max_model_len=esc.max_model_len,
        async_sched=esc.async_sched,
    )
    kvc = construir_kv_cache_config(vc, num_blocks=esc.num_blocks, hibrido=esc.hibrido)
    # normalmente lo setea el worker tras perfilar la VRAM
    vc.cache_config.num_gpu_blocks = esc.num_blocks
    sched, hash_bs = construir_scheduler(vc, kvc)
    hasher = construir_hasher(vc, hash_bs)
    instrumentar(sched, cont, dump=dump)

    # biblioteca de "parrafos" de 1 bloque para armar prefijos compartidos
    trozos = [
        [rng.randrange(10, 60000) for _ in range(BLOCK_SIZE)] for _ in range(12)
    ]

    # ARBOL DE PREFIJOS, no prompts al azar.
    #
    # El assert solo puede romperse cuando una request tiene hit LOCAL (en la
    # VRAM) y ADEMAS hit EXTERNO (en el tier de RAM/disco). Eso pide que la
    # VRAM conserve los PRIMEROS k bloques del prefijo y haya perdido la cola,
    # mientras el tier externo si la tiene.
    #
    # Es alcanzable porque vLLM libera los bloques de una request en orden
    # INVERSO (la cola primero), justamente para que el prefijo sobreviva mas.
    # Asi que la receta es: pedir un prompt largo, dejarlo terminar, meter
    # presion para que se desaloje su COLA, y volver a pedir el mismo prefijo
    # extendido. Con prompts al azar esto casi nunca cae solo (medido: 168
    # allocs, 11 hits externos, 0 con hit local).
    historial: list[list[int]] = []

    def prompt_del_arbol() -> list[int]:
        if historial and rng.random() < 0.6:
            base = list(rng.choice(historial))
            base += trozos[rng.randrange(len(trozos))] * rng.randint(1, 2)
        else:
            base = []
            for _ in range(rng.randint(1, 5)):
                base += trozos[rng.randrange(len(trozos))]
        # cola parcial para que no siempre caiga en borde de bloque
        base = base + [rng.randrange(10, 60000) for _ in range(rng.randint(0, 400))]
        # TOPE POR max_model_len. El engine real rechaza con 400 cualquier prompt
        # mas largo antes de que llegue al scheduler; aca las Request se crean a
        # mano y esa validacion no existe. Sin este tope el banco genera prompts
        # imposibles y dispara un assert (scheduler.py:771) que en produccion no
        # puede pasar: allocate_slots clampea a max_model_len y _build_store_jobs
        # no, asi que las cuentas de bloques no cierran.
        base = base[: esc.max_model_len - 1]
        if len(base) // BLOCK_SIZE >= 2:
            historial.append(base[: (len(base) // BLOCK_SIZE) * BLOCK_SIZE])
        return base

    pendientes = [
        nueva_request(f"r{i}", prompt_del_arbol(), hasher, rng.randint(1, 40))
        for i in range(esc.n_requests)
    ]

    pasos = 0
    try:
        while pendientes or sched.get_num_unfinished_requests() > 0:
            if pendientes and rng.random() < 0.5:
                sched.add_request(pendientes.pop(0))
            salida = sched.schedule()
            kv_out = completar_jobs(sched, rng, prob=0.85)
            sched.update_from_output(salida, salida_falsa(sched, salida, kv_out))
            pasos += 1
            if pasos > 4000:
                break
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        try:
            sched.shutdown()
        except Exception:
            pass


def firma(tb: str) -> str:
    """Resume un traceback a 'archivo:linea' del frame mas profundo de vLLM,
    para poder agrupar fallos distintos en vez de contarlos todos juntos."""
    ult = "?"
    for linea in tb.splitlines():
        linea = linea.strip()
        if linea.startswith('File "') and "dist-packages/vllm" in linea:
            partes = linea.split('"')[1].split("/")[-1]
            num = linea.split("line ")[1].split(",")[0]
            ult = f"{partes}:{num}"
    return ult


def esc_para(semilla: int, *, hibrido: bool, spec: bool | None,
             async_sched: bool = True) -> Escenario:
    r = random.Random(semilla * 7919)
    return Escenario(
        semilla=semilla,
        hibrido=hibrido,
        spec_decode=(r.random() < 0.5) if spec is None else spec,
        num_blocks=r.choice([16, 20, 24, 32, 48]),
        cpu_bytes=r.choice([8, 16, 32, 64]) * 1024 * 1024 * 8,
        con_disco=r.random() < 0.7,
        n_requests=r.randint(15, 60),
        max_model_len=16384,
        async_sched=async_sched,
    )


def barrido(semillas, *, hibrido, spec, dump=False, verboso=True, async_sched=True):
    """Corre un rango de semillas y agrupa los fallos por firma."""
    fallos: dict[str, list[int]] = {}
    total = Contadores()
    for i in semillas:
        esc = esc_para(i, hibrido=hibrido, spec=spec, async_sched=async_sched)
        cont = Contadores()
        tb = correr(esc, cont, dump=dump)
        total.alloc += cont.alloc
        total.con_externo += cont.con_externo
        total.local_y_externo += cont.local_y_externo
        total.max_local = max(total.max_local, cont.max_local)
        if tb:
            f = firma(tb)
            fallos.setdefault(f, []).append(i)
            if verboso and len(fallos[f]) == 1:
                print(f"\n===== primera vez {f} (semilla {i}) =====\n{esc}\n{tb}",
                      flush=True)
        if verboso:
            print(f"[{i}] {'FALLO ' + firma(tb) if tb else 'PASO'}  {cont}", flush=True)
    return fallos, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["fuzz", "seed", "matriz"])
    ap.add_argument("valor", nargs="?", type=int, default=0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-hibrido", action="store_true")
    ap.add_argument("--spec", choices=["si", "no"], default=None)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--sin-async", action="store_true")
    args = ap.parse_args()

    hib = not args.no_hibrido
    spec = None if args.spec is None else (args.spec == "si")

    if args.modo == "seed":
        cont = Contadores()
        esc = esc_para(args.valor, hibrido=hib, spec=spec,
                       async_sched=not args.sin_async)
        tb = correr(esc, cont, dump=args.dump)
        print(f"\n{esc}\n{cont}\n")
        print(tb or "PASO")
        return 1 if tb else 0

    if args.modo == "matriz":
        # Aisla las dos variables sospechosas: modelo hibrido (GDN + atencion)
        # y speculative decoding. Misma lista de semillas en las 4 celdas.
        semillas = range(1, args.iters + 1)
        print(f"{'celda':<22}{'fallos':<10}{'firmas'}")
        for h in (True, False):
            for s in (True, False):
                fal, tot = barrido(semillas, hibrido=h, spec=s, verboso=False,
                                   async_sched=not args.sin_async)
                n = sum(len(v) for v in fal.values())
                det = ", ".join(f"{k} x{len(v)} (ej. {v[0]})" for k, v in fal.items())
                celda = f"hibrido={h} spec={s}"
                print(f"{celda:<22}{n:<10}{det or '-'}   [{tot}]", flush=True)
        return 0

    fal, tot = barrido(range(1, args.iters + 1), hibrido=hib, spec=spec,
                       dump=args.dump, async_sched=not args.sin_async)
    n = sum(len(v) for v in fal.values())
    print(f"\nacumulado: {tot}")
    for k, v in fal.items():
        print(f"  {k}: {len(v)} fallos, semillas {v[:10]}")
    print(f"fallos: {n}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
