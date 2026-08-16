# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N84 — hit de prefijo inconsistente entre grupos (hibrido + MTP).

================================================================
EL SINTOMA
================================================================

`genesis-27b-qwen38-fp8` murio entero el 2026-08-16 a las 20:03:34, despues
de 14 horas arriba y ~60 requests servidos, con 4 trabajos en paralelo y 5
en cola:

    File ".../kv_connector/v1/offloading/scheduler.py", line 612,
      in update_state_after_alloc
        num_locally_computed_tokens
    AssertionError
    vllm.v1.engine.exceptions.EngineDeadError

Un `assert` pelado, sin mensaje, que no atrapa nadie: se lleva puesto el
EngineCore y todas las requests en vuelo con un 500.

================================================================
LA CAUSA (medida, no deducida)
================================================================

El assert que revienta es:

    assert num_locally_computed_tokens <= num_locally_computed_gpu_blocks * gpu_block_size

o sea: "los tokens que vLLM dice tener cacheados en la VRAM tienen que estar
cubiertos por bloques con hash". El banco de pruebas
(tests/repro/offload_partial_hit_harness.py) lo reprodujo y volco el estado
exacto en el momento del fallo:

    --- r19: L=1600 E=6400 ---
      get_computed_blocks devolvio: (1600, [[], [bloque 10]])
      g0 (atencion) bs=1600 nblk=5 borde=0 patron=nnnnn   <<< ROMPE
      g1 (GDN)      bs=1600 nblk=5 borde=4 patron=....n

`get_computed_blocks` reporta **1600 tokens ya computados** pero devuelve
**cero bloques** para el grupo de atencion. El hit sale entero del grupo GDN.
Y no es que el bloque no estuviera: la sonda confirma que el hash del bloque
0 SI estaba cacheado para los dos grupos (`bloque0_en_cache=[[12],[10]]`).

El camino, en `HybridKVCacheCoordinator.find_longest_cache_hit`:

 1. Grupo 0 (atencion completa) es grupo eagle porque hay MTP, asi que entra
    con `drop_eagle_block=True`: matchea 1 bloque y lo **descarta** (eagle
    necesita matchear uno de mas y tirar el ultimo). Queda `[]`, y el
    candidato de largo baja a 0.
 2. Grupo 1 (GDN/mamba) entra con `_max_length = min(0 + block_size, max)`,
    o sea 1600. Y aca esta el bug: `MambaManager.find_longest_cache_hit`
    **ignora `drop_eagle_block`** — no descarta nada. Busca de derecha a
    izquierda, encuentra su bloque de estado y devuelve 1 bloque.
 3. `curr_hit_length` pasa de 0 a **1600**: un grupo SUBIO el candidato. El
    algoritmo asume lo contrario; su propio comentario dice *"Each attention
    type either accepts the current candidate length or reduces it"*.
 4. `is_simple_hybrid` (1 atencion + 1 otro) corta el `while` ahi mismo, sin
    volver a consultar al grupo 0, "porque una iteracion alcanza" — cierto
    solo mientras nadie suba el largo.
 5. El truncado final recorta la lista del grupo de atencion a
    `hit_length // block_size = 1` bloque... pero esa lista esta **vacia**,
    asi que no recorta nada.

Sale `(1600, ([], [estado_gdn]))`.

================================================================
POR QUE IMPORTA MAS DE LO QUE PARECE
================================================================

El crash es la parte AFORTUNADA. El assert del connector de offloading es lo
UNICO que se da cuenta: `num_computed_tokens = 1600` significa que el
scheduler saltea el prefill de esos 1600 tokens, pero el grupo de atencion no
tiene esos bloques — `allocate_slots` le da bloques nuevos y sin escribir. En
el banco se ve directo: el grupo 0 termina con ids `[39, 17, 46, 4, 13]`, ni
rastro del bloque 12 que estaba cacheado.

Sin el connector de offloading no hay assert y el engine **no se cae**:
contesta leyendo KV de atencion basura para el primer bloque del prompt.

================================================================
CUANDO PASA (matriz medida, 60 semillas por celda)
================================================================

    hibrido=SI  spec=SI  -> 3 fallos en scheduler.py:612   <- este parche
    hibrido=SI  spec=NO  -> 0
    hibrido=NO  spec=SI  -> 0
    hibrido=NO  spec=NO  -> 0

Hacen falta LAS DOS cosas: modelo hibrido (atencion + GDN) y speculative
decoding, que es lo que hace que el grupo de atencion sea grupo eagle. Es
justo la config de los cuatro engines qwen38 de este rig.

Ademas hace falta que la request tenga hit LOCAL (VRAM) y ADEMAS hit EXTERNO
(tier de RAM/disco) al mismo tiempo, que es lo raro: por eso tardo 14 horas
en aparecer. En el banco, cuando esa combinacion se da, revienta en ~1 de
cada 3.

================================================================
EL ARREGLO
================================================================

Un grupo no puede pedir mas largo del que el candidato actual permite si su
manager no implementa el descarte de eagle. `MambaManager` no lo implementa
(se puede leer: su `find_longest_cache_hit` ni mira el parametro), asi que
para los grupos MambaSpec no se infla `_max_length` ni se pide el descarte.

Con eso el grupo GDN solo puede ACEPTAR o BAJAR el candidato, que es la
invariante que el algoritmo ya asumia, y las listas de bloques quedan
consistentes con el largo reportado.

No se pierden hits reales: en el caso normal el grupo de atencion matchea N+1
y descarta 1, el candidato queda en N bloques, y el grupo GDN encuentra su
estado en el bloque N-1 buscando de derecha a izquierda. Lo unico que se
pierde es el "hit" de 1 bloque que hoy es directamente falso.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default ON. Kill switch: GENESIS_DISABLE_PN84=1.
- Solo toca el camino de LOOKUP del cache de prefijo, y solo para grupos
  MambaSpec de un coordinador hibrido. Un modelo sin capas mamba nunca entra.
- No puede inventar hits: solo puede reducirlos. El peor caso es re-prefillear
  un bloque que hoy se "ahorra" leyendo KV que no existe.
- Si upstream arregla esto (que `MambaManager` respete `drop_eagle_block`, o
  que el `is_simple_hybrid` deje de cortar), el ancla deriva y el parche hace
  SKIP limpio en vez de aplicarse dos veces.

================================================================
VERIFICACION
================================================================

    tests/repro/offload_partial_hit_harness.py matriz --iters 60

antes: 3 fallos en scheduler.py:612 (celda hibrido+spec)
despues: 0
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN84_MARKER = "[Genesis PN84 hybrid eagle hit consistency]"


ANCHOR_OLD = (
    "                drop_eagle_block = use_eagle and idx not in eagle_verified\n"
    "\n"
    "                _max_length = curr_hit_length\n"
    "                if drop_eagle_block:\n"
)

ANCHOR_NEW = (
    "                # " + GENESIS_PN84_MARKER + "\n"
    "                # MambaManager.find_longest_cache_hit IGNORA\n"
    "                # drop_eagle_block: no descarta el ultimo bloque. Si se le\n"
    "                # infla _max_length en un bloque (que es lo que hace la\n"
    "                # sonda de eagle), el grupo GDN puede devolver un hit MAS\n"
    "                # LARGO que el candidato actual -> curr_hit_length SUBE, y\n"
    "                # con is_simple_hybrid el while corta sin volver a\n"
    "                # consultar al grupo de atencion. Resultado medido:\n"
    "                # hit_length=1600 con CERO bloques en el grupo de atencion,\n"
    "                # o sea 1600 tokens marcados como computados cuyo KV no\n"
    "                # existe. Mata el engine en el assert de\n"
    "                # offloading/scheduler.py:612 y, sin ese connector, da\n"
    "                # salida basura en silencio.\n"
    "                # Ver wiring/hybrid/patch_N84_hybrid_eagle_hit_consistency.py\n"
    "                import os as _g84_os\n"
    "                from vllm.v1.kv_cache_interface import MambaSpec as _g84_Mamba\n"
    "                _g84_honra = (\n"
    "                    not isinstance(spec, _g84_Mamba)\n"
    "                    or _g84_os.environ.get('GENESIS_DISABLE_PN84') == '1'\n"
    "                )\n"
    "                drop_eagle_block = (\n"
    "                    use_eagle and idx not in eagle_verified and _g84_honra\n"
    "                )\n"
    "\n"
    "                _max_length = curr_hit_length\n"
    "                if drop_eagle_block:\n"
)


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/kv_cache_coordinator.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN84 hybrid eagle hit consistency",
        target_file=str(target),
        marker=GENESIS_PN84_MARKER,
        sub_patches=[
            TextPatch(
                name="pn84_no_eagle_probe_for_mamba",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # senales de que upstream ya lo arreglo por su cuenta
            "MambaSpec) and drop_eagle_block",
            "curr_hit_length = min(",
        ],
    )


def apply() -> tuple[str, str]:
    import os

    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN84")
    log_decision("PN84", decision, reason)
    if not decision:
        return "skipped", reason
    if os.environ.get("GENESIS_DISABLE_PN84") == "1":
        return "skipped", "GENESIS_DISABLE_PN84=1 (kill switch)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/core/kv_cache_coordinator.py not found"
    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN84 applied: en modelos hibridos con MTP, el grupo GDN ya no puede "
            "SUBIR el largo del hit de prefijo (MambaManager ignora "
            "drop_eagle_block). Sin esto find_longest_cache_hit devuelve un hit "
            "de N tokens con CERO bloques en el grupo de atencion: mata el engine "
            "en el assert de offloading/scheduler.py:612 y, sin connector de "
            "offloading, contesta con KV basura en silencio. "
            "Kill switch: GENESIS_DISABLE_PN84=1."
        ),
        patch_name="PN84 hybrid eagle hit consistency",
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
            return GENESIS_PN84_MARKER in f.read()
    except Exception:
        return False
