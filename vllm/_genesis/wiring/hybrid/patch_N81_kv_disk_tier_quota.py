# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N81 — cuota y poda del tier de disco del cache de KV.

================================================================
QUÉ RESUELVE
================================================================

`TieringOffloadingSpec` permite una segunda capa de cache de KV en disco
(`secondary_tiers: [{"type": "fs", "root_dir": ...}]`), que es lo que evita
re-prefillear un prefijo largo cuando los subagentes lo desalojan de la VRAM.

Pero `FileSystemTierManager` **no tiene cuota ni limpieza**. Verificado
leyendo su código: los únicos `os.remove` del módulo son manejo de errores
(borrar el temporal si falla una escritura, borrar un archivo ilegible), y
ni `shutdown()` ni `file_mapper` tocan los archivos. La interfaz de un
secondary tier ni siquiera declara un método de desalojo.

Resultado sin este parche: `root_dir` crece indefinidamente y sobrevive a
los reinicios. En una sola sesión de pruebas ya escribió 3,4 GB.

La alternativa —un cron en el host— queda fuera del proyecto y del
contenedor, se desincroniza con la config y no sabe nada del engine.

================================================================
CÓMO
================================================================

`SecondaryTierManager.on_schedule_end()` existe justamente para esto. Su
docstring lo dice: *"Called once at the end of each scheduler step.
Secondary tiers may override this for per-step cleanup or deferred work
submission."* Desde 0.27 `FileSystemTierManager` lo implementa, pero solo
para hacer flush del lookup manager: sigue sin cuota ni limpieza.

PN81 extiende ese hook con dos barridos independientes:

**A. Cuota del directorio propio** — cada `GENESIS_KV_DISK_CHECK_SECS`
   (default 60) recorre `root_dir`, suma tamaños y, si supera
   `GENESIS_KV_DISK_MAX_GB` (default 30), borra los archivos más **viejos
   por mtime** hasta bajar al `GENESIS_KV_DISK_TARGET_RATIO` del límite
   (default 0.85), para no podar en cada paso.

   El gate es por RELOJ, no por pasos del scheduler. `on_schedule_end` suma
   un paso por iteración: con el engine ocioso los pasos no se acumulan y la
   cuota no corría nunca. Medido con el gate viejo (`CHECK_EVERY=2000`):
   22 min sirviendo 61 requests no llegaron a 2000 pasos y el directorio
   tocó 31,4 GiB con límite de 30 sin una sola poda.

**B. Purga de directorios abandonados** — cada
   `GENESIS_KV_DISK_ORPHAN_CHECK_SECS` (default 3600) borra los directorios
   de OTROS modelos sin usar hace más de `GENESIS_KV_DISK_ORPHAN_DAYS`.
   Sin esto, cada modelo que se prueba deja el suyo para siempre: medido,
   142 GiB en total, 54 de un modelo que ya ni estaba descargado.

   Cadencia propia y más lenta que la de la cuota porque recorre TODOS los
   directorios de `root_dir`, no solo el del modelo actual.

⚠️ La cuota NO es por rank — el total en disco es `GENESIS_KV_DISK_MAX_GB`,
no un múltiplo. `TieringOffloadingSpec.get_manager()` se llama desde un
único sitio (`offloading/scheduler.py:457`, `OffloadingConnectorScheduler`),
así que existe UN solo `FileSystemTierManager`, en el proceso scheduler, con
`parallel_config.rank == 0`. En disco solo aparece `{base_path}_r0`, incluso
con TP=2. Lo que sí es por rank es el mmap de RAM, que se crea aparte en
`create_handlers()` (worker-side) — de ahí venía la confusión.

Podar por mtime es una aproximación a LRU: `store_block` saltea el archivo
si ya existe, así que el mtime marca la primera escritura, no el último
uso. Un bloque muy reusado pero viejo podría podarse antes que uno nuevo y
poco usado. Es aceptable porque el costo de un fallo es un re-prefill de
ese bloque, no un error — y el tier de RAM (con ARC) ya protege lo caliente.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default OFF (`GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1`).
- El escaneo de la cuota es 1 cada 60s de reloj, no por paso, y solo recorre
  el directorio del modelo actual.
- El barrido de huérfanos es 1 cada hora y recorre todos los directorios.
  Ambos son `os.walk` + `stat`: no leen contenido de los bloques.
- Todo el cuerpo va en try/except: un fallo de la poda NO puede tumbar el
  scheduler. Se avisa una vez y se sigue.
- No toca el camino de datos: no interviene en store, load ni lookup. Solo
  borra archivos que ya no entran en la cuota.
- Si el tier de disco no está configurado, el parche es inerte: el método
  que agrega nunca se llama.

================================================================
COMPOSICIÓN
================================================================

Ortogonal a PN80 (sonda de VRAM) y a los pools de FFN (PN12/PN25).
Requiere que el compose declare el tier `fs` en `--kv-transfer-config`.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN81_MARKER = "[Genesis PN81 kv disk tier quota]"


ANCHOR_OLD = (
    "    @override\n"
    "    def on_schedule_end(self, context: ScheduleEndContext) -> None:\n"
    "        self._lookup_manager.flush()\n"
)


ANCHOR_NEW = (
    "    # " + GENESIS_PN81_MARKER + "\n"
    "    # Extiende el on_schedule_end de upstream (que solo hace flush del\n"
    "    # lookup manager) con la cuota de disco. Sin esto root_dir crece sin\n"
    "    # techo: el tier fs no tiene poda propia.\n"
    "    # Ver wiring/hybrid/patch_N81_kv_disk_tier_quota.py\n"
    "    @override\n"
    "    def on_schedule_end(self, context: ScheduleEndContext) -> None:\n"
    "        self._lookup_manager.flush()\n"
    "        import os as _g81_os\n"
    "        if _g81_os.environ.get('GENESIS_ENABLE_PN81_KV_DISK_QUOTA') != '1':\n"
    "            return\n"
    "        try:\n"
    "            # GATE POR TIEMPO, no por pasos del scheduler.\n"
    "            # El gate viejo era `paso % GENESIS_KV_DISK_CHECK_EVERY`, y\n"
    "            # on_schedule_end sube un paso por iteracion del scheduler: si el\n"
    "            # engine esta ocioso, los pasos no se acumulan y la cuota NUNCA\n"
    "            # corre. Medido: 22 min sirviendo 61 requests cortos no alcanzaron\n"
    "            # los 2000 pasos, y el directorio llego a 31,4 GiB con limite de 30\n"
    "            # sin una sola poda. Con reloj, la cuota corre haya o no trafico.\n"
    "            import time as _g81_time\n"
    "            _secs = float(_g81_os.environ.get('GENESIS_KV_DISK_CHECK_SECS', '60'))\n"
    "            _now = _g81_time.monotonic()\n"
    "            _last = getattr(self, '_genesis_pn81_last', 0.0)\n"
    "            if _now - _last < _secs:\n"
    "                return\n"
    "            self._genesis_pn81_last = _now\n"
    "\n"
    "            _max = float(_g81_os.environ.get('GENESIS_KV_DISK_MAX_GB', '30'))\n"
    "            _ratio = float(_g81_os.environ.get(\n"
    "                'GENESIS_KV_DISK_TARGET_RATIO', '0.85'))\n"
    "            # El directorio real NO es base_path: FileMapper escribe en\n"
    "            # f'{base_path}_r{rank}' (ver get_file_name). Cada rank poda el\n"
    "            # SUYO, asi que la cuota es POR RANK: con TP=2 el total en disco\n"
    "            # es 2 x GENESIS_KV_DISK_MAX_GB.\n"
    "            _bp = getattr(self.file_mapper, 'base_path', None)\n"
    "            _rk = getattr(self.file_mapper, 'rank', 0)\n"
    "            _root = f'{_bp}_r{_rk}' if _bp else None\n"
    "            if not _root or not _g81_os.path.isdir(_root):\n"
    "                return\n"
    "            # PURGA DE DIRECTORIOS ABANDONADOS. La cuota de arriba solo mira\n"
    "            # el directorio del modelo ACTUAL. Cada modelo que se prueba deja\n"
    "            # el suyo, y nadie los borra nunca: medido, 54 GiB de un modelo que\n"
    "            # ya ni estaba descargado, y 142 GiB en total en /kv-offload.\n"
    "            # Se borran por ANTIGUEDAD (no por 'no es el mio'): un modelo que\n"
    "            # usaste ayer conserva su cache, que es justamente para lo que\n"
    "            # existe el tier de disco.\n"
    "            #\n"
    "            # PERIODICA, NO UNA SOLA VEZ (2026-08-21). El gate viejo era un\n"
    "            # booleano `_genesis_pn81_orphans` que se prendia en la primera\n"
    "            # pasada y no se apagaba nunca: la purga corria al arrancar y\n"
    "            # listo. Un directorio que queda huerfano DESPUES del arranque no\n"
    "            # se revisaba jamas mientras el engine siguiera vivo.\n"
    "            # Medido: al arrancar el 08-19, orcarouter_..._3ef784eef730_r0\n"
    "            # llevaba 1,6 dias sin uso -- por debajo de los 3 del umbral, asi\n"
    "            # que se salteo con razon. Dos dias despues ya calificaba y sus\n"
    "            # 25,47 GiB seguian ahi, porque la purga no volvia a correr.\n"
    "            # Despistaba que su hermano sin datos (solo config.json, 0 GiB) SI\n"
    "            # se borro: nadie le escribe despues de crearlo, asi que su mtime\n"
    "            # ya tenia 3,2 dias. Se borro la cascara y quedo el contenido.\n"
    "            # Ahora corre cada GENESIS_KV_DISK_ORPHAN_CHECK_SECS (default una\n"
    "            # hora). Cadencia propia y no la de la cuota (60s) porque este\n"
    "            # barrido recorre TODOS los directorios de /kv-offload y no solo\n"
    "            # el del modelo actual: son miles de stat() en vez de cientos.\n"
    "            # La primera pasada sigue siendo al arranque (_olast is None).\n"
    "            _dias = float(_g81_os.environ.get('GENESIS_KV_DISK_ORPHAN_DAYS', '0'))\n"
    "            _osecs = float(_g81_os.environ.get(\n"
    "                'GENESIS_KV_DISK_ORPHAN_CHECK_SECS', '3600'))\n"
    "            _olast = getattr(self, '_genesis_pn81_orphan_last', None)\n"
    "            if _dias > 0 and (_olast is None or _now - _olast >= _osecs):\n"
    "                self._genesis_pn81_orphan_last = _now\n"
    "                import sys as _g81_s3\n"
    "                _padre = _g81_os.path.dirname(_root.rstrip('/'))\n"
    "                _mio = _g81_os.path.basename(_root.rstrip('/'))\n"
    "                _corte = _g81_time.time() - _dias * 86400\n"
    "                for _d in sorted(_g81_os.listdir(_padre)):\n"
    "                    if _d == _mio or _d.startswith(_mio.rsplit('_r', 1)[0]):\n"
    "                        continue\n"
    "                    _dp2 = _g81_os.path.join(_padre, _d)\n"
    "                    if not _g81_os.path.isdir(_dp2):\n"
    "                        continue\n"
    "                    _reciente = 0.0\n"
    "                    _peso = 0\n"
    "                    for _r2, _dn2, _fn2 in _g81_os.walk(_dp2):\n"
    "                        for _f2 in _fn2:\n"
    "                            try:\n"
    "                                _st2 = _g81_os.stat(_g81_os.path.join(_r2, _f2))\n"
    "                            except OSError:\n"
    "                                continue\n"
    "                            _reciente = max(_reciente, _st2.st_mtime)\n"
    "                            _peso += _st2.st_size\n"
    "                    if _peso and _reciente < _corte:\n"
    "                        import shutil as _g81_sh\n"
    "                        try:\n"
    "                            _g81_sh.rmtree(_dp2)\n"
    "                            # PN88: sacar el desalojo a metricas. Best-effort\n"
    "                            # y con import local: si PN88 no esta, no pasa nada.\n"
    "                            try:\n"
    "                                from vllm._genesis import kv_tier_metrics as _g88\n"
    "                                _g88.note_eviction('disk', 'orphan', 1, _peso)\n"
    "                            except Exception:\n"
    "                                pass\n"
    "                            print('[PN81] cache abandonada borrada: %s '\n"
    "                                  '(%.2f GiB, sin usar hace %.1f dias)'\n"
    "                                  % (_d, _peso / (1 << 30),\n"
    "                                     (_g81_time.time() - _reciente) / 86400.0),\n"
    "                                  file=_g81_s3.stderr, flush=True)\n"
    "                        except OSError as _e2:\n"
    "                            print('[PN81] no pude borrar %s: %r' % (_d, _e2),\n"
    "                                  file=_g81_s3.stderr, flush=True)\n"

    "            _limit = int(_max * (1 << 30))\n"
    "            _files = []\n"
    "            _total = 0\n"
    "            for _dp, _dn, _fn in _g81_os.walk(_root):\n"
    "                for _f in _fn:\n"
    "                    _p = _g81_os.path.join(_dp, _f)\n"
    "                    try:\n"
    "                        _st = _g81_os.stat(_p)\n"
    "                    except OSError:\n"
    "                        continue\n"
    "                    _files.append((_st.st_mtime, _st.st_size, _p))\n"
    "                    _total += _st.st_size\n"
    "            if not getattr(self, '_genesis_pn81_visto', False):\n"
    "                self._genesis_pn81_visto = True\n"
    "                import sys as _g81_s0\n"
    "                print('[PN81] cuota activa sobre %s: %.2f GiB de %.2f GiB '\n"
    "                      '(chequeo cada %.0fs)'\n"
    "                      % (_root, _total / (1 << 30), _max, _secs),\n"
    "                      file=_g81_s0.stderr, flush=True)\n"
    "            # PN88: ocupacion del tier como gauge. Va ANTES del early return\n"
    "            # para que se publique en cada chequeo, se pode o no.\n"
    "            try:\n"
    "                from vllm._genesis import kv_tier_metrics as _g88\n"
    "                _g88.set_occupancy('disk', _total, _limit)\n"
    "            except Exception:\n"
    "                pass\n"
    "            if _total <= _limit:\n"
    "                return\n"
    "            # podar del mas viejo al mas nuevo hasta el ratio objetivo\n"
    "            _target = int(_limit * _ratio)\n"
    "            _files.sort()\n"
    "            _freed = 0\n"
    "            _n = 0\n"
    "            for _mt, _sz, _p in _files:\n"
    "                if _total - _freed <= _target:\n"
    "                    break\n"
    "                try:\n"
    "                    _g81_os.remove(_p)\n"
    "                    _freed += _sz\n"
    "                    _n += 1\n"
    "                except OSError:\n"
    "                    continue\n"
    "            try:\n"
    "                from vllm._genesis import kv_tier_metrics as _g88\n"
    "                _g88.note_eviction('disk', 'quota', _n, _freed)\n"
    "                _g88.set_occupancy('disk', _total - _freed, _limit)\n"
    "            except Exception:\n"
    "                pass\n"
    "            import sys as _g81_sys\n"
    "            print('[PN81] cuota del tier de disco: %.2f GiB > limite %.2f GiB '\n"
    "                  '-> borrados %d bloques mas viejos, liberados %.2f GiB '\n"
    "                  '(queda %.2f GiB)'\n"
    "                  % (_total / (1 << 30), _max, _n, _freed / (1 << 30),\n"
    "                     (_total - _freed) / (1 << 30)),\n"
    "                  file=_g81_sys.stderr, flush=True)\n"
    "        except Exception as _g81_err:\n"
    "            # la poda NUNCA debe tumbar el scheduler; se avisa una sola vez\n"
    "            if not getattr(self, '_genesis_pn81_failed', False):\n"
    "                self._genesis_pn81_failed = True\n"
    "                import sys as _g81_s2\n"
    "                print('[PN81] la poda fallo y se desactiva (el engine sigue '\n"
    "                      'normal): %r' % (_g81_err,), file=_g81_s2.stderr, flush=True)\n"
)



# ── Sub-parche 2: limpieza de mmaps huerfanos en /dev/shm ────────────────────
# vLLM tiene SharedOffloadRegion.cleanup() que hace unlink del mmap, pero solo
# corre en cierre ORDENADO. Un `docker rm -f` (o un crash) lo saltea, y con
# `ipc: host` el archivo queda en el /dev/shm del HOST, sobreviviendo al
# contenedor. Medido: 4 huerfanos de 5,3 GB llenaron los 16 GB de /dev/shm y el
# arranque siguiente murio con `madvise: Bad address`.
# Se limpia al ARRANCAR (no al cerrar), que es lo unico que se puede garantizar.
SHM_ANCHOR_OLD = (
    '        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"\n'
)

SHM_ANCHOR_NEW = (
    "        # " + GENESIS_PN81_MARKER + " (purga de huerfanos)\n"
    "        # Borra mmaps de corridas anteriores que quedaron por un cierre\n"
    "        # abrupto. Es CONSERVADOR: saltea los que algun proceso visible\n"
    "        # tiene mapeados y los muy recientes, para no pisar a un engine\n"
    "        # hermano que este arrancando.\n"
    "        try:\n"
    "            import os as _g81s_os, glob as _g81s_glob, time as _g81s_t, sys as _g81s_sys\n"
    "            if _g81s_os.environ.get('GENESIS_ENABLE_PN81_KV_DISK_QUOTA') == '1':\n"
    "                # Sin guard de edad: al arrancar, todo mmap que NADIE tenga\n"
    "                # mapeado es basura de una corrida anterior. El chequeo de\n"
    "                # /proc/*/maps es la unica salvaguarda necesaria.\n"
    "                _min_age = float(_g81s_os.environ.get(\n"
    "                    'GENESIS_SHM_ORPHAN_MIN_AGE_S', '0'))\n"
    "                _en_uso = set()\n"
    "                for _mp in _g81s_glob.glob('/proc/*/maps'):\n"
    "                    try:\n"
    "                        with open(_mp) as _fh:\n"
    "                            for _ln in _fh:\n"
    "                                if 'vllm_offload_' in _ln:\n"
    "                                    _en_uso.add(_ln.rsplit(' ', 1)[-1].strip())\n"
    "                    except Exception:\n"
    "                        continue\n"
    "                _ahora = _g81s_t.time()\n"
    "                for _f in _g81s_glob.glob('/dev/shm/vllm_offload_*.mmap'):\n"
    "                    if _f in _en_uso:\n"
    "                        continue\n"
    "                    try:\n"
    "                        _st = _g81s_os.stat(_f)\n"
    "                        if _ahora - _st.st_mtime < _min_age:\n"
    "                            continue\n"
    "                        _g81s_os.unlink(_f)\n"
    "                        print('[PN81] mmap huerfano borrado de /dev/shm: %s '\n"
    "                              '(%.2f GiB liberados)'\n"
    "                              % (_f, _st.st_size / (1 << 30)),\n"
    "                              file=_g81s_sys.stderr, flush=True)\n"
    "                    except OSError:\n"
    "                        continue\n"
    "        except Exception:\n"
    "            pass  # la purga nunca debe impedir el arranque\n"
) + SHM_ANCHOR_OLD


def _is_enabled() -> bool:
    import os

    return os.environ.get("GENESIS_ENABLE_PN81_KV_DISK_QUOTA") == "1"


def _shm_patcher() -> TextPatcher | None:
    """Segundo target: la purga de huerfanos vive en otro archivo, asi que
    necesita su propio TextPatcher (uno por archivo)."""
    target = resolve_vllm_file("v1/kv_offload/cpu/shared_offload_region.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN81 shm orphan purge",
        target_file=str(target),
        marker=GENESIS_PN81_MARKER,
        sub_patches=[
            TextPatch(
                name="pn81_shm_orphan_purge",
                anchor=SHM_ANCHOR_OLD,
                replacement=SHM_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["orphan", "stale_mmap"],
    )


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/fs/manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN81 KV disk tier quota",
        target_file=str(target),
        marker=GENESIS_PN81_MARKER,
        sub_patches=[
            TextPatch(
                name="pn81_on_schedule_end_quota",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Si upstream le agrega cuota propia al tier de disco, este parche
            # sobra y podria pelearse con la de ellos -> SKIP limpio.
            # (on_schedule_end ya existe en 0.27 pero solo hace flush del
            # lookup manager: no es senal de cuota.)
            "max_bytes",
            "capacity_bytes",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN81")
    log_decision("PN81", decision, reason)
    if not decision:
        return "skipped", reason
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN81_KV_DISK_QUOTA not set; default OFF. "
            "Agrega cuota y poda por antiguedad al tier de disco del cache de "
            "KV, que en vLLM no tiene ninguna: root_dir crece sin techo. "
            "Tunear con GENESIS_KV_DISK_MAX_GB / _CHECK_EVERY / _TARGET_RATIO."
        )
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/tiering/fs/manager.py not found"
    result, failure = p.apply()
    # el purgador de /dev/shm es best-effort: si su ancla derivo, la cuota de
    # disco (lo principal) igual se aplica
    sp = _shm_patcher()
    if sp is not None:
        try:
            sp.apply()
        except Exception:
            pass
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN81 applied: on_schedule_end() implementado en FileSystemTierManager "
            "con cuota de disco. Poda por mtime al superar GENESIS_KV_DISK_MAX_GB, "
            "bajando hasta GENESIS_KV_DISK_TARGET_RATIO del limite. Ademas purga mmaps huerfanos de /dev/shm dejados por cierres abruptos."
        ),
        patch_name="PN81 KV disk tier quota",
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
            return GENESIS_PN81_MARKER in f.read()
    except Exception:
        return False
