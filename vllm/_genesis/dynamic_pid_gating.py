# SPDX-License-Identifier: Apache-2.0
"""Genesis PN115 — admisión por headroom de KV + bypass por prioridad.

Tres mecanismos INDEPENDIENTES, cada uno con su switch:

1. Gating por headroom de KV (`GENESIS_PN115_KV_GATING`, default 1)
   Difiere la admisión de un request en espera cuando meterlo dejaría al
   motor sin bloques KV libres. Es protección real: evita que
   `allocate_slots` falle y que el scheduler entre en tormenta de
   preemption. Se apoya SOLO en el BlockPool (verdad física); no estima
   capacidad en tokens ni proyecta uso.

2. PID de latencia (`GENESIS_PN115_LATENCY_PID`, default 0)
   Baja la concurrencia admitida cuando el paso de decode es más lento que
   su línea base PARA ESA MISMA CONCURRENCIA. Va APAGADO por defecto:
   cambia throughput por latencia, y con `--async-scheduling` la señal que
   se puede medir desde `schedule()` es la cadencia del scheduler, no la
   latencia del forward. Con async scheduling se auto-desactiva.

3. Bypass por prioridad
   Un request de prioridad alta puede desalojar a uno de prioridad menor
   cuando el motor está en `max_num_seqs`. La preemption se hace con el
   MISMO rollback de estado que hace vLLM en su propio camino
   (`Scheduler.schedule`), y el paso termina sin admitir a nadie más:
   es el invariante de upstream, cuyo bucle de waiting entero está bajo
   `if not preempted_reqs:`.

Invariantes que el gating respeta siempre (anti-deadlock):
  - Nunca difiere si el motor está por debajo de `min_concurrency`
    corriendo. Sin esto, un request más grande que el headroom queda en la
    cola para siempre y el motor se queda quieto con trabajo pendiente.
  - Nunca difiere por un tamaño que el request no puede evitar: con
    chunked prefill sólo se contabiliza el chunk de ESTE paso, no el
    prompt entero.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

PID_CONTROL_FILE = "/dev/shm/genesis_pid_control.json"
PID_STATUS_FILE = "/dev/shm/genesis_pid_status.json"

if TYPE_CHECKING:  # pragma: no cover
    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401
    from vllm.v1.request import Request  # noqa: F401

log = logging.getLogger("genesis.dynamic_pid_gating")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# Prometheus metrics for Genesis PN115 PID admission tuner
try:
    from prometheus_client import Counter, Gauge

    PROM_PID_EMA_STEP_MS = Gauge(
        "vllm:pid_ema_step_time_ms",
        "Genesis PN115 PID EMA step time in milliseconds",
    )
    PROM_PID_TARGET_STEP_MS = Gauge(
        "vllm:pid_target_step_time_ms",
        "Genesis PN115 PID target step time in milliseconds",
    )
    PROM_PID_ACTIVE_KV_TOKENS = Gauge(
        "vllm:pid_active_kv_tokens",
        "Genesis PN115 active KV cache tokens tracked by PID controller",
    )
    PROM_PID_MAX_KV_TOKENS = Gauge(
        "vllm:pid_max_kv_tokens",
        "Genesis PN115 maximum active KV cache token budget before backoff",
    )
    PROM_PID_CONCURRENCY_LIMIT = Gauge(
        "vllm:pid_concurrency_limit",
        "Genesis PN115 dynamic concurrency admission limit",
    )
    PROM_PID_STATUS = Gauge(
        "vllm:pid_status",
        "Genesis PN115 admission status (1=normal/unthrottled, 0=throttled)",
    )
    PROM_PID_GATED_TOTAL = Counter(
        "vllm:pid_gated_requests_total",
        "Genesis PN115 total count of request admission deferrals/gates",
    )
    PROM_PID_BYPASS_TOTAL = Counter(
        "vllm:pid_priority_bypass_total",
        "Genesis PN115 total count of high-priority emergency bypasses",
    )
    PROM_PID_PREEMPTIONS_TOTAL = Counter(
        "vllm:pid_emergency_preemptions_total",
        "Genesis PN115 total count of emergency preemptions for high-priority admission",
    )
    PROM_PID_ACTIVE_CONCURRENCY = Gauge(
        "vllm:pid_active_concurrency",
        "Genesis PN115 currently running requests in scheduler",
    )
    PROM_PID_FREE_BLOCKS = Gauge(
        "vllm:pid_free_kv_blocks",
        "Genesis PN115 free KV blocks reported by the BlockPool",
    )
    PROM_PID_GATED_PREFILL = Counter(
        "vllm:pid_gated_by_prefill_total",
        "Genesis PN115 requests deferred because another prefill is in flight",
    )
    PROM_PID_SHORT_BYPASS = Counter(
        "vllm:pid_short_prompt_bypass_total",
        "Genesis PN115 short prompts admitted past the prefill serialization",
    )
except Exception as _prom_err:  # pragma: no cover
    log.debug("Prometheus client not initialized for PN115: %s", _prom_err)
    PROM_PID_EMA_STEP_MS = None
    PROM_PID_SHORT_BYPASS = None
    PROM_PID_TARGET_STEP_MS = None
    PROM_PID_ACTIVE_KV_TOKENS = None
    PROM_PID_MAX_KV_TOKENS = None
    PROM_PID_CONCURRENCY_LIMIT = None
    PROM_PID_STATUS = None
    PROM_PID_GATED_TOTAL = None
    PROM_PID_BYPASS_TOTAL = None
    PROM_PID_PREEMPTIONS_TOTAL = None
    PROM_PID_ACTIVE_CONCURRENCY = None
    PROM_PID_FREE_BLOCKS = None
    PROM_PID_GATED_PREFILL = None

# Agent-to-priority heuristic mapping when `request.priority` is 0 (default).
#
# OJO: sólo dispara si el CLIENTE fija el request_id (OpenAI lo permite vía el
# header `X-Request-Id`, que vLLM propaga). Con ids autogenerados
# (`chatcmpl-<uuid>`) este mapa es inerte y la prioridad efectiva es la que
# venga en `request.priority`. No es un bug: es el alcance real del heurístico.
AGENT_PRIORITY_MAP: dict[str, int] = {
    "coach": -10,
    "primary": -8,
    "primary_high": -8,
    "coder": -5,
    "verifier": -5,
    "planner": -5,
    "primary_low": 0,
    "primary_nothink": 0,
    "planner_nothink": 0,
    "utility": 5,
    "explorer": 10,
    "vision": 10,
    "art": 10,
}

# Prefijos que vLLM/OpenAI anteponen al id que manda el cliente.
_ID_PREFIXES = ("chatcmpl-", "cmpl-", "embd-", "rerank-")


def _is_prefilling(request: Any) -> bool:
    """¿Este request todavia esta computando su prompt?

    El predicado tiene que ser `num_computed_tokens < num_prompt_tokens`.
    Usar `num_tokens` es un error silencioso: es una property que vale
    `num_prompt_tokens + len(output_token_ids)`, asi que un request en decode
    SIEMPRE la cumple y todo el batch parece estar en prefill.
    """
    npt = getattr(request, "num_prompt_tokens", None)
    if npt is None:
        return False
    return getattr(request, "num_computed_tokens", 0) < npt


def _is_running(request: Any) -> bool:
    """¿Está el request en estado RUNNING?

    `Scheduler._preempt_request` asertea `status == RequestStatus.RUNNING`, así
    que elegir cualquier otra cosa como víctima revienta el scheduler. Se
    compara por NOMBRE en vez de importar `RequestStatus`: un import que falla
    dejaría el filtro sin efecto justo donde importa, y sólo `RUNNING` se llama
    así en el enum de vLLM.
    """
    status = getattr(request, "status", None)
    if status is None:
        return False
    return (getattr(status, "name", None) or str(status)) == "RUNNING"


class PIDAdmissionController:
    """Admisión por headroom de KV, con PID de latencia opcional."""

    _instance: Optional[PIDAdmissionController] = None

    @classmethod
    def get_instance(cls) -> PIDAdmissionController:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.enabled = _env_flag("GENESIS_ENABLE_PN115_PID_GATING", "0")

        # Los dos controladores son independientes.
        self.kv_gating_enabled = _env_flag("GENESIS_PN115_KV_GATING", "1")
        self.latency_pid_enabled = _env_flag("GENESIS_PN115_LATENCY_PID", "0")
        # Se apaga solo si el motor usa async scheduling (ver _observe_step).
        self.latency_pid_forced_off_reason: str = ""

        env_target = os.environ.get("GENESIS_PID_TARGET_STEP_MS")
        # None = línea base auto, POR NIVEL DE CONCURRENCIA.
        self.target_step_ms: Optional[float] = float(env_target) if env_target else None

        env_min_c = os.environ.get("GENESIS_PID_MIN_CONCURRENCY")
        self.min_concurrency: int = max(1, int(env_min_c)) if env_min_c else 2

        env_max_c = os.environ.get("GENESIS_PID_MAX_CONCURRENCY")
        self.max_concurrency: Optional[int] = int(env_max_c) if env_max_c else None

        self.high_prio_threshold = int(os.environ.get("GENESIS_PID_HIGH_PRIO_THRESHOLD", "0"))
        self.kp = float(os.environ.get("GENESIS_PID_KP", "0.15"))
        self.kd = float(os.environ.get("GENESIS_PID_KD", "0.05"))

        # Cuánto puede empeorar un paso respecto de su línea base A LA MISMA
        # concurrencia antes de considerarlo congestión.
        self.latency_slack = float(os.environ.get("GENESIS_PID_LATENCY_SLACK", "0.30"))
        # Muestras mínimas en un nivel de concurrencia antes de controlar ahí.
        self.min_baseline_samples = int(os.environ.get("GENESIS_PID_MIN_SAMPLES", "12"))
        # La línea base puede derivar hacia arriba: un paso afortunado no la
        # fija para siempre.
        self.baseline_decay = float(os.environ.get("GENESIS_PID_BASELINE_DECAY", "1.02"))

        # Serializacion de prefills. El prefill NO escala con la concurrencia
        # (medido 2026-09-12: 1.504 / 1.511 / 1.489 tok/s agregados con 1, 2 y
        # 4 requests frios de 42k, y lo mismo con el presupuesto de batch en
        # 4096 o en 16384). Correr N prefills a la vez le da 1/N a cada uno y
        # no entrega ninguno hasta el final; de a uno, el tiempo total es el
        # mismo pero cada request queda usable N veces antes.
        # 0 = desactivado. 1 = un prefill por vez.
        self.max_concurrent_prefills = int(
            os.environ.get("GENESIS_PN115_MAX_CONCURRENT_PREFILLS", "0")
        )
        # Prioridad AUTOMATICA: un prompt de hasta N tokens no espera detras de un prefill en
        # curso. La serializacion tiene sentido entre prefills comparables (6 subagentes de
        # 50k); contra un pedido de 300 tokens es pura espera — medido 2026-09-20: 26 s detras
        # de un prefill de 59k, contra 108 ms con el motor libre. Exime SOLO de la
        # serializacion: no saltea el headroom de KV ni desaloja a nadie, que eso sigue siendo
        # de la prioridad forzada (`priority` / `genesis_agent`).
        # Rinde junto con --long-prefill-token-threshold < max-num-batched-tokens: si el largo
        # se lleva el paso entero, el corto admitido igual espera un chunk completo.
        # 0 = desactivado (comportamiento anterior).
        self.short_prompt_tokens = int(
            os.environ.get("GENESIS_PN115_PROMPT_CORTO_TOKENS", "0")
        )
        # Lo mismo por lo que FALTA calcular (06-10): un turno de 101k con 96% en el prefix cache local tiene ~3,5k
        # tokens de trabajo, pero por largo total esperaba detras del prefill de otro pedido (15-24 s de TTFT medidos
        # con 2-4 s de computo). Si lo que falta (prompt menos el acierto local) entra en N tokens, no espera.
        # Solo exime de la serializacion, como el prompt corto. 0 = desactivado.
        self.faltan_tokens = int(os.environ.get("GENESIS_PN115_FALTAN_TOKENS", "0"))
        self.faltan_bypass = 0

        # Headroom de seguridad: fracción de bloques GPU que nunca se compromete.
        self.safety_headroom_ratio = float(os.environ.get("GENESIS_PID_HEADROOM_RATIO", "0.10"))
        self.max_safe_usage_ratio = 1.0 - self.safety_headroom_ratio

        # Estado descubierto de vLLM
        self.is_initialized_from_vllm = False
        self.total_gpu_blocks: int = 0
        self.block_size: int = 16
        self.max_batched_tokens: int = 0
        self.watermark_blocks: int = 0

        # Telemetría
        self.ema_step_ms: float = 0.0
        self.last_error: float = 0.0
        self.current_concurrency_limit: int = 0
        self.total_kv_tokens: int = 0
        self.committed_kv_tokens: int = 0
        self.current_kv_usage: float = 0.0
        self.free_blocks: int = 0

        # baseline por concurrencia: {n_running: [min_ms, n_samples]}
        self._baselines: dict[int, list[float]] = {}

        self.gated_count: int = 0
        self.gated_by_prefill: int = 0
        self.short_bypass: int = 0
        self.bypass_count: int = 0
        self.preempt_count: int = 0

        self.last_control_check: float = 0.0
        self.last_control_mtime: float = 0.0
        self.last_status_write: float = 0.0

    # ───────────────────────── descubrimiento ─────────────────────────

    def init_from_scheduler(self, scheduler: Any) -> None:
        """Detecta topología y límites directamente de vLLM (una sola vez)."""
        if self.is_initialized_from_vllm:
            return
        try:
            sc = getattr(scheduler, "scheduler_config", None)

            sc_max = getattr(scheduler, "max_num_running_reqs", None)
            if sc_max is None:
                sc_max = getattr(sc, "max_num_seqs", 10)
            if self.max_concurrency is None:
                self.max_concurrency = int(sc_max)
            self.current_concurrency_limit = int(self.max_concurrency)
            self.min_concurrency = min(self.min_concurrency, int(self.max_concurrency))

            cache_config = getattr(scheduler, "cache_config", None)
            self.block_size = int(getattr(cache_config, "block_size", None) or
                                  getattr(scheduler, "block_size", 16) or 16)

            gpu_blocks = getattr(cache_config, "num_gpu_blocks", None)
            kv_mgr = getattr(scheduler, "kv_cache_manager", None)
            if not gpu_blocks and kv_mgr is not None:
                block_pool = getattr(kv_mgr, "block_pool", None)
                gpu_blocks = getattr(block_pool, "num_gpu_blocks", None)
            if gpu_blocks:
                self.total_gpu_blocks = int(gpu_blocks)

            if kv_mgr is not None:
                self.watermark_blocks = int(getattr(kv_mgr, "watermark_blocks", 0) or 0)

            # Con chunked prefill un request sólo reserva el chunk del paso.
            self.max_batched_tokens = int(
                getattr(scheduler, "max_num_scheduled_tokens", 0)
                or getattr(sc, "max_num_batched_tokens", 0)
                or 0
            )

            # El PID de latencia no puede medir el forward desde schedule()
            # cuando el scheduling es asíncrono: lo que mide es la cadencia del
            # propio scheduler. Apagarlo es más honesto que controlar con ruido.
            if self.latency_pid_enabled and getattr(sc, "async_scheduling", False):
                self.latency_pid_enabled = False
                self.latency_pid_forced_off_reason = (
                    "async_scheduling activo: schedule() no observa la latencia del forward"
                )
                log.warning("[PN115] PID de latencia desactivado — %s",
                            self.latency_pid_forced_off_reason)

            self.is_initialized_from_vllm = True
            log.info(
                "[PN115] topología vLLM: gpu_blocks=%d block_size=%d max_conc=%d "
                "watermark=%d max_batched_tokens=%d | kv_gating=%s latency_pid=%s "
                "headroom=%.2f",
                self.total_gpu_blocks, self.block_size, self.max_concurrency,
                self.watermark_blocks, self.max_batched_tokens,
                self.kv_gating_enabled, self.latency_pid_enabled,
                self.safety_headroom_ratio,
            )
            if PROM_PID_CONCURRENCY_LIMIT:
                PROM_PID_CONCURRENCY_LIMIT.set(self.current_concurrency_limit)
        except Exception as e:
            log.warning("[PN115] error auto-descubriendo límites de vLLM: %s", e)

    # ─────────────────── control file / status file ───────────────────

    def check_control_updates(self) -> None:
        """Relee el archivo de control. Se llama UNA vez por paso, nunca por
        request: son dos syscalls y el bucle de waiting es camino caliente."""
        now = time.monotonic()
        if now - self.last_control_check < 0.2:
            return
        self.last_control_check = now
        try:
            if not os.path.exists(PID_CONTROL_FILE):
                return
            mtime = os.path.getmtime(PID_CONTROL_FILE)
            if mtime <= self.last_control_mtime:
                return
            self.last_control_mtime = mtime
            with open(PID_CONTROL_FILE, "r") as f:
                data = json.load(f)

            if "enabled" in data:
                new_enabled = bool(data["enabled"])
                if new_enabled != self.enabled:
                    log.info("[PN115] toggle en vivo: enabled %s -> %s", self.enabled, new_enabled)
                    self.enabled = new_enabled
            if "kv_gating" in data:
                self.kv_gating_enabled = bool(data["kv_gating"])
            if "latency_pid" in data:
                want = bool(data["latency_pid"])
                if want and self.latency_pid_forced_off_reason:
                    log.warning("[PN115] latency_pid pedido pero forzado off — %s",
                                self.latency_pid_forced_off_reason)
                else:
                    self.latency_pid_enabled = want
            if "target_step_ms" in data:
                self.target_step_ms = float(data["target_step_ms"])
                if PROM_PID_TARGET_STEP_MS:
                    PROM_PID_TARGET_STEP_MS.set(self.target_step_ms)
            if "min_concurrency" in data:
                self.min_concurrency = max(1, int(data["min_concurrency"]))
            if "max_concurrency" in data:
                self.max_concurrency = int(data["max_concurrency"])
            if "max_concurrent_prefills" in data:
                self.max_concurrent_prefills = int(data["max_concurrent_prefills"])
                log.info("[PN115] max_concurrent_prefills -> %d",
                         self.max_concurrent_prefills)
            if "short_prompt_tokens" in data:
                self.short_prompt_tokens = int(data["short_prompt_tokens"])
                log.info("[PN115] short_prompt_tokens -> %d", self.short_prompt_tokens)
            if "headroom_ratio" in data:
                self.safety_headroom_ratio = float(data["headroom_ratio"])
                self.max_safe_usage_ratio = 1.0 - self.safety_headroom_ratio
            if "high_prio_threshold" in data:
                self.high_prio_threshold = int(data["high_prio_threshold"])
            if "kp" in data:
                self.kp = float(data["kp"])
            if "kd" in data:
                self.kd = float(data["kd"])
            self.write_status(force=True)
        except Exception as err:
            log.debug("[PN115] no se pudo leer el archivo de control: %s", err)

    def write_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_status_write < 0.5):
            return
        self.last_status_write = now
        try:
            tmp_path = f"{PID_STATUS_FILE}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.snapshot(), f)
            os.replace(tmp_path, PID_STATUS_FILE)
        except Exception as err:
            log.debug("[PN115] error escribiendo el status: %s", err)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": "active" if self.enabled else "disabled",
            "kv_gating": self.kv_gating_enabled,
            "latency_pid": self.latency_pid_enabled,
            "latency_pid_forced_off_reason": self.latency_pid_forced_off_reason,
            "target_step_ms": self.target_step_ms,
            "ema_step_ms": round(self.ema_step_ms, 2),
            "active_kv_tokens": self.total_kv_tokens,
            "committed_kv_tokens": self.committed_kv_tokens,
            "kv_usage_ratio": round(self.current_kv_usage, 3),
            "free_kv_blocks": self.free_blocks,
            "total_gpu_blocks": self.total_gpu_blocks,
            "block_size": self.block_size,
            "headroom_ratio": self.safety_headroom_ratio,
            "concurrency_limit": self.current_concurrency_limit,
            "min_concurrency": self.min_concurrency,
            "max_concurrency": self.max_concurrency or 0,
            "high_prio_threshold": self.high_prio_threshold,
            "baselines_ms": {str(k): round(v[0], 2) for k, v in sorted(self._baselines.items())},
            "max_concurrent_prefills": self.max_concurrent_prefills,
            "gated_requests_total": self.gated_count,
            "gated_by_prefill_total": self.gated_by_prefill,
            "short_prompt_tokens": self.short_prompt_tokens,
            "short_prompt_bypass_total": self.short_bypass,
            "priority_bypass_total": self.bypass_count,
            "faltan_bypass_total": getattr(self, "faltan_bypass", 0),
            "emergency_preemptions_total": self.preempt_count,
            "bloques_por_request": getattr(self, "bloques_por_request", None),
            "updated_at": round(time.time(), 3),
        }

    # ───────────────────────────── prioridad ─────────────────────────────

    def resolve_priority(self, request: Any) -> int:
        """Prioridad efectiva. La explícita gana; el mapa de agentes es fallback.

        Se mira `kv_transfer_params` porque es el canal que el cliente YA usa
        para etiquetar agentes (PN88/PN90 leen `genesis_agent` de ahi). Sin
        esto el bypass era inerte en la practica: con ids autogenerados
        (`chatcmpl-<uuid>`) y `priority` sin setear, nunca disparaba.
        """
        explicit = getattr(request, "priority", 0) or 0
        if explicit != 0:
            return explicit

        params = getattr(request, "kv_transfer_params", None)
        if isinstance(params, dict):
            if params.get("priority") is not None:
                try:
                    return int(params["priority"])
                except (TypeError, ValueError):
                    pass
            tagged = params.get("genesis_agent") or params.get("agent")
            if tagged and str(tagged).strip().lower() in AGENT_PRIORITY_MAP:
                return AGENT_PRIORITY_MAP[str(tagged).strip().lower()]

        agent = getattr(request, "agent", None)
        if agent and agent in AGENT_PRIORITY_MAP:
            return AGENT_PRIORITY_MAP[agent]

        req_id = getattr(request, "request_id", "") or ""
        for prefix in _ID_PREFIXES:
            if req_id.startswith(prefix):
                req_id = req_id[len(prefix):]
                break
        for ag_name, prio in AGENT_PRIORITY_MAP.items():
            if req_id.startswith(f"{ag_name}-") or req_id.startswith(f"{ag_name}_"):
                return prio
        return 0

    def is_high_priority(self, request: Any) -> bool:
        return self.resolve_priority(request) < self.high_prio_threshold

    # ─────────────────────── observación por paso ───────────────────────

    def _observe_decode_step(self, concurrency: int, step_ms: float) -> None:
        st = self._baselines.get(concurrency)
        if st is None:
            self._baselines[concurrency] = [step_ms, 1]
            return
        st[0] = min(st[0] * self.baseline_decay, step_ms)
        st[1] += 1

    def _target_for(self, concurrency: int) -> Optional[float]:
        """Objetivo de latencia PARA ESTA concurrencia.

        Comparar el paso a 10 secuencias contra la línea base medida a 1 es lo
        que hacía que el controlador leyera el costo normal del batching como
        congestión y estrangulara hasta min_concurrency.
        """
        if self.target_step_ms is not None:
            return self.target_step_ms
        st = self._baselines.get(concurrency)
        if st is None or st[1] < self.min_baseline_samples:
            return None
        return st[0] * (1.0 + self.latency_slack)

    def _medir_bloques_por_grupo(self, kv_mgr: Any, running: list) -> None:
        """Diagnostico: bloques reales (no nulos) de cada request por grupo KV.

        Sirve para ver cuanto del pool se va en estados GDN (1 vivo +
        num_speculative_blocks para rollback del MTP + el del paso anterior)
        contra los bloques de atencion. Throttleado a 2 s: recorre listas.
        """
        now = time.monotonic()
        if now - getattr(self, "_ultima_medicion_bloques", 0.0) < 2.0:
            return
        self._ultima_medicion_bloques = now
        try:
            managers = kv_mgr.coordinator.single_type_managers
            filas = []
            for r in running:
                rid = r.request_id
                por_grupo = []
                for m in managers:
                    blocks = m.req_to_blocks.get(rid, ())
                    por_grupo.append(sum(1 for b in blocks if not b.is_null))
                filas.append({"tokens": int(getattr(r, "num_computed_tokens", 0)),
                              "bloques": por_grupo})
            self.bloques_por_request = {
                "grupos": [type(m).__name__ for m in managers],
                "requests": filas,
            }
        except Exception as err:
            self.bloques_por_request = {"error": str(err)}

    def update_step_with_scheduler(self, scheduler: Any, step_duration_ms: float) -> None:
        """Hook de un paso del Scheduler de vLLM."""
        self.init_from_scheduler(scheduler)
        self.check_control_updates()

        running = getattr(scheduler, "running", []) or []
        n_running = len(running)

        self.total_kv_tokens = sum(getattr(r, "num_computed_tokens", 0) for r in running)
        # `Request.num_tokens` ya incluye los tokens generados; sumar
        # num_output_tokens los contaba dos veces e inflaba el gate.
        self.committed_kv_tokens = sum(
            max(getattr(r, "num_tokens", 0), getattr(r, "num_computed_tokens", 0))
            for r in running
        )

        kv_mgr = getattr(scheduler, "kv_cache_manager", None)
        self._medir_bloques_por_grupo(kv_mgr, running)
        if kv_mgr is not None:
            self.current_kv_usage = float(getattr(kv_mgr, "usage", 0.0) or 0.0)
            block_pool = getattr(kv_mgr, "block_pool", None)
            if block_pool is not None:
                try:
                    self.free_blocks = int(block_pool.get_num_free_blocks())
                    if PROM_PID_FREE_BLOCKS:
                        PROM_PID_FREE_BLOCKS.set(self.free_blocks)
                except Exception:
                    pass

        if PROM_PID_ACTIVE_KV_TOKENS:
            PROM_PID_ACTIVE_KV_TOKENS.set(self.total_kv_tokens)
        if PROM_PID_ACTIVE_CONCURRENCY:
            PROM_PID_ACTIVE_CONCURRENCY.set(n_running)

        if not self.enabled or not self.latency_pid_enabled or not running:
            self.write_status()
            return

        # Un paso con prefill tarda 100-800 ms y no es comparable con decode.
        has_prefill = any(_is_prefilling(r) for r in running)
        if has_prefill or step_duration_ms < 1.0:
            self.write_status()
            return

        self._observe_decode_step(n_running, step_duration_ms)
        self.ema_step_ms = (
            step_duration_ms if self.ema_step_ms <= 0.0
            else 0.20 * step_duration_ms + 0.80 * self.ema_step_ms
        )

        target = self._target_for(n_running)
        if target is None:
            # Todavía sin línea base confiable en este nivel: observar, no actuar.
            self.write_status()
            return
        if PROM_PID_TARGET_STEP_MS:
            PROM_PID_TARGET_STEP_MS.set(target)

        error = self.ema_step_ms - target
        d_error = error - self.last_error
        self.last_error = error
        control = self.kp * error + self.kd * d_error

        max_c = self.max_concurrency or n_running or 1
        if self.current_concurrency_limit <= 0:
            self.current_concurrency_limit = max_c

        if control > 1.2:
            self.current_concurrency_limit = max(
                self.min_concurrency, self.current_concurrency_limit - max(1, int(control))
            )
        elif control < -0.5:
            self.current_concurrency_limit = min(max_c, self.current_concurrency_limit + 1)

        if PROM_PID_EMA_STEP_MS:
            PROM_PID_EMA_STEP_MS.set(round(self.ema_step_ms, 2))
        if PROM_PID_CONCURRENCY_LIMIT:
            PROM_PID_CONCURRENCY_LIMIT.set(self.current_concurrency_limit)
        if PROM_PID_STATUS:
            PROM_PID_STATUS.set(1.0 if self.current_concurrency_limit >= max_c else 0.0)

        self.write_status()

    # ───────────────────────────── admisión ─────────────────────────────

    def _gate(self) -> bool:
        self.gated_count += 1
        if PROM_PID_GATED_TOTAL:
            PROM_PID_GATED_TOTAL.inc()
        return True

    def _faltan_pocos(self, request: Any, scheduler: Any) -> bool:
        """¿Lo que falta calcular (prompt menos el acierto del prefix cache local) entra en faltan_tokens?
        Lookup puro del coordinador (el mismo que hace el scheduler despues), sin tocar referencias ni metricas."""
        if self.faltan_tokens <= 0:
            return False
        try:
            kvm = scheduler.kv_cache_manager
            if not getattr(kvm, "enable_caching", True):
                return False
            hashes = getattr(request, "block_hashes", None)
            if not hashes:
                return False
            _, hit, _ = kvm.coordinator.find_longest_cache_hit(hashes, request.num_tokens - 1)
            return (request.num_tokens - int(hit)) <= self.faltan_tokens
        except Exception as e:  # nunca rompe el scheduler
            log.debug("[PN115] lookup de faltantes fallo: %s", e)
            return False

    def should_gate_waiting(self, request: Any, scheduler: Any) -> bool:
        """¿Conviene diferir este request al próximo paso?

        NO llama a check_control_updates(): esto corre una vez por request en
        el bucle de waiting, que es camino caliente. El archivo de control se
        relee en el hook de paso.
        """
        if not self.enabled:
            return False
        self.init_from_scheduler(scheduler)

        running = getattr(scheduler, "running", []) or []
        num_running = len(running) + getattr(scheduler, "num_waiting_for_streaming_input", 0)

        # Bypass de prioridad alta: nunca se difiere.
        if self.is_high_priority(request):
            self.bypass_count += 1
            if PROM_PID_BYPASS_TOTAL:
                PROM_PID_BYPASS_TOTAL.inc()
            return False

        # INVARIANTE ABSOLUTO: con el motor vacio no se difiere jamas. Es lo
        # unico que garantiza que siempre haya alguien que pueda progresar.
        if num_running == 0:
            return False

        # 1. Serializacion de prefills.
        #
        # El prefill esta al 80-85% del roofline de computo de las dos placas
        # (medido 2026-09-12: 1.500 tok/s = 39,4 TFLOPS/GPU sobre un modelo
        # DENSO de 26,3e9 parametros). No escala con la concurrencia ni con el
        # presupuesto de batch. Correr N prefills a la vez le da 1/N a cada uno
        # y no entrega NINGUNO hasta que terminan todos; de a uno, el tiempo
        # total es identico pero el primero queda usable N veces antes.
        #
        # Con 6 subagentes de 50k: todos usables recien a los 200 s, contra el
        # primero a los 33 s y un promedio de 117 s.
        #
        # Va despues del bypass de prioridad a proposito: el hilo principal
        # marcado con prioridad alta nunca espera detras de un prefill.
        if self.max_concurrent_prefills > 0:
            prefilling = sum(1 for r in running if _is_prefilling(r))
            if prefilling >= self.max_concurrent_prefills:
                npt = getattr(request, "num_prompt_tokens", None)
                corto = (self.short_prompt_tokens > 0 and npt is not None
                         and npt <= self.short_prompt_tokens)
                if not corto and self._faltan_pocos(request, scheduler):
                    corto = True
                    self.faltan_bypass += 1
                if not corto:
                    self.gated_by_prefill += 1
                    if PROM_PID_GATED_PREFILL:
                        PROM_PID_GATED_PREFILL.inc()
                    return self._gate()
                # Prioridad automatica: solo exime de la serializacion; el gate de
                # headroom de KV de abajo le sigue aplicando.
                self.short_bypass += 1
                if PROM_PID_SHORT_BYPASS:
                    PROM_PID_SHORT_BYPASS.inc()

        # INVARIANTE ANTI-DEADLOCK del gate de KV. Por debajo de
        # min_concurrency no se difiere por headroom, pase lo que pase. Sin
        # esto, un request que no entra en el headroom queda en la cola para
        # siempre y el motor se duerme con trabajo pendiente: no hay quien
        # libere bloques.
        if num_running < max(1, self.min_concurrency):
            return False

        # 2. Límite de concurrencia del PID (sólo si el PID está activo).
        if self.latency_pid_enabled and self.current_concurrency_limit > 0:
            limit = min(
                self.current_concurrency_limit,
                getattr(scheduler, "max_num_running_reqs", self.current_concurrency_limit),
            )
            if num_running >= limit:
                return self._gate()

        if not self.kv_gating_enabled:
            return False

        kv_mgr = getattr(scheduler, "kv_cache_manager", None)
        block_pool = getattr(kv_mgr, "block_pool", None) if kv_mgr is not None else None
        if block_pool is None:
            return False

        # 3. Headroom físico de bloques. Es la única verdad: no se estima
        #    capacidad en tokens ni se proyecta uso futuro (esa proyección era
        #    la que mataba de hambre a cualquier prompt largo con
        #    --max-model-len grande).
        try:
            free_blocks = int(block_pool.get_num_free_blocks())
        except Exception:
            return False
        total_blocks = self.total_gpu_blocks or int(getattr(block_pool, "num_gpu_blocks", 0) or 0)
        if total_blocks <= 0:
            return False

        safety_blocks = max(
            int(getattr(kv_mgr, "watermark_blocks", 0) or 0),
            int(total_blocks * self.safety_headroom_ratio),
        )

        # Con chunked prefill el request sólo reserva el chunk de ESTE paso.
        pending = max(0, getattr(request, "num_tokens", 0) - getattr(request, "num_computed_tokens", 0))
        if self.max_batched_tokens > 0:
            pending = min(pending, self.max_batched_tokens)
        block_size = self.block_size or 16
        req_blocks = (pending + block_size - 1) // block_size

        if (free_blocks - req_blocks) < safety_blocks:
            return self._gate()
        return False

    # ───────────────────── bypass por prioridad ─────────────────────

    def select_preemption_victim(
        self, running_requests: list[Any], incoming_request: Any
    ) -> Optional[Any]:
        """Request corriendo de MENOR prioridad que el entrante, o None."""
        if not self.enabled or not running_requests:
            return None

        incoming_prio = self.resolve_priority(incoming_request)
        candidates = [
            r for r in running_requests
            if self.resolve_priority(r) > incoming_prio and _is_running(r)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: (self.resolve_priority(r), getattr(r, "arrival_time", 0.0)),
        )

    def try_priority_preempt(
        self,
        scheduler: Any,
        timestamp: float,
        preempted_reqs: list[Any],
        scheduled_running_reqs: list[Any],
        num_scheduled_tokens: dict[str, int],
        req_to_new_blocks: dict[str, Any],
        scheduled_spec_decode_tokens: dict[str, Any],
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> tuple[int, int, int]:
        """Libera un slot para el head de la cola si es de prioridad alta.

        Devuelve `(token_budget_delta, encoder_budget_delta, req_index_delta)`
        para que el caller reponga los contadores locales de `schedule()`.

        Replica el rollback que vLLM hace en su PROPIO camino de preemption.
        Para cuando corre el bucle de waiting, la víctima ya fue agendada en
        este mismo paso: sacarla de `self.running` sin deshacer
        `num_scheduled_tokens` / `req_to_new_blocks` /
        `scheduled_spec_decode_tokens` dejaría al model runner ejecutando un
        request cuyos bloques KV se acaban de liberar, y `_update_after_schedule`
        le sumaría tokens computados a un request recién reseteado a cero.
        """
        if not self.enabled:
            return (0, 0, 0)
        try:
            request_queue = scheduler._select_waiting_queue_for_scheduling()
            if request_queue is None:
                return (0, 0, 0)
            head = request_queue.peek_request()
            if head is None or not self.is_high_priority(head):
                return (0, 0, 0)

            victim = self.select_preemption_victim(scheduler.running, head)
            if victim is None:
                return (0, 0, 0)

            vid = victim.request_id
            scheduler.running.remove(victim)

            tok_delta = 0
            enc_delta = 0
            idx_delta = 0
            if victim in scheduled_running_reqs:
                scheduled_running_reqs.remove(victim)
                tok_delta += num_scheduled_tokens.pop(vid, 0)
                req_to_new_blocks.pop(vid, None)
                scheduled_spec_decode_tokens.pop(vid, None)
                enc_inputs = scheduled_encoder_inputs.pop(vid, None)
                if enc_inputs:
                    enc_delta += sum(victim.get_num_encoder_embeds(i) for i in enc_inputs)
                idx_delta -= 1

            scheduler._preempt_request(victim, timestamp)
            preempted_reqs.append(victim)

            self.preempt_count += 1
            if PROM_PID_PREEMPTIONS_TOTAL:
                PROM_PID_PREEMPTIONS_TOTAL.inc()
            log.info(
                "[PN115] bypass de emergencia: se desaloja %s (prio=%d) para %s (prio=%d)",
                vid, self.resolve_priority(victim),
                getattr(head, "request_id", "?"), self.resolve_priority(head),
            )
            return (tok_delta, enc_delta, idx_delta)
        except Exception as e:
            # Nunca tumbar el scheduler por el bypass.
            log.warning("[PN115] try_priority_preempt falló, se ignora: %s", e, exc_info=True)
            return (0, 0, 0)


# ───────────────────────── accesores de módulo ─────────────────────────

_CONTROLLER = PIDAdmissionController.get_instance()


def update_step_with_scheduler(scheduler: Any, step_duration_ms: float) -> None:
    _CONTROLLER.update_step_with_scheduler(scheduler, step_duration_ms)


def should_gate_waiting(request: Any, scheduler_or_count: Any) -> bool:
    if hasattr(scheduler_or_count, "running"):
        return _CONTROLLER.should_gate_waiting(request, scheduler_or_count)
    return False


def try_priority_preempt(
    scheduler: Any,
    timestamp: float,
    preempted_reqs: list[Any],
    scheduled_running_reqs: list[Any],
    num_scheduled_tokens: dict[str, int],
    req_to_new_blocks: dict[str, Any],
    scheduled_spec_decode_tokens: dict[str, Any],
    scheduled_encoder_inputs: dict[str, list[int]],
) -> tuple[int, int, int]:
    return _CONTROLLER.try_priority_preempt(
        scheduler, timestamp, preempted_reqs, scheduled_running_reqs,
        num_scheduled_tokens, req_to_new_blocks, scheduled_spec_decode_tokens,
        scheduled_encoder_inputs,
    )


def select_preemption_victim(running_requests: list[Any], incoming_request: Any) -> Optional[Any]:
    return _CONTROLLER.select_preemption_victim(running_requests, incoming_request)


def resolve_priority(request: Any) -> int:
    return _CONTROLLER.resolve_priority(request)


def is_high_priority(request: Any) -> bool:
    return _CONTROLLER.is_high_priority(request)


def get_pid_status() -> dict[str, Any]:
    if os.path.exists(PID_STATUS_FILE):
        try:
            with open(PID_STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return PIDAdmissionController.get_instance().snapshot()


def set_pid_config(updates: dict[str, Any]) -> dict[str, Any]:
    current: dict[str, Any] = {}
    if os.path.exists(PID_CONTROL_FILE):
        try:
            with open(PID_CONTROL_FILE, "r") as f:
                current = json.load(f)
        except Exception:
            pass
    current.update(updates)
    tmp_path = f"{PID_CONTROL_FILE}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(current, f)
    os.replace(tmp_path, PID_CONTROL_FILE)

    ctrl = PIDAdmissionController.get_instance()
    ctrl.check_control_updates()
    return ctrl.snapshot()
