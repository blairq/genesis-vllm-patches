# Suite del model card de Qwen3.8-27B — hle_low_sincierre_20261010_222448

- Modelo: `qwen3.8` en `http://127.0.0.1:8360/v1` (commit Genesis `651270f`)
- Modo: thinking, reasoning_effort=low, max_tokens=81920, limite por benchmark: ninguno (completo); por benchmark: hle=500

| Capacidad | Benchmark | IdiotSavant | IdiotSavant (rescatado) | Qwen3.8-27b-bf16 | Δ | n | truncadas | sin respuesta (sin </think>) | notas |
|---|---|---|---|---|---|---|---|---|---|
| Multidisciplinary reasoning | HLE | **22.4** | 22.4 | 30.8 | -8.4 | 500/500 | 79 | 5 (5) | juez GPT-4o en el card; juez qwen3.8 |

Con `--limit` el error estandar es grande (n=20 → ±10 puntos): sirve para humo, no para comparar.

## Filas del card no reproducidas

| Capacidad | Benchmark | Card | Motivo |
|---|---|---|---|
| Agentic terminal coding | Terminal Bench 2.1 (Terminus) | 73.0 | harness Terminus + ~90 entornos docker; ver README |
| Agentic coding | SWE-bench Pro | 61.7 | harness Claude Code, imagenes docker por repo, 256K |
| Repo-level code generation | NL2Repo-Bench | 42.3 | harness Claude Code |
| Agentic coding | DeepSWE 1.1 | 42.2 | harness Claude Code |
| Software engineering | QwenSWEBench | 79.0 | interno de Qwen |
| Long-horizon office work | CoWorkBench | 70.7 | interno de Qwen |
| Professional job tasks | JobBench | 33.4 | entorno de agente |
| Frontier agentic tasks | Agents' Last Exam | 20.4 / 42.9 | entorno de agente |
| Computer use | OSWorld-Verified | 84.3 | VMs de escritorio |
| Browser use | WebArena-Verified | 64.8 | sitios web autohospedados |
| Mobile use | AndroidWorld | 81.9 | emulador Android |
| Application recreation | RecreationBench | 47.1 | interno de Qwen |
| Multimodal tool use | ClawEval-MM | 57.4 / 56.9 | entorno de agente |
| Multimodal software engineering | SWE-MM | 38.6 | harness Claude Code |
| Visual web development | Vision2Web | 62.9 | harness Claude Code + juez gpt-5.4 |
| Document intelligence | OmniDocBench 1.5 | 91.1 | pendiente: necesita el toolkit oficial (CDM para formulas, TEDS para tablas) |
