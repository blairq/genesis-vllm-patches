# Suite del model card de Qwen3.8-27B — completa_20261009_180151

- Modelo: `qwen3.8` en `http://127.0.0.1:8360/v1` (commit Genesis `8588289`)
- Modo: thinking, reasoning_effort=xhigh, max_tokens=81920, limite por benchmark: ninguno (completo); por benchmark: hle=500; MathVision split testmini

| Capacidad | Benchmark | IdiotSavant | IdiotSavant (rescatado) | Qwen3.8-27b-bf16 | Δ | n | truncadas | sin respuesta (sin </think>) | notas |
|---|---|---|---|---|---|---|---|---|---|
| Instruction following | IFBench | **83.0** | 83.0 | 79.5 | +3.5 | 300/300 | 2 | 1 (1) |  |
| Scientific reasoning | GPQA Diamond | **91.9** | 92.4 | 89.2 | +2.7 | 198/198 | 2 | 2 (2) |  |
| Multidisciplinary reasoning | HLE | **21.0** | 24.4 | 30.8 | -9.8 | 500/500 | 78 | 58 (58) | juez GPT-4o en el card; juez qwen3.8 |
| Competitive coding | LiveCodeBench v6 | **85.5** | = | 90.3 | -4.8 | 131/131 | 10 | 0 (0) |  |
| Visual math problem solving | MathVision | **81.9** | = | 90.0 | -8.1 | 304/304 | 0 | 0 (0) | Without CI; With CI 94.6 |
| General visual reasoning | BabyVision | **46.1** | = | 65.7 | -19.6 | 388/388 | 1 | 0 (0) | Without CI; With CI 85.6; juez qwen3.8 |
| Scientific chart analysis | CharXiv (RQ) | **81.3** | 81.4 | 83.7 | -2.4 | 1000/1000 | 0 | 1 (1) | Without CI; With CI 90.2; juez qwen3.8 |
| Real-world perception | RealWorldQA | **82.0** | = | 85.9 | -3.9 | 765/765 | 0 | 0 (0) |  |
| Embodied intelligence | ERQA | **67.5** | = | 65.5 | +2.0 | 400/400 | 0 | 0 (0) |  |

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
