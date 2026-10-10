# Suite del model card de Qwen3.8-27B

Replica, contra el vLLM local (`compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml`,
puerto 8360), los benchmarks que Qwen publica en
[huggingface.co/Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), para ver
cuánto pierde idiotSavant (W4A8 + KV int8 + parches Genesis) frente al modelo BF16 oficial.

No reemplaza a `benchmarks/harness/` (compuertas rápidas de regresión antes de
desplegar): esta suite es lenta y mide calidad absoluta contra el card.

## Qué se reproduce

| Capacidad (card) | Benchmark | Card | Módulo | n | Calificación |
|---|---|---|---|---|---|
| Instruction following | IFBench | 79.5 | `ifbench` | 300 | verificador oficial de AllenAI, prompt-level loose |
| Scientific reasoning | GPQA Diamond | 89.2 | `gpqa` | 198 | letra exacta, prompt de simple-evals, opciones barajadas con semilla |
| Multidisciplinary reasoning | HLE | 30.8 | `hle` | 2500 | juez con el prompt oficial (el card usa GPT-4o) |
| Competitive coding | LiveCodeBench v6 | 90.3 | `livecodebench` | 131 | pass@1, tests públicos + privados en contenedor sin red |
| Visual math | MathVision | 90.0 (sin CI) | `mathvision` | 3040 | `\boxed{}` + igualdad normalizada/numérica/sympy |
| General visual reasoning | BabyVision | 65.7 (sin CI) | `babyvision` | 388 | letra exacta; completar con igualdad o juez |
| Scientific chart analysis | CharXiv (RQ) | 83.7 (sin CI) | `charxiv` | 1000 | prompts y juez oficiales de CharXiv |
| Real-world perception | RealWorldQA | 85.9 | `realworldqa` | 765 | exacta normalizada |
| Embodied intelligence | ERQA | 65.5 | `erqa` | 400 | letra exacta (53 ítems con >2 imágenes van en mosaico) |

Las filas restantes del card (Terminal Bench, SWE-bench Pro, NL2Repo, DeepSWE,
QwenSWEBench, CoWorkBench, JobBench, Agents' Last Exam, OSWorld, WebArena,
AndroidWorld, RecreationBench, ClawEval-MM, SWE-MM, Vision2Web, OmniDocBench)
son internas de Qwen o piden un entorno de agente completo (VMs, emuladores,
imágenes docker por repo, el harness de Claude Code a 256K). Quedan listadas
con su motivo en cada `summary.md`. Para código agéntico el proyecto ya tiene
un banco propio con tareas de SWE-rebench (ver memoria "banco de código de
agente"). Terminal Bench se puede correr aparte con su harness oficial y el
agente Terminus apuntado al endpoint OpenAI-compatible del 8360; no está
integrado acá.

## Condiciones, como en el card

- Modo thinking, `reasoning_effort=xhigh` (el default de Qwen; la plantilla del
  server tiene `medium` por defecto, por eso se manda explícito).
- Muestreo de "Best Practices": `temperature=1.0, top_p=0.95, top_k=20,
  min_p=0, presence_penalty=0`. `max_tokens=81920` (ajustable): con 32k los
  problemas difíciles de LiveCodeBench se cortan pensando y cuentan como error.
- Una sola muestra por ítem (el card no aclara avg@k para estos).

Desviaciones conocidas, todas registradas en el resultado:

- **Juez**: HLE, CharXiv y BabyVision usan un juez LLM. Por defecto es el mismo
  modelo en modo instruct a temperatura 0; para comparar en serio con el card
  conviene uno externo: `--judge-endpoint https://api.openai.com/v1
  --judge-model gpt-4o` con `GENESIS_JUDGE_API_KEY`.
- **MathVision / CharXiv**: Qwen corrigió anotaciones erróneas que no publicó.
- **ERQA**: el server acepta 2 imágenes por pedido; los ítems con más van en
  un mosaico rotulado. El resumen informa el acierto sin esos ítems.
- **LiveCodeBench**: el card no publica la ventana; se usa 2025-02..2025-05
  (la "v6" de Qwen), configurable con `--lcb-start/--lcb-end`.

## Uso

Antes de correr, apagar Hermes (pregunta cosas al server y ensucia la corrida);
al terminar, volver a prenderlo con `start`:

```bash
systemctl --user stop hermes-gateway.service hermes-dashboard.service
benchmarks/qwen38_card/setup.sh                        # una vez
python3 -m benchmarks.qwen38_card.run_suite --ping
python3 -m benchmarks.qwen38_card.run_suite --bench all --limit 5      # humo
python3 -m benchmarks.qwen38_card.run_suite --bench gpqa,ifbench       # completos
# corrida "practica" (~12-15 h): muestra fija de HLE y MathVision testmini
python3 -m benchmarks.qwen38_card.run_suite --concurrency 10 --limits hle=500 --mathvision-split testmini
python3 -m benchmarks.qwen38_card.run_suite --resume benchmarks/results/qwen38_card/<dir>
python3 -m benchmarks.qwen38_card.run_suite --resume <dir> --grade-only  # recalificar
```

Las corridas completas tardan horas (xhigh piensa miles de tokens por ítem):
lanzarlas desacopladas, `setsid nohup python3 -m ... > <dir>/stdout.log 2>&1 &`.
Cada respuesta se escribe apenas llega, así que cortar y `--resume` no pierde nada.

Endpoint, clave y modelo: `--endpoint/--api-key/--model` o
`GENESIS_BENCH_ENDPOINT`, `GENESIS_BENCH_API_KEY`, `GENESIS_BENCH_MODEL`; si
no hay clave se lee `VLLM_API_KEY` de `compose/.env` (nunca se escribe en los
resultados).

Ritmo medido con xhigh a concurrencia 6: IFBench ~200/h (mediana 3k tokens),
GPQA ~100/h (mediana 8k). HLE completo (2500) solo llevaria ~1 dia: con
`--limits hle=500` el error queda en ±2 puntos, suficiente contra el 30.8 del card.

`--concurrency` (6 por defecto) deja lugar en el server (`max-num-seqs 11`)
para opencode/Hermes. Ojo que la medición de calidad no depende de la carga,
pero el tiempo sí.

## Datasets

Se bajan de HF a la caché estándar. **GPQA y HLE son gated**: la cuenta del
token de `~/.cache/huggingface/token` tiene que pedir acceso en
[Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) y
[cais/hle](https://huggingface.co/datasets/cais/hle). Hasta entonces esos dos
aparecen como "no disponible" y el resto corre igual.

## Salida

`benchmarks/results/qwen38_card/<fecha>/`:

- `<bench>.responses.jsonl`: respuesta, razonamiento, tokens, finish_reason.
- `<bench>.graded.jsonl`: predicción extraída, oro y puntaje por ítem.
- `summary.json` / `summary.md`: puntaje local vs card, truncadas, desgloses.
- `config.json`, `run.log`.
