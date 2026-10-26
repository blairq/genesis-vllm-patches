# DFlash2-DCOCO: corrector de transiciones para el borrador DFlash2

Entrena la cabeza **DCOCO** ([arXiv 2610.12570](https://arxiv.org/abs/2610.12570),
[two-delta/DCOCO](https://github.com/two-delta/DCOCO)) sobre el borrador DFlash2 de
`qwen3.8_27b_idiotSavant_sm_86`, con las capturas que ya arma `idiotsavant/dflash2.sh`. Se sirve con
el parche **PN178** (`GENESIS_ENABLE_PN178_DCOCO=1`) o con `"dcoco": true` en vllm-idiotsavant.

## Qué es y qué cambia

DFlash2 propone 16 candidatos por posición. Su `CandidateSelector` arma la tabla
`scores[paso, previo, candidato]` con un bilineal de rango bajo:

- tiene codebooks por token;
- solo mira la oculta de la posición destino;
- vino con el borrador base de terceros, y su entrenamiento no es público.

Sobre esa tabla se recorre la cadena o se arma el árbol de 8 nodos.

DCOCO agrega un MLP chico (~2-3 M de parámetros) que puntúa cada transición con dos entradas:

- **un estado de par:** las ocultas de las dos posiciones vecinas;
- **una representación por candidato:** su embedding del target, proyectado y mezclado con la oculta
  de su posición.

Acá va como **corrección residual** de la tabla del selector, con la última capa en cero. Sin
entrenar, el borrador es exactamente el de hoy. La cuenta está en un solo lugar,
`vllm/_genesis/dcoco.py`, que el entrenamiento importa por ruta: es la misma que corre el servidor.

| Archivo | Qué hace |
|---|---|
| `selector.py` | El `CandidateSelector` y el top-k de vLLM en PyTorch. `dflash2/` no carga el selector. |
| `entrenar_dcoco.py` | Entrena la cabeza (`--modo cabeza`) o cabeza + LoRA (`--modo conjunto`). Evalúa y exporta. |
| `dflash2_dcoco.sh` | El pipeline completo, retomable: producción en fp16 → evaluación de base → entrenamiento → cuantización con las escalas de producción → A/B en vLLM. |
| `subir_hf.sh`, `README_hf.md` | Publicación en Hugging Face (`BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco`). |
| `test_entrenar_dcoco.py` | Pruebas en CPU: pérdida, recorridos, árbol contra fuerza bruta y que la cabeza aprende una dependencia que el selector no ve. |

`dflash2/` no se toca: de ahí se importan los datos, la LoRA, la pérdida del borrador y `guardar`.

## La pérdida

Es entropía cruzada forzada por el maestro sobre las transiciones `p-1 → p`:

- el previo es el candidato que coincide con el token real en `p-1`, y se pide el token real en `p`;
- `--perdida fila` (default) normaliza sobre la fila del previo, que es como usa la tabla el árbol;
- `--perdida conjunta` normaliza sobre las K × K transiciones, como en el paper;
- el peso es `e^{-(p-1)/ρ}` con ρ = 4;
- si alguno de los dos tokens reales queda fuera del top-16, la transición no cuenta.

En `--modo conjunto` se suma la pérdida de `dflash2/entrenar.py` (0,9 TV + 0,1 CE, `--auf`) y la
LoRA se entrena junto con la cabeza. En el paper, el entrenamiento conjunto rinde bastante más que
con el borrador congelado.

## Cómo correrlo

Requisitos:

- las capturas de `idiotsavant/dflash2.sh` (paso 3) en `tests/bench/medicion/trazas/captura/entrena_rot`;
- el borrador de producción en el models-cache;
- la imagen `vllm/vllm-openai:v0.29.0`.

Una 3090, con el servidor apagado durante el entrenamiento.

```bash
cd entrenamiento/dflash2-dcoco
PASOS="0 1" ./dflash2_dcoco.sh                                # producción en fp16 + evaluación de base
setsid nohup ./dflash2_dcoco.sh > dflash2_dcoco.log 2>&1 &    # todo (retoma lo hecho)
MODO=conjunto NOMBRE=conjunto ./dflash2_dcoco.sh              # cabeza + LoRA
```

**El paso 1 es la medición del margen, antes de entrenar.** `largo_unario`, `largo_selector` y
`largo_arbol_selector` en los mismos pedidos apartados que `dflash2/entrenar.py`, más
`recall_top16` por posición. La diferencia entre `recall_top16[p]` y `por_posicion_selector[p]` es
la cota de lo que cualquier re-selección puede ganar en la posición `p`.

**Después de cada época,** el log imprime lo mismo para la tabla corregida (`largo_dcoco` y
`largo_arbol_dcoco`). Si `largo_arbol_dcoco` no supera a `largo_arbol_selector`, no vale la pena
servirlo.

El paso 4 compara producción contra DCOCO en vLLM con el banco de agente (40 pedidos, concurrencia
4, dos arranques por brazo), con PN178 prendido en los dos. En producción no hace nada, porque el
checkpoint no trae cabeza, así que lo único que cambia entre brazos es el checkpoint.

Para publicar:

```bash
huggingface-cli login
./subir_hf.sh /home/usuario/Proyectos/models-cache/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco_cabeza
```

Antes de subir, completá la tabla de resultados de `README_hf.md` con el `registro.json` y el A/B.

## Pruebas sin GPU

```bash
python3 -m pytest -q entrenamiento/dflash2-dcoco/test_entrenar_dcoco.py
PYTHONPATH=. python3 -m pytest -q vllm/_genesis/tests/test_pn178_dcoco.py
```
