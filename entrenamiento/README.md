# entrenamiento/

Cómo se arman los modelos que sirven los parches Genesis. Cada carpeta es independiente: lo que
comparte con otra lo importa de ahí, sin copiarlo.

| Carpeta | Qué arma | Publicado en |
|---|---|---|
| [`idiotsavant/`](idiotsavant/) | El target **qwen3.8_27b_idiotSavant_sm_86** (W4 con el residuo rotado) y la orquestación del borrador (`dflash2.sh`). Se publica tal cual como `reproducir/` en la tarjeta del modelo. | [BlairQ/qwen3.8_27b_idiotSavant_sm_86](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86) |
| [`dflash2/`](dflash2/) | El borrador **DFlash2** ajustado (LoRA sobre las 5 capas): DFlash2 en PyTorch, entrenamiento, rotación y cuantización. Antes vivía en `borrador/`. | [BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2) |
| [`dflash2-dcoco/`](dflash2-dcoco/) | El borrador **DFlash2 + DCOCO**: el mismo DFlash2 con un corrector de transiciones entre posiciones (arXiv 2610.12570), entrenado sobre las capturas de `dflash2.sh`. Se sirve con PN178. | BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco (a publicar) |
| [`cuant/`](cuant/) | Las herramientas de cuantización y rotación del target (GPTQ, sensibilidad, fidelidad). | — |
| [`ste_corpus/`](ste_corpus/) | Corpus de calibración. | — |

El pipeline del borrador es `idiotsavant/dflash2.sh`: captura → entrenamiento (`dflash2/entrenar.py`) →
cuantización (`dflash2/cuantizar_rtn.py`) → A/B. `dflash2-dcoco/dflash2_dcoco.sh` reusa sus capturas.
