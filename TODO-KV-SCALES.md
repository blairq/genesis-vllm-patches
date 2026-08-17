# TODO: k_scale / v_scale por capa

**Estado: analizado y medido, no implementado.** No hay evidencia todavía de que
aplicarlos mejore nada, y sí un argumento concreto de que pueden empeorar. Este
archivo existe para no volver a investigar todo de cero.

Contexto: `tests/repro/kv_scale_probe.py` (commits `87dee61`, `af456a4`).

---

## Qué se sabe

El checkpoint no trae escalas, así que vLLM usa **1,0 en las 32** (16 capas de
atención plena × K y V). Las otras 48 capas son GDN y no usan KV cache.

Medido sobre 8000 tokens, 262M de valores, con K tomada **después de `k_norm` y
RoPE** (que es lo que realmente entra al cache):

|                     | resultado |
|---|---|
| `\|max\|` global      | 91,0 → quedan **5×** de margen hasta 448 |
| recortados          | **0** de 262.144.000 |
| en denormales       | 0,22% |

K queda plana (9,2 a 22,8) porque `k_norm` la normaliza. La que crece con la
profundidad es V, que no pasa por ninguna norma: 6,0 en la capa 3 → 91,0 en la 63.

Escalas que calcularía vLLM (`K_SCALE_CONSTANT=200`, `V_SCALE_CONSTANT=100`, no 448):

| capa | k_scale | v_scale | | capa | k_scale | v_scale |
|---|---|---|---|---|---|---|
| 3 | 0,046 | 0,060 | | 35 | 0,095 | 0,343 |
| 7 | 0,068 | 0,080 | | 39 | 0,101 | 0,295 |
| 11 | 0,063 | 0,081 | | 43 | 0,096 | 0,593 |
| 15 | 0,084 | 0,052 | | 47 | 0,091 | 0,552 |
| 19 | 0,086 | 0,072 | | 51 | 0,085 | 0,465 |
| 23 | 0,100 | 0,119 | | 55 | 0,074 | 0,311 |
| 27 | 0,106 | 0,244 | | 59 | 0,080 | 0,830 |
| 31 | 0,114 | 0,385 | | 63 | 0,070 | 0,910 |

---

## Por qué NO se aplicó todavía

**Es un canje, no una mejora gratis.** Con escala `s` se guarda `x/s`, así que el
techo de recorte pasa de 448 a `448·s` y el piso de denormales de 2⁻⁹ a `2⁻⁹·s`.
Los dos bajan juntos.

| | hoy (s=1,0) | calibrado |
|---|---|---|
| capa 3 K (`\|max\|` 9,25) | techo 448 → **48× de aire** | s=0,046 → techo 20,7 → **2,2×** |
| capa 63 V (`\|max\|` 91,0) | techo 448 → **4,9×** | s=0,910 → techo 407,7 → **4,5×** |

Se entrega margen contra outliers a cambio de recuperar 0,22% de denormales —
los valores más chicos del tensor, los que menos pesan en el softmax.

**Y la medición tiene una debilidad conocida:** el prompt era código Python
repetitivo generado. Los outliers de KV no son ruido, son las *massive
activations* / attention sinks, que caen en tokens específicos (BOS,
delimitadores, cambios de tema). Un prompt repetitivo es justo el que menos los
produce. 5× de aire ahí no garantiza 5× de aire en una conversación real de
126k tokens.

---

## Cómo se aplicaría (verificado en el código de vLLM 0.23.0)

La ruta nativa existe y está viva en este modelo:

- `qwen3_5.py:316` llama a `maybe_remap_kv_scale_name` en `load_weights`
- la regla por defecto es `\.([qkv])_scale$` → `.attn.\1_scale`
  (`weight_utils.py:1527`)
- el módulo destino es `model.layers.{i}.self_attn.attn` (`qwen3_next.py:267`)
- `KVCacheScaleParameter.weight_loader` acepta shape `()` o `(1,)`

O sea que el checkpoint sólo necesita escalares llamados
`model.layers.{i}.self_attn.k_scale` (y `.v_scale`, `.q_scale`) para
i ∈ {3, 7, 11 … 63}.

Y FLASHINFER las honra de verdad: hay un kernel Triton que hace
`fp8_k.to(f32) * k_scale_val` al desempacar (`flashinfer.py:135`).

### Opción A — checkpoint propio (destino final)

No hace falta copiar los 31 GB: un directorio local con symlinks a los
safetensors originales + **un shard nuevo y chico** con los 48 escalares +
`model.safetensors.index.json` editado. Ruta nativa, no toca vLLM, sobrevive a
los upgrades.

### Opción B — parche genesis (por acá conviene empezar)

Escribir `layer._k_scale` después de `process_weights_after_loading` (que es
donde se pone el 1,0 por defecto). Se prende y apaga con env var, así que hace
el A/B trivial sin reconstruir nada.

⚠️ **Hay que escribir los dos, el buffer y el float.** FlashInfer lee
`layer._k_scale_float` en unos caminos (`:295`, `:1581`, `:1732`) y
`layer._k_scale` como tensor en otros (`:1662`, `:1858`). Actualizar uno solo
deja prefill y decode con escalas distintas, y el bug es silencioso.

### Opción C — `--calculate-kv-scales`: DESCARTADA

`models/config.py:213` lo fuerza a `False` en modelos híbridos: el estado
recurrente sin inicializar corrompe la pasada de calibración
([issue 37554](https://github.com/vllm-project/vllm/issues/37554)).

### Cuidado con `q_scale`

En las 3090 no hay tensor cores FP8, así que la atención no es FP8 y `q_scale`
no se usa en nuestra ruta. Pero **si se da `k_scale` sin `q_scale`, vLLM copia
`k_scale` dentro de `q_scale`** con un warning. Conviene setearlo explícito en
1,0 para no dejar un 0,046 armado esperando que alguien active esa ruta
(`flashinfer.py:1404`: `bmm1_scale *= q_scale * k_scale`).

---

## Plan de medición

**Nota:** si escribimos nosotros las escalas, NO estamos obligados a usar el
200/100 de vLLM. Esas constantes son heurísticas para calibración al vuelo con
margen para outliers no vistos. Elegir ese número ES el trabajo: define cuánto
techo se entrega por cuánto piso.

### Paso 1 — round-trip numérico, sin engine (barato)

Extensión chica de `kv_scale_probe.py`: cuantizar los K/V reales a
`torch.float8_e4m3fn` y desempacarlos con `s=1,0` y con `s` calibrado, midiendo
error relativo y similitud coseno contra el original.

**Requisito:** correrlo sobre texto real y variado (conversación larga, un repo,
prosa), NO sobre el código sintético de ahora. Si algún set de calibración
muestra recortes, el resultado se da vuelta.

Predicción: diferencia del orden de 1e-4, o sea nada, y ahí se cierra el tema.

### Paso 2 — sólo si el paso 1 se mueve: A/B punta a punta

Comparar `fp8+1,0` contra `fp8+calibrado` dice que difieren, no cuál está mejor.
Hace falta la referencia: correr con `--kv-cache-dtype auto` (KV en fp16, verdad
de terreno) y medir divergencia de logits / perplejidad de las otras dos contra
esa, sobre un documento largo.

Sumarle una prueba de recuperación a contexto largo (needle-in-haystack): el
error de KV se acumula con la distancia y la perplejidad promedio lo esconde.
