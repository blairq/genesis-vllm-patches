# qwen3.8_27b_idiotSavant_sm_86 — decisiones y por qué

Cada decisión con la medición que la respalda. Hardware fijo: 2× RTX 3090 (GA102, sm_86), 32 GB de
RAM. En SM86 el tensor core rinde fp16 1×, int8 4× e int4 8×. La tesis del proyecto es **calcular en
int8 en todo el decode**: pesos en int4 (el decode está en el techo de ancho de banda) y activaciones
en int8 por token (Marlin W4A8).

La métrica de calidad es **KL(BF16 ‖ modelo) sobre las posiciones que genera el asistente** en tráfico
real de agente de código (top-20 del BF16). Las cifras son de 4–48 ventanas de 1024 tokens.

---

## A. Qué se mide y cómo

**A1. KL solo sobre la respuesta, nunca sobre el prompt.** En texto de sistema o de usuario el modelo de
chat predice `<|im_end|>` con 99% (p. ej. `" directory"` a −21 nats): nunca se entrenó para predecir
esos turnos. Ahí la distribución es degenerada y caótica: una sola capa int4, o bf16 contra fp32, da
vuelta el top-1. Sobre prompts, cualquier cuant daba KL ~1,2; sobre respuestas, 0,01–0,04.

**A2. Error local por capa: orienta, no decide.** Correlación 0,6 entre el error local de una capa y el
KL que causa. Para decidir se mide el KL del modelo entero, primero en torch y después servido en vLLM.

**A3. Sensibilidad por capa (una capa cuantizada por vez, KL sobre respuestas).** El perfil es suave:
bajo en 0–18, meseta en 23–51, baja hacia 62; la 63 es la más sensible (2×). Una capa de atención
completa pesa ~1,5× una GDN. Las 8 peores suman solo el 22% del total. Referencia: KL-lens para
híbridos SSM, arXiv 2604.13440.
→ **Sin precisión mixta**: no hay capas culpables; lo que paga es mejorar todas por igual.

**A4. Velocidad con totales, no con medianas.** Entre modelos o borradores distintos, la mediana de
tok/s por pedido engaña: los textos generados son distintos, así que cambia cuáles pedidos son cortos
o largos. Se usan pasos por segundo y tokens por segundo de decode sumados sobre todo el banco.

## B. Punto de partida

**B1. Base: orcarouter/Qwen3.8-27B-Uncensored (BF16).** Es el mismo modelo sin censura que usa noon.
No se parte del cuant de noon por dos razones:
- AutoRound le dejó el 50% de las escalas negativas, y Marlin W4A8 las rompe (hicieron falta PN125 y PN130);
- no se puede recuantizar sin perder.

**B2. Formato: W4 simétrico, grupo 128, escala fp16 = max|w|/7,5, sin act-order.** Es lo que Marlin
W4A8 (PN130) lee directo, sin g_idx y con los grupos contiguos. Empaquetado compressed-tensors
pack-quantized.

**B3. GPTQ en vez de RTN o AutoRound.** Da 18% menos error local que el AutoRound de noon (mejor en
63 de 64 capas). En KL A16 sobre respuestas: noon 0,0155, GPTQ 0,0122.

**B4. Calibración con tráfico real.** 256 muestras × 4096 tokens: la ventana final de prompt +
respuesta de pedidos reales de opencode (agente coder, con tools), tokenizada con el mismo chat
template y el mismo hook P69 que producción (`armar_calibracion.py`, verificado token a token).
- Las 2 primeras muestras quedan apartadas de las Hessianas para medir el error sin favorecer a GPTQ.
- La calibración propaga la salida **BF16** de cada capa, no la cuantizada: cada capa se calibra contra
  su entrada verdadera y el trabajo sirve para cualquier combinación de variantes.

## C. El error de las activaciones int8 (el hallazgo central)

**C1. Con A8 por token, el error de activaciones es tan grande como el de los pesos int4.** La escala
int8 de cada token la fija su máximo, y las entradas que salen del residuo tienen **cresta** (máx/rms)
~56–70. El culpable es un canal con activación masiva: el 3994. Error relativo de la salida con int8
por token, sin rotar:

| lineal | mediana | peor | cresta |
|---|---|---|---|
| k_proj / v_proj | 10–14% | 37–48% (capas 3, 7, 11, 15) | ~56 |
| in_proj_qkv (GDN) | 12% | 24% (capa 6) | ~58 |
| gate / up | 5–7% | 11–12% | ~32 |
| down_proj | 6% | 11% (capas 54–61) | ~27 |
| o_proj / out_proj | 2–3% | 4–8% | 12–21 |

En el modelo entero, W4A8 duplicaba el KL de W4A16 (GPTQ: 0,0122 → 0,0289), y se comía la ventaja de
GPTQ sobre AutoRound (noon W4A8: 0,0300).

**C2. Remedio: rotar el residuo (estilo QuaRot, arXiv 2404.00456).** Una transformación ortogonal
reparte el pico entre todos los canales: la cresta baja de ~70 a ~2,7. En aritmética exacta el modelo
calcula lo mismo, porque RMSNorm(h·R) = RMSNorm(h)·R. Alternativas medidas (error de A8 por lineal):
- suavizado por canal (estilo SmoothQuant): 2–3%;
- Hadamard por bloques de 128: 0,6–1,7%;
- **Hadamard por bloques del tamaño de K: <1% (0,5–0,9%)**;
- suavizado + Hadamard 128: 0,5–0,9%, pero el suavizado no se pliega en todos lados.

Se eligió la rotación porque no cuesta nada en tiempo de ejecución (se hornea en los pesos) y además
mejora la cuantización de los pesos: el error local int4 baja en las 64 capas (mediana −8%, capa 0 −43%).

**C3. R = diag(signos) · Hadamard por bloques de 1024.** Los signos son ±1 con semilla fija 20260924.
- Por bloques y no densa: con 1024 ya queda <1%, y es la misma forma que usan en línea PN148 y PN149,
  sin leer pesos.
- Los signos al azar evitan que la estructura de Sylvester se alinee con la de los datos, como hace
  QuaRot.
- No se usó una rotación aprendida (SpinQuant, arXiv 2405.16406): la de Hadamard ya deja el costo de
  A8 en +12% del KL.
- Para capas tipo Mamba, MambaQuant (arXiv 2501.13484) muestra que Hadamard sola no alcanza y hace falta
  igualar varianzas. Acá el GDN no tuvo ese problema: sus entradas quedan con cresta ~2,7.

**C4. El peso de la norma se pliega.** GemmaRMSNorm calcula x·(1+w), un producto elemento a elemento
que no conmuta con la rotación. Se pliega g = 1+w en la lineal siguiente (W·diag(g)) y la norma queda
en w = 0.

**C5. Cómo se trata cada lineal** (convención de filas, y = x·Wᵀ):

| lineal | qué se cuantiza | Hessiana | por qué |
|---|---|---|---|
| q/k/v_proj, in_proj_qkv/z, gate/up | W·diag(g)·Rᵀ | R·H_n·Rᵀ | leen el residuo: ahí vive el pico |
| o_proj, out_proj | R·W | H | escriben al residuo: la salida tiene que quedar rotada |
| down_proj | R·W·Hd | Hd·H·Hd | escribe al residuo **y** su entrada tiene cresta ~27 |
| in_proj_a/b (compuertas GDN) | bf16: W·diag(g)·Rᵀ | — | chicas y sensibles; como en noon y RedHat, no se cuantizan |
| conv1d, A_log, dt_bias, normas q/k y gated | igual | — | operan dentro de la cabeza, después de la proyección |

**C6. La entrada de o_proj y out_proj no se rota.** Rotarla bajaría su A8 de ~2,3% a ~1,2%, pero habría
que hacerlo en línea: en atención, la compuerta de salida es elemento a elemento; en el GDN, la gated
norm tiene peso por dimensión de cabeza y compuerta z. Ninguna de las dos deja plegar la rotación, y
con cresta 12–21 no vale el kernel.

**C7. Down_proj: Hadamard de 512 en línea (PN148).**
- Sin ella, el KL W4A8 sube 15% (0,0116 → 0,0134).
- 512 y no más: con TP=2 cada GPU tiene 8704 = 17 × 512, así que los bloques no cruzan la partición y
  no hace falta comunicación.
- Primero se hizo en torch (matmul 512×512): costaba 4–5% de prefill.
- Fusionada en Triton con SiluAndMul (H512 = H16 ⊗ H32 con tensor cores fp16): quedó en paridad.
- Después **SK-23** (CUDA): SiluAndMul + Hadamard + int8 por token en registros, directo a Marlin sin
  pasar por Python. Prefill +1,4%, decode +0,6%. En Triton no se puede, porque la escala por token
  necesita la fila entera.

**C8. El canal muerto de la capa 7.** La `post_attention_layernorm` de la capa 7 tiene g = 0 **exacto**
en el canal 3994, el de la activación masiva (rms 70 contra 0,15 de mediana): el modelo lo apaga antes
de la MLP. Al rotar, esa energía entra a gate/up, y como la Hessiana guardada sobre x = g·n no la
incluye, el error de la MLP se multiplicó por 7 y el KL servido saltó a 0,17. Solución: acumular la
Hessiana de n (antes de g) directo en la entrada de la norma, y GPTQ compensa. Lección: validar SIEMPRE
el checkpoint **servible** en torch; la simulación en la base original no ve esa fuga.

**C9. Lo que está fuera de las capas:**
- embedding: E·Rᵀ;
- norma final: w = 0, con su g plegada en el lm_head ((W·diag(g))·Rᵀ); g se guarda en el config
  (`genesis_rotacion.g_final`) para el borrador;
- visión: la torre no se toca y queda en bf16; solo el merger escribe al residuo: fc2 → R·W, bias → b·Rᵀ;
- `mtp.*`: fuera, porque no se usa (el especulativo es DFlash2) y quedaría inválido;
- lm_head: en bf16 en el checkpoint; vLLM lo cuantiza a int4 al cargar (PN139). Rotado da el mismo
  error: 11,85% contra 11,79%.

## D. Resultado

| | torch W4A16 | torch W4A8 | vLLM W4A8 (servido) |
|---|---|---|---|
| noon (AutoRound) | 0,0155 | 0,0300 | 0,0385 |
| GPTQ sin rotar | 0,0122 | 0,0289 | 0,0365 |
| **idiotSavant** | **0,0104** | **0,0116** | **0,0194** |

Top-1 servido: 0,956 contra 0,938 de noon. Prefill y decode, en paridad con el modelo sin rotar.

## E. El borrador DFlash2 (`dflash2.sh`)

**E1. El borrador queda en su base original.** Su k_proj/v_proj se aplica a dos corrientes con normas
distintas (`hidden_norm` sobre las features del target e `input_layernorm` sobre los tokens del bloque),
y en la base rotada un mismo k_proj no puede servir a las dos. Por eso se convierte en tres puntos:
- **fc:** se pliega W_c·Rᵀ en cada uno de los 5 trozos de 5120, así las features salen idénticas y la
  captura y el entrenador no cambian;
- **embedding compartido (PN149):** en línea, e = e_rot·R;
- **lm_head compartido (PN149):** antes de él, (h/g_final)·Rᵀ, tanto para los logits como para el
  top-16 del árbol.

Son Hadamard por bloques: costo por paso no medible (pasos/s 38,3 contra 38,4).

**E2. Se ajusta contra el modelo SERVIDO.** Las etiquetas son el top-16 del modelo en vLLM con W4A8,
árbol y perfil coder; las features, las del fc servido. El entrenador simula exactamente el embedding
y el lm_head efectivos, int4 incluido.

**E3. Base de partida: el DFlash2 original de Inco, ajustado DIRECTO contra idiotSavant.** Offline
(greedy, 3% de pedidos apartados): 5,70 → 6,13. Servido: 5,63 aceptados por paso y ~216 tok/s.
Partir de un borrador ya ajustado contra otro cuant (noon) daba 5,69 y 217: la misma velocidad dentro
del ruido entre réplicas, pero con un linaje que depende de otro modelo. Se publica el directo (el
otro queda en models-cache como `..._dflash2_desde_noon`).

**E4. Receta** (validada antes en otro cuant: +7% offline y +14% de tok/s en agente):
- LoRA r=64 en las 7 lineales, 1 época, lr 1e-4, lote 32;
- pérdida 0,9 TV + 0,1 CE contra el top-16;
- AUF: corta en la primera posición que fallaría, o sea entrena lo que el árbol realmente usa;
- el conv queda congelado: con Adam a 3e-4, siete pasos bajaron el largo aceptado de 4,2 a 2,8;
- el fc y el selector también quedan congelados.

**E5. Cuantización RTN W4A16 g128.** Empata con el GPTQ de syvai sobre este borrador y es determinista:
el fc servido durante la captura es exactamente el del borrador final.

**E6. Captura en una instancia aislada** (puerto 8361, red propia), con el offload de KV en disco
acotado a 5 GB. El tráfico ajeno contamina los contadores de aceptación, que son globales. Con el tope
de 64 GB, el disco se llenó en minutos.

**E7. Resultado** (banco de 40 pedidos de agente, árbol, 3 réplicas, totales): 5,65 / 5,69 / 5,54 →
**5,63 aceptados por paso**, 218 / 218 / 212 → **~216 tok/s de decode**. Es la misma velocidad que tenía
el stack anterior (5,67 / 216), sobre un modelo con la mitad de error.

## F. Ejecución

**F1. Etapas en procesos separados** (calibrar ‖ cuantizar → armar), que se comunican por disco. El
BF16 pesa 55 GB contra 32 GB de RAM, y el estado oculto de la calibración ocupa 10,7 GB de GPU:
calibrar necesita ~14 GB y cuantizar ~8. Con dos GPUs van en paralelo (~85 min); con una, la comparten.

**F2. Caché y retomada.** Hay una marca por capa (CALIBRADA / CUANTIZADA) y todas las escrituras son
atómicas. El estado oculto se guarda cada 8 capas si el disco da. Las Hessianas se borran al cuantizar
(con `--conservar_hessianas` quedan ~50 GB, que permiten armar variantes sin recalibrar).

**F3. Recursos.** Chequeo previo de RAM (la del host y el límite del cgroup), GPU y disco. Antes de cada
capa, si falta algo, espera en vez de reventar. La calibración no se adelanta más de 3 capas a la
cuantización, así las Hessianas pendientes no llenan el disco. Durante el GPTQ el estado oculto baja a
RAM: la primera versión se quedó sin memoria de GPU en el down_proj de la capa 1.

**F4. Entorno local.** Un `.venv` con las mismas versiones que la imagen de vLLM (torch 2.13.0+cu130,
transformers 5.16.1). Con otras, el forward de Qwen3.5 que se calibra podría no ser el que se sirve.
Todas las cachés van en `.cache/` del proyecto, y el acceso al Hub queda apagado. El borrador corre en
la imagen de vLLM porque la captura necesita al servidor.

**F5. La interfaz es la línea de comandos.** Un evento por línea con el prefijo de la etapa; códigos de
salida 0 / 1 / 2; `estado --json` para scripts y LLMs. La TUI es solo un visor opcional que lee lo mismo.
