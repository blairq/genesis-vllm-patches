# Configuración de credenciales (`.env`)

Todo lo que hay que crear a mano para que los composes de `compose/` levanten.
Ninguno de estos archivos está versionado: contienen credenciales.

---

## Único archivo requerido: `compose/.env`

Docker Compose lo lee **automáticamente** por estar en el mismo directorio que
los `docker-compose.*.yml`. No hay que pasarle `--env-file` ni nada: si está
ahí, se usa.

```bash
cp compose/.env.example compose/.env
$EDITOR compose/.env
```

### Contenido

```bash
# ── Clave de la API de vLLM ──────────────────────────────────────────────
# Protege el endpoint OpenAI-compatible. Los clientes la mandan como
# `Authorization: Bearer <valor>`.
# ⚠️ Si el engine se publica a internet (Traefik/Cloudflare), esta clave es
#    LO ÚNICO que separa tu GPU del mundo. Poné algo largo y aleatorio:
#      openssl rand -hex 32
VLLM_API_KEY=

# ── HuggingFace ──────────────────────────────────────────────────────────
# Necesario solo para descargar checkpoints privados o gated. Si los pesos ya
# están en models-cache, se puede dejar vacío.
# Se saca de https://huggingface.co/settings/tokens (permiso: read).
# Las dos variables llevan EL MISMO valor: distintas versiones de
# transformers/vLLM leen una u otra.
HF_TOKEN=
HUGGING_FACE_HUB_TOKEN=

# ── DNS interno ──────────────────────────────────────────────────────────
# IP del resolver DNS de la red Docker. Solo hace falta si los contenedores
# tienen que resolverse entre sí por nombre.
AI_DNS_SERVER=
```

### Variables opcionales

Las usan composes puntuales, no el de vLLM:

| Variable | Quién la usa | Para qué |
|---|---|---|
| `WEBUI_SECRET_KEY` | open-webui | Firma de sesiones. Cualquier string largo. |
| `OPEN_TERMINAL_API_KEY` | open-terminal | Auth del terminal web. |

---

## Dónde viven los pesos

Los composes montan `/home/usuario/Proyectos/models-cache` como
`/root/.cache/huggingface`. Es una **ruta absoluta**: si tu checkout está en
otro lado, hay que ajustarla en los `.yml`.

> Esa ruta está fuera del repo a propósito. Antes los pesos vivían dentro de
> `club-3090/`, y como los contenedores corren como root iban dejando
> directorios `root:root` que rompían git en ese repo. Ver
> [`DIAGNOSTICO-OOM-qwen38-27b.md`](../DIAGNOSTICO-OOM-qwen38-27b.md).

---

## Verificar que quedó bien

```bash
# 1. Compose resuelve las variables (no debe imprimir warnings de "not set")
docker compose -f compose/docker-compose.qwen38-27b-w8a16-mtp.yml config >/dev/null

# 2. El contenedor recibió la clave
docker exec <container> printenv VLLM_API_KEY

# 3. El endpoint responde con esa clave
curl -s localhost:8320/v1/models -H "Authorization: Bearer $VLLM_API_KEY" | head -c 200
```

Si el paso 3 devuelve `401 Unauthorized`, la clave del `.env` y la que estás
mandando no coinciden. Si devuelve vacío o falla la conexión, el engine todavía
está cargando pesos — puede tardar 1-2 minutos.

---

## Reglas

- **Nunca commitear un `.env`.** `.gitignore` ya los excluye; no lo fuerces con
  `git add -f`.
- **No usar el valor de ejemplo en producción.** Un `VLLM_API_KEY` de ejemplo
  publicado en un repo equivale a no tener clave.
- **Rotar es barato**: cambiar el valor, `docker rm -f <container>` y volver a
  levantar. Acordate de actualizar también los clientes (p. ej. los canales de
  `opencode.jsonc`).
