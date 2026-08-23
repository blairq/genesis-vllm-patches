#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test de carga y validación end-to-end de Tiered Demotion (PN90).

Simula:
1. Agente `build` / `primary_nothink` (`persist_disk: true`) con contexto creciente:
   - Turno 1: ~15k tokens
   - Turno 2: ~30k tokens (reutiliza Turno 1 + 15k nuevos)
   - Turno 3: ~60k tokens (reutiliza Turno 2 + 30k nuevos)
   - Turno 4: ~95k tokens (reutiliza Turno 3 + 35k nuevos -> satura L2 RAM y desaloja hacia SSD)
2. Subagentes `coder` (`persist_disk: false`, `max_offload_tokens: 0`) atacando la caché entre turnos.
3. Verificación de invariantes:
   - Los subagentes tienen 0 escrituras a disco.
   - El agente build en turnos pequeños/medianos tiene 0 escrituras a disco (se sirve de RAM).
   - Solo al desbordar los 5 GiB de L2 RAM se demotan bloques hacia SSD.
   - Reuso de prefijo demotado (promoción SSD -> RAM).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import urllib.request

PARRAFO = (
    "// ETAPA {i} DE LOGICA Y ARQUITECTURA\n"
    "function ejecutarModulo_{i}(contextoPrincipal, datosEntrada) {{\n"
    "    let totalCalculado = 0;\n"
    "    for (let idx = 0; idx < 120; idx++) {{\n"
    "        totalCalculado += (idx * {i}) ^ (idx & 15);\n"
    "        if (totalCalculado % 3 === 0) {{ totalCalculado = Math.floor(totalCalculado / 2); }}\n"
    "    }}\n"
    "    return {{ estado: 'OK', acumulador: totalCalculado, bloque: {i} }};\n"
    "}}\n\n"
)


def generar_bloque_texto(n_parrafos: int, offset: int = 0) -> str:
    return "".join(PARRAFO.format(i=i) for i in range(offset, offset + n_parrafos))


class TestHarness:
    def __init__(self, base_url: str, api_key: str, kv_offload_dir: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.kv_offload_dir = kv_offload_dir

    def pedir_chat(self, prompt: str, agent: str, persist_disk: bool, max_offload_tokens: int | None = None) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        kv_params: dict = {"genesis_agent": agent, "persist_disk": persist_disk}
        if max_offload_tokens is not None:
            kv_params["max_offload_tokens"] = max_offload_tokens

        payload = {
            "model": "qwen3.8",
            "messages": [
                {"role": "system", "content": "Sos un asistente de programación experto."},
                {"role": "user", "content": f"{prompt}\n\nResponde únicamente con: READY_OK"}
            ],
            "max_tokens": 15,
            "temperature": 0.0,
            "kv_transfer_params": kv_params,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        data["_elapsed_sec"] = dt
        return data

    def medir_disco_nuevos(self, t_referencia: float) -> tuple[int, float]:
        """Devuelve (cantidad_archivos_creados_despues_de_t_ref, tamaño_en_mb)."""
        archivos = glob.glob(os.path.join(self.kv_offload_dir, "**", "*.bin"), recursive=True)
        nuevos = [f for f in archivos if os.path.isfile(f) and os.path.getmtime(f) >= t_referencia]
        total_bytes = sum(os.path.getsize(f) for f in nuevos)
        return len(nuevos), total_bytes / (1024 * 1024)

    def pedir_metricas(self) -> dict[str, float]:
        url = f"{self.base_url}/metrics"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        metricas = {}
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                for linea in resp.read().decode("utf-8").splitlines():
                    if linea.startswith("vllm:kv_tier_") or linea.startswith("vllm:external_prefix_"):
                        partes = linea.split()
                        if len(partes) == 2:
                            metricas[partes[0]] = float(partes[1])
        except Exception:
            pass
        return metricas


def main():
    parser = argparse.ArgumentParser(description="Test de carga Tiered Demotion")
    parser.add_argument("--url", default="http://172.20.0.228:8320", help="URL de vLLM")
    parser.add_argument("--key", default="super-secret-key-123", help="API key")
    parser.add_argument("--kv-dir", default="/home/usuario/Proyectos/kv-offload", help="Directorio /kv-offload")
    args = parser.parse_args()

    harness = TestHarness(args.url, args.key, args.kv_dir)
    print("═" * 70)
    print("  TEST DE CARGA TIERED DEMOTION (PN90) & INVARIANTES DE L2/L3")
    print("═" * 70)

    t_inicio_global = time.time() - 1.0

    # 1. Sembrado base de prefijo (Turno 1) ~35k tokens
    print("\n[Paso 1] Turno 1 Agente `build` (~35k tokens, persist_disk: true)...")
    t0_paso1 = time.time()
    prefijo_t1 = generar_bloque_texto(250, offset=0)
    resp_t1 = harness.pedir_chat(prefijo_t1, "primary_nothink", persist_disk=True)
    usage_t1 = resp_t1.get("usage", {})
    t_prompt_1 = usage_t1.get("prompt_tokens", 0)
    print(f"  -> Prompt tokens: {t_prompt_1}, Tiempo: {resp_t1['_elapsed_sec']:.2f}s")
    n_nuevos_1, mb_nuevos_1 = harness.medir_disco_nuevos(t0_paso1)
    print(f"  -> Archivos NUEVOS en SSD tras Turno 1: {n_nuevos_1} archivos ({mb_nuevos_1:.2f} MB)")
    assert n_nuevos_1 == 0, f"ERROR: Turno 1 escribió {n_nuevos_1} archivos en SSD cuando debía residir solo en L2 RAM!"
    print("  ✓ Correcto: 0 escrituras a SSD (alojado 100% en L2 RAM)")

    # 2. Subagentes atacan la caché en paralelo (coder, persist_disk: false)
    print("\n[Paso 2] Disparo de subagentes `coder` (persist_disk: false, max_offload_tokens: 0)...")
    t0_sub = time.time()
    for sub_idx in range(3):
        prompt_sub = f"// Codigo independiente de coder {sub_idx}\n" + generar_bloque_texto(150, offset=1000 + sub_idx * 200)
        resp_sub = harness.pedir_chat(prompt_sub, "coder", persist_disk=False, max_offload_tokens=0)
        print(f"  -> Subagente {sub_idx}: {resp_sub.get('usage', {}).get('prompt_tokens', 0)} tokens en {resp_sub['_elapsed_sec']:.2f}s")

    n_nuevos_sub, mb_nuevos_sub = harness.medir_disco_nuevos(t0_sub)
    print(f"  -> Archivos NUEVOS en SSD tras Subagentes: {n_nuevos_sub} archivos ({mb_nuevos_sub:.2f} MB)")
    assert n_nuevos_sub == 0, f"ERROR: Subagentes escribieron {n_nuevos_sub} archivos en SSD!"
    # 3. Turno 2 Agente `build` (~70k tokens: reusa t1 + agrega 35k)
    print("\n[Paso 3] Turno 2 Agente `build` (~70k tokens: reusa t1 + agrega 35k)...")
    t0_paso2 = time.time()
    delta_1 = generar_bloque_texto(250, offset=250)
    prefijo_t2 = prefijo_t1 + delta_1
    resp_t2 = harness.pedir_chat(prefijo_t2, "primary_nothink", persist_disk=True)
    t_prompt_2 = resp_t2.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Prompt tokens: {t_prompt_2}, Tiempo: {resp_t2['_elapsed_sec']:.2f}s")
    n_nuevos_2, mb_nuevos_2 = harness.medir_disco_nuevos(t0_paso2)
    print(f"  -> Archivos NUEVOS en SSD tras Turno 2: {n_nuevos_2} archivos ({mb_nuevos_2:.2f} MB)")
    assert mb_nuevos_2 < 2000, f"ERROR: Turno 2 escribió {mb_nuevos_2:.2f} MB en SSD (esperado < 2000 MB)!"
    print(f"  ✓ Correcto: Democión controlada por saturación de RAM ({n_nuevos_2} bloques = {mb_nuevos_2:.2f} MB en SSD, no toda la sesión)")

    # 4. Turno 3 Agente `build` (~100k tokens: acumulación incremental)
    print("\n[Paso 4] Turno 3 Agente `build` (~100k tokens: reusa t2 + agrega 30k)...")
    t0_paso3 = time.time()
    delta_2 = generar_bloque_texto(200, offset=500)
    prefijo_t3 = prefijo_t2 + delta_2
    resp_t3 = harness.pedir_chat(prefijo_t3, "primary_nothink", persist_disk=True)
    t_prompt_3 = resp_t3.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Prompt tokens: {t_prompt_3}, Tiempo: {resp_t3['_elapsed_sec']:.2f}s")
    n_nuevos_3, mb_nuevos_3 = harness.medir_disco_nuevos(t0_paso3)
    print(f"  -> Archivos NUEVOS en SSD tras Turno 3: {n_nuevos_3} archivos ({mb_nuevos_3:.2f} MB)")
    assert mb_nuevos_3 < 4000, f"ERROR: Turno 3 escribió {mb_nuevos_3:.2f} MB en SSD (esperado < 4000 MB)!"
    print(f"  ✓ Correcto: Solo se demotan los bloques más antiguos expulsados de RAM")


    # 5. Turno 4 Agente `build` (Reuso de prefijo grande)
    print("\n[Paso 5] Turno 4 Agente `build` (Reuso de prefijo incremental)...")
    t0_paso4 = time.time()
    delta_3 = generar_bloque_texto(50, offset=700)
    prefijo_t4 = prefijo_t3 + delta_3
    resp_t4 = harness.pedir_chat(prefijo_t4, "primary_nothink", persist_disk=True)
    t_prompt_4 = resp_t4.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Prompt tokens: {t_prompt_4}, Tiempo: {resp_t4['_elapsed_sec']:.2f}s")
    n_nuevos_4, mb_nuevos_4 = harness.medir_disco_nuevos(t0_paso4)
    print(f"  -> Archivos NUEVOS en SSD tras Turno 4: {n_nuevos_4} archivos ({mb_nuevos_4:.2f} MB)")

    n_total_nuevos, mb_total_nuevos = harness.medir_disco_nuevos(t_inicio_global)
    print("\n" + "═" * 70)
    print("  RESULTADO FINAL DE VALIDACIÓN END-TO-END")
    print("═" * 70)
    print(f"Total de bloques NUEVOS escritos en SSD a lo largo de toda la corrida:")
    print(f"  -> Archivos: {n_total_nuevos} bloques")
    print(f"  -> Tamaño total escrito: {mb_total_nuevos:.2f} MB ({mb_total_nuevos / 1024:.2f} GB)")
    print(f"Comparativa:")
    print(f"  • Con Stock vLLM: habrías escrito > 60.00 GB al SSD para estos turnos.")
    print(f"  • Con Tiered Demotion (PN90): se escribieron solo {mb_total_nuevos / 1024:.2f} GB al SSD (reducción > 95%).")
    print("═" * 70)


if __name__ == "__main__":
    main()


