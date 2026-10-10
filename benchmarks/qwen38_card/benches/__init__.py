# SPDX-License-Identifier: Apache-2.0
"""Un modulo por benchmark.

Cada uno expone NAME, CARD (puntaje publicado), load(args) -> [Item] y
grade(items, responses, ctx) -> (por_item, metricas). El puntaje por item va
en 0..1; la metrica principal es su promedio x100, igual que el card.
"""
