#!/usr/bin/env python3
"""Ažuriran dijagram toka izrade BALANSIRANOG skupa (ćirilica)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
OUT = os.path.join(os.path.dirname(__file__), "..")

koraci = [
    "Извори: закони, одлуке Уставног суда,\nкадровска решења (PDF/DOCX)",
    "Чишћење и сегментација\n(спајање редова, транслитерација)",
    "Полуаутоматско предобележавање\n(regex + bcms-bertic-ner)",
    "Ручна анотација и исправка\n(Label Studio, 8 типова)",
    "Балансирање: проширење домена,\nпотискивање честих типова",
    "Поновна сегментација\n(премапирање ознака)",
    "Балансиран скуп (2.375 реченица,\nоднос 282:1 → ≈3,5:1)",
    "Примена модела — промптовање\n(Qwen3 инстр./мислећи) + NER4Legal",
    "Евалуација (P, R, F1; строго и ублажено;\nмикро и макро)",
]
boje = ["#DDEBF7","#DDEBF7","#FCE4D6","#FCE4D6","#FFF2CC","#FFF2CC","#E2EFDA","#FCE4D6","#F2DCDB"]
fig, ax = plt.subplots(figsize=(6.8, 10.2)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, len(koraci)*1.5+0.5)
for i, (txt, col) in enumerate(zip(koraci, boje)):
    y = (len(koraci)-1-i)*1.5 + 0.6
    ax.add_patch(FancyBboxPatch((1.0, y), 8.0, 1.05, boxstyle="round,pad=0.08,rounding_size=0.12",
                                linewidth=1.2, edgecolor="#666", facecolor=col))
    ax.text(5.0, y+0.52, txt, ha="center", va="center", fontsize=9.5)
    if i < len(koraci)-1:
        ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y-0.45), arrowstyle="-|>", mutation_scale=16, color="#444"))
ax.set_title("Ток израде и евалуације балансираног скупа података", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "slika_pipeline.png"), dpi=150)
fig.savefig(os.path.join(OUT, "figures", "slika_pipeline.png"), dpi=150)
print("Napravljen azuriran slika_pipeline.png")
