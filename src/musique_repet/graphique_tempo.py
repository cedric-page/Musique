"""Graphique JPG des variations de tempo d'un morceau."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .graphique_commun import (
    ENCRE,
    ENCRE_DISCRETE,
    ENCRE_SECONDAIRE,
    SERIE,
    SERIE_CLAIRE,
    SURFACE,
    axe_du_temps,
    date_lisible,
    enregistrer_jpg,
    pourcent,
)
from .tempo import AnalyseTempo

def tracer(analyse: AnalyseTempo, chemin: Path, titre: str, date: str | None = None) -> None:
    """Enregistre en JPG la courbe du tempo au cours du morceau.

    ``date`` : date de la répétition au format AAAAMMJJ (facultative).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    tolerance = analyse.tolerance_pct / 100
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = plt.figure(figsize=(12, 5.2), dpi=150, facecolor=SURFACE)
        ax = fig.add_axes((0.06, 0.11, 0.85, 0.6))
        ax.set_facecolor(SURFACE)

        # En-tête : titre, contexte, chiffres clés.
        fig.text(0.06, 0.935, titre, color=ENCRE, fontsize=16, fontweight="bold", va="center")
        contexte = ["Tempo au cours du morceau"]
        if date_lisible(date):
            contexte.append(f"répétition du {date_lisible(date)}")
        if analyse.sources:
            contexte.append("mesuré sur " + ", ".join(analyse.sources))
        fig.text(0.06, 0.88, "  ·  ".join(contexte), color=ENCRE_SECONDAIRE, fontsize=10, va="center")

        if analyse.detecte:
            mediane = analyse.mediane
            debut, fin = analyse.debut_fin
            chiffres = (
                f"Médiane {mediane:.0f} BPM   ·   de {analyse.minimum:.0f} à {analyse.maximum:.0f} BPM   ·   "
                f"{pourcent(analyse.part_stable)} du morceau à ±{analyse.tolerance_pct:g} % de la médiane   ·   "
                f"début {debut:.0f} → fin {fin:.0f} BPM"
            )
            fig.text(0.06, 0.825, chiffres, color=ENCRE, fontsize=11, va="center")

            ax.axhspan(mediane * (1 - tolerance), mediane * (1 + tolerance), facecolor=SERIE, alpha=0.1, lw=0, zorder=1)
            ax.axhline(mediane, color=ENCRE_DISCRETE, lw=1, zorder=2)
            ax.text(
                1.006, mediane, f"médiane\n{mediane:.0f} BPM", transform=ax.get_yaxis_transform(),
                color=ENCRE_SECONDAIRE, fontsize=9, va="center", ha="left", linespacing=1.3,
            )
            ax.plot(analyse.temps, analyse.bpm_brut, color=SERIE_CLAIRE, lw=1, zorder=3)
            ax.plot(
                analyse.temps, analyse.bpm_lisse, color=SERIE, lw=2,
                solid_capstyle="round", solid_joinstyle="round", zorder=4,
            )

            # Repères sur les extrêmes, seulement s'ils se distinguent vraiment.
            if analyse.maximum - analyse.minimum >= 1:
                for indice, decalage, alignement in (
                    (int(np.nanargmax(analyse.bpm_lisse)), 9, "bottom"),
                    (int(np.nanargmin(analyse.bpm_lisse)), -9, "top"),
                ):
                    x, y = analyse.temps[indice], analyse.bpm_lisse[indice]
                    ax.scatter([x], [y], s=60, color=SERIE, edgecolors=SURFACE, linewidths=2, zorder=5)
                    ax.annotate(
                        f"{y:.0f}", (x, y), xytext=(0, decalage), textcoords="offset points",
                        ha="center", va=alignement, color=ENCRE_SECONDAIRE, fontsize=9, zorder=6,
                    )

            bas = min(analyse.minimum, mediane * (1 - tolerance))
            haut = max(analyse.maximum, mediane * (1 + tolerance))
            marge = max((haut - bas) * 0.3, mediane * 0.04)
            ax.set_ylim(bas - marge, haut + marge)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.set_ylabel("BPM", color=ENCRE_DISCRETE, fontsize=9)

            legende = [
                Line2D([], [], color=SERIE, lw=2, label=f"Tempo (moyenne sur {analyse.lissage_temps} temps)"),
                Line2D([], [], color=SERIE_CLAIRE, lw=1, label="Tempo entre deux temps"),
                Patch(facecolor=SERIE, alpha=0.1, label=f"±{analyse.tolerance_pct:g} % autour de la médiane"),
            ]
            fig.legend(
                handles=legende, loc="center left", bbox_to_anchor=(0.055, 0.765), ncol=3, frameon=False,
                fontsize=9, labelcolor=ENCRE_SECONDAIRE, handlelength=2.2, columnspacing=2.5,
            )
        else:
            fig.text(0.06, 0.825, "Aucune pulsation régulière détectée", color=ENCRE, fontsize=11, va="center")
            ax.text(
                0.5, 0.5, "Pas de rythme assez régulier pour mesurer un tempo\n(passage libre, silence ou instruments sans attaques nettes)",
                transform=ax.transAxes, ha="center", va="center", color=ENCRE_SECONDAIRE, fontsize=11, linespacing=1.5,
            )
            ax.set_yticks([])

        axe_du_temps(ax, analyse.duree)
        enregistrer_jpg(fig, chemin)
