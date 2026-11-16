"""Graphique JPG de la justesse d'une voix (lead ou harmonie), note par note et toutes les 8 secondes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .graphique_commun import (
    ATTENTION,
    BON,
    CRITIQUE,
    ENCRE,
    ENCRE_DISCRETE,
    ENCRE_SECONDAIRE,
    SURFACE,
    axe_du_temps,
    date_lisible,
    enregistrer_jpg,
    minutes_secondes,
    pourcent,
)
from .justesse import AnalyseJustesse

COULEURS = {"juste": BON, "approximatif": ATTENTION, "faux": CRITIQUE}
LIMITE_CENTS = 60  # au-delà, la note est dessinée au bord avec une flèche
ETIQUETTES_MAX = 8  # fausses notes hors harmonie nommées sur le graphique


def _nombre(valeur: float) -> str:
    """Nombre signé à la française : « −8 », « +12 »."""
    return f"{valeur:+.0f}".replace("-", "−")


def _tendance(ecart: float) -> str:
    if math.isnan(ecart) or abs(ecart) < 5:
        return "ni trop haut ni trop bas en moyenne"
    sens = "bas" if ecart < 0 else "haut"
    force = "un peu " if abs(ecart) < 15 else "trop "
    return f"{force}{sens} en moyenne ({_nombre(ecart)} cents)"


def _couleur_phrase(part: float) -> str:
    return BON if part >= 0.8 else ATTENTION if part >= 0.6 else CRITIQUE


def tracer(analyse: AnalyseJustesse, chemin: Path, titre: str, date: str | None = None) -> None:
    """Enregistre en JPG la justesse de chaque note et la part juste de chaque tranche de 8 s."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    tol, faux = analyse.tolerance_cents, analyse.faux_cents
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = plt.figure(figsize=(12, 6.8), dpi=150, facecolor=SURFACE)
        ax = fig.add_axes((0.07, 0.31, 0.86, 0.45))
        ax_phrases = fig.add_axes((0.07, 0.08, 0.86, 0.13), sharex=ax)

        # En-tête.
        fig.text(0.07, 0.955, titre, color=ENCRE, fontsize=16, fontweight="bold", va="center")
        contexte = ["Justesse de la voix lead" if analyse.voix == "lead" else "Justesse de l'harmonie (seconde voix)"]
        if date_lisible(date):
            contexte.append(f"répétition du {date_lisible(date)}")
        if analyse.reference == "instruments":
            la = f"{analyse.la_hz:.1f}".replace(".", ",")
            contexte.append(f"comparée à l'accordage du groupe (La = {la} Hz)")
        else:
            contexte.append("comparée à l'accordage moyen des voix (aucun instrument mélodique)")
        fig.text(0.07, 0.912, "  ·  ".join(contexte), color=ENCRE_SECONDAIRE, fontsize=10, va="center")

        if analyse.detecte:
            fausses = analyse.fausses
            hors = [n for n in analyse.notes if n.hors_harmonie]
            chiffres = [
                f"{pourcent(analyse.part_juste)} juste sur {minutes_secondes(analyse.duree_chantee)} de chant",
                _tendance(analyse.ecart_moyen),
                f"{len(fausses)} note{'s' if len(fausses) > 1 else ''} fausse{'s' if len(fausses) > 1 else ''}"
                + (f" dont {len(hors)} hors de l'harmonie" if hors else ""),
            ]
            if analyse.tonalite:
                chiffres.append(f"tonalité : {analyse.tonalite}")
            fig.text(0.07, 0.868, "   ·   ".join(chiffres), color=ENCRE, fontsize=11, va="center")

            # Zone juste, repères et notes.
            ax.axhspan(-tol, tol, facecolor=BON, alpha=0.1, lw=0, zorder=1)
            ax.axhline(0, color=ENCRE_DISCRETE, lw=1, zorder=2)
            for categorie, couleur in COULEURS.items():
                notes = [n for n in analyse.notes if analyse.categorie(n) == categorie]
                if not notes:
                    continue
                y = np.clip([n.ecart for n in notes], -LIMITE_CENTS + 3, LIMITE_CENTS - 3)
                ax.hlines(
                    y, [n.debut for n in notes], [n.fin for n in notes],
                    colors=couleur, lw=3.5, capstyle="round", zorder=3,
                )
                hors_limite = [(n, v) for n, v in zip(notes, y) if abs(n.ecart) > LIMITE_CENTS]
                for n, v in hors_limite:
                    ax.scatter(
                        [(n.debut + n.fin) / 2], [v + (4 if v > 0 else -4)], marker="^" if v > 0 else "v",
                        s=30, color=couleur, edgecolors=SURFACE, linewidths=1, zorder=4,
                    )
            # Fausses notes hors de l'harmonie : croix et nom de la note (les plus longues).
            for n in sorted(hors, key=lambda n: n.duree, reverse=True)[:ETIQUETTES_MAX]:
                x, y = (n.debut + n.fin) / 2, float(np.clip(n.ecart, -LIMITE_CENTS + 3, LIMITE_CENTS - 3))
                ax.scatter([x], [y], marker="x", s=36, color=ENCRE, linewidths=1.5, zorder=5)
                ax.annotate(
                    n.nom.rstrip("0123456789-"), (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", color=ENCRE_SECONDAIRE, fontsize=8, zorder=6,
                )
            for n in hors[ETIQUETTES_MAX:]:
                ax.scatter([(n.debut + n.fin) / 2], [float(np.clip(n.ecart, -57, 57))], marker="x",
                           s=36, color=ENCRE, linewidths=1.5, zorder=5)

            legende = [
                Line2D([], [], color=BON, lw=3.5, label=f"Juste (±{tol:g} cents)"),
                Line2D([], [], color=ATTENTION, lw=3.5, label=f"Approximatif ({tol:g} à {faux:g} cents)"),
                Line2D([], [], color=CRITIQUE, lw=3.5, label=f"Faux (plus de {faux:g} cents)"),
                Line2D([], [], color=ENCRE, marker="x", lw=0, markersize=6, label="Note hors de l'harmonie"),
            ]
            fig.legend(
                handles=legende, loc="center left", bbox_to_anchor=(0.065, 0.815), ncol=4, frameon=False,
                fontsize=9, labelcolor=ENCRE_SECONDAIRE, handlelength=2.0, columnspacing=2.2,
            )

            # Part juste de chaque tranche de 8 s.
            for debut, fin, part in analyse.tranches():
                largeur = max(fin - debut - 0.4, 0.2)
                # Hauteur minimale : une tranche chantée à 0 % de justesse reste visible (en rouge),
                # pour ne pas la confondre avec une tranche où l'on ne chante pas.
                ax_phrases.bar(
                    debut, max(part * 100, 3), width=largeur, align="edge", color=_couleur_phrase(part), lw=0, zorder=3
                )
        else:
            fig.text(
                0.07, 0.868, "Pas de chant détecté" if analyse.voix == "lead" else "Pas d'harmonie détectée",
                color=ENCRE, fontsize=11, va="center",
            )
            ax.text(
                0.5, 0.5, "Aucune note chantée assez longue pour mesurer la justesse",
                transform=ax.transAxes, ha="center", va="center", color=ENCRE_SECONDAIRE, fontsize=11,
            )

        # Panneau principal : écart en cents.
        axe_du_temps(ax, analyse.duree)
        ax.set_ylim(-LIMITE_CENTS, LIMITE_CENTS)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "0" if v == 0 else _nombre(v)))
        ax.set_ylabel("écart à la note juste (cents)", color=ENCRE_DISCRETE, fontsize=9)
        ax.tick_params(axis="x", labelbottom=False)
        for texte, y, alignement in (("trop haut", LIMITE_CENTS - 4, "top"), ("trop bas", -LIMITE_CENTS + 4, "bottom")):
            ax.text(0.004, y, texte, transform=ax.get_yaxis_transform(), color=ENCRE_SECONDAIRE,
                    fontsize=9, va=alignement, ha="left", zorder=6)
        ax.text(1.006, 0, "100 cents\n= ½ ton", transform=ax.get_yaxis_transform(), color=ENCRE_DISCRETE,
                fontsize=8, va="center", ha="left", linespacing=1.3)

        # Panneau du bas : part juste par tranche de 8 s.
        axe_du_temps(ax_phrases, analyse.duree)
        ax_phrases.set_ylim(0, 100)
        ax_phrases.set_yticks([0, 50, 100])
        ax_phrases.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f} %"))
        ax_phrases.set_title(
            "Part du chant juste, toutes les 8 secondes (vert : 80 % ou plus, jaune : 60 à 80 %, rouge : moins de 60 %)",
            loc="left", color=ENCRE_SECONDAIRE, fontsize=9, pad=6,
        )
        enregistrer_jpg(fig, chemin)
