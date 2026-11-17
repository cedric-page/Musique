"""Palette et mise en forme communes aux graphiques JPG (fond clair)."""

from __future__ import annotations

from pathlib import Path

SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"
SERIE = "#2a78d6"  # bleu, première série
SERIE_CLAIRE = "#9ec5f4"  # même bleu, plus clair : données en retrait


def minutes_secondes(secondes: float, _position=None) -> str:
    s = int(round(secondes))
    return f"{s // 60}:{s % 60:02d}"


def pas_graduation(duree: float) -> float:
    for pas in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800):
        if duree / pas <= 10:
            return pas
    return 3600


def pourcent(valeur: float) -> str:
    return f"{round(valeur * 100)} %"


def date_lisible(date: str | None) -> str | None:
    """« 20260312 » -> « 12/03/2026 »."""
    if date and len(date) == 8 and date.isdigit():
        return f"{date[6:8]}/{date[4:6]}/{date[0:4]}"
    return date


def axe_du_temps(ax, duree: float) -> None:
    """Axe horizontal en minutes:secondes, grille horizontale fine, sans cadre."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, max(duree, 1.0))
    ax.xaxis.set_major_locator(MultipleLocator(pas_graduation(duree)))
    ax.xaxis.set_major_formatter(FuncFormatter(minutes_secondes))
    ax.grid(axis="y", color=GRILLE, lw=0.6)
    ax.set_axisbelow(True)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color(AXE)
    ax.tick_params(colors=ENCRE_DISCRETE, labelsize=9, length=0, pad=6)


def enregistrer_jpg(fig, chemin: Path) -> None:
    """Enregistre la figure en JPG sans laisser de fichier à moitié écrit, puis la ferme."""
    import matplotlib.pyplot as plt

    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    fig.savefig(temporaire, format="jpg", dpi=150, facecolor=SURFACE, pil_kwargs={"quality": 90})
    plt.close(fig)
    temporaire.replace(chemin)
