"""Repérage de la chanson dans un enregistrement : on retire ce qui précède et suit
(discussions, accordage, silences).

Le repérage s'appuie sur les pistes séparées :

* le **cœur** de la chanson est l'ensemble des moments où plusieurs sources
  jouent en même temps (batterie, basse, guitare, piano, voix), par rapport à
  ce que le groupe atteint au plus fort du morceau ;
* il est **prolongé** tant qu'un instrument joue sans interruption (intro au
  piano seul, résonance du dernier accord) ;
* une voix seule (discussion) ou un instrument isolé séparé de la chanson par
  un silence (accordage, échauffement) reste en dehors.

Seul l'extérieur est coupé : les pauses à l'intérieur de la chanson sont gardées.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

log = logging.getLogger(__name__)

INSTRUMENTS = ("batterie", "basse", "guitare", "piano")


@dataclass(frozen=True)
class ParametresRognage:
    pas_s: float = 0.25
    """Résolution de l'analyse."""
    lissage_s: float = 3.0
    """Une activité plus courte est ignorée (coup isolé, bruit)."""
    seuil_relatif_db: float = 20.0
    """Une piste joue quand elle est à moins de 20 dB de son niveau de jeu habituel."""
    plancher_db: float = -55.0
    """En dessous de ce niveau absolu (dBFS), une piste est silencieuse."""
    ecart_piste_db: float = 25.0
    """Une piste plus de 25 dB sous la plus forte ne contient que des résidus de séparation."""
    trou_max_s: float = 10.0
    """Pause tolérée à l'intérieur de la chanson (break, silence avant un couplet)."""
    duree_min_s: float = 15.0
    """Durée minimale d'un passage joué par le groupe pour être une chanson."""
    prolongement_trou_s: float = 1.5
    """Silence toléré entre la chanson et un instrument seul qui la précède ou la suit."""
    prolongement_max_s: float = 30.0
    """Durée maximale gardée d'un instrument seul avant ou après le groupe (intro, outro)."""
    marge_debut_s: float = 0.5
    marge_fin_s: float = 1.5
    """Marge gardée après la fin, pour la résonance du dernier accord (avec un fondu)."""
    gain_min_s: float = 2.0
    """On ne coupe pas un bord pour gagner moins de 2 s."""


@dataclass(frozen=True)
class Extrait:
    debut: float
    fin: float
    duree_totale: float

    @property
    def rogne(self) -> bool:
        return self.debut > 0 or self.fin < self.duree_totale

    @property
    def duree(self) -> float:
        return self.fin - self.debut

    def description(self) -> str:
        if not self.rogne:
            return "enregistrement gardé en entier"
        return (
            f"chanson de {_mmss(self.debut)} à {_mmss(self.fin)} : "
            f"{self.debut:.0f} s coupées au début, {self.duree_totale - self.fin:.0f} s à la fin"
        )


def _mmss(secondes: float) -> str:
    s = int(round(secondes))
    return f"{s // 60}:{s % 60:02d}"


def niveau_db_par_trame(chemin: Path, pas_s: float) -> tuple[np.ndarray, float]:
    """Niveau (dBFS) de chaque trame de ``pas_s`` secondes, lu par blocs. Renvoie (niveaux, durée)."""
    with sf.SoundFile(chemin) as f:
        taille = max(1, int(round(pas_s * f.samplerate)))
        duree = f.frames / f.samplerate
        niveaux = []
        for bloc in f.blocks(blocksize=taille * 240, dtype="float32", always_2d=True):
            mono = bloc.mean(axis=1, dtype=np.float64)
            nb = mono.size // taille
            if nb:
                puissance = (mono[: nb * taille].reshape(nb, taille) ** 2).mean(axis=1)
                niveaux.append(10 * np.log10(puissance + 1e-12))
            if mono.size % taille:
                reste = mono[nb * taille :]
                niveaux.append(np.array([10 * math.log10(float(np.mean(reste**2)) + 1e-12)]))
    return (np.concatenate(niveaux) if niveaux else np.zeros(0)), duree


def _plages(masque: np.ndarray) -> list[tuple[int, int]]:
    """Plages [début, fin[ où ``masque`` est vrai."""
    bords = np.diff(np.concatenate([[0], masque.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(bords == 1).tolist(), np.flatnonzero(bords == -1).tolist()))


def _prolonger_avant(masque: np.ndarray, debut: int, trou: int, maximum: int) -> int:
    garde, j = debut, debut - 1
    while j >= 0 and debut - j <= maximum:
        if masque[j]:
            garde = j
        elif garde - j > trou:
            break
        j -= 1
    return garde


def _prolonger_apres(masque: np.ndarray, fin: int, trou: int, maximum: int) -> int:
    garde, j = fin, fin
    while j < masque.size and j - fin < maximum:
        if masque[j]:
            garde = j + 1
        elif j - garde >= trou:
            break
        j += 1
    return garde


def detecter_chanson(pistes: dict[str, Path], params: ParametresRognage = ParametresRognage()) -> Extrait:
    """Trouve le début et la fin de la chanson à partir des pistes séparées."""
    noms = [n for n in (*INSTRUMENTS, "voix") if n in pistes]
    if not noms:
        noms = list(pistes)
    niveaux: dict[str, np.ndarray] = {}
    duree = 0.0
    for nom in noms:
        niveaux[nom], duree_piste = niveau_db_par_trame(pistes[nom], params.pas_s)
        duree = max(duree, duree_piste)
    n = min(v.size for v in niveaux.values())
    tout = Extrait(0.0, duree, duree)
    if n == 0:
        return tout

    jeu = {nom: float(np.percentile(v[:n], 95)) for nom, v in niveaux.items()}
    plus_fort = max(jeu.values())
    retenues = [nom for nom in noms if jeu[nom] >= max(plus_fort - params.ecart_piste_db, params.plancher_db)]
    lissage = max(1, int(round(params.lissage_s / params.pas_s))) | 1  # taille impaire
    actif = {
        nom: median_filter(
            (niveaux[nom][:n] > max(jeu[nom] - params.seuil_relatif_db, params.plancher_db)).astype(np.uint8),
            size=lissage,
            mode="nearest",
        ).astype(bool)
        for nom in retenues
    }
    if not actif:
        return tout
    instruments = sum((actif[nom].astype(np.int8) for nom in retenues if nom != "voix"), np.zeros(n, np.int8))
    sources = instruments + actif.get("voix", np.zeros(n, bool)).astype(np.int8)

    # Nombre de sources qui jouent ensemble au plus fort du morceau (tenu au moins
    # la moitié de la durée minimale, quelle que soit la longueur des bavardages
    # autour), puis seuil du « cœur » : 3 sur 4 ou 5, 2 sur 2 ou 3.
    trames_par_s = 1.0 / params.pas_s
    tenue = params.duree_min_s / 2 * trames_par_s
    ensemble = next(
        (k for k in range(len(retenues), 1, -1) if any(f - d >= tenue for d, f in _plages(sources >= k))), 1
    )
    seuil = max(2, math.ceil(0.6 * ensemble))
    plages = _plages(sources >= seuil)
    fusion: list[list[int]] = []
    for debut, fin in plages:
        if fusion and debut - fusion[-1][1] <= params.trou_max_s * trames_par_s:
            fusion[-1][1] = fin
        else:
            fusion.append([debut, fin])
    coeur = [(d, f) for d, f in fusion if f - d >= params.duree_min_s * trames_par_s]

    if coeur:
        joue = instruments >= 1
        trou = int(round(params.prolongement_trou_s * trames_par_s))
        maximum = int(round(params.prolongement_max_s * trames_par_s))
        debut = _prolonger_avant(joue, coeur[0][0], trou, maximum)
        fin = _prolonger_apres(joue, coeur[-1][1], trou, maximum)
    else:
        # Pas de passage joué par le groupe (instrument seul, voix seule) :
        # on retire seulement les silences au début et à la fin.
        quelque_chose = np.flatnonzero(sources >= 1)
        if quelque_chose.size == 0:
            return tout
        debut, fin = int(quelque_chose[0]), int(quelque_chose[-1]) + 1

    debut_s = max(0.0, debut * params.pas_s - params.marge_debut_s)
    fin_s = min(duree, fin * params.pas_s + params.marge_fin_s)
    if debut_s < params.gain_min_s:
        debut_s = 0.0
    if duree - fin_s < params.gain_min_s:
        fin_s = duree
    return Extrait(debut_s, fin_s, duree)
