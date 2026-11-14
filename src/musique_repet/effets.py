"""Effets audio appliqués aux pistes avant le mixage."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParametresChorus:
    """Chorus léger : quelques copies du signal légèrement retardées et désaccordées."""

    melange: float = 0.3
    """Proportion du signal traité (0 = pas d'effet, 1 = autant de traité que d'original)."""
    retards_ms: tuple[float, ...] = (15.0, 22.0)
    """Retard moyen de chaque voix du chorus."""
    profondeur_ms: float = 2.0
    """Amplitude de la variation du retard (désaccord)."""
    vitesses_hz: tuple[float, ...] = (0.6, 0.9)
    """Vitesse de modulation de chaque voix du chorus."""


def chorus(audio: np.ndarray, sr: int, params: ParametresChorus = ParametresChorus()) -> np.ndarray:
    """Applique un chorus à ``audio`` (échantillons, canaux) et renvoie un nouveau tableau.

    Chaque voix est une ligne à retard modulée par une sinusoïde, décalée d'un
    quart de période entre canaux pour élargir l'image stéréo. Le niveau global
    est conservé. Le traitement se fait par blocs pour limiter la mémoire.
    """
    if params.melange <= 0 or audio.size == 0:
        return audio.copy()
    if len(params.retards_ms) != len(params.vitesses_hz):
        raise ValueError("retards_ms et vitesses_hz doivent avoir la même longueur")

    nb, canaux = audio.shape
    sortie = np.empty_like(audio)
    nb_voix = len(params.retards_ms)
    normalisation = 1.0 / (1.0 + params.melange)
    taille_bloc = sr * 10

    for debut in range(0, nb, taille_bloc):
        fin = min(nb, debut + taille_bloc)
        n = np.arange(debut, fin, dtype=np.float64)
        for c in range(canaux):
            x = audio[:, c]
            traite = np.zeros(fin - debut, dtype=np.float64)
            for v in range(nb_voix):
                phase = v * math.pi / nb_voix + c * math.pi / 2
                retard = (
                    params.retards_ms[v]
                    + params.profondeur_ms * np.sin(2 * math.pi * params.vitesses_hz[v] * n / sr + phase)
                ) * sr / 1000.0
                position = n - retard
                i0 = np.floor(position).astype(np.int64)
                frac = position - i0
                valide = i0 >= 0  # avant le début du fichier : silence
                i0c = np.clip(i0, 0, nb - 1)
                i1c = np.clip(i0 + 1, 0, nb - 1)
                echantillon = x[i0c] * (1.0 - frac) + x[i1c] * frac
                traite += np.where(valide, echantillon, 0.0)
            sec = audio[debut:fin, c].astype(np.float64)
            sortie[debut:fin, c] = (sec + params.melange * traite / nb_voix) * normalisation
    return sortie
