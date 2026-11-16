"""Rythmes de batterie synthétiques au tempo connu, pour tester l'analyse du tempo."""

import numpy as np

SR = 44100


def _kick(rng):
    t = np.arange(int(0.25 * SR)) / SR
    freq = 50 + 90 * np.exp(-t / 0.03)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.12)


def _caisse_claire(rng):
    t = np.arange(int(0.2 * SR)) / SR
    return 0.5 * rng.standard_normal(t.size) * np.exp(-t / 0.06) + 0.3 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)


def _charleston(rng):
    t = np.arange(int(0.06 * SR)) / SR
    bruit = rng.standard_normal(t.size)
    bruit = np.diff(bruit, prepend=0.0)  # aigus
    return 0.15 * bruit * np.exp(-t / 0.015)


def temps_des_battements(duree, tempo):
    """Instants des temps pour un tempo constant (nombre) ou variable (fonction du temps en s)."""
    if not callable(tempo):
        valeur = float(tempo)
        tempo = lambda t: np.full_like(t, valeur)  # noqa: E731
    t = np.arange(int(duree * 1000)) / 1000.0
    phase = np.cumsum(tempo(t) / 60.0) / 1000.0  # en nombre de temps
    entiers = np.arange(0, int(phase[-1]))
    return np.interp(entiers, phase, t)


def batterie(duree, tempo, croches=True, silence=None, humain_ms=0.0, graine=0):
    """Rythme rock : grosse caisse sur 1 et 3, caisse claire sur 2 et 4, charleston en croches.

    ``silence`` = (début, fin) en secondes sans aucune note (break).
    ``humain_ms`` = imprécision aléatoire du batteur (écart-type en ms).
    """
    rng = np.random.default_rng(graine)
    sortie = np.zeros(int(duree * SR) + SR)
    kick, caisse, hh = _kick(rng), _caisse_claire(rng), _charleston(rng)
    temps = temps_des_battements(duree, tempo)

    def poser(son, instant):
        if silence and silence[0] <= instant < silence[1]:
            return
        i = int(round((instant + rng.normal(0, humain_ms / 1000)) * SR))
        if 0 <= i < sortie.size - son.size:
            sortie[i : i + son.size] += son

    for k, instant in enumerate(temps):
        poser(kick if k % 2 == 0 else caisse, instant)
        poser(hh, instant)
        if croches and k + 1 < len(temps):
            poser(hh * 0.7, (instant + temps[k + 1]) / 2)
    return sortie[: int(duree * SR)].astype(np.float32)
