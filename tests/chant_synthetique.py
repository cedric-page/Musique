"""Chant et accompagnement synthétiques dont on connaît chaque écart de justesse."""

import numpy as np

SR = 44100


def frequence(midi: float, accordage_cents: float = 0.0) -> float:
    """Fréquence d'une note MIDI (69 = La 440) dans un accordage décalé de ``accordage_cents``."""
    return 440.0 * 2 ** ((midi - 69 + accordage_cents / 100) / 12)


def _ton(f0: np.ndarray, harmoniques: int, pente: float) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f0) / SR
    return sum((pente**k) * np.sin((k + 1) * phase) for k in range(harmoniques))


def voix(duree, notes, accordage_cents=0.0, vibrato_cents=35.0, graine=0):
    """``notes`` : liste de (début, durée, note MIDI visée, écart en cents).

    Chaque note a un vibrato (5,5 Hz), une attaque glissée par en dessous
    (80 ms) et une enveloppe douce, comme une voix chantée.
    """
    rng = np.random.default_rng(graine)
    sortie = np.zeros(int(duree * SR))
    for debut, longueur, midi, ecart in notes:
        n = int(longueur * SR)
        t = np.arange(n) / SR
        attaque = -60 * np.exp(-t / 0.03)  # arrive par en dessous
        vib = vibrato_cents * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi)) * np.minimum(1, t / 0.3)
        cents = ecart + attaque + vib
        f0 = frequence(midi, accordage_cents) * 2 ** (cents / 1200)
        enveloppe = np.minimum(1, t / 0.04) * np.minimum(1, (longueur - t) / 0.06)
        i = int(debut * SR)
        sortie[i : i + n] += 0.2 * enveloppe * _ton(f0, 8, 0.6)
    return sortie.astype(np.float32)


def accompagnement(duree, accords, accordage_cents=0.0):
    """``accords`` : liste de (début, durée, [notes MIDI]) ; sons de piano qui s'éteignent."""
    sortie = np.zeros(int(duree * SR))
    for debut, longueur, notes in accords:
        n = int(longueur * SR)
        t = np.arange(n) / SR
        for midi in notes:
            f0 = np.full(n, frequence(midi, accordage_cents))
            son = _ton(f0, 6, 0.5) * np.exp(-t / 1.2) * np.minimum(1, t / 0.005)
            i = int(debut * SR)
            sortie[i : i + n] += 0.08 * son
    return sortie.astype(np.float32)


# Grille Do – Sol – La mineur – Fa, deux fois (2 s par accord).
GRILLE = [(0, [48, 60, 64, 67]), (2, [43, 59, 62, 67]), (4, [45, 60, 64, 69]), (6, [41, 60, 65, 69])]
ACCORDS = [(k * 8 + d, 2.0, notes) for k in range(2) for d, notes in GRILLE]

# Mélodie : (début, durée, note visée, écart en cents). Notes de l'accord ou de la gamme de Do majeur.
MELODIE = [
    (0.1, 0.9, 72, 0),      # Do4  juste
    (1.1, 0.8, 76, 5),      # Mi   juste
    (2.1, 0.9, 74, -8),     # Ré   juste
    (3.1, 0.8, 71, 30),     # Si   approximatif (trop haut)
    (4.1, 0.9, 72, 0),      # Do   juste
    (5.1, 0.8, 76, -50),    # Mi   faux (trop bas, entre Mi bémol et Mi)
    (6.1, 0.9, 77, 10),     # Fa   juste
    (7.1, 0.8, 72, 0),      # Do   juste
    (8.1, 0.9, 79, 0),      # Sol  juste
    (9.1, 0.8, 78, 0),      # Fa dièse sur l'accord de Do : fausse note (hors gamme), chantée bien droite
    (10.1, 0.9, 69, -25),   # La   approximatif (trop bas)
    (11.1, 0.8, 72, 15),    # Do   juste
    (12.1, 0.9, 77, -42),   # Fa   faux (trop bas)
    (13.1, 0.8, 76, 0),     # Mi   juste
    (14.1, 1.6, 72, 0),     # Do   juste (note tenue)
]
