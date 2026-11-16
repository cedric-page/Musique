"""Répétition synthétique dont les pistes sont déjà « séparées », pour tester le
repérage de la chanson : on sait exactement où la chanson commence et finit."""

import numpy as np

from batterie_synthetique import SR, batterie, temps_des_battements


def _db(niveau_dbfs: float) -> float:
    return 10 ** (niveau_dbfs / 20)


def _note(freq, duree, amorti, rng=None, vibrato=0.0):
    t = np.arange(int(duree * SR)) / SR
    phase = 2 * np.pi * freq * t
    if vibrato:
        phase += vibrato * np.sin(2 * np.pi * 5.5 * t)
    son = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    return son * np.exp(-t / amorti) * np.minimum(1, t / 0.01)


def _poser(piste, son, instant, gain=1.0):
    i = int(instant * SR)
    if i < 0 or i >= piste.size:
        return
    n = min(son.size, piste.size - i)
    piste[i : i + n] += gain * son[:n]


def _normaliser(piste, niveau_dbfs, debut, fin):
    zone = piste[int(debut * SR) : int(fin * SR)]
    rms = np.sqrt(np.mean(zone**2)) if zone.size else 0
    return piste * (_db(niveau_dbfs) / rms) if rms > 0 else piste


def parole(piste, debut, fin, rng, niveau=-26):
    """Syllabes courtes (~4 par seconde) à hauteur variable, avec des pauses."""
    t = debut
    morceau = np.zeros_like(piste)
    while t < fin - 0.3:
        if rng.random() < 0.15:
            t += rng.uniform(0.4, 1.2)  # pause entre deux phrases
            continue
        duree = rng.uniform(0.12, 0.25)
        freq = rng.uniform(110, 220)
        _poser(morceau, _note(freq, duree, 0.1) * np.hanning(int(duree * SR)), t)
        t += duree + rng.uniform(0.03, 0.1)
    return piste + _normaliser(morceau, niveau, debut, fin)


def chant(piste, debut, fin, rng, niveau=-18):
    """Notes tenues avec vibrato."""
    morceau = np.zeros_like(piste)
    t = debut
    while t < fin - 0.5:
        duree = min(rng.uniform(0.5, 1.6), fin - t)
        freq = rng.choice([262, 294, 330, 349, 392, 440])
        enveloppe = np.minimum(1, np.minimum(np.arange(int(duree * SR)) / (0.05 * SR), 1))
        _poser(morceau, _note(freq, duree, 5.0, vibrato=0.6) * enveloppe, t)
        t += duree + 0.05
    return piste + _normaliser(morceau, niveau, debut, fin)


def accordage(piste, instants, niveau=-24):
    morceau = np.zeros_like(piste)
    for k, instant in enumerate(instants):
        _poser(morceau, _note([82, 110, 147, 196, 247, 330][k % 6], 2.5, 0.8), instant)
    return piste + _normaliser(morceau, niveau, instants[0], instants[-1] + 1)


def accords(piste, debut, fin, bpm, freqs, niveau, amorti=0.6):
    """Un accord sur chaque temps entre ``debut`` et ``fin``."""
    morceau = np.zeros_like(piste)
    for instant in temps_des_battements(fin - debut, bpm) + debut:
        for f in freqs:
            _poser(morceau, _note(f, 1.5, amorti), instant, 1 / len(freqs))
    return piste + _normaliser(morceau, niveau, debut, fin)


def repetition(graine=0, bavardage_avant=0.0):
    """120 s (+ ``bavardage_avant``) :

    - 2-18 s : discussion (voix), accordage de la guitare à 4, 7, 10 et 13 s ;
    - 22-30 s : intro au piano seul, décompte de baguettes à 28-29,5 s ;
    - 30-95 s : la chanson (batterie, basse, guitare, piano, chant) ;
    - 100-116 s : discussion.

    Renvoie (pistes séparées en mono, début attendu, fin attendue des instruments).
    """
    rng = np.random.default_rng(graine)
    d = bavardage_avant
    n = int((120 + d) * SR)
    pistes = {nom: np.zeros(n) for nom in ("voix", "batterie", "basse", "guitare", "piano", "autre")}
    if d:
        pistes["voix"] = parole(pistes["voix"], 1, d, rng)
    pistes["voix"] = parole(pistes["voix"], d + 2, d + 18, rng)
    pistes["guitare"] = accordage(pistes["guitare"], [d + 4, d + 7, d + 10, d + 13])
    pistes["piano"] = accords(pistes["piano"], d + 22, d + 30, 100, [262, 330, 392], -24)
    for k in range(4):
        _poser(pistes["batterie"], _note(2000, 0.05, 0.01), d + 28 + 0.5 * k, 0.3)
    morceau = np.zeros(n)
    morceau[int((d + 30) * SR) : int((d + 95) * SR)] = batterie(65, 120, graine=graine)
    pistes["batterie"] += _normaliser(morceau, -15, d + 30, d + 95)
    pistes["basse"] = accords(pistes["basse"], d + 30, d + 95, 120, [55], -18, amorti=0.4)
    pistes["guitare"] = accords(pistes["guitare"], d + 30, d + 95, 120, [196, 247, 294], -20)
    pistes["piano"] = accords(pistes["piano"], d + 30, d + 95, 120, [262, 330, 392], -22)
    pistes["voix"] = chant(pistes["voix"], d + 30, d + 95, rng)
    pistes["voix"] = parole(pistes["voix"], d + 100, d + 116, rng)
    # Résidus de séparation : un peu de tout dans « autre ».
    pistes["autre"] = 0.01 * sum(pistes[nom] for nom in ("voix", "guitare", "piano"))
    return {nom: v.astype(np.float32) for nom, v in pistes.items()}, d + 22.0, d + 95.0


class SeparateurScenario:
    """Ignore l'audio reçu et renvoie les pistes du scénario (en stéréo)."""

    identifiant = "scenario:v1"

    def __init__(self, pistes):
        self.pistes = pistes

    def separer(self, audio, sr):
        assert sr == SR
        return {
            nom: np.repeat(v[: audio.shape[0], None], 2, axis=1).astype(np.float32) for nom, v in self.pistes.items()
        }, sr
