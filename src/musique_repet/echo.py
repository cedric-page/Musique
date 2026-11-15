"""Atténuation de la réverbération (écho de la salle) par soustraction spectrale.

Méthode de Lebart et al. (2001), reprise par Habets : dans une grande salle la
réverbération décroît de façon exponentielle, avec une durée caractérisée par
le RT60 (temps pour perdre 60 dB). La puissance de la réverbération tardive à
l'instant ``t`` est donc estimée à partir de la puissance du signal un peu plus
tôt (``t - retard``), atténuée selon cette décroissance. Cette part est ensuite
retirée de chaque bande de fréquence, avec un plancher pour éviter les
artefacts (« bruit musical »).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from . import audio_io

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParametresAntiEcho:
    rt60: float = 0.4
    """Durée de réverbération de la salle en secondes (grande salle : 1,5 à 3 s)."""
    attenuation_max_db: float = 12.0
    """Atténuation maximale de la réverbération ; 0 désactive le traitement."""
    retard_ms: float = 50.0
    """Ce qui arrive moins de 50 ms après le son direct est conservé (premières
    réflexions, qui donnent du corps au son) ; seule la traîne est atténuée."""
    surestimation: float = 1.0
    """Multiplie la réverbération estimée : > 1 traitement plus fort, < 1 plus doux."""
    lissage_ms: float = 30.0
    """Lissage temporel des puissances, limite les artefacts."""
    taille_fft: int = 2048
    pas: int = 512


def _traiter_canal(x: np.ndarray, sr: int, p: ParametresAntiEcho) -> np.ndarray:
    n_fft, pas = p.taille_fft, p.pas
    fenetre = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)  # racine de Hann périodique
    longueur = x.shape[0]
    signal = np.concatenate([np.zeros(n_fft, np.float32), x.astype(np.float32), np.zeros(n_fft, np.float32)])
    nb_trames = 1 + (signal.shape[0] - n_fft) // pas
    sortie = np.zeros(signal.shape[0], np.float32)
    norme = np.zeros(signal.shape[0], np.float32)

    alpha = math.exp(-pas / (sr * p.lissage_ms / 1000.0))
    decalage = max(1, round(p.retard_ms / 1000.0 * sr / pas))  # en trames
    delta = 3.0 * math.log(10.0) / p.rt60
    decroissance = p.surestimation * math.exp(-2.0 * delta * decalage * pas / sr)
    gain_min2 = 10.0 ** (-p.attenuation_max_db / 10.0)

    nb_bandes = n_fft // 2 + 1
    historique = np.zeros((decalage, nb_bandes))
    etat = np.zeros((1, nb_bandes))
    groupes = n_fft // pas  # les trames d'un même groupe ne se chevauchent pas
    lot = 1024  # trames traitées à la fois (mémoire bornée)
    decalages = np.arange(n_fft)

    for debut in range(0, nb_trames, lot):
        indices = np.arange(debut, min(nb_trames, debut + lot))
        positions = indices[:, None] * pas + decalages
        spectre = np.fft.rfft(signal[positions] * fenetre, axis=1)

        puissance = spectre.real.astype(np.float64) ** 2 + spectre.imag.astype(np.float64) ** 2
        lissee, etat = lfilter([1.0 - alpha], [1.0, -alpha], puissance, axis=0, zi=etat)
        avec_passe = np.concatenate([historique, lissee], axis=0)
        reverberation = decroissance * avec_passe[: indices.size]
        historique = avec_passe[-decalage:]

        gain2 = np.clip(1.0 - reverberation / np.maximum(lissee, 1e-12), gain_min2, 1.0)
        trames = np.fft.irfft(spectre * np.sqrt(gain2), n=n_fft, axis=1).astype(np.float32) * fenetre

        for g in range(groupes):
            selection = (indices % groupes) == g
            sortie[positions[selection]] += trames[selection]
            norme[positions[selection]] += fenetre * fenetre

    sortie /= np.maximum(norme, 1e-6)
    return sortie[n_fft : n_fft + longueur]


def attenuer_echo(audio: np.ndarray, sr: int, params: ParametresAntiEcho = ParametresAntiEcho()) -> np.ndarray:
    """Atténue la réverbération de ``audio`` (échantillons, canaux)."""
    if params.attenuation_max_db <= 0 or audio.size == 0:
        return audio.copy()
    return np.stack([_traiter_canal(audio[:, c], sr, params) for c in range(audio.shape[1])], axis=1)


FICHIER_META = "anti_echo.json"


def attenuer_echo_avec_cache(
    pistes: dict[str, Path], dossier: Path, params: ParametresAntiEcho
) -> dict[str, Path]:
    """Traite chaque piste séparée et garde le résultat dans ``dossier``.

    Le résultat est réutilisé tant que les pistes d'origine et les réglages
    n'ont pas changé.
    """
    if params.attenuation_max_db <= 0:
        return pistes
    signature = {
        "parametres": asdict(params),
        "sources": {nom: [c.stat().st_size, c.stat().st_mtime_ns] for nom, c in sorted(pistes.items())},
    }
    resultat = {nom: dossier / f"{nom}.wav" for nom in pistes}
    meta = dossier / FICHIER_META
    try:
        if json.loads(meta.read_text(encoding="utf-8")) == signature and all(p.exists() for p in resultat.values()):
            log.info("  Écho déjà atténué, réutilisation de %s", dossier)
            return resultat
    except (OSError, json.JSONDecodeError):
        pass

    log.info("  Atténuation de l'écho (RT60 %.1f s, jusqu'à -%.0f dB)", params.rt60, params.attenuation_max_db)
    meta.unlink(missing_ok=True)
    for nom, chemin in pistes.items():
        audio, sr = audio_io.lire(chemin)
        audio_io.ecrire_wav(resultat[nom], attenuer_echo(audio, sr, params), sr, sous_type="FLOAT")
        del audio
    meta.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return resultat
