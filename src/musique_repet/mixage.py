"""Mesure de sonie, équilibrage des pistes et mixage final."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyloudnorm

from . import audio_io
from .effets import CHORUS_BASSE, ParametresChorus, chorus

log = logging.getLogger(__name__)


def mesurer_lufs(audio: np.ndarray, sr: int) -> float:
    """Sonie intégrée (ITU-R BS.1770) en LUFS ; ``-inf`` pour un signal muet."""
    if audio.size == 0 or not np.any(audio):
        return -math.inf
    taille_bloc = 0.4
    if audio.shape[0] < int(taille_bloc * sr) + 1:
        # Trop court pour la mesure normalisée : complément de silence.
        audio = np.pad(audio, ((0, int(taille_bloc * sr) + 1 - audio.shape[0]), (0, 0)))
    valeur = pyloudnorm.Meter(sr, block_size=taille_bloc).integrated_loudness(audio.astype(np.float64))
    return float(valeur)


def db_vers_lineaire(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lineaire_vers_db(facteur: float) -> float:
    return 20.0 * math.log10(facteur) if facteur > 0 else -math.inf


@dataclass
class ParametresMixage:
    seuil_silence_lufs: float = -50.0
    """En dessous, une piste est considérée vide (instrument absent) : pas d'équilibrage."""
    gain_max_db: float = 12.0
    """Correction maximale (en + ou en -) appliquée par l'équilibrage."""
    exclues_equilibrage: set[str] = field(default_factory=lambda: {"autre"})
    """Pistes laissées à leur niveau d'origine (résidus de séparation, etc.)."""
    lufs_cible: float = -14.0
    """Sonie du mixage final."""
    crete_max_db: float = -1.0
    """Crête maximale du mixage final (dBFS)."""
    chorus: dict[str, ParametresChorus] = field(
        default_factory=lambda: {"voix": ParametresChorus(), "basse": CHORUS_BASSE}
    )
    """Pistes recevant un chorus, avec leurs réglages."""


@dataclass
class ReglagePiste:
    nom: str
    lufs: float
    gain_equilibrage_db: float
    facteur_config: float

    @property
    def facteur_total(self) -> float:
        return db_vers_lineaire(self.gain_equilibrage_db) * self.facteur_config


def calculer_equilibrage(
    sonies: dict[str, float], facteurs_config: dict[str, float], params: ParametresMixage
) -> list[ReglagePiste]:
    """Calcule le gain de chaque piste pour les amener à la même sonie.

    La cible est la médiane des pistes actives, ce qui conserve le niveau
    global. Les pistes muettes ou exclues ne sont pas modifiées.
    """
    actives = {
        nom: lufs
        for nom, lufs in sonies.items()
        if lufs > params.seuil_silence_lufs and nom not in params.exclues_equilibrage
    }
    cible = float(np.median(list(actives.values()))) if actives else None
    reglages = []
    for nom, lufs in sonies.items():
        gain = 0.0
        if cible is not None and nom in actives:
            gain = float(np.clip(cible - lufs, -params.gain_max_db, params.gain_max_db))
        reglages.append(ReglagePiste(nom, lufs, gain, facteurs_config.get(nom, 1.0)))
    return reglages


def mixer(
    pistes: dict[str, Path],
    facteurs_config: dict[str, float],
    params: ParametresMixage,
) -> tuple[np.ndarray, int, list[ReglagePiste]]:
    """Équilibre, applique les volumes configurés, additionne et normalise.

    Les pistes sont lues une par une pour limiter la mémoire utilisée.
    """
    sonies: dict[str, float] = {}
    sr_ref = None
    for nom, chemin in pistes.items():
        audio, sr = audio_io.lire(chemin)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"fréquences d'échantillonnage différentes entre les pistes ({sr} / {sr_ref})")
        sonies[nom] = mesurer_lufs(audio, sr)
        del audio

    reglages = calculer_equilibrage(sonies, facteurs_config, params)

    mix: np.ndarray | None = None
    for reglage in reglages:
        if reglage.facteur_total == 0:
            continue
        audio, sr = audio_io.lire(pistes[reglage.nom])
        if reglage.nom in params.chorus:
            audio = chorus(audio, sr, params.chorus[reglage.nom])
        audio *= reglage.facteur_total
        if mix is None:
            mix = audio
        else:
            n = max(mix.shape[0], audio.shape[0])
            if mix.shape[0] < n:
                mix = np.pad(mix, ((0, n - mix.shape[0]), (0, 0)))
            mix[: audio.shape[0]] += audio
        del audio

    if mix is None:
        # Toutes les pistes sont muettes ou coupées dans la configuration.
        audio, _ = audio_io.lire(next(iter(pistes.values())))
        mix = np.zeros_like(audio)

    mix = normaliser(mix, sr_ref, params)
    return mix, sr_ref, reglages


def normaliser(mix: np.ndarray, sr: int, params: ParametresMixage) -> np.ndarray:
    """Amène le mixage à la sonie cible sans dépasser la crête maximale."""
    lufs = mesurer_lufs(mix, sr)
    if not math.isfinite(lufs):
        return mix
    gain = db_vers_lineaire(params.lufs_cible - lufs)
    crete = float(np.max(np.abs(mix))) * gain
    plafond = db_vers_lineaire(params.crete_max_db)
    if crete > plafond:
        gain *= plafond / crete
        log.info(
            "Crête limitée à %.1f dBFS : sonie finale %.1f LUFS au lieu de %.1f",
            params.crete_max_db,
            lufs + lineaire_vers_db(gain),
            params.lufs_cible,
        )
    return mix * gain
