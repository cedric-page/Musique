"""Analyse du tempo (BPM) et de ses variations au cours d'un morceau.

Étapes :

1. **Signal rythmique** : somme des pistes séparées batterie, basse, guitare et
   piano (les voix, souvent rubato, brouillent la pulsation).
2. **Enveloppe d'attaques** : force des attaques (coups de batterie, notes)
   environ 86 fois par seconde. Les graves et médiums (grosse caisse, caisse
   claire, basse, accords) pèsent plus que les aigus : le charleston joue
   souvent en croches et ferait sinon prendre le double du tempo.
3. **Tempo local** toutes les 0,5 s sur une fenêtre de 8 s. Deux mesures de
   périodicité sont combinées car elles se trompent dans des sens opposés :
   l'autocorrélation favorise la moitié du tempo, la transformée de Fourier le
   double. Leur produit désigne le vrai tempo. Un algorithme de Viterbi choisit
   ensuite la suite de tempos la plus continue, ce qui évite de sauter du
   simple au double d'un instant à l'autre.
4. **Battements** : suivi des temps guidé par ce tempo local, position de
   chaque temps affinée sous la trame, puis BPM mesuré entre chaque temps et
   moyenné sur quelques temps.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_io

log = logging.getLogger(__name__)

PISTES_RYTHMIQUES = ("batterie", "basse", "guitare", "piano")
SR_ANALYSE = 22050
PAS_TRAME = 256  # ≈ 11,6 ms entre deux mesures de l'enveloppe d'attaques
PAS_TEMPO_S = 0.5  # pas de l'estimation du tempo local
# Poids des bandes de fréquences dans l'enveloppe d'attaques : (fréquence haute, poids).
BANDES = ((200.0, 1.0), (3000.0, 1.0), (math.inf, 0.35))


@dataclass(frozen=True)
class ParametresTempo:
    bpm_min: float = 50.0
    bpm_max: float = 200.0
    bpm_probable: float = 110.0
    """Tempo le plus vraisemblable a priori ; départage les hésitations restantes
    entre un tempo et son double."""
    fenetre_s: float = 8.0
    """Durée sur laquelle chaque tempo local est estimé."""
    lissage_temps: int = 8
    """Nombre de temps de la moyenne glissante (8 = deux mesures à 4 temps)."""
    tolerance_pct: float = 3.0
    """Écart au tempo médian considéré comme « stable »."""
    clarte_min: float = 0.25
    """Régularité minimale de la pulsation (0 à 1) pour afficher un tempo. Ordres de
    grandeur : bruit ≈ 0, notes jouées librement ≈ 0,2, batterie ≈ 0,5 à 0,9."""


@dataclass
class AnalyseTempo:
    duree: float
    """Durée du morceau en secondes."""
    battements: np.ndarray
    """Instants des temps détectés (s)."""
    temps: np.ndarray
    """Milieu de chaque intervalle entre deux temps (s)."""
    bpm_brut: np.ndarray
    """Tempo mesuré entre deux temps consécutifs (NaN si peu fiable ou silence)."""
    bpm_lisse: np.ndarray
    """Médiane glissante de ``bpm_brut`` (NaN là où aucune pulsation n'est jouée)."""
    sources: tuple[str, ...] = ()
    """Pistes utilisées pour l'analyse."""
    tolerance_pct: float = 3.0
    lissage_temps: int = 8

    @property
    def _valeurs(self) -> np.ndarray:
        return self.bpm_lisse[np.isfinite(self.bpm_lisse)]

    @property
    def detecte(self) -> bool:
        """Vrai si une pulsation régulière a été trouvée."""
        return self._valeurs.size >= 8

    @property
    def mediane(self) -> float:
        return float(np.median(self._valeurs)) if self.detecte else math.nan

    @property
    def minimum(self) -> float:
        return float(self._valeurs.min()) if self.detecte else math.nan

    @property
    def maximum(self) -> float:
        return float(self._valeurs.max()) if self.detecte else math.nan

    @property
    def part_stable(self) -> float:
        """Proportion des temps à moins de ``tolerance_pct`` % du tempo médian."""
        if not self.detecte:
            return math.nan
        return float(np.mean(np.abs(self._valeurs / self.mediane - 1) <= self.tolerance_pct / 100))

    @property
    def debut_fin(self) -> tuple[float, float]:
        """Tempo médian du début et de la fin du morceau (15 % des temps, 8 au moins)."""
        if not self.detecte:
            return math.nan, math.nan
        v = self._valeurs
        n = max(8, int(v.size * 0.15))
        return float(np.median(v[:n])), float(np.median(v[-n:]))

    def resume(self) -> str:
        if not self.detecte:
            return "aucune pulsation régulière détectée"
        debut, fin = self.debut_fin
        return (
            f"{self.mediane:.0f} BPM (de {self.minimum:.0f} à {self.maximum:.0f}), "
            f"{self.part_stable:.0%} du morceau à ±{self.tolerance_pct:g} %, "
            f"début {debut:.0f} → fin {fin:.0f} BPM"
        )


def signal_rythmique(
    pistes: dict[str, Path], zone: tuple[float, float] | None = None
) -> tuple[np.ndarray, int, tuple[str, ...]]:
    """Additionne les pistes rythmiques en mono (sur ``zone`` seulement si fournie).
    Renvoie (signal, fréquence, pistes jouées)."""
    noms = [n for n in PISTES_RYTHMIQUES if n in pistes]
    if not noms:
        noms = [n for n in pistes if n != "voix"] or list(pistes)
    total: np.ndarray | None = None
    sr = None
    niveaux: dict[str, float] = {}
    for nom in noms:
        audio, sr_piste = audio_io.lire(pistes[nom], *(zone or ()))
        if sr is None:
            sr = sr_piste
        elif sr_piste != sr:
            raise ValueError(f"fréquences d'échantillonnage différentes entre les pistes ({sr_piste} / {sr})")
        mono = audio.mean(axis=1)
        niveaux[nom] = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
        if total is None:
            total = mono
        else:
            n = max(total.size, mono.size)
            total = np.pad(total, (0, n - total.size))
            total[: mono.size] += mono
    plus_fort = max(niveaux.values(), default=0.0)
    # Une piste plus de 30 dB sous la plus forte ne joue pas vraiment : elle n'est pas citée.
    jouees = tuple(n for n in noms if plus_fort > 0 and niveaux[n] >= plus_fort * 10 ** (-30 / 20))
    return total if total is not None else np.zeros(0, np.float32), sr or SR_ANALYSE, jouees


def enveloppe_attaques(y: np.ndarray) -> np.ndarray:
    """Force des attaques par trame, bandes graves et médiums privilégiées (voir ``BANDES``)."""
    import librosa

    nb_mels = 128
    frequences = librosa.mel_frequencies(n_mels=nb_mels, fmax=SR_ANALYSE / 2)
    limites = [0] + [int(np.searchsorted(frequences, f)) for f, _ in BANDES[:-1]] + [nb_mels]
    par_bande = librosa.onset.onset_strength_multi(
        y=y, sr=SR_ANALYSE, hop_length=PAS_TRAME, n_mels=nb_mels, channels=limites
    )
    poids = np.array([p for _, p in BANDES])[:, None]
    return (par_bande * poids).sum(axis=0) / poids.sum()


def _affiner(enveloppe: np.ndarray, trames: np.ndarray) -> np.ndarray:
    """Position des pics de l'enveloppe à une fraction de trame près (interpolation parabolique)."""
    i = np.clip(trames, 1, enveloppe.size - 2)
    a, b, c = enveloppe[i - 1], enveloppe[i], enveloppe[i + 1]
    courbure = a - 2 * b + c
    decalage = np.where(courbure < 0, 0.5 * (a - c) / np.where(courbure < 0, courbure, -1.0), 0.0)
    return i + np.clip(decalage, -0.5, 0.5)


def _viterbi(log_obs: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Chemin d'états le plus probable. ``log_obs`` : (états, pas) ; ``log_transition`` : (de, vers)."""
    nb_etats, nb_pas = log_obs.shape
    score = log_obs[:, 0].copy()
    retour = np.zeros((nb_pas, nb_etats), dtype=np.int32)
    colonnes = np.arange(nb_etats)
    for t in range(1, nb_pas):
        candidats = score[:, None] + log_transition
        retour[t] = np.argmax(candidats, axis=0)
        score = candidats[retour[t], colonnes] + log_obs[:, t]
    chemin = np.empty(nb_pas, dtype=np.int64)
    chemin[-1] = int(np.argmax(score))
    for t in range(nb_pas - 1, 0, -1):
        chemin[t - 1] = retour[t, chemin[t]]
    return chemin


def _sur_grille(valeurs: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpole les lignes de ``valeurs`` (lignes, trames) aux positions fractionnaires données."""
    i0 = np.clip(np.floor(positions).astype(np.int64), 0, valeurs.shape[0] - 2)
    frac = np.clip(positions - i0, 0.0, 1.0)[:, None]
    return valeurs[i0] * (1.0 - frac) + valeurs[i0 + 1] * frac


def _par_pas(valeurs: np.ndarray, taille: int) -> np.ndarray:
    """Moyenne des colonnes par blocs de ``taille`` trames."""
    nb_lignes, n = valeurs.shape
    nb_pas = math.ceil(n / taille)
    complet = np.pad(valeurs, ((0, 0), (0, nb_pas * taille - n)), mode="edge")
    return complet.reshape(nb_lignes, nb_pas, taille).mean(axis=2)


def tempo_local(enveloppe: np.ndarray, fps: float, params: ParametresTempo) -> tuple[np.ndarray, np.ndarray]:
    """Tempo local (BPM) et clarté de la pulsation (0 à 1) pour chaque trame de l'enveloppe."""
    import librosa

    n = enveloppe.size
    fenetre = min(int(round(params.fenetre_s * fps)), n - 1)
    nb_bins = int(math.ceil(math.log(params.bpm_max / params.bpm_min) / math.log(1.005))) + 1
    grille = np.geomspace(params.bpm_min, params.bpm_max, nb_bins)  # 0,5 % par case

    # Autocorrélation : pics au tempo et à ses sous-multiples (moitié, tiers...).
    # La moyenne locale est retirée, sinon un signal sans rythme (bruit, nappe)
    # paraîtrait corrélé à tous les décalages.
    une_seconde = max(1, int(round(fps)))
    centree = enveloppe - np.convolve(enveloppe, np.ones(une_seconde) / une_seconde, mode="same")
    acf_brute = _sur_grille(
        librosa.feature.tempogram(onset_envelope=centree, sr=SR_ANALYSE, hop_length=PAS_TRAME, win_length=fenetre),
        60.0 * fps / grille,
    )
    acf = np.clip(acf_brute, 0.0, None)

    # Fourier : pics au tempo et à ses multiples (double, triple...).
    dft = np.abs(
        librosa.feature.fourier_tempogram(
            onset_envelope=enveloppe, sr=SR_ANALYSE, hop_length=PAS_TRAME, win_length=fenetre
        )
    )[:, :n]
    resolution_bpm = 60.0 * fps / fenetre
    dft = _sur_grille(dft, grille / resolution_bpm)

    pas = max(1, int(round(PAS_TEMPO_S * fps)))
    acf_brute, acf, dft = _par_pas(acf_brute, pas), _par_pas(acf, pas), _par_pas(dft, pas)
    acf /= np.maximum(acf.max(axis=0, keepdims=True), 1e-9)
    dft /= np.maximum(dft.max(axis=0, keepdims=True), 1e-9)
    probable = min(max(params.bpm_probable, params.bpm_min), params.bpm_max)
    a_priori = -0.5 * np.log2(grille / probable) ** 2  # log-normale, écart-type 1 octave
    log_obs = np.log(acf * dft + 1e-3) + a_priori[:, None]

    # Transitions : variations lentes probables (~1,5 % par demi-seconde),
    # changement brusque de tempo possible mais rare.
    ecart = np.log(grille)[:, None] - np.log(grille)[None, :]
    continuite = np.exp(-0.5 * (ecart / 0.015) ** 2)
    continuite /= continuite.sum(axis=1, keepdims=True)
    log_transition = np.log(0.999 * continuite + 0.001 / nb_bins)

    chemin = _viterbi(log_obs, log_transition)
    # Clarté : autocorrélation (normalisée, 1 = parfaitement périodique) au tempo retenu,
    # en prenant le meilleur pic à ±4 % pour ne pas pénaliser un tempo retenu un peu décalé.
    voisins = np.clip(chemin[:, None] + np.arange(-8, 9)[None, :], 0, nb_bins - 1)
    clarte = acf_brute[voisins, np.arange(chemin.size)[:, None]].max(axis=1)
    centres = (np.arange(chemin.size) + 0.5) * pas
    trames = np.arange(n)
    return np.interp(trames, centres, grille[chemin]), np.interp(trames, centres, clarte)


def _moyenne_glissante(valeurs: np.ndarray, largeur: int, minimum_valides: float = 0.6) -> np.ndarray:
    """Moyenne sur ``largeur`` + 1 valeurs centrées, en ignorant les NaN."""
    demi = max(1, largeur // 2)
    complet = np.pad(valeurs, demi, constant_values=np.nan)
    fenetres = np.lib.stride_tricks.sliding_window_view(complet, 2 * demi + 1)
    assez = np.isfinite(fenetres).sum(axis=1) >= minimum_valides * (2 * demi + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # fenêtres entièrement vides
        moyenne = np.nanmean(fenetres, axis=1)
    return np.where(assez, moyenne, np.nan)


def analyser(
    signal: np.ndarray, sr: int, params: ParametresTempo = ParametresTempo(), sources: tuple[str, ...] = ()
) -> AnalyseTempo:
    """Analyse le tempo d'un signal mono."""
    import librosa

    duree = signal.shape[0] / sr if sr else 0.0
    vide = np.zeros(0)
    resultat_vide = AnalyseTempo(duree, vide, vide, vide, vide, sources, params.tolerance_pct, params.lissage_temps)
    y = np.ascontiguousarray(signal, dtype=np.float32)
    if y.size < 4 * sr or not np.any(y):
        return resultat_vide
    if sr != SR_ANALYSE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_ANALYSE)
    fps = SR_ANALYSE / PAS_TRAME

    enveloppe = enveloppe_attaques(y)
    n = enveloppe.size

    # Passages joués : niveau (lissé sur 1 s) à moins de 30 dB des passages forts.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=PAS_TRAME)[0]
    rms = np.pad(rms, (0, max(0, n - rms.size)))[:n]
    une_seconde = max(1, int(round(fps)))
    niveau_db = 20 * np.log10(np.convolve(rms, np.ones(une_seconde) / une_seconde, mode="same") + 1e-10)
    actif = niveau_db > max(float(np.percentile(niveau_db, 95)) - 30.0, -70.0)
    if actif.sum() < 4 * fps:
        return resultat_vide

    guide, clarte = tempo_local(enveloppe, fps, params)
    actif &= clarte >= params.clarte_min  # pas de pulsation régulière : pas de tempo affiché
    _, trames_battements = librosa.beat.beat_track(
        onset_envelope=enveloppe, sr=SR_ANALYSE, hop_length=PAS_TRAME, bpm=guide, units="frames", trim=True
    )
    trames_battements = np.asarray(trames_battements, dtype=np.int64)
    if trames_battements.size < 3:
        return resultat_vide
    battements = _affiner(enveloppe, trames_battements) / fps

    intervalles = np.diff(battements)
    temps = battements[:-1] + intervalles / 2
    trames = np.clip(np.round(temps * fps).astype(np.int64), 0, n - 1)
    # Un intervalle très éloigné du tempo local est un temps manqué ou en trop.
    coherent = np.abs(np.log(60.0 / intervalles / guide[trames])) < math.log(1.15)
    intervalles = np.where(coherent & actif[trames], intervalles, np.nan)
    brut = 60.0 / intervalles
    # Moyenne des durées des temps (et non des BPM) : c'est le tempo réellement joué.
    lisse = np.where(actif[trames], 60.0 / _moyenne_glissante(intervalles, params.lissage_temps), np.nan)
    return AnalyseTempo(
        duree, battements, temps, brut, lisse, sources, params.tolerance_pct, params.lissage_temps
    )


def analyser_pistes(
    pistes: dict[str, Path], params: ParametresTempo = ParametresTempo(), zone: tuple[float, float] | None = None
) -> AnalyseTempo:
    signal, sr, jouees = signal_rythmique(pistes, zone)
    return analyser(signal, sr, params, jouees)
