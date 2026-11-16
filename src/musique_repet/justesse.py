"""Justesse du chant par rapport à la musique.

1. **Accordage du groupe** : estimé sur les pistes piano, guitare et basse. La
   voix est jugée par rapport aux instruments, pas par rapport au La 440
   théorique : un groupe accordé à La = 437 Hz n'a pas tort, et les
   chanteuses doivent chanter avec lui.
2. **Hauteur chantée** : suivie sur la piste voix séparée, pour une voix seule
   ou deux voix ensemble (harmonies, voir ``hauteur.py``), puis découpée en
   notes. Chaque note est mesurée sur son centre : l'attaque glissée et le
   vibrato ne sont pas comptés comme des erreurs.
3. **Écart** de chaque note à la note juste la plus proche, en cents
   (100 cents = un demi-ton). Quand une note tombe entre deux, la tonalité
   départage la note visée (Mi chanté très bas plutôt que Mi bémol trop haut).
4. **Fausse note** : note tenue hors de la tonalité et absente de ce que jouent
   les instruments au même moment, même chantée bien droite.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from . import audio_io
from .hauteur import SR as SR_VOIX
from .hauteur import suivre_voix

log = logging.getLogger(__name__)

PISTES_HARMONIQUES = ("piano", "guitare", "basse")
SR_HARMONIE = 22050
PAS_HARMONIE = 512

NOMS_NOTES = ("Do", "Do♯", "Ré", "Mi♭", "Mi", "Fa", "Fa♯", "Sol", "La♭", "La", "Si♭", "Si")
# Profils de tonalité de Krumhansl et Kessler (poids de chaque degré dans une tonalité).
PROFIL_MAJEUR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PROFIL_MINEUR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
GAMME_MAJEURE = (0, 2, 4, 5, 7, 9, 11)
GAMME_MINEURE = (0, 2, 3, 5, 7, 8, 10, 11)  # naturelle, plus la sensible de la mineure harmonique


@dataclass(frozen=True)
class ParametresJustesse:
    tolerance_cents: float = 20.0
    """Écart jusqu'auquel une note est juste."""
    faux_cents: float = 40.0
    """Écart au-delà duquel une note est fausse (entre les deux : approximative)."""
    duree_note_min_s: float = 0.15
    """Les notes plus courtes (ornements, passages) ne sont pas évaluées."""
    fmin: float = 110.0
    """Note la plus grave suivie (La 2)."""
    fmax: float = 1100.0
    """Note la plus aiguë suivie (Do♯ 6)."""


@dataclass(frozen=True)
class Note:
    debut: float
    fin: float
    hauteur: float
    """Hauteur chantée en demi-tons (numérotation MIDI), dans l'accordage du groupe."""
    visee: int
    """Note juste visée (MIDI)."""
    ecart: float
    """Écart en cents : positif = trop haut, négatif = trop bas."""
    hors_harmonie: bool = False

    @property
    def duree(self) -> float:
        return self.fin - self.debut

    @property
    def nom(self) -> str:
        return f"{NOMS_NOTES[self.visee % 12]}{self.visee // 12 - 1}"

    def categorie(self, tolerance: float, faux: float) -> str:
        if self.hors_harmonie or abs(self.ecart) > faux:
            return "faux"
        return "approximatif" if abs(self.ecart) > tolerance else "juste"


@dataclass
class AnalyseJustesse:
    duree: float
    notes: list[Note]
    accordage_cents: float
    """Accordage de référence par rapport à La 440, en cents."""
    reference: str
    """« instruments » ou « voix » (aucun instrument mélodique : la voix sert de référence)."""
    tonalite: str | None
    tolerance_cents: float = 20.0
    faux_cents: float = 40.0

    @property
    def detecte(self) -> bool:
        return len(self.notes) >= 3

    @property
    def la_hz(self) -> float:
        return 440.0 * 2 ** (self.accordage_cents / 1200)

    def categorie(self, note: Note) -> str:
        return note.categorie(self.tolerance_cents, self.faux_cents)

    def _part(self, notes: list[Note], categorie: str) -> float:
        total = sum(n.duree for n in notes)
        return sum(n.duree for n in notes if self.categorie(n) == categorie) / total if total else math.nan

    @property
    def part_juste(self) -> float:
        """Part du temps chanté sur des notes justes."""
        return self._part(self.notes, "juste")

    @property
    def ecart_moyen(self) -> float:
        """Tendance générale (cents, pondérée par la durée) : négatif = chante bas."""
        retenues = [n for n in self.notes if not n.hors_harmonie]
        total = sum(n.duree for n in retenues)
        return sum(n.ecart * n.duree for n in retenues) / total if total else math.nan

    @property
    def fausses(self) -> list[Note]:
        return [n for n in self.notes if self.categorie(n) == "faux"]

    def tranches(self, duree_s: float = 8.0, chant_min_s: float = 1.0) -> list[tuple[float, float, float]]:
        """Part juste du chant par tranche de ``duree_s`` secondes : (début, fin, part juste).

        Les tranches où l'on chante moins de ``chant_min_s`` secondes sont omises.
        """
        resultat = []
        debut = 0.0
        while debut < self.duree:
            fin = min(debut + duree_s, self.duree)
            total = juste = 0.0
            for note in self.notes:
                recouvrement = min(note.fin, fin) - max(note.debut, debut)
                if recouvrement > 0:
                    total += recouvrement
                    if self.categorie(note) == "juste":
                        juste += recouvrement
            if total >= chant_min_s:
                resultat.append((debut, fin, juste / total))
            debut = fin
        return resultat

    def resume(self) -> str:
        if not self.detecte:
            return "pas de chant détecté"
        return (
            f"{self.part_juste:.0%} du chant juste, tendance {self.ecart_moyen:+.0f} cents, "
            f"{len(self.fausses)} note(s) fausse(s) dont {sum(n.hors_harmonie for n in self.notes)} hors de l'harmonie"
        )


def _lire_mono(chemins: list[Path], zone: tuple[float, float] | None, sr_cible: int) -> np.ndarray:
    import librosa

    total = None
    sr = None
    for chemin in chemins:
        audio, sr = audio_io.lire(chemin, *(zone or ()))
        mono = audio.mean(axis=1)
        if total is None:
            total = mono
        else:
            n = max(total.size, mono.size)
            total = np.pad(total, (0, n - total.size))
            total[: mono.size] += mono
    if total is None:
        return np.zeros(0, np.float32)
    return librosa.resample(total, orig_sr=sr, target_sr=sr_cible) if sr != sr_cible else total


def _plages(masque: np.ndarray) -> list[tuple[int, int]]:
    bords = np.diff(np.concatenate([[0], masque.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(bords == 1).tolist(), np.flatnonzero(bords == -1).tolist()))


def tonalite(chroma_moyen: np.ndarray) -> tuple[int, str]:
    """Tonique (0 = Do) et mode (« majeur » / « mineur ») les plus probables (Krumhansl-Kessler)."""
    meilleur = (-2.0, 0, "majeur")
    for tonique in range(12):
        for mode, profil in (("majeur", PROFIL_MAJEUR), ("mineur", PROFIL_MINEUR)):
            correlation = float(np.corrcoef(chroma_moyen, np.roll(profil, tonique))[0, 1])
            if correlation > meilleur[0]:
                meilleur = (correlation, tonique, mode)
    return meilleur[1], meilleur[2]


def combler_trous(hauteurs: np.ndarray, trames_max: int) -> np.ndarray:
    """Relie les courtes interruptions d'un contour (voix un instant non détectée) quand la
    hauteur est la même de part et d'autre (moins d'un demi-ton d'écart)."""
    resultat = hauteurs.copy()
    valide = np.isfinite(hauteurs)
    trous = _plages(~valide)
    for a, b in trous:
        if 0 < a and b < hauteurs.size and b - a <= trames_max and abs(hauteurs[b] - hauteurs[a - 1]) < 0.5:
            resultat[a:b] = np.linspace(hauteurs[a - 1], hauteurs[b], b - a + 2)[1:-1]
    return resultat


def decouper_notes(hauteurs: np.ndarray, fps: float, duree_min_s: float) -> list[tuple[int, int]]:
    """Découpe un contour de hauteur (demi-tons, NaN = pas de voix) en notes [début, fin[ (trames).

    Le contour est lissé sur 150 ms (le vibrato ne crée pas de nouvelles notes) ;
    une nouvelle note commence quand la hauteur s'écarte durablement (30 ms)
    de plus de 0,6 demi-ton de la note en cours.
    """
    valide = np.isfinite(hauteurs)
    lisse = hauteurs.copy()
    taille = max(3, int(round(0.15 * fps)) | 1)
    for a, b in _plages(valide):
        if b - a >= 3:
            lisse[a:b] = median_filter(hauteurs[a:b], size=min(taille, (b - a) | 1), mode="nearest")
    confirmation = max(1, int(round(0.03 * fps)))
    notes = []
    for a, b in _plages(valide):
        debut, somme, nombre = a, float(lisse[a]), 1
        i = a + 1
        while i < b:
            reference = somme / nombre
            fin_test = min(b, i + confirmation)
            if abs(lisse[i] - reference) > 0.6 and np.all(np.abs(lisse[i:fin_test] - reference) > 0.6):
                notes.append((debut, i))
                debut, somme, nombre = i, float(lisse[i]), 1
            else:
                somme += float(lisse[i])
                nombre += 1
            i += 1
        notes.append((debut, b))
    minimum = int(round(duree_min_s * fps))
    return [(a, b) for a, b in notes if b - a >= minimum]


def _hauteur_centrale(hauteurs: np.ndarray, fps: float) -> float | None:
    """Hauteur d'une note sur son centre : sans l'attaque (glissé) ni la fin, moyenne
    des valeurs proches de la médiane (le vibrato se compense)."""
    n = hauteurs.size
    debut = max(int(round(0.08 * fps)), int(0.15 * n))
    fin = n - max(int(round(0.05 * fps)), int(0.1 * n))
    centre = hauteurs[debut:fin]
    if centre.size < 5:
        return None
    mediane = float(np.median(centre))
    proches = centre[np.abs(centre - mediane) <= 0.5]
    return float(proches.mean()) if proches.size else mediane


def analyser_justesse(
    pistes: dict[str, Path], params: ParametresJustesse = ParametresJustesse(), zone: tuple[float, float] | None = None
) -> AnalyseJustesse:
    """Analyse la justesse de la piste « voix » par rapport aux instruments."""
    import librosa

    voix = _lire_mono([pistes["voix"]], zone, SR_VOIX) if "voix" in pistes else np.zeros(0, np.float32)
    duree = voix.size / SR_VOIX
    vide = AnalyseJustesse(duree, [], 0.0, "instruments", None, params.tolerance_cents, params.faux_cents)
    if voix.size < SR_VOIX or not np.any(voix):
        return vide

    # Référence : accordage et harmonie des instruments mélodiques.
    instruments = _lire_mono([pistes[n] for n in PISTES_HARMONIQUES if n in pistes], zone, SR_HARMONIE)
    niveau_voix = float(np.sqrt(np.mean(voix.astype(np.float64) ** 2)))
    niveau_instr = float(np.sqrt(np.mean(instruments.astype(np.float64) ** 2))) if instruments.size else 0.0
    avec_instruments = niveau_instr > max(niveau_voix * 10 ** (-40 / 20), 1e-5)
    chroma = None
    nom_tonalite = None
    gamme: set[int] | None = None
    accordage = 0.0
    if avec_instruments:
        accordage = 100.0 * float(librosa.estimate_tuning(y=instruments, sr=SR_HARMONIE, n_fft=4096))
        with warnings.catch_warnings():
            # Extrait très court : librosa prévient que ses fenêtres graves dépassent le signal.
            warnings.simplefilter("ignore", UserWarning)
            chroma = librosa.feature.chroma_cqt(
                y=instruments, sr=SR_HARMONIE, hop_length=PAS_HARMONIE, tuning=accordage / 100, norm=None
            )
        tonique, mode = tonalite(chroma.sum(axis=1))
        gamme = {(tonique + d) % 12 for d in (GAMME_MAJEURE if mode == "majeur" else GAMME_MINEURE)}
        nom_tonalite = f"{NOMS_NOTES[tonique]} {mode}"
        chroma = chroma / np.maximum(chroma.max(axis=0, keepdims=True), 1e-9)

    # Hauteur chantée (une ou deux voix), seulement là où la voix est bien présente,
    # découpée en notes mesurées sur leur centre (en demi-tons, par rapport à La 440).
    contours = suivre_voix(voix, params.fmin, params.fmax)
    fps = contours.fps
    mesures: list[tuple[float, float, float]] = []  # (début, fin, hauteur centrale)
    for contour in (contours.haute, contours.basse):
        hauteurs = combler_trous(contour, int(round(0.065 * fps)))
        for a, b in decouper_notes(hauteurs, fps, params.duree_note_min_s):
            centrale = _hauteur_centrale(hauteurs[a:b], fps)
            if centrale is not None:
                mesures.append((a / fps, b / fps, centrale))
    mesures.sort()
    if not avec_instruments and mesures:
        # A cappella : la référence est l'accordage moyen des notes chantées (moyenne
        # circulaire de leur écart au demi-ton, pondérée par la durée).
        ecarts = np.array([h - round(h) for _, _, h in mesures])
        durees = np.array([f - d for d, f, _ in mesures])
        accordage = 100.0 * float(np.angle(np.sum(durees * np.exp(2j * np.pi * ecarts)))) / (2 * np.pi)

    notes: list[Note] = []
    for debut, fin, centrale in mesures:
        hauteur = centrale - accordage / 100
        visee = int(round(hauteur))
        ecart = 100.0 * (hauteur - visee)
        # Entre deux notes : la tonalité départage la note visée.
        if gamme is not None and abs(ecart) > 35 and visee % 12 not in gamme:
            voisine = visee + (1 if ecart > 0 else -1)
            if voisine % 12 in gamme:
                visee, ecart = voisine, 100.0 * (hauteur - voisine)
        hors_harmonie = False
        if gamme is not None and chroma is not None and fin - debut >= 0.25 and visee % 12 not in gamme:
            i0 = int(debut * SR_HARMONIE / PAS_HARMONIE)
            i1 = max(i0 + 1, int(fin * SR_HARMONIE / PAS_HARMONIE))
            local = chroma[:, i0:i1].mean(axis=1) if i0 < chroma.shape[1] else np.zeros(12)
            hors_harmonie = bool(local[visee % 12] < 0.3 * max(float(local.max()), 1e-9))
        notes.append(Note(debut, fin, hauteur, visee, ecart, hors_harmonie))

    return AnalyseJustesse(
        duree,
        notes,
        accordage,
        "instruments" if avec_instruments else "voix",
        nom_tonalite,
        params.tolerance_cents,
        params.faux_cents,
    )
