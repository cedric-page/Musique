"""Justesse du chant par rapport aux accords joués par le groupe.

1. **Accords en cours** : reconnus sur les pistes piano, guitare et basse (voir
   ``accords.py``), avec l'accordage des instruments pendant chaque accord. Le
   chant est comparé à ce que joue le groupe à ce moment-là, pas au La 440
   théorique ni à la tonalité du morceau.
2. **Hauteur chantée** : suivie sur la piste voix séparée, pour une voix seule
   ou deux voix ensemble (voir ``hauteur.py``), puis répartie entre la **voix
   lead** et la voix d'**harmonie**, analysées séparément. Chaque voix est
   découpée en notes, mesurées sur leur centre : l'attaque glissée et le
   vibrato ne sont pas comptés comme des erreurs.
3. **Écart** de chaque note à la note juste visée, en cents (100 cents = un
   demi-ton). Quand une note tombe entre deux, la note de l'accord en cours
   est retenue comme note visée (Mi chanté très bas sur un accord de La m,
   plutôt que Mi bémol trop haut).
4. **Note qui frotte avec l'accord** : note tenue qui n'est pas dans l'accord
   et se trouve à un demi-ton d'une de ses notes (Fa sur un accord de Do, qui
   frotte avec le Mi), même chantée bien droite.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from . import audio_io
from .accords import NOMS_NOTES, Accord, accord_pendant, reconnaitre_accords
from .accords import SR as SR_HARMONIE
from .hauteur import SR as SR_VOIX
from .hauteur import attribuer_voix, suivre_voix

log = logging.getLogger(__name__)

PISTES_HARMONIQUES = ("piano", "guitare", "basse")
DUREE_DISSONANCE_S = 0.45
"""Durée à partir de laquelle une note hors de l'accord qui frotte s'entend comme une fausse note."""


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
    """Hauteur chantée en demi-tons (numérotation MIDI), dans l'accordage des instruments."""
    visee: int
    """Note juste visée (MIDI)."""
    ecart: float
    """Écart en cents : positif = trop haut, négatif = trop bas."""
    dissonante: bool = False
    """Note tenue hors de l'accord, à un demi-ton d'une de ses notes : elle frotte."""
    accord: str = ""
    """Nom de l'accord joué pendant la note (vide s'il n'y en a pas)."""
    degre: str | None = None
    """Place de la note dans l'accord (« tierce »...), None si elle n'en fait pas partie."""

    @property
    def duree(self) -> float:
        return self.fin - self.debut

    @property
    def nom(self) -> str:
        return f"{NOMS_NOTES[self.visee % 12]}{self.visee // 12 - 1}"

    def categorie(self, tolerance: float, faux: float) -> str:
        if self.dissonante or abs(self.ecart) > faux:
            return "faux"
        return "approximatif" if abs(self.ecart) > tolerance else "juste"


@dataclass
class AnalyseJustesse:
    duree: float
    notes: list[Note]
    accordage_cents: float
    """Accordage général des instruments (ou des voix, a cappella) par rapport à La 440, en cents."""
    reference: str
    """« instruments » ou « voix » (aucun instrument mélodique : la voix sert de référence)."""
    accords: list[Accord] = field(default_factory=list)
    """Accords joués pendant le morceau."""
    tolerance_cents: float = 20.0
    faux_cents: float = 40.0
    voix: str = "lead"
    """« lead » (la mélodie) ou « harmonie » (la seconde voix)."""

    @property
    def detecte(self) -> bool:
        return len(self.notes) >= 3

    @property
    def duree_chantee(self) -> float:
        return sum(n.duree for n in self.notes)

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
        retenues = [n for n in self.notes if not n.dissonante]
        total = sum(n.duree for n in retenues)
        return sum(n.ecart * n.duree for n in retenues) / total if total else math.nan

    @property
    def fausses(self) -> list[Note]:
        return [n for n in self.notes if self.categorie(n) == "faux"]

    @property
    def dissonantes(self) -> list[Note]:
        return [n for n in self.notes if n.dissonante]

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
            return "pas de chant détecté" if self.voix == "lead" else "pas d'harmonie détectée"
        return (
            f"{self.part_juste:.0%} du chant juste, tendance {self.ecart_moyen:+.0f} cents, "
            f"{len(self.fausses)} note(s) fausse(s) dont {len(self.dissonantes)} qui frotte(nt) avec l'accord"
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


@dataclass
class JustesseDesVoix:
    lead: AnalyseJustesse
    harmonie: AnalyseJustesse


def _mesurer_notes(contour: np.ndarray, fps: float, params: ParametresJustesse) -> list[tuple[float, float, float]]:
    """Notes d'un contour : (début, fin, hauteur centrale en demi-tons par rapport à La 440)."""
    hauteurs = combler_trous(contour, int(round(0.065 * fps)))
    mesures = []
    for a, b in decouper_notes(hauteurs, fps, params.duree_note_min_s):
        centrale = _hauteur_centrale(hauteurs[a:b], fps)
        if centrale is not None:
            mesures.append((a / fps, b / fps, centrale))
    return mesures


def evaluer_note(debut: float, fin: float, centrale: float, accordage: float, accords: list[Accord]) -> Note:
    """Compare une note chantée (hauteur centrale, en demi-tons par rapport à La 440) à l'accord en cours."""
    accord = accord_pendant(accords, debut, fin)
    reference = accord.accordage_cents if accord is not None else accordage
    hauteur = centrale - reference / 100
    visee = int(round(hauteur))
    ecart = 100.0 * (hauteur - visee)
    if accord is None:
        return Note(debut, fin, hauteur, visee, ecart)
    notes_accord = accord.notes
    if visee % 12 not in notes_accord and abs(ecart) > 35:
        # Entre deux notes : la note de l'accord est la note visée, chantée très faux.
        voisine = visee + (1 if ecart > 0 else -1)
        if voisine % 12 in notes_accord:
            visee, ecart = voisine, 100.0 * (hauteur - voisine)
    hors_accord = visee % 12 not in notes_accord
    frotte = hors_accord and any((visee - n) % 12 in (1, 11) for n in notes_accord)
    return Note(
        debut, fin, hauteur, visee, ecart,
        dissonante=frotte and fin - debut >= DUREE_DISSONANCE_S,
        accord=accord.nom,
        degre=accord.degre(visee),
    )


def analyser_justesse(
    pistes: dict[str, Path], params: ParametresJustesse = ParametresJustesse(), zone: tuple[float, float] | None = None
) -> JustesseDesVoix:
    """Analyse la justesse de la voix lead et de la voix d'harmonie par rapport aux accords joués."""
    import librosa

    voix = _lire_mono([pistes["voix"]], zone, SR_VOIX) if "voix" in pistes else np.zeros(0, np.float32)
    duree = voix.size / SR_VOIX

    def analyse(nom: str, notes: list[Note], accordage: float, reference: str, accords: list[Accord]):
        return AnalyseJustesse(
            duree, notes, accordage, reference, accords, params.tolerance_cents, params.faux_cents, nom
        )

    if voix.size < SR_VOIX or not np.any(voix):
        return JustesseDesVoix(
            analyse("lead", [], 0.0, "instruments", []), analyse("harmonie", [], 0.0, "instruments", [])
        )

    # Référence : accords joués par les instruments mélodiques, et leur accordage.
    instruments = _lire_mono([pistes[n] for n in PISTES_HARMONIQUES if n in pistes], zone, SR_HARMONIE)
    niveau_voix = float(np.sqrt(np.mean(voix.astype(np.float64) ** 2)))
    niveau_instr = float(np.sqrt(np.mean(instruments.astype(np.float64) ** 2))) if instruments.size else 0.0
    avec_instruments = niveau_instr > max(niveau_voix * 10 ** (-40 / 20), 1e-5)
    accordage = 0.0
    accords: list[Accord] = []
    if avec_instruments:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            accordage = 100.0 * float(librosa.estimate_tuning(y=instruments, sr=SR_HARMONIE, n_fft=4096))
        accords = reconnaitre_accords(instruments, accordage)

    # Hauteur chantée (une ou deux voix), seulement là où la voix est bien présente,
    # répartie entre lead et harmonie, puis découpée en notes mesurées sur leur centre.
    contours = suivre_voix(voix, params.fmin, params.fmax)
    lead, harmonie = attribuer_voix(contours)
    mesures = {
        "lead": _mesurer_notes(lead, contours.fps, params),
        "harmonie": _mesurer_notes(harmonie, contours.fps, params),
    }
    toutes = mesures["lead"] + mesures["harmonie"]
    if not avec_instruments and toutes:
        # A cappella : la référence est l'accordage moyen des notes chantées (moyenne
        # circulaire de leur écart au demi-ton, pondérée par la durée).
        ecarts = np.array([h - round(h) for _, _, h in toutes])
        durees = np.array([f - d for d, f, _ in toutes])
        accordage = 100.0 * float(np.angle(np.sum(durees * np.exp(2j * np.pi * ecarts)))) / (2 * np.pi)

    reference = "instruments" if avec_instruments else "voix"
    return JustesseDesVoix(
        *(
            analyse(
                nom,
                [evaluer_note(d, f, h, accordage, accords) for d, f, h in mesures[nom]],
                accordage,
                reference,
                accords,
            )
            for nom in ("lead", "harmonie")
        )
    )
