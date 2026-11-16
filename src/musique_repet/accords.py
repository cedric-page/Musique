"""Reconnaissance des accords joués par les instruments (piano, guitare, basse).

À chaque instant (toutes les 93 ms), les notes entendues (chromagramme) sont
comparées à des modèles d'accords : majeur, mineur, septième, mineur
septième, sur les 12 fondamentales. Un algorithme de Viterbi lisse le
résultat : un accord dure, il ne change pas à chaque instant. Les passages
où les instruments ne jouent pas, ou jouent autre chose qu'un accord
reconnaissable, n'ont pas d'accord.

Pour chaque accord, l'accordage des instruments est mesuré pendant l'accord
(sur au moins 4 s) : le chant est comparé à ce qui est réellement joué à ce
moment-là, même si un instrument se désaccorde en cours de morceau.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

SR = 22050
PAS = 2048  # 93 ms
NOMS_NOTES = ("Do", "Do♯", "Ré", "Mi♭", "Mi", "Fa", "Fa♯", "Sol", "La♭", "La", "Si♭", "Si")
# Intervalles (demi-tons au-dessus de la fondamentale) et poids dans le modèle.
TYPES = {
    "": ((0, 1.0), (4, 1.0), (7, 0.8)),
    "m": ((0, 1.0), (3, 1.0), (7, 0.8)),
    "7": ((0, 1.0), (4, 1.0), (7, 0.8), (10, 0.8)),
    "m7": ((0, 1.0), (3, 1.0), (7, 0.8), (10, 0.8)),
}
DEGRES = {0: "fondamentale", 3: "tierce", 4: "tierce", 7: "quinte", 10: "septième"}
SCORE_SANS_ACCORD = 0.55
"""Ressemblance minimale (cosinus) avec un modèle pour reconnaître un accord."""


@dataclass(frozen=True)
class Accord:
    debut: float
    fin: float
    fondamentale: int | None
    """Classe de hauteur (0 = Do) ; None quand aucun accord n'est joué."""
    type: str = ""
    accordage_cents: float = 0.0
    """Accordage des instruments pendant cet accord, par rapport à La 440."""

    @property
    def notes(self) -> frozenset[int]:
        if self.fondamentale is None:
            return frozenset()
        return frozenset((self.fondamentale + i) % 12 for i, _ in TYPES[self.type])

    @property
    def nom(self) -> str:
        if self.fondamentale is None:
            return ""
        return NOMS_NOTES[self.fondamentale] + ({"": "", "m": " m", "7": "7", "m7": " m7"}[self.type])

    def degre(self, classe: int) -> str | None:
        """« fondamentale », « tierce », « quinte », « septième » ou None si la note n'est pas dans l'accord."""
        if classe % 12 not in self.notes:
            return None
        return DEGRES[(classe - self.fondamentale) % 12]


def _modeles() -> tuple[np.ndarray, list[tuple[int, str]]]:
    vecteurs, etiquettes = [], []
    for type_, intervalles in TYPES.items():
        for fondamentale in range(12):
            v = np.zeros(12)
            for intervalle, poids in intervalles:
                v[(fondamentale + intervalle) % 12] = poids
            vecteurs.append(v / np.linalg.norm(v))
            etiquettes.append((fondamentale, type_))
    return np.array(vecteurs), etiquettes


def reconnaitre_accords(instruments: np.ndarray, accordage_cents: float, duree_min_s: float = 0.4) -> list[Accord]:
    """Accords joués dans ``instruments`` (mono, 22 050 Hz), du début à la fin."""
    import librosa

    if instruments.size < SR // 2 or not np.any(instruments):
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # extrait très court
        chroma = librosa.feature.chroma_cqt(
            y=instruments, sr=SR, hop_length=PAS, tuning=accordage_cents / 100, norm=None
        )
    energie = chroma.sum(axis=0)
    norme = np.linalg.norm(chroma, axis=0)
    chroma_n = chroma / np.maximum(norme, 1e-9)
    modeles, etiquettes = _modeles()
    scores = modeles @ chroma_n  # (modèles, trames) : ressemblance cosinus
    silence = energie < 0.02 * max(float(np.percentile(energie, 95)), 1e-9)
    sans_accord = np.where(silence, 1.0, SCORE_SANS_ACCORD)
    scores = np.vstack([scores, sans_accord[None, :]])
    scores[:-1, silence] = 0.0

    # Viterbi : un accord se maintient ; en changer coûte.
    nb_etats, nb_trames = scores.shape
    log_obs = 25.0 * scores
    reste, change = math.log(0.95), math.log(0.05 / (nb_etats - 1))
    score = log_obs[:, 0].copy()
    retour = np.zeros((nb_trames, nb_etats), dtype=np.int32)
    for t in range(1, nb_trames):
        meilleur = int(np.argmax(score))
        venir_du_meilleur = score[meilleur] + change
        garder = score + reste
        retour[t] = np.where(garder >= venir_du_meilleur, np.arange(nb_etats), meilleur)
        score = np.maximum(garder, venir_du_meilleur) + log_obs[:, t]
    chemin = np.empty(nb_trames, dtype=np.int64)
    chemin[-1] = int(np.argmax(score))
    for t in range(nb_trames - 1, 0, -1):
        chemin[t - 1] = retour[t, chemin[t]]

    # Plages d'un même accord ; les plus courtes rejoignent leur voisine.
    trame_s = PAS / SR
    plages: list[list[int]] = []
    for t, etat in enumerate(chemin.tolist()):
        if plages and plages[-1][0] == etat:
            plages[-1][2] = t + 1
        else:
            plages.append([etat, t, t + 1])
    minimum = max(1, int(round(duree_min_s / trame_s)))
    fusion: list[list[int]] = []
    for plage in plages:
        if fusion and (plage[2] - plage[1] < minimum or fusion[-1][0] == plage[0]):
            fusion[-1][2] = plage[2]
        else:
            fusion.append(plage)

    duree = instruments.size / SR
    accords = []
    for etat, a, b in fusion:
        debut, fin = a * trame_s, min(b * trame_s, duree)
        if etat == nb_etats - 1:
            accords.append(Accord(debut, fin, None))
            continue
        fondamentale, type_ = etiquettes[etat]
        accords.append(Accord(debut, fin, fondamentale, type_, _accordage_local(instruments, debut, fin, accordage_cents)))
    return accords


def _accordage_local(instruments: np.ndarray, debut: float, fin: float, defaut: float, fenetre_min_s: float = 4.0) -> float:
    """Accordage des instruments autour de l'accord (au moins ``fenetre_min_s`` secondes)."""
    import librosa

    centre, demi = (debut + fin) / 2, max((fin - debut) / 2, fenetre_min_s / 2)
    a, b = int(max(0.0, centre - demi) * SR), int(min(instruments.size / SR, centre + demi) * SR)
    extrait = instruments[a:b]
    if extrait.size < SR or not np.any(extrait):
        return defaut
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local = 100.0 * float(librosa.estimate_tuning(y=extrait, sr=SR, n_fft=4096))
    # Un accordage local très éloigné de l'accordage général est une erreur de mesure.
    return local if abs(local - defaut) <= 30 else defaut


def accord_pendant(accords: list[Accord], debut: float, fin: float) -> Accord | None:
    """L'accord qui couvre le plus longtemps l'intervalle [début, fin] (None s'il n'y en a pas)."""
    meilleur, recouvrement_max = None, 0.0
    for accord in accords:
        recouvrement = min(accord.fin, fin) - max(accord.debut, debut)
        if recouvrement > recouvrement_max:
            meilleur, recouvrement_max = accord, recouvrement
    return meilleur if meilleur is not None and meilleur.fondamentale is not None else None
