"""Suivi de la hauteur d'une ou deux voix qui chantent en même temps (piste voix séparée).

À chaque instant (toutes les 16 ms) :

1. la hauteur dont les harmoniques (fondamentale, double, triple...) sont les
   plus présentes dans le spectre donne la voix principale ;
2. ses harmoniques sont effacées du spectre ; s'il reste une hauteur bien
   présente, c'est une seconde voix (harmonie en tierce, sixte...) ;
3. chaque hauteur est ensuite mesurée précisément à partir de la position de
   ses harmoniques, en écartant celles qui se superposent à l'autre voix.

Seuls les instants où la voix est bien présente (moins de 25 dB sous les
passages forts) et nettement harmonique (pas du bruit) sont gardés.

Les deux voix sont ensuite attribuées au **lead** et à l'**harmonie** (voir
``attribuer_voix``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
N_FFT = 2048  # 128 ms
PAS = 256  # 16 ms
DF = SR / N_FFT
FREQ_MAX_PARTIELS = 5000.0
NB_HARMONIQUES = 8
SEUIL_HARMONICITE = 0.03
"""Part du spectre portée par les harmoniques d'une hauteur : chant ≈ 0,05 à 0,13, bruit ≈ 0,01."""
RAPPORT_SECONDE_VOIX = 0.35
"""Présence minimale de la seconde voix par rapport à la première (≈ -18 dB)."""
ECART_NIVEAU_DB = 25.0


@dataclass
class Contours:
    """Hauteurs (demi-tons, numérotation MIDI ; NaN = pas de voix) à chaque instant."""

    temps: np.ndarray
    principale: np.ndarray
    """Voix seule, ou la plus présente des deux."""
    seconde: np.ndarray
    """L'autre voix, quand deux voix chantent ensemble."""
    energie_principale: np.ndarray | None = None
    """Énergie de chaque voix (somme des carrés de ses harmoniques) : sert à savoir qui chante le plus fort."""
    energie_seconde: np.ndarray | None = None

    @property
    def fps(self) -> float:
        return SR / PAS


def _spectre(mono: np.ndarray) -> np.ndarray:
    fenetre = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    x = np.pad(mono.astype(np.float32), (N_FFT // 2, N_FFT // 2))
    nb = 1 + (x.size - N_FFT) // PAS
    resultat = np.empty((nb, N_FFT // 2 + 1), np.float32)
    for debut in range(0, nb, 2048):
        indices = np.arange(debut, min(nb, debut + 2048))[:, None] * PAS + np.arange(N_FFT)
        resultat[debut : debut + indices.shape[0]] = np.abs(np.fft.rfft(x[indices] * fenetre, axis=1))
    return resultat


def _grille(fmin: float, fmax: float):
    """Hauteurs candidates (pas de 10 cents) et position de leurs harmoniques dans le spectre."""
    candidats = fmin * 2 ** (np.arange(0, 1200 * np.log2(fmax / fmin), 10) / 1200)
    rangs = np.arange(1, NB_HARMONIQUES + 1)
    frequences = candidats[:, None] * rangs[None, :]
    cases = np.clip(np.round(frequences / DF).astype(np.int64), 1, N_FFT // 2 - 1)
    poids = np.where(frequences < FREQ_MAX_PARTIELS, 0.84 ** (rangs - 1)[None, :], 0.0)
    return candidats, cases, poids


def _salience(spectre_c: np.ndarray, cases: np.ndarray, poids: np.ndarray) -> np.ndarray:
    """Présence de chaque hauteur candidate : somme pondérée de ses harmoniques (± 1 case)."""
    m = np.maximum(np.maximum(spectre_c[:, cases - 1], spectre_c[:, cases]), spectre_c[:, cases + 1])
    return (m * poids[None]).sum(axis=2)


def _effacer_harmoniques(spectre_c: np.ndarray, f0: np.ndarray) -> np.ndarray:
    residu = spectre_c.copy()
    lignes = np.arange(residu.shape[0])
    for rang in range(1, 2 * NB_HARMONIQUES + 1):
        case = np.round(rang * f0 / DF).astype(np.int64)
        ok = case < residu.shape[1] - 3
        for decalage in range(-2, 3):
            residu[lignes[ok], np.maximum(case[ok] + decalage, 0)] = 0
    return residu


def _affiner(spectre: np.ndarray, f0: np.ndarray, autre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hauteur précise d'après la position (interpolée) de chaque harmonique ; les harmoniques
    qui tombent à moins de 3 % d'une harmonique de ``autre`` (seconde voix, NaN sinon) sont écartées.
    Renvoie aussi l'énergie de la voix (somme des carrés de ses harmoniques retenues)."""
    nb = f0.size
    lignes = np.arange(nb)
    estimations = np.full((nb, NB_HARMONIQUES), np.nan)
    poids = np.zeros((nb, NB_HARMONIQUES))
    for i, rang in enumerate(range(1, NB_HARMONIQUES + 1)):
        frequence = rang * f0
        utilisable = frequence < FREQ_MAX_PARTIELS
        avec_autre = np.isfinite(autre)
        multiple = np.where(avec_autre, np.round(frequence / np.where(avec_autre, autre, 1.0)), 0)
        collision = avec_autre & (multiple >= 1) & (np.abs(frequence - multiple * autre) / frequence < 0.03)
        centre = np.round(frequence / DF).astype(np.int64)
        voisinage = np.clip(centre[:, None] + np.arange(-2, 3)[None, :], 1, spectre.shape[1] - 2)
        valeurs = spectre[lignes[:, None], voisinage]
        pic = voisinage[lignes, np.argmax(valeurs, axis=1)]
        a = np.log(spectre[lignes, pic - 1] + 1e-9)
        b = np.log(spectre[lignes, pic] + 1e-9)
        c = np.log(spectre[lignes, pic + 1] + 1e-9)
        courbure = a - 2 * b + c
        decalage = np.where(courbure < 0, 0.5 * (a - c) / np.where(courbure < 0, courbure, -1.0), 0.0)
        interieur = (pic > voisinage[:, 0]) & (pic < voisinage[:, -1])  # vrai pic, pas un bord
        valide = utilisable & ~collision & interieur
        estimations[valide, i] = (pic[valide] + np.clip(decalage[valide], -0.5, 0.5)) * DF / rang
        poids[valide, i] = spectre[lignes[valide], pic[valide]]
    mediane = np.nanmedian(np.where(np.isfinite(estimations), estimations, np.nan), axis=1)
    proches = np.isfinite(estimations) & (np.abs(1200 * np.log2(estimations / mediane[:, None])) < 15)
    somme = (np.where(proches, poids, 0)).sum(axis=1)
    moyenne = np.where(proches, estimations * poids, 0).sum(axis=1) / np.maximum(somme, 1e-12)
    energie = (np.where(np.isfinite(estimations), poids, 0) ** 2).sum(axis=1)
    return np.where(somme > 0, moyenne, f0), energie


def suivre_voix(mono: np.ndarray, fmin: float = 110.0, fmax: float = 1100.0) -> Contours:
    """Hauteur d'une ou deux voix dans ``mono`` (16 kHz)."""
    import warnings

    spectre = _spectre(mono)
    nb = spectre.shape[0]
    temps = np.arange(nb) * PAS / SR
    if nb == 0:
        return Contours(temps, np.zeros(0), np.zeros(0))
    spectre_c = np.sqrt(spectre)  # compression : les harmoniques faibles comptent aussi
    candidats, cases, poids = _grille(fmin, fmax)

    niveau_db = 10 * np.log10((spectre.astype(np.float64) ** 2).sum(axis=1) + 1e-12)
    presente = niveau_db > max(float(np.percentile(niveau_db, 95)) - ECART_NIVEAU_DB, niveau_db.max() - 80)
    total = spectre_c[:, : int(FREQ_MAX_PARTIELS / DF)].sum(axis=1)

    f1 = np.full(nb, np.nan)
    f2 = np.full(nb, np.nan)
    for debut in range(0, nb, 1024):
        bloc = slice(debut, min(nb, debut + 1024))
        sal = _salience(spectre_c[bloc], cases, poids)
        i1 = np.argmax(sal, axis=1)
        s1 = sal[np.arange(i1.size), i1]
        voisee = presente[bloc] & (s1 / np.maximum(total[bloc], 1e-9) > SEUIL_HARMONICITE)
        premiere = candidats[i1]
        residu = _effacer_harmoniques(spectre_c[bloc], premiere)
        sal2 = _salience(residu, cases, poids)
        i2 = np.argmax(sal2, axis=1)
        s2 = sal2[np.arange(i2.size), i2]
        seconde = candidats[i2]
        ecart = 1200 * np.log2(seconde / premiere)
        modulo_octave = np.abs((np.abs(ecart) + 50) % 1200 - 50)
        deux_voix = voisee & (s2 >= RAPPORT_SECONDE_VOIX * s1) & (np.abs(ecart) > 70) & (modulo_octave > 50)
        f1[bloc] = np.where(voisee, premiere, np.nan)
        f2[bloc] = np.where(deux_voix, seconde, np.nan)

    energie1 = np.zeros(nb)
    energie2 = np.zeros(nb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # trames sans harmonique utilisable
        avec_voix = np.isfinite(f1)
        precis1, energie1[avec_voix] = _affiner(spectre[avec_voix], f1[avec_voix], f2[avec_voix])
        avec_deux = np.isfinite(f2)
        precis2, energie2[avec_deux] = _affiner(spectre[avec_deux], f2[avec_deux], f1[avec_deux])
    f1[avec_voix], f2[avec_deux] = precis1, precis2

    return Contours(
        temps, 69 + 12 * np.log2(f1 / 440.0), 69 + 12 * np.log2(f2 / 440.0), energie1, energie2
    )


# --- Lead et harmonie ------------------------------------------------------------


def _plages(masque: np.ndarray) -> list[tuple[int, int]]:
    bords = np.diff(np.concatenate([[0], masque.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(bords == 1).tolist(), np.flatnonzero(bords == -1).tolist()))


def _fermer(masque: np.ndarray, trames: int) -> np.ndarray:
    """Comble les trous d'au plus ``trames`` trames entre deux zones vraies."""
    resultat = masque.copy()
    vrais = np.flatnonzero(masque)
    for a, b in zip(vrais[:-1], vrais[1:]):
        if 1 < b - a <= trames + 1:
            resultat[a + 1 : b] = True
    return resultat


def _cout(valeur: float, precedente: float, age_s: float) -> float:
    """Coût de rattacher ``valeur`` à une piste : carré de l'écart à sa dernière hauteur,
    pondéré par la fraîcheur de celle-ci (une note tenue à l'instant pèse bien plus qu'une
    note d'il y a 200 ms, après laquelle la voix peut repartir n'importe où)."""
    if np.isnan(precedente):
        return 0.0
    return (valeur - precedente) ** 2 * max(np.exp(-age_s / 0.1), 0.05)


def attribuer_voix(
    contours: Contours, pause_s: float = 0.3, meme_note: float = 0.5, duo_min_s: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Répartit les hauteurs entre la voix **lead** et la voix d'**harmonie**.

    - Une voix qui chante seule est le lead (c'est elle qui porte la mélodie).
    - Dans un passage à deux voix (pauses de moins de ``pause_s`` comprises), les
      deux voix sont suivies par continuité de hauteur (le carré des écarts
      favorise les mouvements parallèles, fréquents en harmonie), puis la voix
      qui chante le plus fort sur l'ensemble du passage est le lead. À force
      égale (moins de 60/40), c'est la voix la plus proche, en hauteur, de celle
      qui chante seule dans le reste du morceau.
    - Une note de l'harmonie tenue juste après le passage (le lead s'est arrêté)
      ou commencée juste avant (elle entre avant le lead) reste à l'harmonie.
    - Un passage où l'on n'entend deux voix que moins de ``duo_min_s`` secondes en
      tout (attaque d'une note, résonance) n'est pas une harmonie : la voix seule
      y reste au lead.

    Renvoie (lead, harmonie), en demi-tons, NaN quand la voix ne chante pas.
    """
    p, q = contours.principale, contours.seconde
    n = p.size
    ep = contours.energie_principale if contours.energie_principale is not None else np.ones(n)
    eq = contours.energie_seconde if contours.energie_seconde is not None else np.zeros(n)
    pas_s = 1.0 / contours.fps
    lead = np.full(n, np.nan)
    harmonie = np.full(n, np.nan)
    duo = np.isfinite(p) & np.isfinite(q)
    passages = [
        (a, b)
        for a, b in _plages(_fermer(duo, int(round(pause_s * contours.fps))))
        if duo[a:b].sum() >= duo_min_s * contours.fps
    ]
    dans_passage = np.zeros(n, bool)
    for a, b in passages:
        dans_passage[a:b] = True
    seule = ~dans_passage & np.isfinite(p)
    registre_solo = float(np.median(p[seule])) if seule.any() else np.nan

    for a, b in passages:
        x = np.full(b - a, np.nan)
        y = np.full(b - a, np.nan)
        dernier_x = dernier_y = np.nan
        age_x = age_y = 0.0
        energie_x = energie_y = 0.0
        for k, i in enumerate(range(a, b)):
            age_x += pas_s
            age_y += pas_s
            if np.isfinite(p[i]) and np.isfinite(q[i]):
                garde = _cout(p[i], dernier_x, age_x) + _cout(q[i], dernier_y, age_y)
                echange = _cout(q[i], dernier_x, age_x) + _cout(p[i], dernier_y, age_y)
                if echange < garde:
                    x[k], y[k] = q[i], p[i]
                    energie_x, energie_y = energie_x + eq[i], energie_y + ep[i]
                else:
                    x[k], y[k] = p[i], q[i]
                    energie_x, energie_y = energie_x + ep[i], energie_y + eq[i]
                dernier_x, dernier_y, age_x, age_y = x[k], y[k], 0.0, 0.0
            elif np.isfinite(p[i]):  # une seule voix entendue : rattachée à la piste la plus proche
                if np.isnan(dernier_y) or _cout(p[i], dernier_x, age_x) <= _cout(p[i], dernier_y, age_y):
                    x[k], dernier_x, age_x = p[i], p[i], 0.0
                else:
                    y[k], dernier_y, age_y = p[i], p[i], 0.0
        part_x = energie_x / max(energie_x + energie_y, 1e-12)
        if 0.4 <= part_x <= 0.6 and np.isfinite(registre_solo) and np.isfinite(x).any() and np.isfinite(y).any():
            # Force égale : le lead est la voix dont la hauteur rappelle celle qui chante seule.
            x_est_lead = abs(np.nanmedian(x) - registre_solo) <= abs(np.nanmedian(y) - registre_solo)
        else:
            x_est_lead = part_x >= 0.5
        if x_est_lead:
            lead[a:b], harmonie[a:b] = x, y
        else:
            lead[a:b], harmonie[a:b] = y, x

    lead[seule] = p[seule]

    for a, b in passages:
        for sens in (1, -1):  # après puis avant le passage
            zone = range(b, n) if sens == 1 else range(a - 1, -1, -1)
            bord = range(b - 1, a - 1, -1) if sens == 1 else range(a, b)
            dernier_h = next((harmonie[i] for i in bord if np.isfinite(harmonie[i])), np.nan)
            dernier_l = next((lead[i] for i in bord if np.isfinite(lead[i])), np.nan)
            if np.isnan(dernier_h):
                continue
            for j in zone:
                if dans_passage[j] or np.isnan(lead[j]):
                    break
                v = lead[j]
                plus_proche_de_l_harmonie = np.isnan(dernier_l) or abs(v - dernier_l) > abs(v - dernier_h) + 0.2
                if abs(v - dernier_h) >= meme_note or not plus_proche_de_l_harmonie:
                    break
                harmonie[j], lead[j] = v, np.nan
                dernier_h = v
    return lead, harmonie
