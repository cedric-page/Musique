"""Reconnaissance du titre d'une chanson à partir de ses paroles transcrites.

1. Si un **répertoire** (liste des titres du groupe) est fourni, on cherche
   lequel est chanté pendant la chanson ou annoncé juste avant (« allez, on
   fait Le Sud »). C'est la méthode la plus fiable.
2. Sinon, la phrase la plus répétée du **refrain** est proposée comme titre,
   ce qui marche pour beaucoup de chansons (« Hallelujah », « Let it be »...).
3. Sinon, l'appelant garde le nom du fichier.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .paroles import SegmentParoles, Transcripteur, transcrire_zone

log = logging.getLogger(__name__)


# --- Texte ------------------------------------------------------------------


def normaliser(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation : « J'ai rêvé ! » -> « j ai reve »."""
    sans_accents = "".join(c for c in unicodedata.normalize("NFKD", texte.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", sans_accents).split())


def replier(texte: str) -> str:
    """Forme approximative de la prononciation, pour tolérer les graphies de la transcription :
    « Hallelujah » et « Alléluia » donnent tous deux « aleluia »."""
    texte = normaliser(texte).replace("h", "").replace("j", "i").replace("y", "i")
    return re.sub(r"([a-z])\1+", r"\1", texte)


_MOT = re.compile(r"[^\W_]+", re.UNICODE)


def _mots(texte: str) -> list[tuple[str, int, int]]:
    """Mots du texte : (forme normalisée, début, fin dans le texte d'origine)."""
    resultat = []
    for m in _MOT.finditer(texte):
        for morceau in normaliser(m.group()).split():
            resultat.append((morceau, m.start(), m.end()))
    return resultat


def _phrases(segments: list[SegmentParoles]) -> list[list[str]]:
    return [[replier(m) for m, _, _ in _mots(s.texte)] for s in segments]


def compter_occurrences(cible: str, phrases: list[list[str]], seuil: float = 85.0) -> int:
    """Nombre de phrases qui contiennent ``cible`` (à quelques fautes de transcription près)."""
    from rapidfuzz import fuzz

    mots_cible = [replier(m) for m in normaliser(cible).split()]
    if not mots_cible:
        return 0
    texte_cible = " ".join(mots_cible)
    n = len(mots_cible)
    total = 0
    for mots in phrases:
        for i in range(len(mots)):
            tailles = [m for m in (n - 1, n, n + 1) if m >= 1 and i + m <= len(mots)]
            if any(fuzz.ratio(" ".join(mots[i : i + m]), texte_cible) >= seuil for m in tailles):
                total += 1
                break
    return total


# --- Répertoire ---------------------------------------------------------------


@dataclass(frozen=True)
class EntreeRepertoire:
    titre: str
    paroles: tuple[str, ...] = ()
    """Extraits de paroles ou mots caractéristiques (facultatif)."""


def lire_repertoire(chemin: Path) -> list[EntreeRepertoire]:
    """Une chanson par ligne : ``Titre`` ou ``Titre ; extrait de paroles ; autre extrait``.

    Les lignes vides et celles commençant par ``#`` sont ignorées.
    """
    entrees = []
    for ligne in chemin.read_text(encoding="utf-8-sig").splitlines():
        if not ligne.strip() or ligne.lstrip().startswith("#"):
            continue
        champs = [c.strip() for c in ligne.split(";")]
        if champs[0]:
            entrees.append(EntreeRepertoire(champs[0], tuple(c for c in champs[1:] if c)))
    return entrees


@dataclass(frozen=True)
class Reconnaissance:
    titre: str
    methode: str
    """« répertoire » ou « refrain »."""
    detail: str


def chercher_dans_repertoire(
    chanson: list[SegmentParoles], avant: list[SegmentParoles], repertoire: list[EntreeRepertoire]
) -> Reconnaissance | None:
    """Titre du répertoire le plus présent dans les paroles et dans ce qui est dit juste avant."""
    phrases_chanson, phrases_avant = _phrases(chanson), _phrases(avant)
    resultats = []
    for entree in repertoire:
        chante = compter_occurrences(entree.titre, phrases_chanson)
        annonce = compter_occurrences(entree.titre, phrases_avant)
        extraits = sum(1 for p in entree.paroles if compter_occurrences(p, phrases_chanson, 80) > 0)
        # L'annonce compte autant qu'un refrain : c'est le seul indice pour un titre
        # absent des paroles (« Bohemian Rhapsody »).
        score = chante + annonce + 2 * extraits
        # Un titre court (« Home », « Toi ») se retrouve facilement par hasard : il faut plus d'indices.
        court = len(normaliser(entree.titre).replace(" ", "")) < 5
        if score >= (2 if court else 1):
            detail = f"titre chanté {chante} fois, annoncé {annonce} fois, {extraits} extrait(s) de paroles retrouvé(s)"
            resultats.append((score, chante, entree.titre, detail))
    if not resultats:
        return None
    resultats.sort(key=lambda r: (r[0], r[1]), reverse=True)
    if len(resultats) > 1 and resultats[0][:2] == resultats[1][:2]:
        log.info("  Titre ambigu entre « %s » et « %s » : aucun n'est retenu", resultats[0][2], resultats[1][2])
        return None
    _, _, titre, detail = resultats[0]
    return Reconnaissance(titre, "répertoire", detail)


# --- Refrain ----------------------------------------------------------------

# Mots qui ne terminent pas un titre (articles, prépositions, conjonctions, possessifs).
_FIN_INTERDITE = set(
    "the a an and or of to in for with at from by my your his her our their "
    "le la les l un une des du de d et ou au aux en dans pour avec par sur que qui qu "
    "mon ma mes ton ta tes son sa ses notre votre nos vos leur leurs ce cet cette".split()
)
# Mots qui ne commencent pas un titre.
_DEBUT_INTERDIT = set("and or but of to for with at from by et ou mais de du des d au aux pour avec par que qui qu".split())
_VOCALISES = set(
    "oh ah ooh oooh ouh ou wo woh whoa yeah yeh ye hey ho ha hum hmm mm mmm uh eh na nana la lala "
    "da dada doo dou ba sha ya yo ay".split()
)


def _repetition(groupe: tuple[str, ...]) -> bool:
    """Vrai si le groupe répète un motif plus court (« hallelujah hallelujah »)."""
    n = len(groupe)
    return any(n % p == 0 and groupe == groupe[:p] * (n // p) for p in range(1, n))


def _candidat_valable(groupe: tuple[str, ...]) -> bool:
    if _repetition(groupe):
        return False
    if groupe[-1] in _FIN_INTERDITE or groupe[0] in _DEBUT_INTERDIT:
        return False
    if groupe[0] in _VOCALISES or groupe[-1] in _VOCALISES:
        return False
    return any(len(m) >= 4 and m not in _VOCALISES for m in groupe)


def deviner_par_refrain(
    chanson: list[SegmentParoles], repetitions_min: int = 3, etalement_min_s: float = 20.0
) -> Reconnaissance | None:
    """Propose la phrase (1 à 4 mots) qui revient le plus d'une phrase chantée à l'autre.

    Elle doit revenir dans au moins ``repetitions_min`` phrases différentes,
    étalées sur au moins ``etalement_min_s`` secondes : un vrai refrain, pas une
    ligne répétée d'affilée.
    """
    compte: Counter[tuple[str, ...]] = Counter()
    instants: dict[tuple[str, ...], list[float]] = defaultdict(list)
    formes: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for segment in chanson:
        mots = _mots(segment.texte)
        vus: set[tuple[str, ...]] = set()
        for n in range(1, 5):
            for i in range(len(mots) - n + 1):
                groupe = tuple(replier(m) for m, _, _ in mots[i : i + n])
                normal = tuple(m for m, _, _ in mots[i : i + n])
                if groupe in vus or not _candidat_valable(normal):
                    continue
                vus.add(groupe)
                compte[groupe] += 1
                instants[groupe].append(segment.debut)
                formes[groupe][segment.texte[mots[i][1] : mots[i + n - 1][2]]] += 1
    candidats = [
        (nb * (1 + 0.25 * (len(groupe) - 1)), nb, groupe)
        for groupe, nb in compte.items()
        if nb >= repetitions_min and max(instants[groupe]) - min(instants[groupe]) >= etalement_min_s
    ]
    if not candidats:
        return None
    _, nb, groupe = max(candidats)
    forme = formes[groupe].most_common(1)[0][0].strip()
    return Reconnaissance(forme[:1].upper() + forme[1:], "refrain", f"phrase chantée dans {nb} phrases différentes")


# --- Ensemble -----------------------------------------------------------------


@dataclass
class ParametresTitre:
    repertoire: list[EntreeRepertoire] = field(default_factory=list)
    deviner_par_refrain: bool = True
    annonce_s: float = 60.0
    """Durée écoutée avant la chanson, où le titre est souvent annoncé (avec un répertoire)."""


def reconnaitre_titre(
    voix: Path,
    debut: float,
    fin: float,
    dossier_cache: Path,
    transcripteur: Transcripteur,
    params: ParametresTitre,
) -> Reconnaissance | None:
    """Reconnaît le titre de la chanson chantée entre ``debut`` et ``fin`` (s) sur la piste ``voix``."""
    chanson = transcrire_zone(voix, debut, fin, dossier_cache / "paroles_chanson.json", transcripteur)
    avant: list[SegmentParoles] = []
    if params.repertoire and debut >= 2:
        avant = transcrire_zone(
            voix, max(0.0, debut - params.annonce_s), debut, dossier_cache / "paroles_avant.json", transcripteur
        )
    if log.isEnabledFor(logging.DEBUG):
        for s in avant + chanson:
            log.debug("    [%6.1f s] %s", s.debut, s.texte)
    if params.repertoire:
        trouve = chercher_dans_repertoire(chanson, avant, params.repertoire)
        if trouve:
            return trouve
    if params.deviner_par_refrain:
        return deviner_par_refrain(chanson)
    return None
