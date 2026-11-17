"""Reconnaissance d'un morceau par empreinte audio, façon Shazam.

Des extraits de 12 s de la chanson sont soumis à Shazam (bibliothèque
``shazamio``) : l'empreinte est calculée sur l'ordinateur, et seule elle est
envoyée, pas l'audio. Shazam reconnaît des **enregistrements** : il trouve le
morceau quand on joue par-dessus l'original, ou sur une reprise très fidèle,
mais souvent pas sur la version d'un groupe qui répète.

Pour ne pas nommer un MP3 d'après une mauvaise réponse, un morceau n'est
retenu que s'il est trouvé sur deux extraits, ou sur un seul extrait avec une
correspondance exacte (sans décalage de vitesse ni de hauteur : d'après
shazamio, les mauvaises réponses en ont presque toujours un).
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from . import audio_io
from .titre import EntreeRepertoire, Reconnaissance, normaliser

log = logging.getLogger(__name__)

DUREE_EXTRAIT_S = 12.0  # fenêtre analysée par Shazam
POSITIONS_EXTRAITS = (0.30, 0.55, 0.80)  # dans la chanson : on évite l'intro
DECALAGE_EXACT = 1e-3


@dataclass(frozen=True)
class MorceauReconnu:
    titre: str
    artiste: str = ""
    cle: str = ""
    """Identifiant du morceau chez le service de reconnaissance."""


class ReconnaisseurAudio(Protocol):
    identifiant: str

    def reconnaitre(self, audio: np.ndarray, sr: int) -> tuple[MorceauReconnu, float] | None:
        """Morceau reconnu dans l'extrait et décalage de la correspondance (0 = identique), ou None."""
        ...


class ReconnaisseurShazam:
    """Interroge Shazam avec l'empreinte d'un extrait (connexion Internet nécessaire)."""

    identifiant = "shazam"

    def reconnaitre(self, audio: np.ndarray, sr: int) -> tuple[MorceauReconnu, float] | None:
        import asyncio

        from shazamio import Shazam
        from shazamio_core import SearchParams

        tampon = io.BytesIO()
        sf.write(tampon, audio, sr, format="WAV", subtype="PCM_16")

        async def rechercher() -> dict:
            # Le moteur d'empreinte de shazamio se crée dans la boucle asynchrone ;
            # « async with » referme la connexion à la fin.
            async with Shazam() as shazam:
                return await shazam.recognize(
                    tampon.getvalue(), options=SearchParams(segment_duration_seconds=int(DUREE_EXTRAIT_S))
                )

        return lire_reponse_shazam(asyncio.run(rechercher()))


def lire_reponse_shazam(reponse: dict) -> tuple[MorceauReconnu, float] | None:
    piste = reponse.get("track") or {}
    correspondances = reponse.get("matches") or []
    titre = str(piste.get("title", "")).strip()
    if not titre or not correspondances:
        return None
    premiere = correspondances[0]
    decalage = max(abs(float(premiere.get("timeskew", 0.0))), abs(float(premiere.get("frequencyskew", 0.0))))
    return MorceauReconnu(titre, str(piste.get("subtitle", "")).strip(), str(piste.get("key", ""))), decalage


def extraits(debut: float, fin: float) -> list[tuple[float, float]]:
    """Extraits de 12 s à soumettre, pris à 30, 55 et 80 % de la chanson."""
    longueur = fin - debut
    if longueur <= DUREE_EXTRAIT_S + 1:
        return [(debut, fin)]
    resultat = []
    for position in POSITIONS_EXTRAITS:
        centre = debut + position * longueur
        a = min(max(debut, centre - DUREE_EXTRAIT_S / 2), fin - DUREE_EXTRAIT_S)
        resultat.append((a, a + DUREE_EXTRAIT_S))
    return resultat


def decider(resultats: list[tuple[MorceauReconnu, float]]) -> tuple[MorceauReconnu, str] | None:
    """Morceau retenu et explication, ou None si les réponses ne sont pas assez sûres."""
    groupes: dict[str, list[tuple[MorceauReconnu, float]]] = {}
    for morceau, decalage in resultats:
        groupes.setdefault(morceau.cle or normaliser(morceau.titre), []).append((morceau, decalage))
    for groupe in groupes.values():
        if len(groupe) >= 2:
            return groupe[0][0], f"trouvé sur {len(groupe)} extraits"
    for morceau, decalage in resultats:
        if decalage <= DECALAGE_EXACT:
            return morceau, "correspondance exacte avec l'enregistrement"
    return None


# Mentions de version ajoutées par les catalogues : « (Remastered 2011) », « - Live »...
_MENTION_VERSION = re.compile(
    r"\s*(?:[\(\[][^\(\)\[\]]*(?:remaster|live|version|edit|mix|mono|stereo|acoustic|acoustique|demo|radio|"
    r"single|deluxe|feat|bonus|en direct)[^\(\)\[\]]*[\)\]]|-\s+[^-]*(?:remaster|live|version|edit|mix|mono|stereo|"
    r"acoustic|acoustique|demo|radio|single|en direct)[^-]*)\s*$",
    re.IGNORECASE,
)


def nettoyer_titre(titre: str) -> str:
    """Retire les mentions de version en fin de titre : « Le Sud (Remastered) » -> « Le Sud »."""
    precedent = None
    while precedent != titre:
        precedent, titre = titre, _MENTION_VERSION.sub("", titre).strip()
    return titre


def titre_du_repertoire(titre: str, repertoire: list[EntreeRepertoire]) -> str:
    """Reprend l'orthographe du répertoire du groupe quand le titre y figure."""
    from rapidfuzz import fuzz

    cible = normaliser(titre)
    meilleur = max(repertoire, key=lambda e: fuzz.ratio(normaliser(e.titre), cible), default=None)
    if meilleur is not None and fuzz.ratio(normaliser(meilleur.titre), cible) >= 85:
        return meilleur.titre
    return titre


def identifier(
    wav: Path,
    debut: float,
    fin: float,
    reconnaisseur: ReconnaisseurAudio,
    cache: Path,
    repertoire: list[EntreeRepertoire] | None = None,
) -> Reconnaissance | None:
    """Reconnaît la chanson jouée entre ``debut`` et ``fin`` (s) de ``wav``.

    Le résultat, même négatif, est gardé dans ``cache`` : une relance n'interroge
    pas le service à nouveau. Une erreur (pas de connexion...) n'est pas gardée.
    """
    signature = {
        "source": [wav.stat().st_size, wav.stat().st_mtime_ns],
        "zone": [round(debut, 2), round(fin, 2)],
        "reconnaisseur": reconnaisseur.identifiant,
    }
    decision: tuple[MorceauReconnu, str] | None
    try:
        donnees = json.loads(cache.read_text(encoding="utf-8"))
        if donnees.get("signature") != signature:
            raise ValueError("cache périmé")
        decision = (MorceauReconnu(**donnees["morceau"]), donnees["detail"]) if donnees["morceau"] else None
    except (OSError, ValueError, KeyError, TypeError):
        resultats: list[tuple[MorceauReconnu, float]] = []
        decision = None
        for a, b in extraits(debut, fin):
            audio, sr = audio_io.lire(wav, a, b)
            reponse = reconnaisseur.reconnaitre(audio, sr)
            if reponse is not None:
                log.debug("    %s : %s (%s), décalage %.4f", reconnaisseur.identifiant, reponse[0].titre,
                          reponse[0].artiste, reponse[1])
                resultats.append(reponse)
            decision = decider(resultats)
            if decision is not None:
                break
        if decision is None and resultats:
            log.info("  %s a proposé « %s » sans confirmation : non retenu", reconnaisseur.identifiant.capitalize(),
                     resultats[0][0].titre)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "morceau": asdict(decision[0]) if decision else None,
                    "detail": decision[1] if decision else "",
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    if decision is None:
        return None
    morceau, detail = decision
    titre = nettoyer_titre(morceau.titre)
    if repertoire:
        titre = titre_du_repertoire(titre, repertoire)
    artiste = f"{morceau.artiste}, " if morceau.artiste else ""
    return Reconnaissance(titre, reconnaisseur.identifiant.capitalize(), f"{artiste}{detail}")
