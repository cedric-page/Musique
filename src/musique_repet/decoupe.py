"""Lecture des fichiers CSV de découpe (<titre>;<debut>;<fin>)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Morceau:
    """Un extrait à découper dans l'enregistrement (temps en secondes)."""

    titre: str
    debut: float
    fin: float


def parser_temps(texte: str) -> float:
    """Convertit un temps en secondes.

    Formats acceptés : ``83``, ``83.5``, ``83,5``, ``1:23``, ``1:23.5``,
    ``01:02:03`` et ``01:02:03.250``.
    """
    brut = texte.strip().replace(",", ".")
    if not brut:
        raise ValueError("temps vide")
    parties = brut.split(":")
    if len(parties) > 3:
        raise ValueError(f"temps invalide : {texte!r}")
    try:
        valeurs = [float(p) for p in parties]
    except ValueError as exc:
        raise ValueError(f"temps invalide : {texte!r}") from exc
    if any(v < 0 for v in valeurs):
        raise ValueError(f"temps négatif : {texte!r}")
    secondes = 0.0
    for v in valeurs:
        secondes = secondes * 60 + v
    return secondes


def nom_fichier_sur(titre: str) -> str:
    """Rend un titre utilisable comme nom de fichier."""
    nom = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", titre).strip().strip(".")
    return nom or "sans_titre"


def _lire_texte(chemin: Path) -> str:
    for encodage in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return chemin.read_text(encoding=encodage)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"encodage illisible : {chemin}")


def lire_csv(chemin: Path, duree: float | None = None) -> list[Morceau]:
    """Lit un CSV ``titre;debut;fin``.

    Une éventuelle ligne d'en-tête est ignorée, ainsi que les lignes vides et
    celles commençant par ``#``. Si ``duree`` est fourni, les fins dépassant
    l'enregistrement sont ramenées à la durée réelle.
    Les titres en double reçoivent un suffixe ``(2)``, ``(3)``...
    """
    morceaux: list[Morceau] = []
    vus: dict[str, int] = {}
    for num, ligne in enumerate(_lire_texte(chemin).splitlines(), start=1):
        if not ligne.strip() or ligne.lstrip().startswith("#"):
            continue
        champs = [c.strip() for c in ligne.split(";")]
        if len(champs) != 3:
            raise ValueError(f"{chemin.name}:{num} : 3 champs attendus (titre;debut;fin), lu {ligne!r}")
        titre, debut_txt, fin_txt = champs
        try:
            debut, fin = parser_temps(debut_txt), parser_temps(fin_txt)
        except ValueError:
            if not morceaux and num == 1:
                log.debug("%s : ligne d'en-tête ignorée", chemin.name)
                continue
            raise
        if not titre:
            raise ValueError(f"{chemin.name}:{num} : titre vide")
        if duree is not None and fin > duree:
            if debut >= duree:
                log.warning("%s:%d : « %s » commence après la fin de l'audio, ignoré", chemin.name, num, titre)
                continue
            log.warning("%s:%d : fin de « %s » ramenée à la durée de l'audio", chemin.name, num, titre)
            fin = duree
        if fin <= debut:
            raise ValueError(f"{chemin.name}:{num} : la fin doit être après le début pour « {titre} »")
        n = vus.get(titre, 0) + 1
        vus[titre] = n
        morceaux.append(Morceau(titre if n == 1 else f"{titre} ({n})", debut, fin))
    return morceaux
