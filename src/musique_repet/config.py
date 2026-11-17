"""Fichier de configuration des volumes : ``{"instrument": facteur}``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .separation import NOMS_PISTES

log = logging.getLogger(__name__)

# Pistes réglables : la voix est séparée en « lead » et « choeurs » (ou reste en une
# seule piste « voix » si cette séparation est désactivée).
PISTES = ["lead", "choeurs", "voix", *[n for n in dict.fromkeys(NOMS_PISTES.values()) if n != "voix"]]

# Noms alternatifs acceptés dans la configuration (minuscules, « œ » écrit « oe »).
ALIAS = {
    **{anglais: francais for anglais, francais in NOMS_PISTES.items()},
    "lead": "lead",
    "voix lead": "lead",
    "voix principale": "lead",
    "chant lead": "lead",
    "chant principal": "lead",
    "lead vocals": "lead",
    "choeurs": "choeurs",
    "choeur": "choeurs",
    "choristes": "choeurs",
    "harmonies": "choeurs",
    "harmonie": "choeurs",
    "backing": "choeurs",
    "backing vocals": "choeurs",
    "voix": "voix",
    "chant": "voix",
    "chanteuses": "voix",
    "vocal": "voix",
    "batterie": "batterie",
    "drum": "batterie",
    "basse": "basse",
    "guitare": "guitare",
    "piano": "piano",
    "clavier": "piano",
    "autre": "autre",
    "autres": "autre",
}

CONFIG_EXEMPLE = {
    "lead": 1.0,
    "choeurs": 0.9,
    "piano": 1.0,
    "batterie": 0.8,
    "basse": 1.0,
    "guitare": 1.0,
    "autre": 0.5,
}


def _cle(nom: str) -> str:
    return " ".join(str(nom).strip().lower().replace("œ", "oe").replace("_", " ").replace("-", " ").split())


def charger_config(chemin: Path | None) -> dict[str, float]:
    """Renvoie un facteur de volume (linéaire) par piste.

    1.0 = inchangé, 0.5 = moitié (-6 dB), 2.0 = double (+6 dB), 0 = muet.
    Les pistes absentes du fichier restent à 1.0. « voix » règle à la fois le
    lead et les chœurs, sauf si l'un d'eux a son propre réglage.
    """
    gains = {piste: 1.0 for piste in PISTES}
    if chemin is None:
        return gains
    donnees = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if not isinstance(donnees, dict):
        raise ValueError(f"{chemin} : un objet JSON {{\"instrument\": nombre}} est attendu")
    reglages: dict[str, float] = {}
    for cle, valeur in donnees.items():
        piste = ALIAS.get(_cle(cle))
        if piste is None:
            log.warning("%s : instrument inconnu « %s » ignoré (connus : %s)", chemin, cle, ", ".join(PISTES))
            continue
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)) or valeur < 0:
            raise ValueError(f"{chemin} : « {cle} » doit être un nombre positif ou nul, lu {valeur!r}")
        reglages[piste] = float(valeur)
    if "voix" in reglages:
        for piste in ("voix", "lead", "choeurs"):
            gains[piste] = reglages["voix"]
    gains.update({piste: valeur for piste, valeur in reglages.items() if piste != "voix"})
    return gains


def generer_config_exemple(chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(CONFIG_EXEMPLE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
