"""Fichier de configuration des volumes : ``{"instrument": facteur}``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .separation import NOMS_PISTES

log = logging.getLogger(__name__)

PISTES = list(dict.fromkeys(NOMS_PISTES.values()))

# Noms alternatifs acceptés dans la configuration.
ALIAS = {
    **{anglais: francais for anglais, francais in NOMS_PISTES.items()},
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
    "voix": 1.0,
    "piano": 1.0,
    "batterie": 0.8,
    "basse": 1.0,
    "guitare": 1.0,
    "autre": 0.5,
}


def charger_config(chemin: Path | None) -> dict[str, float]:
    """Renvoie un facteur de volume (linéaire) par piste.

    1.0 = inchangé, 0.5 = moitié (-6 dB), 2.0 = double (+6 dB), 0 = muet.
    Les pistes absentes du fichier restent à 1.0.
    """
    gains = {piste: 1.0 for piste in PISTES}
    if chemin is None:
        return gains
    donnees = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if not isinstance(donnees, dict):
        raise ValueError(f"{chemin} : un objet JSON {{\"instrument\": nombre}} est attendu")
    for cle, valeur in donnees.items():
        piste = ALIAS.get(str(cle).strip().lower())
        if piste is None:
            log.warning("%s : instrument inconnu « %s » ignoré (connus : %s)", chemin, cle, ", ".join(PISTES))
            continue
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)) or valeur < 0:
            raise ValueError(f"{chemin} : « {cle} » doit être un nombre positif ou nul, lu {valeur!r}")
        gains[piste] = float(valeur)
    return gains


def generer_config_exemple(chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(CONFIG_EXEMPLE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
