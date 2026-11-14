"""Enchaînement complet : découpe -> séparation -> équilibrage -> volumes -> MP3."""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import audio_io
from .decoupe import Morceau, lire_csv, nom_fichier_sur
from .mixage import ParametresMixage, ReglagePiste, mixer
from .separation import Separateur, separer_avec_cache

log = logging.getLogger(__name__)


@dataclass
class Tache:
    """Un extrait à traiter : un morceau du CSV ou l'enregistrement complet."""

    nom_affiche: str
    wav: Path
    dossier_pistes: Path
    mp3: Path


@dataclass
class Options:
    entree: Path
    sortie: Path
    travail: Path
    facteurs: dict[str, float]
    mixage: ParametresMixage
    debit_kbps: int = 192
    forcer_separation: bool = False


def _cles_sources(fichiers: list[Path]) -> dict[Path, str]:
    """Nom de base de chaque fichier ; l'extension est ajoutée en cas de doublon (a.wav / a.mp3)."""
    compte = Counter(f.stem for f in fichiers)
    return {
        f: f.stem if compte[f.stem] == 1 else f"{f.stem}_{f.suffix.lstrip('.').lower()}"
        for f in fichiers
    }


def date_creation(chemin: Path) -> str:
    """Date de création du fichier au format AAAAMMJJ (heure locale).

    Utilise la vraie date de création quand le système la fournit (macOS,
    Windows, certains systèmes de fichiers Linux avec Python >= 3.12), sinon
    la date de dernière modification.
    """
    st = chemin.stat()
    horodatage = getattr(st, "st_birthtime", None)
    if horodatage is None and sys.platform == "win32":
        horodatage = st.st_ctime  # date de création sous Windows
    if horodatage is None:
        horodatage = st.st_mtime
    return datetime.fromtimestamp(horodatage).strftime("%Y%m%d")


def _nom_libre(chemin: Path, pris: set[str]) -> Path:
    """Ajoute « (2) », « (3) »... si un autre extrait a déjà ce nom de sortie."""
    candidat, n = chemin, 1
    while candidat.name.lower() in pris:
        n += 1
        candidat = chemin.with_name(f"{chemin.stem} ({n}){chemin.suffix}")
    pris.add(candidat.name.lower())
    return candidat


def preparer_taches(source: Path, cle: str, options: Options) -> list[Tache]:
    """Découpe la source selon son CSV (s'il existe) et renvoie les extraits à traiter."""
    lisible = audio_io.source_lisible(source, options.travail / "conversions")
    date = date_creation(source)
    csv = source.with_suffix(".csv")
    if not csv.exists():
        csv = next((p for p in source.parent.glob(f"{source.stem}.*") if p.suffix.lower() == ".csv"), csv)

    if not csv.exists():
        if lisible.suffix.lower() == ".wav":
            wav = lisible
        else:
            wav = options.travail / "decoupes" / f"{cle}.wav"
            audio, sr = audio_io.lire(lisible)
            audio_io.ecrire_wav(wav, audio, sr)
        log.info("%s : pas de CSV, traitement de l'enregistrement complet", source.name)
        return [Tache(source.name, wav, options.travail / "pistes" / cle, options.sortie / f"{date}-{cle}.mp3")]

    morceaux: list[Morceau] = lire_csv(csv, duree=audio_io.duree(lisible))
    if not morceaux:
        log.warning("%s : aucun morceau valide dans %s, rien à produire", source.name, csv.name)
    log.info("%s : %d morceau(x) dans %s", source.name, len(morceaux), csv.name)
    taches = []
    noms_vus: set[str] = set()
    for morceau in morceaux:
        nom = nom_fichier_sur(morceau.titre)
        base, n = nom, 1
        while nom.lower() in noms_vus:
            n += 1
            nom = f"{base} ({n})"
        noms_vus.add(nom.lower())
        wav = options.travail / "decoupes" / cle / f"{nom}.wav"
        audio, sr = audio_io.lire(lisible, morceau.debut, morceau.fin)
        audio_io.ecrire_wav(wav, audio, sr)
        taches.append(
            Tache(
                f"{source.name} › {morceau.titre}",
                wav,
                options.travail / "pistes" / cle / nom,
                options.sortie / f"{date}-{nom}.mp3",
            )
        )
    return taches


def _journaliser_reglages(reglages: list[ReglagePiste]) -> None:
    for r in reglages:
        lufs = f"{r.lufs:6.1f} LUFS" if math.isfinite(r.lufs) else "   muet    "
        log.info(
            "    %-9s %s  équilibrage %+5.1f dB  × config %.2f",
            r.nom,
            lufs,
            r.gain_equilibrage_db,
            r.facteur_config,
        )


def traiter_tache(tache: Tache, separateur: Separateur, options: Options) -> None:
    log.info("▶ %s", tache.nom_affiche)
    pistes = separer_avec_cache(tache.wav, tache.dossier_pistes, separateur, options.forcer_separation)
    mix, sr, reglages = mixer(pistes, options.facteurs, options.mixage)
    _journaliser_reglages(reglages)
    audio_io.ecrire_mp3(tache.mp3, mix, sr, options.debit_kbps)
    log.info("  ✔ %s", tache.mp3)


def executer(options: Options, separateur: Separateur) -> int:
    """Traite tous les audios du dossier d'entrée. Renvoie le nombre d'échecs."""
    sources = audio_io.lister_audios(options.entree)
    if not sources:
        log.warning("Aucun fichier audio dans %s", options.entree)
        return 0
    options.sortie.mkdir(parents=True, exist_ok=True)
    cles = _cles_sources(sources)
    echecs = 0
    noms_pris: set[str] = set()
    for source in sources:
        try:
            taches = preparer_taches(source, cles[source], options)
        except Exception:
            log.exception("Échec de la préparation de %s", source.name)
            echecs += 1
            continue
        for tache in taches:
            tache.mp3 = _nom_libre(tache.mp3, noms_pris)
            try:
                traiter_tache(tache, separateur, options)
            except Exception:
                log.exception("Échec du traitement de %s", tache.nom_affiche)
                echecs += 1
    return echecs
