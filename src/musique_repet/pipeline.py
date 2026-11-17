"""Enchaînement complet : découpe -> séparation -> anti-écho -> (sans CSV : repérage de la
chanson et reconnaissance du titre, façon Shazam puis d'après les paroles) -> équilibrage ->
volumes -> MP3, puis graphique du tempo."""

from __future__ import annotations

import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import audio_io
from .chanson import ParametresRognage, detecter_chanson
from .decoupe import Morceau, lire_csv, nom_fichier_sur
from .echo import ParametresAntiEcho, attenuer_echo_avec_cache
from .graphique_tempo import tracer as tracer_tempo
from .mixage import ParametresMixage, ReglagePiste, fondus, mixer
from .empreinte import ReconnaisseurAudio, identifier
from .paroles import Transcripteur
from .separation import Separateur, separer_avec_cache
from .tempo import ParametresTempo, analyser_pistes
from .titre import ParametresTitre, reconnaitre_titre
from .voix import SeparateurVoix, separer_voix_avec_cache

log = logging.getLogger(__name__)


@dataclass
class Tache:
    """Un extrait à traiter : un morceau du CSV ou l'enregistrement complet."""

    nom_affiche: str
    wav: Path
    dossier_pistes: Path
    mp3: Path
    titre: str = ""
    date: str = ""
    """Date de l'enregistrement d'origine, AAAAMMJJ."""
    sans_csv: bool = False
    """Enregistrement complet : la chanson y est repérée et son titre reconnu."""


@dataclass
class Options:
    entree: Path
    sortie: Path
    travail: Path
    facteurs: dict[str, float]
    mixage: ParametresMixage
    debit_kbps: int = 192
    forcer_separation: bool = False
    anti_echo: ParametresAntiEcho = field(default_factory=ParametresAntiEcho)
    tempo: ParametresTempo | None = field(default_factory=ParametresTempo)
    """Réglages de l'analyse du tempo ; ``None`` pour ne pas produire de graphique."""
    rognage: ParametresRognage | None = field(default_factory=ParametresRognage)
    """Sans CSV : retrait de ce qui précède et suit la chanson ; ``None`` pour tout garder."""
    titres: ParametresTitre | None = field(default_factory=ParametresTitre)
    """Sans CSV : reconnaissance du titre ; ``None`` pour garder le nom du fichier."""


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
        return [
            Tache(
                source.name, wav, options.travail / "pistes" / cle, options.sortie / f"{date}-{cle}.mp3",
                titre=source.stem, date=date, sans_csv=True,
            )
        ]

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
                titre=morceau.titre,
                date=date,
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


def _reconnaitre(
    tache: Tache,
    pistes: dict[str, Path],
    zone: tuple[float, float] | None,
    options: Options,
    transcripteur: Transcripteur | None,
    reconnaisseur: ReconnaisseurAudio | None,
) -> None:
    """Renomme la tâche d'après le titre reconnu (le nom du fichier est gardé sinon).

    D'abord par empreinte audio (façon Shazam), puis d'après les paroles.
    """
    debut, fin = zone or (0.0, audio_io.duree(tache.wav))
    trouve = None
    if reconnaisseur is not None:
        try:
            trouve = identifier(
                tache.wav, debut, fin, reconnaisseur, tache.dossier_pistes / "empreinte.json",
                options.titres.repertoire,
            )
        except Exception as exc:  # pas de connexion, service indisponible...
            log.warning("  Reconnaissance par %s impossible (%s)", reconnaisseur.identifiant.capitalize(), exc)
            log.debug("  Détail :", exc_info=True)
        if trouve is None and transcripteur is not None:
            log.info(
                "  Pas de réponse sûre de %s : reconnaissance d'après les paroles", reconnaisseur.identifiant.capitalize()
            )
    # Les paroles se transcrivent mieux sur la voix lead seule que sur toutes les voix.
    chant = pistes.get("lead") or pistes.get("voix")
    if trouve is None and transcripteur is not None and chant is not None:
        try:
            trouve = reconnaitre_titre(chant, debut, fin, tache.dossier_pistes, transcripteur, options.titres)
        except Exception as exc:  # modèle introuvable, pas de connexion au premier usage...
            log.warning("  Reconnaissance d'après les paroles impossible (%s)", exc)
            log.debug("  Détail :", exc_info=True)
    if trouve is None:
        log.info("  Titre non reconnu : le nom du fichier est gardé")
        return
    log.info("  Titre reconnu (%s) : « %s » — %s", trouve.methode, trouve.titre, trouve.detail)
    tache.titre = trouve.titre
    tache.mp3 = tache.mp3.with_name(f"{tache.date}-{nom_fichier_sur(trouve.titre)}.mp3")


def traiter_tache(
    tache: Tache,
    separateur: Separateur,
    options: Options,
    transcripteur: Transcripteur | None = None,
    noms_pris: set[str] | None = None,
    reconnaisseur: ReconnaisseurAudio | None = None,
    separateur_voix: SeparateurVoix | None = None,
) -> None:
    log.info("▶ %s", tache.nom_affiche)
    pistes = separer_avec_cache(tache.wav, tache.dossier_pistes, separateur, options.forcer_separation)
    pistes = separer_voix_avec_cache(
        pistes, tache.dossier_pistes / "lead_choeurs", separateur_voix, options.forcer_separation
    )
    pistes = attenuer_echo_avec_cache(pistes, tache.dossier_pistes / "sans_echo", options.anti_echo)

    zone = None
    extrait = None
    if tache.sans_csv and options.rognage is not None:
        extrait = detecter_chanson(pistes, options.rognage)
        log.info("  %s", extrait.description()[:1].upper() + extrait.description()[1:])
        if extrait.rogne:
            zone = (extrait.debut, extrait.fin)
    if tache.sans_csv and options.titres is not None and (transcripteur or reconnaisseur) is not None:
        _reconnaitre(tache, pistes, zone, options, transcripteur, reconnaisseur)
    if noms_pris is not None:
        tache.mp3 = _nom_libre(tache.mp3, noms_pris)

    mix, sr, reglages = mixer(pistes, options.facteurs, options.mixage, zone)
    if extrait is not None and zone is not None:
        mix = fondus(
            mix,
            sr,
            entree_s=0.1 if extrait.debut > 0 else 0.0,
            sortie_s=1.0 if extrait.fin < extrait.duree_totale else 0.0,
        )
    _journaliser_reglages(reglages)
    audio_io.ecrire_mp3(tache.mp3, mix, sr, options.debit_kbps)
    log.info("  ✔ %s", tache.mp3)
    if options.tempo is not None:
        analyse = analyser_pistes(pistes, options.tempo, zone)
        log.info("  Tempo : %s", analyse.resume())
        graphique = tache.mp3.with_name(f"{tache.mp3.stem}-tempo.jpg")
        tracer_tempo(analyse, graphique, tache.titre or tache.mp3.stem, tache.date or None)
        log.info("  ✔ %s", graphique)


def executer(
    options: Options,
    separateur: Separateur,
    transcripteur: Transcripteur | None = None,
    reconnaisseur: ReconnaisseurAudio | None = None,
    separateur_voix: SeparateurVoix | None = None,
) -> int:
    """Traite tous les audios du dossier d'entrée. Renvoie le nombre d'échecs.

    ``separateur_voix`` : sépare la piste voix en voix lead et chœurs (``None`` :
    une seule piste voix).

    Pour reconnaître le titre des enregistrements sans CSV : ``reconnaisseur``
    (empreinte audio, façon Shazam) puis ``transcripteur`` (paroles). Sans l'un
    ni l'autre, le nom du fichier est gardé.
    """
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
            try:
                traiter_tache(tache, separateur, options, transcripteur, noms_pris, reconnaisseur, separateur_voix)
            except Exception:
                log.exception("Échec du traitement de %s", tache.nom_affiche)
                echecs += 1
    return echecs
