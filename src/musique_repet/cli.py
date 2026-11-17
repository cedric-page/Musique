"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import charger_config, generer_config_exemple
from .echo import ParametresAntiEcho
from .effets import CHORUS_BASSE, ParametresChorus
from .mixage import ParametresMixage
from .pipeline import Options, executer
from .paroles import TranscripteurWhisper
from .separation import MODELE_PAR_DEFAUT, SeparateurDemucs
from .tempo import ParametresTempo
from .titre import ParametresTitre, lire_repertoire


def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="musique-repet",
        description=(
            "Découpe des enregistrements de répétition (CSV titre;debut;fin), sépare les "
            "instruments, équilibre les volumes, applique la configuration et exporte en MP3."
        ),
    )
    p.add_argument("entree", type=Path, nargs="?", help="dossier contenant les audios (et les CSV)")
    p.add_argument("sortie", type=Path, nargs="?", help="dossier où écrire les MP3")
    p.add_argument("-c", "--config", type=Path, help='fichier JSON {"instrument": facteur} (1.0 = inchangé)')
    p.add_argument(
        "--generer-config",
        type=Path,
        metavar="FICHIER",
        help="écrit un fichier de configuration d'exemple puis quitte",
    )
    p.add_argument(
        "--travail",
        type=Path,
        help="dossier des fichiers intermédiaires (découpes, pistes séparées). Défaut : <sortie>/_travail",
    )
    p.add_argument("--modele", default=MODELE_PAR_DEFAUT, help=f"modèle Demucs (défaut : {MODELE_PAR_DEFAUT})")
    p.add_argument("--appareil", choices=["cpu", "cuda", "mps"], help="calcul de la séparation (défaut : auto)")
    p.add_argument("--shifts", type=int, default=1, help="passes de séparation, plus = meilleur mais plus lent")
    p.add_argument("--forcer-separation", action="store_true", help="refait la séparation même si elle est en cache")
    p.add_argument("--debit", type=int, default=192, help="débit MP3 en kb/s (défaut : 192)")
    p.add_argument("--lufs-cible", type=float, default=-14.0, help="sonie du MP3 final (défaut : -14 LUFS)")
    p.add_argument("--gain-max-db", type=float, default=12.0, help="correction max. de l'équilibrage (défaut : 12 dB)")
    p.add_argument(
        "--exclure-equilibrage",
        nargs="*",
        default=["autre"],
        metavar="PISTE",
        help="pistes laissées à leur niveau d'origine par l'équilibrage (défaut : autre)",
    )
    p.add_argument(
        "--anti-echo",
        type=float,
        default=ParametresAntiEcho.attenuation_max_db,
        metavar="DB",
        help=(
            "atténuation max. de l'écho de la salle sur chaque piste, 0 pour désactiver "
            f"(défaut : {ParametresAntiEcho.attenuation_max_db:g} dB)"
        ),
    )
    p.add_argument(
        "--rt60",
        type=float,
        default=ParametresAntiEcho.rt60,
        metavar="SECONDES",
        help=f"durée de réverbération de la salle (défaut : {ParametresAntiEcho.rt60:g} s)",
    )
    p.add_argument(
        "--chorus",
        type=float,
        default=ParametresChorus.melange,
        metavar="DOSAGE",
        help=f"dosage du chorus sur les voix, 0 pour le désactiver (défaut : {ParametresChorus.melange})",
    )
    p.add_argument(
        "--chorus-basse",
        type=float,
        default=CHORUS_BASSE.melange,
        metavar="DOSAGE",
        help=(
            f"dosage du chorus sur la basse (au-dessus de {CHORUS_BASSE.coupure_hz:g} Hz), "
            f"0 pour le désactiver (défaut : {CHORUS_BASSE.melange})"
        ),
    )
    p.add_argument(
        "--sans-rognage",
        action="store_true",
        help="sans CSV : garde tout l'enregistrement au lieu de couper ce qui précède et suit la chanson",
    )
    p.add_argument(
        "--sans-titre",
        action="store_true",
        help="sans CSV : garde le nom du fichier au lieu de reconnaître le titre d'après les paroles",
    )
    p.add_argument(
        "--repertoire",
        type=Path,
        metavar="FICHIER",
        help="liste des titres du groupe, un par ligne (défaut : repertoire.txt du dossier d'entrée s'il existe)",
    )
    p.add_argument(
        "--modele-paroles",
        default="small",
        metavar="MODELE",
        help="modèle Whisper de transcription : tiny, base, small, medium, large-v3 (défaut : small)",
    )
    p.add_argument("--sans-tempo", action="store_true", help="ne produit pas le graphique du tempo (JPG)")
    p.add_argument(
        "--bpm-min",
        type=float,
        default=ParametresTempo.bpm_min,
        help=f"tempo minimal recherché (défaut : {ParametresTempo.bpm_min:g})",
    )
    p.add_argument(
        "--bpm-max",
        type=float,
        default=ParametresTempo.bpm_max,
        help=(
            f"tempo maximal recherché (défaut : {ParametresTempo.bpm_max:g}). Le baisser si une ballade "
            "est mesurée au double de son tempo"
        ),
    )
    p.add_argument("-v", "--verbeux", action="store_true", help="affiche plus de détails")
    return p


def main(argv: list[str] | None = None) -> int:
    parseur = construire_parseur()
    args = parseur.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbeux else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("musique_repet")

    if args.generer_config:
        generer_config_exemple(args.generer_config)
        log.info("Configuration d'exemple écrite dans %s", args.generer_config)
        if args.entree is None:
            return 0

    if args.entree is None or args.sortie is None:
        parseur.error("les dossiers d'entrée et de sortie sont obligatoires")
    if not args.entree.is_dir():
        parseur.error(f"dossier d'entrée introuvable : {args.entree}")

    if args.rt60 <= 0:
        parseur.error("--rt60 doit être positif")
    if not 30 <= args.bpm_min < args.bpm_max <= 400:
        parseur.error("il faut 30 <= --bpm-min < --bpm-max <= 400")
    try:
        facteurs = charger_config(args.config)
    except (OSError, ValueError) as exc:
        parseur.error(str(exc))

    options = Options(
        entree=args.entree,
        sortie=args.sortie,
        travail=args.travail or args.sortie / "_travail",
        facteurs=facteurs,
        mixage=ParametresMixage(
            gain_max_db=args.gain_max_db,
            exclues_equilibrage=set(args.exclure_equilibrage),
            lufs_cible=args.lufs_cible,
            chorus={
                nom: params
                for nom, params in (
                    ("voix", ParametresChorus(melange=args.chorus)),
                    ("basse", replace(CHORUS_BASSE, melange=args.chorus_basse)),
                )
                if params.melange > 0
            },
        ),
        debit_kbps=args.debit,
        anti_echo=ParametresAntiEcho(rt60=args.rt60, attenuation_max_db=args.anti_echo),
        tempo=None if args.sans_tempo else ParametresTempo(bpm_min=args.bpm_min, bpm_max=args.bpm_max),
        forcer_separation=args.forcer_separation,
    )
    if args.sans_rognage:
        options.rognage = None
    transcripteur = None
    if args.sans_titre:
        options.titres = None
    else:
        repertoire = args.repertoire or args.entree / "repertoire.txt"
        if args.repertoire and not repertoire.is_file():
            parseur.error(f"répertoire introuvable : {repertoire}")
        options.titres = ParametresTitre(repertoire=lire_repertoire(repertoire) if repertoire.is_file() else [])
        if options.titres.repertoire:
            log.info("Répertoire : %d titre(s) lus dans %s", len(options.titres.repertoire), repertoire)
        # ctranslate2 (Whisper) ne gère pas « mps » : calcul sur processeur dans ce cas.
        transcripteur = TranscripteurWhisper(
            args.modele_paroles, appareil=args.appareil if args.appareil in ("cpu", "cuda") else None
        )
    separateur = SeparateurDemucs(args.modele, appareil=args.appareil, shifts=args.shifts)
    echecs = executer(options, separateur, transcripteur)
    if echecs:
        log.error("%d élément(s) en échec", echecs)
        return 1
    log.info("Terminé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
