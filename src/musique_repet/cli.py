"""Interface en ligne de commande."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import charger_config, generer_config_exemple
from .effets import ParametresChorus
from .mixage import ParametresMixage
from .pipeline import Options, executer
from .separation import MODELE_PAR_DEFAUT, SeparateurDemucs


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
        "--chorus",
        type=float,
        default=ParametresChorus.melange,
        metavar="DOSAGE",
        help=f"dosage du chorus sur les voix, 0 pour le désactiver (défaut : {ParametresChorus.melange})",
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
            chorus={"voix": ParametresChorus(melange=args.chorus)} if args.chorus > 0 else {},
        ),
        debit_kbps=args.debit,
        forcer_separation=args.forcer_separation,
    )
    separateur = SeparateurDemucs(args.modele, appareil=args.appareil, shifts=args.shifts)
    echecs = executer(options, separateur)
    if echecs:
        log.error("%d élément(s) en échec", echecs)
        return 1
    log.info("Terminé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
