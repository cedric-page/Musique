"""Séparation de la piste voix en deux pistes : la voix **lead** et les **chœurs**.

Demucs isole les voix ensemble, sans les distinguer. Un modèle « karaoké » de
la bibliothèque audio-separator (Mel-Roformer Karaoke par défaut), appliqué à
cette piste voix, en extrait la voix lead ; le reste forme les chœurs.

Les modèles « BVE » (Backing Vocal Extractor) peuvent aussi être choisis, mais
ils reconnaissent les chœurs à leur écho : dans une prise de salle, où toutes
les voix ont de l'écho, ils rangent tout dans les chœurs.

Si le modèle ne trouve pas de voix lead nette (lead plus de 10 dB sous les
chœurs sur toute la chanson), la séparation est jugée ratée et la voix reste en
une seule piste. Le résultat est gardé sur disque : une relance ne refait la
séparation que si la piste voix ou le modèle a changé.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from . import audio_io

log = logging.getLogger(__name__)

MODELE_PAR_DEFAUT = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
ECART_LEAD_MIN_DB = -10.0
"""En dessous (lead par rapport aux chœurs, sur toute la chanson), la séparation est jugée ratée."""
FICHIER_META = "separation_voix.json"
DOSSIER_MODELES = Path.home() / ".cache" / "musique-repet" / "modeles"


class SeparateurVoix(Protocol):
    identifiant: str

    def separer(self, voix: Path) -> tuple[np.ndarray, np.ndarray, int]:
        """(lead, chœurs, fréquence d'échantillonnage) à partir du fichier de la piste voix."""
        ...


class SeparateurLeadChoeurs:
    """Sépare lead et chœurs avec un modèle d'audio-separator, téléchargé au premier usage
    dans ``~/.cache/musique-repet/modeles``.

    Un modèle « karaoké » extrait la voix lead : sa piste « Vocals » est le lead,
    l'autre les chœurs. Un modèle « BVE » extrait les chœurs : sa piste « Vocals »
    contient les chœurs, l'autre le lead.
    """

    def __init__(self, modele: str = MODELE_PAR_DEFAUT, dossier_modeles: Path = DOSSIER_MODELES) -> None:
        self.modele = modele
        self.dossier_modeles = dossier_modeles
        self.identifiant = f"audio-separator:{modele}"
        self.choeurs_dans_vocals = "bve" in modele.lower()
        self._separateur = None
        self._sortie: tempfile.TemporaryDirectory | None = None

    def _charger(self):
        if self._separateur is None:
            from audio_separator.separator import Separator

            self._sortie = tempfile.TemporaryDirectory(prefix="musique-repet-voix-")
            self.dossier_modeles.mkdir(parents=True, exist_ok=True)
            log.info("Chargement du modèle de séparation lead / chœurs « %s »", self.modele)
            self._separateur = Separator(
                log_level=logging.ERROR,  # ses messages d'information encombrent la sortie
                model_file_dir=str(self.dossier_modeles),
                output_dir=self._sortie.name,
                output_format="WAV",
                normalization_threshold=1.0,  # ne pas renormaliser chaque piste : garder l'équilibre lead / chœurs
                use_soundfile=True,
            )
            self._separateur.load_model(self.modele)
        return self._separateur

    def separer(self, voix: Path) -> tuple[np.ndarray, np.ndarray, int]:
        separateur = self._charger()
        dossier = Path(self._sortie.name)
        for ancien in dossier.iterdir():
            ancien.unlink()
        separateur.separate(str(voix))
        fichiers = sorted(dossier.iterdir())
        vocals = [f for f in fichiers if "(vocals)" in f.name.lower()]
        autres = [f for f in fichiers if f not in vocals]
        if len(vocals) != 1 or len(autres) != 1:
            raise RuntimeError(
                f"le modèle « {self.modele} » n'a pas produit une piste « Vocals » et une autre : "
                + ", ".join(f.name for f in fichiers)
            )
        fichier_lead, fichier_choeurs = (autres[0], vocals[0]) if self.choeurs_dans_vocals else (vocals[0], autres[0])
        audio_lead, sr = sf.read(fichier_lead, dtype="float32", always_2d=True)
        audio_choeurs, sr_choeurs = sf.read(fichier_choeurs, dtype="float32", always_2d=True)
        for f in fichiers:
            f.unlink()
        if sr_choeurs != sr:
            raise RuntimeError(f"fréquences différentes entre lead et chœurs ({sr} / {sr_choeurs})")
        return audio_lead, audio_choeurs, sr


def _ajuster(audio: np.ndarray, longueur: int, canaux: int) -> np.ndarray:
    """Même longueur et même nombre de canaux que la piste voix d'origine."""
    if audio.shape[1] != canaux:
        audio = np.repeat(audio.mean(axis=1, keepdims=True), canaux, axis=1)
    if audio.shape[0] < longueur:
        audio = np.pad(audio, ((0, longueur - audio.shape[0]), (0, 0)))
    return audio[:longueur]


def _niveau_db(audio: np.ndarray) -> float:
    return float(10 * np.log10(float(np.mean(np.square(audio, dtype=np.float64))) + 1e-20))


def _position_choeurs(ecart_db: float) -> str:
    """Niveau des chœurs par rapport au lead, pour le journal (``ecart_db`` = lead - chœurs)."""
    if ecart_db > 40:
        return "quasi muets (pas de chœurs dans ce morceau ?)"
    if abs(ecart_db) < 0.5:
        return "au même niveau que le lead"
    position = "sous le" if ecart_db > 0 else "au-dessus du"
    return f"{abs(ecart_db):.0f} dB {position} lead"


def separer_voix_avec_cache(
    pistes: dict[str, Path], dossier: Path, separateur: SeparateurVoix | None, forcer: bool = False
) -> dict[str, Path]:
    """Remplace la piste ``voix`` par les pistes ``lead`` et ``choeurs``.

    Inchangé sans séparateur, si la piste voix est muette, ou si le modèle ne trouve
    pas de voix lead nette (la voix reste alors en une seule piste).
    """
    if separateur is None or "voix" not in pistes:
        return pistes
    voix = pistes["voix"]
    signature = {"source": [voix.stat().st_size, voix.stat().st_mtime_ns], "separateur": separateur.identifiant}
    resultat = {nom: chemin for nom, chemin in pistes.items() if nom != "voix"}
    resultat["lead"], resultat["choeurs"] = dossier / "lead.wav", dossier / "choeurs.wav"
    meta = dossier / FICHIER_META
    if not forcer:
        try:
            donnees = json.loads(meta.read_text(encoding="utf-8"))
            if donnees.get("signature") == signature:
                if not donnees.get("separee"):
                    return pistes
                if all(resultat[n].exists() for n in ("lead", "choeurs")):
                    log.info("  Voix lead et chœurs déjà séparés, réutilisation de %s", dossier)
                    if "ecart_db" in donnees:
                        log.info("  Chœurs : %s", _position_choeurs(donnees["ecart_db"]))
                    return resultat
        except (OSError, json.JSONDecodeError):
            pass

    audio_voix, _ = audio_io.lire(voix)
    if _niveau_db(audio_voix) < -70:
        return pistes  # personne ne chante
    del audio_voix
    log.info("  Séparation de la voix lead et des chœurs")
    reference = sf.info(voix)
    lead, choeurs, sr = separateur.separer(voix)
    if sr != reference.samplerate:
        raise RuntimeError(f"la séparation lead / chœurs a changé la fréquence ({reference.samplerate} -> {sr})")
    dossier.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    ecart = _niveau_db(lead) - _niveau_db(choeurs)
    separee = bool(ecart >= ECART_LEAD_MIN_DB)
    if separee:
        for nom, audio in (("lead", lead), ("choeurs", choeurs)):
            audio_io.ecrire_wav(
                resultat[nom], _ajuster(audio, reference.frames, reference.channels), sr, sous_type="FLOAT"
            )
        log.info("  Chœurs : %s", _position_choeurs(ecart))
    else:
        log.warning(
            "  Pas de voix lead nette (%.0f dB sous les chœurs) : la voix reste en une seule piste "
            "(un autre modèle peut être essayé avec --modele-choeurs)",
            -ecart,
        )
    # Écrit en dernier : le cache n'est valide que si les pistes sont sur disque.
    meta.write_text(
        json.dumps({"signature": signature, "separee": separee, "ecart_db": round(ecart, 1)}, indent=2),
        encoding="utf-8",
    )
    return resultat if separee else pistes
