"""Transcription des paroles chantées (et des échanges parlés) de la piste voix séparée.

Whisper (faster-whisper) tourne en local ; le modèle est téléchargé au premier
usage. Le filtre de détection de parole de Whisper écarte souvent le chant :
les passages à transcrire sont donc repérés d'après le niveau de la piste voix
séparée, puis seuls ces passages sont transcrits.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import audio_io

log = logging.getLogger(__name__)

SR_TRANSCRIPTION = 16000

# Phrases que Whisper invente sur de la musique ou du silence (sous-titres de vidéos).
HALLUCINATIONS = re.compile(
    r"amara|sous.?titr|merci d.avoir regard|abonnez.vous|thanks? (?:you )?for watching|subtitles? by|"
    r"transcri(?:bed|ption) by|like and subscribe|www\.|\.com\b",
    re.IGNORECASE,
)
SEULEMENT_BRUIT = re.compile(r"^[\s\W]*(?:musique|music|applaudissements|applause|rires|laughter)?[\s\W]*$", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentParoles:
    """Une phrase transcrite ; temps en secondes depuis le début de l'enregistrement."""

    debut: float
    fin: float
    texte: str


class Transcripteur(Protocol):
    """Tout objet capable de transcrire des passages d'une piste voix."""

    identifiant: str

    def transcrire(self, audio: np.ndarray, sr: int, zones: list[tuple[float, float]]) -> list[SegmentParoles]:
        """``audio`` : mono à 16 kHz ; ``zones`` : passages à transcrire (s, depuis le début de ``audio``)."""
        ...


def est_hallucination(texte: str) -> bool:
    sans_notes = texte.replace("♪", "").replace("♫", "").strip(" .…-")
    return bool(SEULEMENT_BRUIT.match(sans_notes)) or bool(HALLUCINATIONS.search(sans_notes))


class TranscripteurWhisper:
    """Transcription avec faster-whisper, sur processeur (int8) ou carte graphique (float16)."""

    def __init__(self, modele: str = "small", appareil: str | None = None) -> None:
        self.modele_nom = modele
        self.appareil = appareil
        self._modele = None
        self.identifiant = f"faster-whisper:{modele}"

    def _charger(self):
        if self._modele is None:
            from faster_whisper import WhisperModel

            appareil = self.appareil
            if appareil is None:
                import ctranslate2

                appareil = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            log.info("Chargement du modèle de transcription Whisper « %s » sur %s", self.modele_nom, appareil)
            try:
                self._modele = WhisperModel(
                    self.modele_nom, device=appareil, compute_type="float16" if appareil == "cuda" else "int8"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"modèle Whisper « {self.modele_nom} » indisponible ; il est téléchargé au premier usage "
                    f"depuis huggingface.co ({exc})"
                ) from exc
        return self._modele

    def transcrire(self, audio: np.ndarray, sr: int, zones: list[tuple[float, float]]) -> list[SegmentParoles]:
        if sr != SR_TRANSCRIPTION:
            raise ValueError(f"fréquence attendue : {SR_TRANSCRIPTION} Hz")
        if not zones:
            return []
        segments, info = self._charger().transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            beam_size=5,
            vad_filter=False,  # les zones viennent de la piste voix (le filtre écarte souvent le chant)
            clip_timestamps=[t for zone in zones for t in zone],
            condition_on_previous_text=False,  # limite les boucles de répétition inventées
        )
        resultat = []
        for s in segments:
            texte = s.text.strip()
            peu_fiable = s.no_speech_prob > 0.6 and s.avg_logprob < -1.0
            if peu_fiable or s.compression_ratio > 2.4 or est_hallucination(texte):
                continue
            resultat.append(SegmentParoles(float(s.start), float(s.end), texte))
        log.info("  Paroles : %d phrase(s) transcrite(s), langue « %s »", len(resultat), info.language)
        return resultat


def zones_de_voix(
    mono: np.ndarray, sr: int, pas_s: float = 0.25, ecart_db: float = 30.0, pause_max_s: float = 2.0
) -> list[tuple[float, float]]:
    """Passages où la voix est présente (chant ou parole), pauses de moins de 2 s comprises."""
    taille = max(1, int(round(pas_s * sr)))
    nb = mono.size // taille
    if nb == 0:
        return []
    puissance = (mono[: nb * taille].astype(np.float64).reshape(nb, taille) ** 2).mean(axis=1)
    niveaux = 10 * np.log10(puissance + 1e-12)
    fort = float(np.percentile(niveaux, 95))
    if fort < -60:
        return []
    actif = niveaux >= max(fort - ecart_db, -60.0)
    indices = np.flatnonzero(actif)
    zones: list[list[float]] = []
    for i in indices:
        debut, fin = i * pas_s, (i + 1) * pas_s
        if zones and debut - zones[-1][1] <= pause_max_s:
            zones[-1][1] = fin
        else:
            zones.append([debut, fin])
    duree = mono.size / sr
    return [
        (round(max(0.0, d - 0.3), 2), round(min(duree, f + 0.3), 2)) for d, f in zones if f - d >= 0.5
    ]


def transcrire_zone(
    voix: Path, debut: float, fin: float, cache: Path, transcripteur: Transcripteur
) -> list[SegmentParoles]:
    """Transcrit la piste ``voix`` entre ``debut`` et ``fin`` (s), en réutilisant un résultat déjà calculé.

    Chaque zone est transcrite séparément : la langue est détectée pour la
    chanson d'un côté, pour la discussion qui la précède de l'autre.
    """
    signature = {
        "transcripteur": transcripteur.identifiant,
        "source": [voix.stat().st_size, voix.stat().st_mtime_ns],
        "zone": [round(debut, 2), round(fin, 2)],
    }
    try:
        donnees = json.loads(cache.read_text(encoding="utf-8"))
        if donnees.get("signature") == signature:
            return [SegmentParoles(**s) for s in donnees["segments"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass

    import librosa

    audio, sr = audio_io.lire(voix, debut, fin)
    segments: list[SegmentParoles] = []
    if audio.shape[0] >= sr:  # au moins une seconde
        mono = librosa.resample(audio.mean(axis=1), orig_sr=sr, target_sr=SR_TRANSCRIPTION)
        del audio
        zones = zones_de_voix(mono, SR_TRANSCRIPTION)
        if zones:
            segments = [
                SegmentParoles(s.debut + debut, s.fin + debut, s.texte)
                for s in transcripteur.transcrire(mono, SR_TRANSCRIPTION, zones)
            ]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {"signature": signature, "segments": [asdict(s) for s in segments]}, indent=1, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return segments
