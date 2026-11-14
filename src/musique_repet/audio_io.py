"""Lecture / écriture audio : WAV via soundfile, MP3 via lameenc."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

import lameenc
import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

EXTENSIONS_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".wma"}


def lister_audios(dossier: Path) -> list[Path]:
    return sorted(
        p for p in dossier.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS_AUDIO
    )


def source_lisible(chemin: Path, dossier_conversion: Path) -> Path:
    """Renvoie un fichier lisible (et positionnable) par soundfile.

    Les formats que libsndfile ne lit pas (m4a, aac, wma...) sont convertis
    une fois pour toutes en WAV avec ffmpeg.
    """
    try:
        with sf.SoundFile(chemin):
            return chemin
    except sf.LibsndfileError:
        pass
    cible = dossier_conversion / f"{chemin.stem}.wav"
    if cible.exists() and cible.stat().st_mtime >= chemin.stat().st_mtime:
        return cible
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(f"{chemin.name} : format non lu nativement et ffmpeg est introuvable")
    dossier_conversion.mkdir(parents=True, exist_ok=True)
    log.info("Conversion de %s en WAV avec ffmpeg", chemin.name)
    temporaire = cible.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(chemin), "-vn", "-c:a", "pcm_s24le", str(temporaire)],
        check=True,
    )
    temporaire.replace(cible)
    return cible


def duree(chemin: Path) -> float:
    info = sf.info(chemin)
    return info.frames / info.samplerate


def lire(chemin: Path, debut: float | None = None, fin: float | None = None) -> tuple[np.ndarray, int]:
    """Lit un fichier (ou un extrait) en float32, forme (échantillons, canaux)."""
    with sf.SoundFile(chemin) as f:
        sr = f.samplerate
        premier = 0 if debut is None else int(round(debut * sr))
        dernier = f.frames if fin is None else min(f.frames, int(round(fin * sr)))
        f.seek(premier)
        donnees = f.read(max(0, dernier - premier), dtype="float32", always_2d=True)
    return donnees, sr


def ecrire_wav(chemin: Path, audio: np.ndarray, sr: int, sous_type: str = "PCM_24") -> None:
    """Écrit un WAV de façon atomique (pas de fichier à moitié écrit en cas d'arrêt)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    sf.write(temporaire, audio, sr, subtype=sous_type, format="WAV")
    temporaire.replace(chemin)


def ecrire_mp3(chemin: Path, audio: np.ndarray, sr: int, debit_kbps: int = 192) -> None:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    encodeur = lameenc.Encoder()
    encodeur.set_bit_rate(debit_kbps)
    encodeur.set_in_sample_rate(sr)
    encodeur.set_channels(pcm.shape[1])
    encodeur.set_quality(2)
    donnees = encodeur.encode(pcm.tobytes()) + encodeur.flush()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    temporaire = chemin.with_name(chemin.name + ".tmp")
    temporaire.write_bytes(donnees)
    temporaire.replace(chemin)


def empreinte(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()
