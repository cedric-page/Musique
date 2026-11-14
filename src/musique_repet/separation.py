"""Séparation des pistes d'instruments (Demucs) avec cache sur disque."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from . import audio_io

log = logging.getLogger(__name__)

# Noms des sources Demucs -> noms des pistes (et des clés de configuration).
NOMS_PISTES = {
    "vocals": "voix",
    "drums": "batterie",
    "bass": "basse",
    "guitar": "guitare",
    "piano": "piano",
    "other": "autre",
}

MODELE_PAR_DEFAUT = "htdemucs_6s"
FICHIER_META = "separation.json"


class Separateur(Protocol):
    """Tout objet capable de séparer un signal en pistes."""

    identifiant: str

    def separer(self, audio: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], int]:
        """``audio`` : (échantillons, canaux). Renvoie ({piste: audio}, fréquence)."""
        ...


class SeparateurDemucs:
    """Séparation avec Demucs. ``htdemucs_6s`` sépare voix, batterie, basse,
    guitare, piano et « autre »."""

    def __init__(
        self,
        modele: str = MODELE_PAR_DEFAUT,
        appareil: str | None = None,
        shifts: int = 1,
        overlap: float = 0.25,
    ) -> None:
        self.modele_nom = modele
        self.appareil = appareil
        self.shifts = shifts
        self.overlap = overlap
        self._modele = None
        self.identifiant = f"demucs:{modele}:shifts={shifts}:overlap={overlap}"

    def _charger(self):
        if self._modele is None:
            import torch
            from demucs.pretrained import get_model

            if self.appareil is None:
                self.appareil = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Chargement du modèle Demucs %s sur %s", self.modele_nom, self.appareil)
            self._modele = get_model(self.modele_nom)
            self._modele.eval()
        return self._modele

    def separer(self, audio: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], int]:
        import julius
        import torch
        from demucs.apply import apply_model

        modele = self._charger()
        signal = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
        # Adapter le nombre de canaux au modèle (stéréo).
        if signal.shape[0] == 1 and modele.audio_channels == 2:
            signal = signal.repeat(2, 1)
        elif signal.shape[0] > modele.audio_channels:
            signal = signal[: modele.audio_channels]
        if sr != modele.samplerate:
            signal = julius.resample_frac(signal, sr, modele.samplerate)

        reference = signal.mean(0)
        moyenne, ecart = reference.mean(), reference.std()
        if float(ecart) < 1e-8:
            ecart = torch.tensor(1.0)
        signal = (signal - moyenne) / ecart

        with torch.no_grad():
            sources = apply_model(
                modele,
                signal[None],
                device=self.appareil,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=True,
            )[0]
        sources = sources * ecart + moyenne

        pistes = {
            NOMS_PISTES.get(nom, nom): sources[i].cpu().numpy().T.astype(np.float32)
            for i, nom in enumerate(modele.sources)
        }
        return pistes, modele.samplerate


def separer_avec_cache(
    source_wav: Path,
    dossier_pistes: Path,
    separateur: Separateur,
    forcer: bool = False,
) -> dict[str, Path]:
    """Sépare ``source_wav`` dans ``dossier_pistes`` sauf si un résultat valide existe.

    Le cache est valide si le fichier ``separation.json`` correspond à la même
    empreinte audio et au même séparateur, et que toutes les pistes sont présentes.
    """
    empreinte = audio_io.empreinte(source_wav)
    meta_chemin = dossier_pistes / FICHIER_META

    if not forcer and meta_chemin.exists():
        try:
            meta = json.loads(meta_chemin.read_text(encoding="utf-8"))
            pistes = {nom: dossier_pistes / f"{nom}.wav" for nom in meta["pistes"]}
            if (
                meta.get("empreinte") == empreinte
                and meta.get("separateur") == separateur.identifiant
                and all(p.exists() for p in pistes.values())
            ):
                log.info("Pistes déjà séparées, réutilisation de %s", dossier_pistes)
                return pistes
        except (json.JSONDecodeError, KeyError, TypeError):
            log.warning("Cache de séparation illisible, nouvelle séparation : %s", meta_chemin)

    log.info("Séparation des pistes de %s", source_wav.name)
    audio, sr = audio_io.lire(source_wav)
    resultats, sr_sortie = separateur.separer(audio, sr)

    dossier_pistes.mkdir(parents=True, exist_ok=True)
    meta_chemin.unlink(missing_ok=True)
    chemins: dict[str, Path] = {}
    for nom, piste in resultats.items():
        chemin = dossier_pistes / f"{nom}.wav"
        # Flottant 32 bits : aucune perte et pas d'écrêtage des pistes séparées.
        audio_io.ecrire_wav(chemin, piste, sr_sortie, sous_type="FLOAT")
        chemins[nom] = chemin
    # Le fichier méta est écrit en dernier : il ne valide le cache que si tout est sur disque.
    meta_chemin.write_text(
        json.dumps(
            {"empreinte": empreinte, "separateur": separateur.identifiant, "pistes": sorted(chemins)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return chemins
