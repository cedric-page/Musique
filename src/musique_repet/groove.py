"""Piste « groove » : la batterie, la basse et un click qui suit le tempo du groupe.

Le click marque chaque temps, plus aigu sur le premier temps de la mesure. Son
tempo est constant pendant 4 mesures, calé sur les temps réellement joués dans
ces 4 mesures, puis il se réajuste pour les 4 suivantes : il colle à la musique
sans la suivre note à note, si bien qu'on entend si le groupe presse ou
ralentit à l'intérieur d'un bloc.

Le premier temps de la mesure est repéré sur les pistes séparées : la grosse
caisse frappe surtout sur les temps 1 et 3, la caisse claire sur 2 et 4, et la
basse change de note en début de mesure (changement d'accord).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio_io
from .mixage import ParametresMixage, ReglagePiste, additionner, normaliser, piste_traitee
from .tempo import AnalyseTempo

log = logging.getLogger(__name__)

SR_ANALYSE = 22050
LATENCE_ANALYSE_S = 0.017
"""Retard moyen des temps détectés par rapport à la frappe (fenêtre d'analyse), si la batterie
ne permet pas de le mesurer."""


@dataclass(frozen=True)
class ParametresGroove:
    temps_par_mesure: int = 4
    mesures_par_bloc: int = 4
    """Le tempo du click est réajusté toutes les ``mesures_par_bloc`` mesures."""
    volume_click_db: float = 0.0
    """Volume du click par rapport à son réglage automatique (+6 = plus fort, -6 = plus doux)."""
    gain_basse_db: float = 3.0
    """Renfort de la basse par rapport au mixage complet : sans les autres instruments, elle
    doit ressortir nettement face à la batterie (ses notes graves passent mal sur de petits
    haut-parleurs)."""


@dataclass
class Click:
    temps: np.ndarray
    """Instants des clicks (s)."""
    accents: np.ndarray
    """Vrai pour le premier temps de chaque mesure."""
    blocs: list[tuple[float, float, float]] = field(default_factory=list)
    """(début, fin, BPM) de chaque bloc de mesures à tempo constant."""

    def resume(self, mesures_par_bloc: int) -> str:
        if not self.blocs:
            return "pas de click"
        bpm = [b for _, _, b in self.blocs]
        if max(bpm) - min(bpm) < 0.5:
            return f"click à {bpm[0]:.0f} BPM"
        return (
            f"click de {min(bpm):.0f} à {max(bpm):.0f} BPM, réajusté toutes les {mesures_par_bloc} mesures "
            f"({len(self.blocs)} blocs)"
        )


def _normaliser_z(valeurs: np.ndarray) -> np.ndarray:
    ecart = float(np.std(valeurs))
    return (valeurs - float(np.mean(valeurs))) / ecart if ecart > 1e-12 else np.zeros_like(valeurs)


def _mono_22k(audio: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    return librosa.resample(np.ascontiguousarray(mono, dtype=np.float32), orig_sr=sr, target_sr=SR_ANALYSE)


def _attaques_par_bande(y: np.ndarray, bandes: list[tuple[float, float]], pas: int) -> list[np.ndarray]:
    """Force des attaques (hausse d'énergie, en log) dans chaque bande de fréquences."""
    import librosa

    spectre = np.abs(librosa.stft(y, n_fft=1024, hop_length=pas)) ** 2
    frequences = librosa.fft_frequencies(sr=SR_ANALYSE, n_fft=1024)
    resultat = []
    for bas, haut in bandes:
        energie = np.log(spectre[(frequences >= bas) & (frequences < haut)].sum(axis=0) + 1e-10)
        resultat.append(np.maximum(0.0, np.diff(energie, prepend=energie[:1])))
    return resultat


def premier_temps(
    battements: np.ndarray, batterie: np.ndarray | None, basse: np.ndarray | None, sr: int, temps_par_mesure: int
) -> int:
    """Rang (0 à ``temps_par_mesure`` - 1) du premier battement qui tombe sur un premier temps."""
    import librosa

    if battements.size < 2 * temps_par_mesure:
        return 0
    pas = 256
    fps = SR_ANALYSE / pas
    trames = np.round(battements * fps).astype(np.int64)
    score = np.zeros(battements.size)
    if batterie is not None and np.any(batterie):
        grave, medium = _attaques_par_bande(_mono_22k(batterie, sr), [(30, 150), (200, 2000)], pas)
        voisinage = np.arange(-3, 6)  # de -35 à +70 ms autour du temps

        def autour(courbe: np.ndarray) -> np.ndarray:
            return courbe[np.clip(trames[:, None] + voisinage[None, :], 0, courbe.size - 1)].max(axis=1)

        score += _normaliser_z(autour(grave)) - 0.5 * _normaliser_z(autour(medium))
    if basse is not None and np.any(basse):
        chroma = librosa.feature.chroma_cqt(y=_mono_22k(basse, sr), sr=SR_ANALYSE, hop_length=512)
        fps_chroma = SR_ANALYSE / 512
        periode = float(np.median(np.diff(battements)))
        changements = np.zeros(battements.size)
        for k, t in enumerate(battements):
            a = int((t - periode / 2) * fps_chroma)
            b = int(t * fps_chroma)
            c = int((t + periode / 2) * fps_chroma)
            if a < 0 or c > chroma.shape[1] or b <= a or c <= b:
                continue
            avant, apres = chroma[:, a:b].mean(axis=1), chroma[:, b:c].mean(axis=1)
            cosinus = float(avant @ apres / (np.linalg.norm(avant) * np.linalg.norm(apres) + 1e-12))
            changements[k] = 1.0 - cosinus
        score += _normaliser_z(changements)
    phases = [float(np.mean(score[phase::temps_par_mesure])) for phase in range(temps_par_mesure)]
    return int(np.argmax(phases))


def caler_sur_attaques(battements: np.ndarray, batterie: np.ndarray | None, sr: int) -> np.ndarray:
    """Recale les temps détectés sur les frappes de la batterie.

    L'analyse du tempo place les temps avec un léger retard (sa fenêtre d'analyse) :
    pour qu'un click tombe pile avec la batterie, on mesure le décalage médian entre
    les temps et le début des frappes les plus proches, et on le retire.
    """
    import librosa

    if batterie is None or not np.any(batterie) or battements.size == 0:
        return battements - LATENCE_ANALYSE_S
    y = _mono_22k(batterie, sr)
    attaques = librosa.onset.onset_detect(y=y, sr=SR_ANALYSE, hop_length=64, backtrack=True, units="time")
    if attaques.size < 8:
        return battements - LATENCE_ANALYSE_S
    position = np.clip(np.searchsorted(attaques, battements), 1, attaques.size - 1)
    voisines = np.where(
        np.abs(attaques[position - 1] - battements) <= np.abs(attaques[position] - battements),
        attaques[position - 1],
        attaques[position],
    )
    decalages = battements - voisines
    proches = decalages[np.abs(decalages) < 0.06]
    if proches.size < 8:
        return battements - LATENCE_ANALYSE_S
    return battements - float(np.median(proches))


def _ajuster_droite(position: np.ndarray, instants: np.ndarray, periode: float) -> tuple[float, float]:
    """Tempo constant (départ, période) le plus proche des temps joués, points aberrants écartés."""
    depart = float(np.median(instants - periode * position))
    gardes = np.abs(instants - (depart + periode * position)) < 0.06
    if gardes.sum() >= 4:
        periode, depart = (float(v) for v in np.polyfit(position[gardes], instants[gardes], 1))
    return depart, periode


def placer_clicks(battements: np.ndarray, phase: int, params: ParametresGroove) -> Click:
    """Clicks à tempo constant par blocs de mesures, calés sur les temps joués dans chaque bloc."""
    battements = np.sort(np.asarray(battements, dtype=np.float64))
    n = battements.size
    taille_bloc = params.temps_par_mesure * params.mesures_par_bloc
    temps: list[float] = []
    accents: list[bool] = []
    blocs: list[tuple[float, float, float]] = []
    periode_precedente = float(np.median(np.diff(battements))) if n >= 2 else None
    for debut in range(phase, n, taille_bloc):
        rangs = np.arange(debut, min(debut + taille_bloc, n))
        position = rangs - debut
        instants = battements[rangs]
        # Tempo constant le plus proche des temps joués dans le bloc (un temps mal
        # détecté est écarté) ; un bloc trop court (fin du morceau) garde le tempo précédent.
        periode = float(np.median(np.diff(instants))) if rangs.size >= 4 else periode_precedente
        if periode is None or periode <= 0:
            break
        if rangs.size >= 4:
            depart, periode = _ajuster_droite(position.astype(np.float64), instants, periode)
        else:
            depart = float(np.median(instants - periode * position))
        clicks = depart + periode * np.arange(rangs.size)
        temps.extend(clicks.tolist())
        accents.extend(((np.arange(rangs.size)) % params.temps_par_mesure == 0).tolist())
        blocs.append((float(clicks[0]), float(clicks[-1] + periode), 60.0 / periode))
        periode_precedente = periode
    if phase > 0 and blocs:
        # Levée : les quelques temps joués avant le premier « 1 », sans accent.
        levee = battements[:phase]
        temps = levee.tolist() + temps
        accents = [False] * levee.size + accents
    return Click(np.array(temps), np.array(accents, dtype=bool), blocs)


def son_click(sr: int, accent: bool) -> np.ndarray:
    """Click court (30 ms) : aigu sur le premier temps de la mesure, plus grave et plus doux sinon."""
    t = np.arange(int(0.03 * sr)) / sr
    frequence, volume = (1500.0, 1.0) if accent else (1000.0, 0.7)
    return (volume * np.sin(2 * np.pi * frequence * t) * np.exp(-t / 0.008) * np.minimum(1.0, t / 5e-4)).astype(
        np.float32
    )


def piste_click(click: Click, longueur: int, sr: int, amplitude: float) -> np.ndarray:
    piste = np.zeros(longueur, np.float32)
    sons = {True: amplitude * son_click(sr, True), False: amplitude * son_click(sr, False)}
    for instant, accent in zip(click.temps, click.accents):
        i = int(round(instant * sr))
        son = sons[bool(accent)]
        if 0 <= i < longueur:
            fin = min(longueur, i + son.size)
            piste[i:fin] += son[: fin - i]
    return np.repeat(piste[:, None], 2, axis=1)


def _niveau_rms(audio: np.ndarray, sr: int) -> float:
    """Niveau efficace des passages joués (les silences ne comptent pas)."""
    taille = max(1, int(0.05 * sr))
    nb = audio.shape[0] // taille
    if nb == 0:
        return 0.0
    rms = np.sqrt((audio[: nb * taille].astype(np.float64) ** 2).reshape(nb, taille, -1).mean(axis=(1, 2)))
    joue = rms > 0.01 * rms.max() if rms.max() > 0 else rms > 0
    return float(np.sqrt(np.mean(rms[joue] ** 2))) if joue.any() else 0.0


def mixer_groove(
    pistes: dict[str, Path],
    reglages: list[ReglagePiste],
    params_mixage: ParametresMixage,
    zone: tuple[float, float] | None,
    analyse: AnalyseTempo | None,
    params: ParametresGroove = ParametresGroove(),
) -> tuple[np.ndarray, int, Click | None]:
    """Batterie + basse (réglées comme dans le mixage complet, basse renforcée) + click, normalisés."""
    par_nom = {r.nom: r for r in reglages}
    mix = None
    sr = None
    for nom in ("batterie", "basse"):
        if nom in pistes and nom in par_nom and par_nom[nom].facteur_total > 0:
            audio, sr = piste_traitee(pistes[nom], par_nom[nom], params_mixage, zone)
            if nom == "basse":
                audio *= 10 ** (params.gain_basse_db / 20)
            mix = additionner(mix, audio)
    if mix is None:
        reference, sr = audio_io.lire(next(iter(pistes.values())), *(zone or ()))
        mix = np.zeros_like(reference)

    click = None
    if analyse is not None and analyse.detecte and analyse.battements.size >= 2 * params.temps_par_mesure:
        brutes = {nom: audio_io.lire(pistes[nom], *(zone or ()))[0] for nom in ("batterie", "basse") if nom in pistes}
        battements = caler_sur_attaques(analyse.battements, brutes.get("batterie"), sr)
        phase = premier_temps(battements, brutes.get("batterie"), brutes.get("basse"), sr, params.temps_par_mesure)
        click = placer_clicks(battements, phase, params)
        niveau = _niveau_rms(mix, sr)
        amplitude = niveau * 10 ** ((12.0 + params.volume_click_db) / 20) if niveau > 0 else 0.3
        mix = mix + piste_click(click, mix.shape[0], sr, amplitude)
    return normaliser(mix, sr, params_mixage), sr, click
