import logging

import numpy as np
import pytest
import soundfile as sf

from batterie_synthetique import SR, basse_par_mesure, batterie, temps_des_battements
from musique_repet.config import charger_config
from musique_repet.echo import ParametresAntiEcho
from musique_repet.groove import (
    LATENCE_ANALYSE_S,
    ParametresGroove,
    caler_sur_attaques,
    mixer_groove,
    placer_clicks,
    premier_temps,
    son_click,
)
from musique_repet.mixage import ParametresMixage, mixer
from musique_repet.pipeline import Options, executer
from musique_repet.tempo import analyser

PARAMS = ParametresGroove()


# --- Placement du click --------------------------------------------------------------


def test_click_a_tempo_constant():
    battements = np.arange(64) * 0.5 + 1.0  # 120 BPM
    click = placer_clicks(battements, 0, PARAMS)
    assert np.allclose(click.temps, battements, atol=1e-9)
    assert list(np.flatnonzero(click.accents)) == list(range(0, 64, 4))  # le « 1 » de chaque mesure
    assert len(click.blocs) == 4 and click.resume(4) == "click à 120 BPM"


def test_click_reajuste_toutes_les_4_mesures():
    # Le groupe accélère de 100 à 124 BPM : le click change de tempo à chaque bloc de 16 temps,
    # reste régulier à l'intérieur, et repart calé sur la musique au début de chaque bloc.
    acceleration = lambda t: 100 + 24 * np.clip(t / 64, 0, 1)  # noqa: E731
    battements = temps_des_battements(64, acceleration)
    click = placer_clicks(battements, 0, PARAMS)
    tempos = [bpm for _, _, bpm in click.blocs]
    assert all(b > a for a, b in zip(tempos, tempos[1:]))
    for k, (debut, fin, bpm) in enumerate(click.blocs):
        dans_bloc = click.temps[16 * k : 16 * (k + 1)]
        assert np.allclose(np.diff(dans_bloc), 60 / bpm, atol=1e-9)  # tempo constant dans le bloc
        assert abs(dans_bloc[0] - battements[16 * k]) < 0.03  # recalé sur la musique
    assert np.max(np.abs(click.temps - battements[: click.temps.size])) < 0.04


def test_un_temps_mal_detecte_ne_derange_pas_le_click():
    battements = np.arange(32) * 0.5
    battements[5] += 0.12
    click = placer_clicks(battements, 0, PARAMS)
    assert np.allclose(click.temps, np.arange(32) * 0.5, atol=1e-6)


def test_mesures_et_blocs_reglables():
    battements = np.arange(24) * 0.6
    click = placer_clicks(battements, 0, ParametresGroove(temps_par_mesure=3, mesures_par_bloc=2))
    assert list(np.flatnonzero(click.accents)) == list(range(0, 24, 3))  # valse
    assert len(click.blocs) == 4  # blocs de 2 mesures de 3 temps


def test_levee_avant_le_premier_temps():
    battements = np.arange(20) * 0.5
    click = placer_clicks(battements, 2, PARAMS)
    assert np.allclose(click.temps, battements)
    assert not click.accents[:2].any() and click.accents[2]  # le « 1 » est le troisième temps


def test_son_du_click():
    accent, normal = son_click(SR, True), son_click(SR, False)
    frequence = lambda s: np.argmax(np.abs(np.fft.rfft(s, 8 * SR))) / 8  # noqa: E731
    assert frequence(accent) == pytest.approx(1500, abs=5) and frequence(normal) == pytest.approx(1000, abs=5)
    assert np.max(np.abs(accent)) > np.max(np.abs(normal))


# --- Analyse de la batterie et de la basse -------------------------------------------------


@pytest.mark.parametrize("coupe", [0, 1, 2, 3])
def test_premier_temps_de_la_mesure(coupe):
    # Grosse caisse sur 1 et 3, caisse claire sur 2 et 4, la basse change de note au « 1 ».
    # On coupe le morceau à partir du temps ``coupe`` d'une mesure : le premier « 1 » est
    # alors le temps de rang (4 - coupe) % 4.
    vrais = temps_des_battements(48, 110)
    drums, basse = batterie(48, 110), basse_par_mesure(48, 110)
    i = int(round(vrais[coupe] * SR))
    battements = vrais[coupe:] - vrais[coupe]
    phase = premier_temps(battements, drums[i:, None], basse[i:, None], SR, 4)
    assert phase == (4 - coupe) % 4


def test_recalage_sur_les_frappes():
    drums = batterie(30, 120)
    vrais = temps_des_battements(30, 120)
    en_retard = vrais + LATENCE_ANALYSE_S + 0.004  # comme les temps donnés par l'analyse du tempo
    recales = caler_sur_attaques(en_retard, drums[:, None], SR)
    assert np.max(np.abs(recales - vrais)) < 0.005
    sans_batterie = caler_sur_attaques(en_retard, None, SR)
    assert np.allclose(sans_batterie, en_retard - LATENCE_ANALYSE_S)


# --- Piste groove --------------------------------------------------------------------------


def ecrire(dossier, **pistes):
    chemins = {}
    for nom, audio in pistes.items():
        chemins[nom] = dossier / f"{nom}.wav"
        sf.write(chemins[nom], np.repeat(audio[:, None], 2, axis=1), SR, subtype="FLOAT")
    return chemins


def energie(audio, frequence, debut, fin):
    morceau = audio[int(debut * SR) : int(fin * SR)].mean(axis=1)
    spectre = np.abs(np.fft.rfft(morceau)) ** 2
    frequences = np.fft.rfftfreq(morceau.size, 1 / SR)
    return float(spectre[np.abs(frequences - frequence) < 15].sum() / spectre.sum())


def test_groove_contient_batterie_basse_et_click_sans_les_voix(tmp_path):
    t = np.arange(20 * SR) / SR
    voix = (0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)  # une « voix » facile à repérer
    drums, basse = batterie(20, 120), basse_par_mesure(20, 120)
    pistes = ecrire(tmp_path, batterie=drums, basse=basse, voix=voix)
    params = ParametresMixage()
    complet, sr, reglages = mixer(pistes, charger_config(None), params)
    analyse = analyser(drums + basse, SR)
    groove, sr_groove, click = mixer_groove(pistes, reglages, params, None, analyse, PARAMS)
    assert sr_groove == SR and groove.shape == complet.shape
    # La « voix » pèse 11 % du mix complet ; dans le groove il ne reste que le peu
    # d'énergie que la batterie a elle-même vers 3 kHz.
    assert energie(complet, 3000, 2, 18) > 0.05
    assert energie(groove, 3000, 2, 18) < energie(complet, 3000, 2, 18) / 100
    assert click is not None and click.resume(4) == "click à 120 BPM"
    assert energie(groove, 1000, 2, 18) > 1e-3 and energie(groove, 1500, 2, 18) > 1e-3  # le click
    assert np.max(np.abs(groove)) <= 10 ** (-1 / 20) + 1e-6  # normalisé comme le MP3 complet


def test_groove_sans_pulsation_reguliere(tmp_path):
    rng = np.random.default_rng(0)
    pistes = ecrire(
        tmp_path, batterie=(0.05 * rng.standard_normal(10 * SR)).astype(np.float32),
        basse=np.zeros(10 * SR, np.float32),
    )
    _, _, reglages = mixer(pistes, charger_config(None), ParametresMixage())
    analyse = analyser(sf.read(pistes["batterie"])[0].mean(axis=1), SR)
    groove, _, click = mixer_groove(pistes, reglages, ParametresMixage(), None, analyse, PARAMS)
    assert click is None and groove.shape[0] == 10 * SR


class SeparateurBatterieBasse:
    identifiant = "batterie-basse:v1"

    def __init__(self, drums, basse):
        self.drums, self.basse = drums, basse

    def separer(self, audio, sr):
        stereo = lambda s: np.repeat(s[: audio.shape[0], None], 2, axis=1)  # noqa: E731
        muet = np.zeros_like(audio)
        return {"batterie": stereo(self.drums), "basse": stereo(self.basse), "voix": muet, "piano": muet}, sr


def test_piste_groove_a_cote_du_mp3(tmp_path, monkeypatch, caplog):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    drums, basse = batterie(24, 110), basse_par_mesure(24, 110)
    sf.write(entree / "repet.wav", np.repeat((drums + basse)[:, None], 2, axis=1), SR)
    (entree / "repet.csv").write_text("Le Sud;0;24\n", encoding="utf-8")
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    options = Options(
        entree=entree, sortie=sortie, travail=sortie / "_travail", facteurs=charger_config(None),
        mixage=ParametresMixage(), anti_echo=ParametresAntiEcho(attenuation_max_db=0), tempo=None,
    )
    with caplog.at_level(logging.INFO):
        assert executer(options, SeparateurBatterieBasse(drums, basse)) == 0
    complet, groove = sortie / "20260312-Le Sud.mp3", sortie / "20260312-Le Sud-groove.mp3"
    assert complet.exists() and groove.exists()
    assert sf.info(groove).duration == pytest.approx(sf.info(complet).duration, abs=0.1)
    assert "Groove : click à 110 BPM" in caplog.text
    assert not (sortie / "20260312-Le Sud-tempo.jpg").exists()  # le graphique reste désactivable à part
