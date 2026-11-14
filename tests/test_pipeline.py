import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musique_repet import audio_io
from musique_repet.cli import main
from musique_repet.config import CONFIG_EXEMPLE, charger_config
from musique_repet.decoupe import lire_csv, parser_temps
from musique_repet.mixage import ParametresMixage, calculer_equilibrage, mesurer_lufs
from musique_repet.pipeline import Options, executer

SR = 44100


class FauxSeparateur:
    """Renvoie des copies de l'entrée à des niveaux différents, et compte les appels."""

    identifiant = "faux:v1"
    niveaux = {"voix": 1.0, "piano": 0.5, "batterie": 0.25, "basse": 0.5, "guitare": 0.5, "autre": 0.0}

    def __init__(self):
        self.appels = 0

    def separer(self, audio, sr):
        self.appels += 1
        return {nom: (audio * n).astype(np.float32) for nom, n in self.niveaux.items()}, sr


def sinus(duree, freq=440.0, amplitude=0.3, canaux=2):
    t = np.arange(int(duree * SR)) / SR
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return np.repeat(signal[:, None], canaux, axis=1).astype(np.float32)


def options(entree: Path, sortie: Path, **kw) -> Options:
    return Options(
        entree=entree,
        sortie=sortie,
        travail=sortie / "_travail",
        facteurs=charger_config(None),
        mixage=ParametresMixage(),
        **kw,
    )


# --- CSV ------------------------------------------------------------------


@pytest.mark.parametrize(
    "texte,attendu",
    [("83", 83), ("83.5", 83.5), ("83,5", 83.5), ("1:23", 83), ("01:02:03.25", 3723.25), (" 0:05 ", 5)],
)
def test_parser_temps(texte, attendu):
    assert parser_temps(texte) == pytest.approx(attendu)


@pytest.mark.parametrize("texte", ["", "abc", "1:2:3:4", "-5"])
def test_parser_temps_invalide(texte):
    with pytest.raises(ValueError):
        parser_temps(texte)


def test_lire_csv(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(
        "titre;debut;fin\nIntro;0;0:10\n\n# commentaire\nRefrain;0:10;0:25\nIntro;30;40\nFin;0:50;9:00\n",
        encoding="utf-8",
    )
    morceaux = lire_csv(csv, duree=60)
    assert [(m.titre, m.debut, m.fin) for m in morceaux] == [
        ("Intro", 0, 10),
        ("Refrain", 10, 25),
        ("Intro (2)", 30, 40),
        ("Fin", 50, 60),  # ramené à la durée
    ]


def test_lire_csv_fin_avant_debut(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("A;20;10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="après le début"):
        lire_csv(csv)


# --- Config ---------------------------------------------------------------


def test_config(tmp_path):
    chemin = tmp_path / "c.json"
    chemin.write_text(json.dumps({"Voix": 1.5, "drums": 0.5, "triangle": 2}), encoding="utf-8")
    gains = charger_config(chemin)
    assert gains["voix"] == 1.5 and gains["batterie"] == 0.5 and gains["piano"] == 1.0
    assert "triangle" not in gains


def test_config_invalide(tmp_path):
    chemin = tmp_path / "c.json"
    chemin.write_text('{"voix": "fort"}', encoding="utf-8")
    with pytest.raises(ValueError):
        charger_config(chemin)


def test_generer_config(tmp_path):
    chemin = tmp_path / "exemple.json"
    assert main(["--generer-config", str(chemin)]) == 0
    assert json.loads(chemin.read_text(encoding="utf-8")) == CONFIG_EXEMPLE


# --- Équilibrage ----------------------------------------------------------


def test_equilibrage_vers_mediane():
    sonies = {"voix": -20.0, "piano": -26.0, "batterie": -14.0, "basse": -60.0, "autre": -30.0}
    reglages = {r.nom: r for r in calculer_equilibrage(sonies, {"voix": 2.0}, ParametresMixage())}
    assert reglages["voix"].gain_equilibrage_db == pytest.approx(0)
    assert reglages["piano"].gain_equilibrage_db == pytest.approx(6)
    assert reglages["batterie"].gain_equilibrage_db == pytest.approx(-6)
    assert reglages["basse"].gain_equilibrage_db == 0  # muette : pas touchée
    assert reglages["autre"].gain_equilibrage_db == 0  # exclue par défaut
    assert reglages["voix"].facteur_total == pytest.approx(2.0)


def test_equilibrage_plafonne():
    sonies = {"voix": -10.0, "piano": -40.0, "guitare": -10.0}
    reglages = {r.nom: r for r in calculer_equilibrage(sonies, {}, ParametresMixage(gain_max_db=12))}
    assert reglages["piano"].gain_equilibrage_db == pytest.approx(12)


# --- Chaîne complète ------------------------------------------------------


def test_chaine_complete_avec_csv_et_cache(tmp_path):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "repet.wav", sinus(12), SR, subtype="PCM_24")
    (entree / "repet.csv").write_text("Chanson A;0;5\nChanson/B;0:05;0:11\n", encoding="utf-8")
    sf.write(entree / "jam.flac", sinus(3, 220), SR)

    separateur = FauxSeparateur()
    assert executer(options(entree, sortie), separateur) == 0

    mp3_a = sortie / "repet" / "Chanson A.mp3"
    mp3_b = sortie / "repet" / "Chanson_B.mp3"
    mp3_jam = sortie / "jam.mp3"
    for mp3 in (mp3_a, mp3_b, mp3_jam):
        assert mp3.exists() and mp3.stat().st_size > 1000
    assert separateur.appels == 3

    # Découpes en WAV et pistes séparées conservées sur disque.
    decoupe_b = sortie / "_travail" / "decoupes" / "repet" / "Chanson_B.wav"
    assert sf.info(decoupe_b).duration == pytest.approx(6, abs=0.01)
    pistes_a = sortie / "_travail" / "pistes" / "repet" / "Chanson A"
    assert {p.name for p in pistes_a.glob("*.wav")} == {f"{n}.wav" for n in FauxSeparateur.niveaux}

    # MP3 lisible et à la bonne sonie.
    audio, sr = sf.read(mp3_a, always_2d=True)
    assert audio.shape[0] / sr == pytest.approx(5, abs=0.1)
    assert mesurer_lufs(audio, sr) == pytest.approx(-14, abs=1.0)
    assert np.max(np.abs(audio)) <= 10 ** (-1 / 20) + 0.02

    # Relance : aucune nouvelle séparation.
    assert executer(options(entree, sortie), separateur) == 0
    assert separateur.appels == 3

    # Modification d'un seul morceau : seul celui-ci est re-séparé.
    (entree / "repet.csv").write_text("Chanson A;0;5\nChanson/B;0:06;0:11\n", encoding="utf-8")
    assert executer(options(entree, sortie), separateur) == 0
    assert separateur.appels == 4


def test_volume_config_coupe_une_piste(tmp_path):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "r.wav", sinus(3), SR)

    class SeparateurDeuxPistes(FauxSeparateur):
        niveaux = {"voix": 1.0, "piano": 0.0}

        def separer(self, audio, sr):
            # voix et piano à des fréquences différentes pour distinguer les pistes
            t = np.arange(audio.shape[0]) / sr
            voix = 0.3 * np.sin(2 * np.pi * 440 * t)
            piano = 0.3 * np.sin(2 * np.pi * 1000 * t)
            stereo = lambda s: np.repeat(s[:, None], 2, axis=1).astype(np.float32)  # noqa: E731
            return {"voix": stereo(voix), "piano": stereo(piano)}, sr

    opts = options(entree, sortie)
    opts.facteurs["piano"] = 0.0
    assert executer(opts, SeparateurDeuxPistes()) == 0
    audio, sr = sf.read(sortie / "r.mp3", always_2d=True)
    spectre = np.abs(np.fft.rfft(audio[sr : 2 * sr, 0]))
    freqs = np.fft.rfftfreq(sr, 1 / sr)
    energie = lambda f: spectre[(freqs > f - 20) & (freqs < f + 20)].sum()  # noqa: E731
    assert energie(1000) < energie(440) / 100


def test_fichier_en_echec_n_arrete_pas_les_autres(tmp_path):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "ok.wav", sinus(2), SR)
    sf.write(entree / "ko.wav", sinus(2), SR)
    (entree / "ko.csv").write_text("A;1.5;0.5\n", encoding="utf-8")
    assert executer(options(entree, sortie), FauxSeparateur()) == 1
    assert (sortie / "ok.mp3").exists()


def test_mesure_lufs_silence():
    assert mesurer_lufs(np.zeros((SR, 2), np.float32), SR) == -math.inf
    assert math.isfinite(mesurer_lufs(sinus(0.1), SR))


def test_ecrire_mp3_mono(tmp_path):
    chemin = tmp_path / "m.mp3"
    audio_io.ecrire_mp3(chemin, sinus(1, canaux=1)[:, 0], SR)
    assert sf.info(chemin).channels == 1
