import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from batterie_synthetique import SR, batterie
from musique_repet.graphique_tempo import tracer
from musique_repet.pipeline import Options, executer
from musique_repet.config import charger_config
from musique_repet.mixage import ParametresMixage
from musique_repet.tempo import ParametresTempo, analyser, signal_rythmique


def erreur_moyenne(analyse, attendu):
    ok = np.isfinite(analyse.bpm_lisse)
    return float(np.mean(np.abs(analyse.bpm_lisse[ok] - attendu(analyse.temps[ok]))))


@pytest.mark.parametrize("bpm", [72, 90, 128, 165])
def test_tempo_constant_sans_confusion_avec_le_double(bpm):
    # Le charleston en croches ne doit pas faire mesurer le double du tempo.
    analyse = analyser(batterie(40, bpm), SR)
    assert analyse.detecte
    assert analyse.mediane == pytest.approx(bpm, abs=0.5)
    assert analyse.part_stable > 0.95


def test_batteur_humain():
    analyse = analyser(batterie(40, 128, humain_ms=15), SR)
    assert analyse.mediane == pytest.approx(128, abs=1)
    assert np.isfinite(analyse.bpm_lisse).mean() > 0.9


def test_suit_une_acceleration():
    rampe = lambda t: 100 + 20 * np.clip(t, 0, 90) / 90  # noqa: E731
    analyse = analyser(batterie(90, rampe), SR)
    assert erreur_moyenne(analyse, rampe) < 1
    debut, fin = analyse.debut_fin
    assert debut < 104 and fin > 116


def test_suit_un_ralenti():
    ralenti = lambda t: np.where(t < 40, 120, 120 - 30 * np.clip((t - 40) / 10, 0, 1))  # noqa: E731
    analyse = analyser(batterie(60, ralenti), SR)
    assert erreur_moyenne(analyse, ralenti) < 1
    assert analyse.minimum == pytest.approx(90, abs=2)


def test_break_sans_tempo_affiche():
    analyse = analyser(batterie(50, 128, silence=(20, 30)), SR)
    dans_le_break = (analyse.temps > 21) & (analyse.temps < 29)
    assert not np.isfinite(analyse.bpm_lisse[dans_le_break]).any()
    assert analyse.mediane == pytest.approx(128, abs=0.5)


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros(30 * SR, np.float32),
        0.1 * np.random.default_rng(0).standard_normal(30 * SR).astype(np.float32),
        batterie(3, 120),  # trop court
    ],
    ids=["silence", "bruit", "trop_court"],
)
def test_pas_de_tempo_invente(signal):
    analyse = analyser(signal, SR)
    assert not analyse.detecte
    assert analyse.resume() == "aucune pulsation régulière détectée"


def test_plage_de_tempo_reglable():
    # Une ballade lue au double peut être corrigée en baissant le tempo maximal.
    analyse = analyser(batterie(40, 75), SR, ParametresTempo(bpm_min=40, bpm_max=100))
    assert analyse.mediane == pytest.approx(75, abs=0.5)


def test_signal_rythmique_sans_les_voix(tmp_path):
    t = np.arange(SR * 2) / SR
    pistes = {}
    for nom, amplitude in [("voix", 0.5), ("batterie", 0.3), ("basse", 0.2), ("piano", 0.0), ("autre", 0.1)]:
        pistes[nom] = tmp_path / f"{nom}.wav"
        sf.write(pistes[nom], np.repeat((amplitude * np.sin(2 * np.pi * 110 * t))[:, None], 2, axis=1), SR)
    signal, sr, jouees = signal_rythmique(pistes)
    assert sr == SR
    assert jouees == ("batterie", "basse")  # piano muet non cité, voix et autre exclues
    assert np.max(np.abs(signal)) == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("avec_rythme", [True, False])
def test_graphique_jpg(tmp_path, avec_rythme):
    signal = batterie(30, 110, humain_ms=10) if avec_rythme else np.zeros(30 * SR, np.float32)
    analyse = analyser(signal, SR, sources=("batterie",))
    chemin = tmp_path / "20260312-Essai.jpg"
    tracer(analyse, chemin, "Essai", "20260312")
    with Image.open(chemin) as image:
        assert image.format == "JPEG"
        assert image.size == (1800, 780)
    assert not list(tmp_path.glob("*.tmp"))


class SeparateurCopie:
    """Renvoie l'entrée telle quelle comme piste de batterie, les autres pistes muettes."""

    identifiant = "copie:v1"

    def separer(self, audio, sr):
        muet = np.zeros_like(audio)
        return {"batterie": audio, "basse": muet, "voix": muet, "autre": muet}, sr


def _options(entree, sortie, **kw):
    return Options(
        entree=entree, sortie=sortie, travail=sortie / "_travail",
        facteurs=charger_config(None), mixage=ParametresMixage(), **kw,
    )


def test_pipeline_produit_le_graphique_a_cote_du_mp3(tmp_path, monkeypatch):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "repet.wav", np.repeat(batterie(40, 96)[:, None], 2, axis=1), SR)
    (entree / "repet.csv").write_text("Le Sud;0;20\nHallelujah;20;40\n", encoding="utf-8")
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    vus = {}
    import musique_repet.pipeline as pipeline

    original = pipeline.tracer_tempo

    def espion(analyse, chemin, titre, date=None):
        vus[titre] = (analyse, chemin, date)
        original(analyse, chemin, titre, date)

    monkeypatch.setattr(pipeline, "tracer_tempo", espion)
    assert executer(_options(entree, sortie), SeparateurCopie()) == 0

    assert sorted(p.name for p in sortie.glob("*-tempo.jpg")) == [
        "20260312-Hallelujah-tempo.jpg",
        "20260312-Le Sud-tempo.jpg",
    ]
    analyse, chemin, date = vus["Le Sud"]
    assert chemin == sortie / "20260312-Le Sud-tempo.jpg" and date == "20260312"
    assert analyse.sources == ("batterie",)
    assert analyse.mediane == pytest.approx(96, abs=0.5)


def test_pipeline_sans_tempo(tmp_path):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "r.wav", np.repeat(batterie(10, 120)[:, None], 2, axis=1), SR)
    assert executer(_options(entree, sortie, tempo=None), SeparateurCopie()) == 0
    assert len(list(sortie.glob("*.mp3"))) == 1
    assert not list(sortie.glob("*-tempo.jpg"))
