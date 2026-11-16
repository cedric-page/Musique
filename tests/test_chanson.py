import numpy as np
import pytest
import soundfile as sf

from scenario_repetition import SR, accords, chant, parole, repetition
from musique_repet.chanson import detecter_chanson
from musique_repet.mixage import fondus


def ecrire(dossier, pistes):
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = {}
    for nom, audio in pistes.items():
        chemins[nom] = dossier / f"{nom}.wav"
        sf.write(chemins[nom], audio, SR, subtype="PCM_16")
    return chemins


@pytest.fixture(scope="module")
def scenario():
    return repetition()


def test_repetition_type(tmp_path, scenario):
    # Discussion et accordage avant, intro au piano seul, chanson, discussion après.
    pistes, debut_attendu, fin_instruments = scenario
    extrait = detecter_chanson(ecrire(tmp_path, pistes))
    assert extrait.debut == pytest.approx(debut_attendu - 0.5, abs=0.6)  # l'intro au piano est gardée
    assert fin_instruments + 0.5 <= extrait.fin <= fin_instruments + 4.5  # avec la résonance finale
    assert extrait.rogne and "coupées au début" in extrait.description()


def test_long_bavardage_avant(tmp_path):
    pistes, debut_attendu, fin_instruments = repetition(bavardage_avant=300)
    extrait = detecter_chanson(ecrire(tmp_path, pistes))
    assert extrait.debut == pytest.approx(debut_attendu - 0.5, abs=0.6)
    assert fin_instruments + 0.5 <= extrait.fin <= fin_instruments + 4.5


def test_chanson_seule_rien_a_couper(tmp_path, scenario):
    pistes, _, _ = scenario
    chanson = {nom: v[30 * SR : 95 * SR] for nom, v in pistes.items()}
    extrait = detecter_chanson(ecrire(tmp_path, chanson))
    assert not extrait.rogne
    assert extrait.description() == "enregistrement gardé en entier"


def test_discussion_seule(tmp_path, scenario):
    # Pas de chanson : seuls les silences du début et de la fin sont retirés.
    pistes, _, _ = scenario
    voix = {nom: (v if nom == "voix" else np.zeros_like(v)) for nom, v in pistes.items()}
    extrait = detecter_chanson(ecrire(tmp_path, voix))
    assert extrait.debut <= 2.0 and extrait.fin >= 115


def test_silence(tmp_path, scenario):
    pistes, _, _ = scenario
    extrait = detecter_chanson(ecrire(tmp_path, {nom: np.zeros_like(v) for nom, v in pistes.items()}))
    assert not extrait.rogne


def test_ballade_piano_voix(tmp_path, scenario):
    # Deux sources seulement : le seuil s'adapte à la taille de la formation.
    pistes, _, _ = scenario
    rng = np.random.default_rng(5)
    ballade = {nom: np.zeros_like(v) for nom, v in pistes.items()}
    ballade["piano"] = accords(ballade["piano"], 30, 95, 70, [262, 330, 392], -20).astype(np.float32)
    ballade["voix"] = parole(ballade["voix"], 2, 20, rng)
    ballade["voix"] = chant(ballade["voix"], 34, 94, rng)
    ballade["voix"] = parole(ballade["voix"], 100, 115, rng).astype(np.float32)
    extrait = detecter_chanson(ecrire(tmp_path, ballade))
    assert extrait.debut == pytest.approx(29.5, abs=1)
    assert 95 <= extrait.fin <= 99


def test_fondus():
    audio = np.ones((SR, 2), np.float32)
    resultat = fondus(audio, SR, 0.1, 0.5)
    assert resultat[0, 0] == 0 and resultat[-1, 0] == 0
    assert resultat[SR // 2, 0] == 1
    assert np.all(np.diff(resultat[: int(0.1 * SR), 0]) >= 0)  # montée régulière
    assert np.array_equal(audio, np.ones((SR, 2), np.float32))  # l'original n'est pas modifié
