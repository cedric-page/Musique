import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from chant_synthetique import ACCORDS, MELODIE, SR, accompagnement, voix
from musique_repet.graphique_justesse import tracer
from musique_repet.hauteur import Contours, attribuer_voix
from musique_repet.justesse import (
    AnalyseJustesse,
    Note,
    analyser_justesse,
    combler_trous,
    decouper_notes,
    tonalite,
)

ACCORDAGE = -14  # le groupe est accordé un peu bas : La = 436,5 Hz


def ecrire(tmp_path, **pistes):
    chemins = {}
    for nom, audio in pistes.items():
        chemins[nom] = tmp_path / f"{nom}.wav"
        sf.write(chemins[nom], np.repeat(audio[:, None], 2, axis=1), SR, subtype="FLOAT")
    return chemins


def note_proche(analyse, debut):
    return min(analyse.notes, key=lambda n: abs(n.debut - debut))


@pytest.fixture(scope="module")
def melodie(tmp_path_factory):
    dossier = tmp_path_factory.mktemp("melodie")
    pistes = ecrire(
        dossier, voix=voix(16, MELODIE, ACCORDAGE), piano=accompagnement(16, ACCORDS, ACCORDAGE)
    )
    resultat = analyser_justesse(pistes)
    assert not resultat.harmonie.detecte  # une seule voix : tout est au lead
    return resultat.lead


def test_accordage_et_tonalite_du_groupe(melodie):
    assert melodie.reference == "instruments"
    assert melodie.accordage_cents == pytest.approx(ACCORDAGE, abs=3)
    assert melodie.tonalite == "Do majeur"


def test_chaque_note_mesuree(melodie):
    assert len(melodie.notes) == len(MELODIE)
    for debut, _, visee, ecart in MELODIE:
        note = note_proche(melodie, debut)
        assert note.visee == visee
        assert note.ecart == pytest.approx(ecart, abs=6)


def test_categories(melodie):
    categories = {debut: melodie.categorie(note_proche(melodie, debut)) for debut, *_ in MELODIE}
    assert categories[0.1] == "juste"
    assert categories[3.1] == "approximatif"  # +30 cents
    assert categories[5.1] == "faux"  # -50 cents : Mi chanté très bas, pas Mi bémol trop haut
    assert note_proche(melodie, 5.1).visee == 76
    assert categories[10.1] == "approximatif"  # -25 cents


def test_fausse_note_hors_harmonie(melodie):
    hors = [n for n in melodie.notes if n.hors_harmonie]
    assert len(hors) == 1 and hors[0].nom == "Fa♯5"  # chanté bien droit, mais faux sur l'accord de Do
    assert abs(hors[0].ecart) < 10 and melodie.categorie(hors[0]) == "faux"


# --- Lead et harmonie -------------------------------------------------------------

GAMME = [55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84]
LEAD = [(d, l, m, 0) for d, l, m, _ in MELODIE if m in GAMME]  # mélodie chantée juste


def tierces(notes, sens, ecart=-30, depuis=0.0):
    """Harmonie à la tierce (au-dessus : sens = 1, en dessous : -1), chantée ``ecart`` cents trop bas."""
    return [(d, l, GAMME[GAMME.index(m) + 2 * sens], ecart) for d, l, m, _ in notes if d >= depuis]


def duo(tmp_path, lead, harmonie, volume_harmonie=0.6):
    pistes = ecrire(
        tmp_path,
        voix=voix(16, lead, ACCORDAGE, graine=1) + volume_harmonie * voix(16, harmonie, ACCORDAGE, graine=2),
        piano=accompagnement(16, ACCORDS, ACCORDAGE),
    )
    return analyser_justesse(pistes)


def verifier(resultat, nb_lead, nb_harmonie, ecart_harmonie=-30):
    lead, harmonie = resultat.lead, resultat.harmonie
    assert lead.voix == "lead" and harmonie.voix == "harmonie"
    assert len(lead.notes) == nb_lead and all(abs(n.ecart) < 8 for n in lead.notes)
    assert len(harmonie.notes) >= nb_harmonie
    # Les éventuels fragments en plus restent dans l'harmonie : même écart.
    assert all(abs(n.ecart - ecart_harmonie) < 10 for n in harmonie.notes)


def test_harmonie_en_dessous_a_partir_du_refrain(tmp_path):
    harmonie = tierces(LEAD, -1, depuis=8)
    resultat = duo(tmp_path, LEAD, harmonie)
    verifier(resultat, len(LEAD), len(harmonie))
    assert resultat.harmonie.categorie(resultat.harmonie.notes[0]) == "approximatif"
    assert min(n.debut for n in resultat.harmonie.notes) >= 7.9


def test_harmonie_au_dessus_plus_douce(tmp_path):
    verifier(duo(tmp_path, LEAD, tierces(LEAD, 1)), len(LEAD), len(LEAD))


def test_harmonie_qui_tient_sa_note_apres_le_lead(tmp_path):
    lead = [n for n in LEAD if not 6 <= n[0] < 8] + [(6.1, 0.8, 77, 0)]  # le lead s'arrête à 6,9 s
    resultat = duo(tmp_path, lead, [(6.1, 1.8, 74, -30)])  # l'harmonie tient jusqu'à 7,9 s
    verifier(resultat, len(lead), 1)
    assert resultat.harmonie.notes[0].duree == pytest.approx(1.8, abs=0.15)


def test_harmonie_qui_entre_avant_le_lead(tmp_path):
    lead = [n for n in LEAD if not 10 <= n[0] < 12]  # le lead se tait de 10 à 12 s
    resultat = duo(tmp_path, lead, [(11.0, 2.0, 74, -30)])  # l'harmonie entre seule à 11 s
    verifier(resultat, len(lead), 1)
    assert resultat.harmonie.notes[0].debut == pytest.approx(11.0, abs=0.15)


def test_volume_egal_departage_par_le_couplet_chante_seul(tmp_path):
    # Couplet seul (0-8 s), puis refrain à deux voix au même volume, harmonie au-dessus :
    # le lead est la voix qui prolonge le registre du couplet.
    harmonie = tierces(LEAD, 1, depuis=8)
    verifier(duo(tmp_path, LEAD, harmonie, volume_harmonie=1.0), len(LEAD), len(harmonie))


def test_attribution_stable_quand_la_voix_dominante_change_d_un_instant_a_l_autre():
    # Deux voix tenues (Do et La) ; la plus forte selon le spectre change au hasard
    # d'une trame à l'autre, mais le lead (Do) a plus d'énergie au total.
    rng = np.random.default_rng(0)
    n = 300
    do_est_principale = rng.random(n) < 0.6
    principale = np.where(do_est_principale, 72.0, 69.0)
    seconde = np.where(do_est_principale, 69.0, 72.0)
    energie_do, energie_la = 1.0, 0.7
    contours = Contours(
        np.arange(n) / 62.5, principale, seconde,
        np.where(do_est_principale, energie_do, energie_la), np.where(do_est_principale, energie_la, energie_do),
    )
    lead, harmonie = attribuer_voix(contours)
    assert np.all(lead == 72.0) and np.all(harmonie == 69.0)


def test_voix_seule_toujours_au_lead():
    n = 100
    contours = Contours(np.arange(n) / 62.5, np.full(n, 67.0), np.full(n, np.nan))
    lead, harmonie = attribuer_voix(contours)
    assert np.all(lead == 67.0) and np.isnan(harmonie).all()


def test_a_cappella_reference_des_voix(tmp_path):
    # Sans instrument, la référence est l'accordage moyen des voix elles-mêmes.
    notes = [(d, l, m, 0) for d, l, m, _ in MELODIE if m != 78]
    analyse = analyser_justesse(ecrire(tmp_path, voix=voix(16, notes, 30))).lead
    assert analyse.reference == "voix" and analyse.tonalite is None
    assert analyse.accordage_cents == pytest.approx(30, abs=4)
    assert analyse.part_juste > 0.95


@pytest.mark.parametrize(
    "signal",
    [np.zeros(10 * SR, np.float32), 0.05 * np.random.default_rng(0).standard_normal(10 * SR).astype(np.float32)],
    ids=["silence", "bruit"],
)
def test_pas_de_chant(tmp_path, signal):
    resultat = analyser_justesse(ecrire(tmp_path, voix=signal, piano=accompagnement(10, ACCORDS[:5])))
    assert not resultat.lead.detecte and not resultat.harmonie.detecte
    assert resultat.lead.resume() == "pas de chant détecté"
    assert resultat.harmonie.resume() == "pas d'harmonie détectée"


def test_decouper_notes_ignore_le_vibrato():
    fps = 62.5
    t = np.arange(int(0.6 * fps)) / fps
    vibrato = 0.35 * np.sin(2 * np.pi * 5.5 * t)
    contour = np.concatenate([69 + vibrato, [np.nan] * 3, 72 + vibrato, 74 + vibrato])
    notes = decouper_notes(contour, fps, 0.15)
    assert len(notes) == 3
    assert [b - a for a, b in notes] == [t.size] * 3


def test_combler_trous():
    contour = np.array([60.0, 60.1, np.nan, np.nan, 60.2, np.nan, 65.0, np.nan, np.nan, np.nan, np.nan, 65.0])
    resultat = combler_trous(contour, 2)
    assert np.isfinite(resultat[2:4]).all()  # même note de part et d'autre : relié
    assert np.isnan(resultat[5])  # changement de note : pas relié
    assert np.isnan(resultat[7:11]).all()  # trou trop long


def test_tonalite():
    do_majeur = np.zeros(12)
    do_majeur[[0, 4, 7]] = 3
    do_majeur[[2, 5, 9, 11]] = 1
    assert tonalite(do_majeur) == (0, "majeur")
    la_mineur = np.zeros(12)
    la_mineur[[9, 0, 4]] = 3
    la_mineur[[11, 2, 5, 7]] = 1
    assert tonalite(la_mineur) == (9, "mineur")


def test_tranches():
    notes = [Note(1, 3, 60, 60, 0), Note(3, 7, 62, 62, -30), Note(9, 11, 64, 64, 5), Note(17.5, 17.8, 60, 60, 0)]
    analyse = AnalyseJustesse(20, notes, 0.0, "instruments", "Do majeur")
    tranches = analyse.tranches(8)
    assert [(a, b) for a, b, _ in tranches] == [(0, 8), (8, 16)]  # la dernière tranche chante trop peu
    assert tranches[0][2] == pytest.approx(2 / 6)
    assert tranches[1][2] == 1


@pytest.mark.parametrize("cas", ["lead", "harmonie", "vide"])
def test_graphique_jpg(tmp_path, melodie, cas):
    if cas == "lead":
        analyse = melodie
    elif cas == "harmonie":
        analyse = AnalyseJustesse(16, melodie.notes[:5], -14.0, "instruments", "Do majeur", voix="harmonie")
    else:
        analyse = AnalyseJustesse(16, [], 0.0, "instruments", None)
    chemin = tmp_path / "20260312-Essai-justesse.jpg"
    tracer(analyse, chemin, "Essai", "20260312")
    with Image.open(chemin) as image:
        assert image.format == "JPEG" and image.size == (1800, 1020)
