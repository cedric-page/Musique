import numpy as np
import pytest
import soundfile as sf

from scenario_repetition import SR, SeparateurScenario, repetition
from musique_repet.config import charger_config
from musique_repet.echo import ParametresAntiEcho
from musique_repet.mixage import ParametresMixage
from musique_repet.paroles import SegmentParoles, est_hallucination, transcrire_zone, zones_de_voix
from musique_repet.pipeline import Options, executer
from musique_repet.tempo import ParametresTempo
from musique_repet.titre import (
    EntreeRepertoire,
    ParametresTitre,
    chercher_dans_repertoire,
    compter_occurrences,
    deviner_par_refrain,
    lire_repertoire,
    normaliser,
    replier,
)


def segments(*lignes, ecart=10.0, debut=0.0):
    return [SegmentParoles(debut + k * ecart, debut + k * ecart + 3, texte) for k, texte in enumerate(lignes)]


# --- Texte --------------------------------------------------------------------


def test_normaliser_et_replier():
    assert normaliser("J'ai rêvé d'Élise !") == "j ai reve d elise"
    assert replier("Hallelujah") == replier("Alléluia") == "aleluia"


def test_occurrences_tolerantes():
    phrases = [[replier(m) for m in normaliser(t).split()] for t in ("Alléluia, alléluia", "et on dirait le su", "rien")]
    assert compter_occurrences("Hallelujah", phrases) == 1  # une fois par phrase au plus
    assert compter_occurrences("On dirait le Sud", phrases) == 1  # « su » au lieu de « sud »
    assert compter_occurrences("Imagine", phrases) == 0


# --- Répertoire -----------------------------------------------------------------


def test_lire_repertoire(tmp_path):
    chemin = tmp_path / "repertoire.txt"
    chemin.write_text(
        "# Nos chansons\nHallelujah\n\nLe Sud ; on dirait le sud ; le temps dure longtemps\n", encoding="utf-8"
    )
    assert lire_repertoire(chemin) == [
        EntreeRepertoire("Hallelujah"),
        EntreeRepertoire("Le Sud", ("on dirait le sud", "le temps dure longtemps")),
    ]


REPERTOIRE = [EntreeRepertoire("Hallelujah"), EntreeRepertoire("Le Sud"), EntreeRepertoire("Imagine")]


def test_titre_chante():
    chanson = segments("I heard there was a secret chord", "Hallelujah, hallelujah", "Alléluia")
    trouve = chercher_dans_repertoire(chanson, [], REPERTOIRE)
    assert trouve.titre == "Hallelujah" and trouve.methode == "répertoire"


def test_titre_annonce_avant_la_chanson():
    # Titre absent des paroles, mais annoncé juste avant.
    chanson = segments("c'est un endroit qui ressemble à la Louisiane", "à l'Italie")
    avant = segments("bon allez on fait le Sud", debut=-30)
    assert chercher_dans_repertoire(chanson, avant, REPERTOIRE).titre == "Le Sud"


def test_extraits_de_paroles():
    repertoire = [EntreeRepertoire("Chanson sans titre chanté", ("le temps dure longtemps",))]
    chanson = segments("et le temps dure longtemps", "et la vie sûrement")
    assert chercher_dans_repertoire(chanson, [], repertoire).titre == "Chanson sans titre chanté"


def test_titre_court_demande_plus_d_indices():
    repertoire = [EntreeRepertoire("Home")]
    assert chercher_dans_repertoire(segments("going home tonight"), [], repertoire) is None
    assert chercher_dans_repertoire(segments("going home", "take me home"), [], repertoire).titre == "Home"


def test_titre_ambigu_non_retenu():
    chanson = segments("hallelujah", "imagine")
    assert chercher_dans_repertoire(chanson, [], REPERTOIRE) is None


def test_rien_du_repertoire():
    assert chercher_dans_repertoire(segments("la la la", "une autre chanson"), [], REPERTOIRE) is None


# --- Refrain --------------------------------------------------------------------


def test_refrain_hallelujah():
    chanson = segments(
        "Now I've heard there was a secret chord",
        "Hallelujah, hallelujah",
        "Your faith was strong but you needed proof",
        "Hallelujah, hallelujah",
        "Maybe there's a God above",
        "Hallelujah",
    )
    trouve = deviner_par_refrain(chanson)
    assert trouve.titre == "Hallelujah" and trouve.methode == "refrain"


def test_refrain_garde_accents_et_apostrophes():
    chanson = segments(
        "C'est un endroit qui ressemble à la Louisiane",
        "On dirait le Sud",
        "Le temps dure longtemps",
        "On dirait le Sud",
        "Et la vie sûrement",
        "on dirait le sud",
    )
    assert deviner_par_refrain(chanson).titre == "On dirait le Sud"


def test_refrain_mot_repete_dans_chaque_ligne():
    chanson = segments(*["Hallelujah, hallelujah"] * 4)
    assert deviner_par_refrain(chanson).titre == "Hallelujah"


def test_refrain_ignore_les_vocalises():
    chanson = segments("oh oh oh", "la la la", "oh oh oh", "yeah yeah", "oh oh oh")
    assert deviner_par_refrain(chanson) is None


def test_refrain_trop_peu_repete_ou_d_affilee():
    assert deviner_par_refrain(segments("Hallelujah", "autre chose", "Hallelujah")) is None
    assert deviner_par_refrain(segments("Hallelujah", "Hallelujah", "Hallelujah", ecart=2)) is None


# --- Transcription ----------------------------------------------------------------


@pytest.mark.parametrize(
    "texte,attendu",
    [
        ("Sous-titres réalisés par la communauté d'Amara.org", True),
        ("Merci d'avoir regardé cette vidéo !", True),
        ("♪ ♪ ♪", True),
        ("[Musique]", True),
        ("Thanks for watching!", True),
        ("Hallelujah", False),
        ("On dirait le Sud", False),
    ],
)
def test_hallucinations(texte, attendu):
    assert est_hallucination(texte) is attendu


def test_zones_de_voix():
    sr = 16000
    mono = np.zeros(20 * sr, np.float32)
    t = np.arange(3 * sr) / sr
    for debut in (2, 6, 14):  # 6 s suit 2 s de près : un seul passage
        mono[debut * sr : debut * sr + t.size] = 0.3 * np.sin(2 * np.pi * 220 * t)
    zones = zones_de_voix(mono, sr, pause_max_s=2.0)
    assert len(zones) == 2
    assert zones[0][0] == pytest.approx(1.7, abs=0.3) and zones[0][1] == pytest.approx(9.3, abs=0.3)
    assert zones[1][0] == pytest.approx(13.7, abs=0.3)


class FauxTranscripteur:
    """Renvoie, appel après appel, des phrases prévues (temps relatifs au début de la zone)."""

    identifiant = "faux:v1"

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.appels = []

    def transcrire(self, audio, sr, zones):
        self.appels.append((audio.size / sr, zones))
        lignes = self.reponses.pop(0) if self.reponses else []
        return [SegmentParoles(t, t + 3, texte) for t, texte in lignes]


def test_transcription_par_zone_et_cache(tmp_path):
    voix = tmp_path / "voix.wav"
    t = np.arange(10 * SR) / SR
    sf.write(voix, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)
    faux = FauxTranscripteur([(1.0, "bonjour")])
    cache = tmp_path / "paroles.json"
    premiers = transcrire_zone(voix, 4.0, 9.0, cache, faux)
    assert premiers == [SegmentParoles(5.0, 8.0, "bonjour")]  # temps depuis le début de l'enregistrement
    assert faux.appels[0][0] == pytest.approx(5.0, abs=0.01)  # seule la zone demandée est transcrite
    assert transcrire_zone(voix, 4.0, 9.0, cache, faux) == premiers
    assert len(faux.appels) == 1  # réutilisé depuis le cache


# --- Chaîne complète (sans CSV) -------------------------------------------------------


@pytest.fixture(scope="module")
def scenario():
    return repetition()


def preparer(tmp_path, pistes, nom="repet_mardi.wav"):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    mix = sum(pistes.values())
    sf.write(entree / nom, np.repeat(mix[:, None], 2, axis=1), SR, subtype="PCM_24")
    return entree, sortie


def options(entree, sortie, **kw):
    # Anti-écho et tempo, testés par ailleurs, sont coupés pour accélérer ces tests.
    reglages = {"anti_echo": ParametresAntiEcho(attenuation_max_db=0), "tempo": None, **kw}
    return Options(
        entree=entree, sortie=sortie, travail=sortie / "_travail",
        facteurs=charger_config(None), mixage=ParametresMixage(), **reglages,
    )


REFRAIN = [(k * 12.0, "Hallelujah, hallelujah") for k in range(5)] + [(6.0, "I heard there was a secret chord")]


def test_sans_csv_chanson_coupee_et_titre_reconnu(tmp_path, scenario, monkeypatch):
    pistes, debut, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    faux = FauxTranscripteur(REFRAIN)
    avec_tempo = options(entree, sortie, tempo=ParametresTempo())
    assert executer(avec_tempo, SeparateurScenario(pistes), faux) == 0

    mp3 = sortie / "20260312-Hallelujah.mp3"
    assert mp3.exists() and (sortie / "20260312-Hallelujah-tempo.jpg").exists()
    duree = sf.info(mp3).duration
    assert 72 <= duree <= 80  # 120 s d'enregistrement, ~76 s de chanson
    # Seule la chanson a été transcrite (pas de répertoire : rien à chercher avant).
    assert len(faux.appels) == 1 and faux.appels[0][0] == pytest.approx(duree, abs=1)

    # Relance : transcription réutilisée, même nom.
    assert executer(options(entree, sortie), SeparateurScenario(pistes), faux) == 0
    assert len(faux.appels) == 1
    assert sorted(p.name for p in sortie.glob("*.mp3")) == ["20260312-Hallelujah.mp3"]


def test_sans_csv_titre_du_repertoire_annonce(tmp_path, scenario, monkeypatch):
    pistes, _, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    faux = FauxTranscripteur(
        [(5.0, "c'est un endroit qui ressemble à la Louisiane")],  # la chanson
        [(20.0, "bon, on fait le Sud ?"), (25.0, "ok")],  # juste avant
    )
    opts = options(entree, sortie, titres=ParametresTitre(repertoire=REPERTOIRE_TEST))
    assert executer(opts, SeparateurScenario(pistes), faux) == 0
    assert (sortie / "20260312-Le Sud.mp3").exists()
    assert len(faux.appels) == 2


REPERTOIRE_TEST = [EntreeRepertoire("Le Sud"), EntreeRepertoire("Hallelujah")]


def test_sans_csv_titre_non_reconnu_garde_le_nom(tmp_path, scenario, monkeypatch):
    pistes, _, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    assert executer(options(entree, sortie), SeparateurScenario(pistes), FauxTranscripteur([])) == 0
    assert (sortie / "20260312-repet_mardi.mp3").exists()


def test_transcription_impossible_n_empeche_pas_le_mp3(tmp_path, scenario, monkeypatch, caplog):
    class TranscripteurHorsLigne:
        identifiant = "hors-ligne"

        def transcrire(self, audio, sr, zones):
            raise RuntimeError("impossible de télécharger le modèle")

    pistes, _, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    assert executer(options(entree, sortie), SeparateurScenario(pistes), TranscripteurHorsLigne()) == 0
    assert (sortie / "20260312-repet_mardi.mp3").exists()
    assert "Reconnaissance d'après les paroles impossible" in caplog.text


def test_sans_rognage_garde_tout(tmp_path, scenario, monkeypatch):
    pistes, _, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    assert executer(options(entree, sortie, rognage=None, titres=None), SeparateurScenario(pistes)) == 0
    assert sf.info(sortie / "20260312-repet_mardi.mp3").duration == pytest.approx(120, abs=0.5)


def test_avec_csv_ni_coupe_ni_renomme(tmp_path, scenario, monkeypatch):
    pistes, _, _ = scenario
    entree, sortie = preparer(tmp_path, pistes)
    (entree / "repet_mardi.csv").write_text("Mon titre;0;1:00\n", encoding="utf-8")
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    faux = FauxTranscripteur(REFRAIN)
    assert executer(options(entree, sortie), SeparateurScenario(pistes), faux) == 0
    assert sf.info(sortie / "20260312-Mon titre.mp3").duration == pytest.approx(60, abs=0.5)
    assert faux.appels == []
