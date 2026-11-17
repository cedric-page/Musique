import io

import numpy as np
import pytest
import soundfile as sf

from scenario_repetition import SR, SeparateurScenario, repetition
from musique_repet.config import charger_config
from musique_repet.echo import ParametresAntiEcho
from musique_repet.empreinte import (
    MorceauReconnu,
    ReconnaisseurShazam,
    decider,
    extraits,
    identifier,
    lire_reponse_shazam,
    nettoyer_titre,
    titre_du_repertoire,
)
from musique_repet.mixage import ParametresMixage
from musique_repet.paroles import SegmentParoles
from musique_repet.pipeline import Options, executer
from musique_repet.titre import EntreeRepertoire, ParametresTitre

HALLELUJAH = MorceauReconnu("Hallelujah", "Jeff Buckley", "41191926")
LE_SUD = MorceauReconnu("Le Sud", "Nino Ferrer", "52312")


# --- Choix des extraits et décision ------------------------------------------------


def test_extraits_dans_la_chanson():
    assert extraits(0, 120) == [(30, 42), (60, 72), (90, 102)]
    assert extraits(10, 30) == [(10, 22), (15, 27), (18, 30)]  # le dernier ne dépasse pas la fin
    assert extraits(5, 14) == [(5, 14)]  # chanson très courte : un seul extrait


def test_deux_extraits_d_accord():
    assert decider([(HALLELUJAH, 0.02), (HALLELUJAH, 0.03)]) == (HALLELUJAH, "trouvé sur 2 extraits")


def test_un_seul_extrait_exact():
    morceau, detail = decider([(LE_SUD, 0.0004)])
    assert morceau == LE_SUD and "exacte" in detail


@pytest.mark.parametrize(
    "resultats",
    [[(HALLELUJAH, 0.02)], [(HALLELUJAH, 0.02), (LE_SUD, 0.05)], []],
    ids=["une-reponse-decalee", "deux-reponses-differentes", "rien"],
)
def test_reponses_pas_assez_sures(resultats):
    assert decider(resultats) is None


@pytest.mark.parametrize(
    "brut,propre",
    [
        ("Le Sud (Remastered 2011)", "Le Sud"),
        ("Hallelujah - Live", "Hallelujah"),
        ("Imagine - Remastered 2010", "Imagine"),
        ("Hey Jude [Mono]", "Hey Jude"),
        ("Song 2 (Live) [Remastered]", "Song 2"),
        ("Let It Be (feat. Quelqu'un)", "Let It Be"),
        ("(I Can't Get No) Satisfaction", "(I Can't Get No) Satisfaction"),
        ("Sweet Child O' Mine", "Sweet Child O' Mine"),
    ],
)
def test_nettoyer_titre(brut, propre):
    assert nettoyer_titre(brut) == propre


def test_orthographe_du_repertoire():
    repertoire = [EntreeRepertoire("Le Sud"), EntreeRepertoire("Alléluia")]
    assert titre_du_repertoire("le sud", repertoire) == "Le Sud"
    assert titre_du_repertoire("Imagine", repertoire) == "Imagine"


# --- Réponse de Shazam (sans réseau) -------------------------------------------------

REPONSE_SHAZAM = {
    "matches": [{"id": "1", "offset": 31.2, "timeskew": 0.00021, "frequencyskew": -0.0004}],
    "track": {"key": "41191926", "title": "Hallelujah", "subtitle": "Jeff Buckley"},
}


def test_lire_reponse_shazam():
    assert lire_reponse_shazam(REPONSE_SHAZAM) == (HALLELUJAH, pytest.approx(0.0004))
    assert lire_reponse_shazam({"matches": [], "tagid": "x"}) is None


def test_reconnaisseur_shazam_envoie_un_extrait(monkeypatch):
    recu = {}

    async def faux_recognize(self, data, proxy=None, options=None, *, max_skew=None):
        recu["audio"], recu["sr"] = sf.read(io.BytesIO(data))
        recu["fenetre"] = options.segment_duration_seconds
        return REPONSE_SHAZAM

    monkeypatch.setattr("shazamio.Shazam.recognize", faux_recognize)
    extrait = np.zeros((12 * SR, 2), np.float32)
    assert ReconnaisseurShazam().reconnaitre(extrait, SR) == (HALLELUJAH, pytest.approx(0.0004))
    assert recu["sr"] == SR and recu["audio"].shape[0] == 12 * SR and recu["fenetre"] == 12


# --- Identification avec cache ---------------------------------------------------------


class FauxReconnaisseur:
    """Renvoie les réponses prévues, appel après appel (None = pas de résultat ; une exception est levée)."""

    identifiant = "shazam"

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.appels = 0

    def reconnaitre(self, audio, sr):
        self.appels += 1
        reponse = self.reponses.pop(0) if self.reponses else None
        if isinstance(reponse, Exception):
            raise reponse
        return reponse


@pytest.fixture
def wav(tmp_path):
    chemin = tmp_path / "repet.wav"
    sf.write(chemin, np.zeros((60 * 8000, 1), np.float32), 8000)
    return chemin


def test_identifier_s_arrete_des_que_c_est_sur_et_garde_le_resultat(tmp_path, wav):
    faux = FauxReconnaisseur((HALLELUJAH, 0.02), (HALLELUJAH, 0.01), (LE_SUD, 0.0))
    cache = tmp_path / "empreinte.json"
    trouve = identifier(wav, 0, 60, faux, cache)
    assert trouve.titre == "Hallelujah" and trouve.methode == "Shazam"
    assert trouve.detail == "Jeff Buckley, trouvé sur 2 extraits"
    assert faux.appels == 2  # le troisième extrait n'a pas été nécessaire
    assert identifier(wav, 0, 60, faux, cache).titre == "Hallelujah"
    assert faux.appels == 2  # relance : réponse gardée


def test_resultat_negatif_garde_mais_pas_les_erreurs(tmp_path, wav):
    cache = tmp_path / "empreinte.json"
    hors_ligne = FauxReconnaisseur(OSError("pas de connexion"))
    with pytest.raises(OSError):
        identifier(wav, 0, 60, hors_ligne, cache)
    assert not cache.exists()  # une erreur n'est pas gardée : on réessaiera
    rien = FauxReconnaisseur(None, None, None)
    assert identifier(wav, 0, 60, rien, cache) is None
    assert identifier(wav, 0, 60, rien, cache) is None
    assert rien.appels == 3  # « pas trouvé » est gardé


# --- Chaîne complète (sans CSV) --------------------------------------------------------


@pytest.fixture(scope="module")
def scenario():
    return repetition()


def lancer(tmp_path, pistes, reconnaisseur, transcripteur=None, repertoire=None):
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "repet_mardi.wav", np.repeat(sum(pistes.values())[:, None], 2, axis=1), SR, subtype="PCM_24")
    options = Options(
        entree=entree, sortie=sortie, travail=sortie / "_travail", facteurs=charger_config(None),
        mixage=ParametresMixage(), anti_echo=ParametresAntiEcho(attenuation_max_db=0), tempo=None,
        titres=ParametresTitre(repertoire=repertoire or []),
    )
    assert executer(options, SeparateurScenario(pistes), transcripteur, reconnaisseur) == 0
    return sorted(p.name for p in sortie.glob("*.mp3"))


class FauxTranscripteur:
    identifiant = "faux:v1"

    def __init__(self, lignes):
        self.lignes = lignes
        self.appels = 0

    def transcrire(self, audio, sr, zones):
        self.appels += 1
        return [SegmentParoles(t, t + 3, texte) for t, texte in self.lignes]


REFRAIN_LE_SUD = [(k * 12.0, "On dirait le Sud") for k in range(5)]


@pytest.fixture(autouse=True)
def date_fixe(monkeypatch):
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")


def test_titre_trouve_par_shazam(tmp_path, scenario):
    pistes, _, _ = scenario
    shazam = FauxReconnaisseur((MorceauReconnu("Hallelujah (Live)", "Jeff Buckley", "1"), 0.01),
                               (MorceauReconnu("Hallelujah (Live)", "Jeff Buckley", "1"), 0.02))
    paroles = FauxTranscripteur(REFRAIN_LE_SUD)
    assert lancer(tmp_path, pistes, shazam, paroles) == ["20260312-Hallelujah.mp3"]
    assert paroles.appels == 0  # Shazam a suffi : pas de transcription


def test_sans_reponse_de_shazam_les_paroles_prennent_le_relais(tmp_path, scenario):
    pistes, _, _ = scenario
    shazam = FauxReconnaisseur((HALLELUJAH, 0.03), None, None)  # une seule réponse, décalée : écartée
    paroles = FauxTranscripteur(REFRAIN_LE_SUD)
    assert lancer(tmp_path, pistes, shazam, paroles) == ["20260312-On dirait le Sud.mp3"]
    assert shazam.appels == 3 and paroles.appels == 1


def test_shazam_injoignable_les_paroles_prennent_le_relais(tmp_path, scenario, caplog):
    pistes, _, _ = scenario
    shazam = FauxReconnaisseur(OSError("pas de connexion"))
    noms = lancer(tmp_path, pistes, shazam, FauxTranscripteur(REFRAIN_LE_SUD), [EntreeRepertoire("Le Sud")])
    assert noms == ["20260312-Le Sud.mp3"]  # trouvé par le répertoire, d'après les paroles
    assert "Reconnaissance par Shazam impossible" in caplog.text


def test_shazam_seul_reprend_l_orthographe_du_repertoire(tmp_path, scenario):
    pistes, _, _ = scenario
    shazam = FauxReconnaisseur((MorceauReconnu("LE SUD - Remastered", "Nino Ferrer", "52312"), 0.0))
    assert lancer(tmp_path, pistes, shazam, repertoire=[EntreeRepertoire("Le Sud")]) == ["20260312-Le Sud.mp3"]
