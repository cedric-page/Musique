import json
import math

import numpy as np
import pytest
import soundfile as sf

from scenario_repetition import SR, SeparateurScenario, chant, repetition
from musique_repet.chanson import detecter_chanson
from musique_repet.config import charger_config
from musique_repet.echo import ParametresAntiEcho
from musique_repet.mixage import ParametresMixage, calculer_equilibrage
from musique_repet.pipeline import Options, executer
from musique_repet.voix import SeparateurLeadChoeurs, separer_voix_avec_cache


def sinus(duree, freq, amplitude, sr=SR):
    t = np.arange(int(duree * sr)) / sr
    return np.repeat((amplitude * np.sin(2 * np.pi * freq * t))[:, None], 2, axis=1).astype(np.float32)


class FauxSeparateurVoix:
    """Lead = 70 % de la piste voix, chœurs = 30 % (ou ce que renvoie ``fonction``)."""

    identifiant = "faux-voix:v1"

    def __init__(self, fonction=None):
        self.fonction = fonction or (lambda audio: (0.7 * audio, 0.3 * audio))
        self.appels = 0

    def separer(self, voix):
        self.appels += 1
        audio, sr = sf.read(voix, dtype="float32", always_2d=True)
        lead, choeurs = self.fonction(audio)
        return lead, choeurs, sr


@pytest.fixture
def pistes(tmp_path):
    chemins = {}
    for nom, audio in (("voix", sinus(3, 440, 0.3)), ("piano", sinus(3, 262, 0.2))):
        chemins[nom] = tmp_path / f"{nom}.wav"
        sf.write(chemins[nom], audio, SR, subtype="FLOAT")
    return chemins


# --- Séparation et cache ---------------------------------------------------------------


def test_voix_remplacee_par_lead_et_choeurs(tmp_path, pistes):
    faux = FauxSeparateurVoix()
    resultat = separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux)
    assert sorted(resultat) == ["choeurs", "lead", "piano"]
    lead, _ = sf.read(resultat["lead"], always_2d=True)
    choeurs, _ = sf.read(resultat["choeurs"], always_2d=True)
    voix, _ = sf.read(pistes["voix"], always_2d=True)
    assert lead.shape == choeurs.shape == voix.shape
    assert np.allclose(lead + choeurs, voix, atol=1e-6)  # rien n'est perdu
    # Relance : réutilisé ; --forcer-separation : refait.
    assert separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux) == resultat and faux.appels == 1
    separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux, forcer=True)
    assert faux.appels == 2


def test_longueur_et_canaux_ajustes(tmp_path, pistes):
    # Le modèle rend un peu plus court, en mono : ramené au format de la piste voix.
    faux = FauxSeparateurVoix(lambda a: (a[:-100, :1] * 0.5, a[:-100, :1] * 0.5))
    resultat = separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux)
    assert sf.info(resultat["lead"]).frames == sf.info(pistes["voix"]).frames
    assert sf.info(resultat["lead"]).channels == 2


def test_sans_lead_net_la_voix_reste_entiere(tmp_path, pistes, caplog):
    # Le modèle met tout dans les chœurs : séparation jugée ratée, et cette décision est gardée.
    faux = FauxSeparateurVoix(lambda a: (np.zeros_like(a), a))
    assert separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux) == pistes
    assert "Pas de voix lead nette" in caplog.text
    assert separer_voix_avec_cache(pistes, tmp_path / "lead_choeurs", faux) == pistes
    assert faux.appels == 1
    meta = json.loads((tmp_path / "lead_choeurs" / "separation_voix.json").read_text())
    assert meta["separee"] is False


def test_pas_de_separation_sans_chant_ni_separateur(tmp_path, pistes):
    faux = FauxSeparateurVoix()
    assert separer_voix_avec_cache(pistes, tmp_path / "a", None) == pistes
    sf.write(pistes["voix"], np.zeros((SR, 2), np.float32), SR, subtype="FLOAT")
    assert separer_voix_avec_cache(pistes, tmp_path / "b", faux) == pistes
    assert faux.appels == 0  # personne ne chante : rien à séparer


# --- Modèle audio-separator (simulé, sans téléchargement) ---------------------------------


class FauxSeparator:
    """Imite audio_separator.separator.Separator : écrit une piste « (Vocals) » et une autre."""

    derniers_reglages = {}

    def __init__(self, **reglages):
        FauxSeparator.derniers_reglages = reglages
        self.output_dir = reglages["output_dir"]

    def load_model(self, nom):
        self.modele = nom

    def separate(self, chemin):
        audio, sr = sf.read(chemin, always_2d=True)
        base = f"{self.output_dir}/voix"
        sf.write(f"{base}_(Vocals)_{self.modele}.wav", audio * 0.25, sr, subtype="FLOAT")
        sf.write(f"{base}_(Instrumental)_{self.modele}.wav", audio * 0.75, sr, subtype="FLOAT")
        return []


@pytest.mark.parametrize(
    "modele,part_lead",
    [("UVR-BVE-4B_SN-44100-2.pth", 0.75), ("mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", 0.25)],
    ids=["bve", "karaoke"],
)
def test_quelle_piste_est_le_lead_selon_le_modele(tmp_path, pistes, monkeypatch, modele, part_lead):
    # Un modèle BVE extrait les chœurs (« Vocals » = chœurs) ; un modèle karaoké extrait le lead.
    monkeypatch.setattr("audio_separator.separator.Separator", FauxSeparator)
    lead, choeurs, sr = SeparateurLeadChoeurs(modele, dossier_modeles=tmp_path / "modeles").separer(pistes["voix"])
    voix, _ = sf.read(pistes["voix"], always_2d=True)
    assert sr == SR
    assert np.allclose(lead, part_lead * voix, atol=1e-6)
    assert np.allclose(choeurs, (1 - part_lead) * voix, atol=1e-6)
    # Pas de renormalisation séparée des deux pistes : l'équilibre entre chanteuses est gardé.
    assert FauxSeparator.derniers_reglages["normalization_threshold"] == 1.0


def test_sorties_inattendues(tmp_path, pistes, monkeypatch):
    class UneSeulePiste(FauxSeparator):
        def separate(self, chemin):
            audio, sr = sf.read(chemin, always_2d=True)
            sf.write(f"{self.output_dir}/voix_(Other).wav", audio, sr)
            return []

    monkeypatch.setattr("audio_separator.separator.Separator", UneSeulePiste)
    with pytest.raises(RuntimeError, match="Vocals"):
        SeparateurLeadChoeurs(dossier_modeles=tmp_path / "modeles").separer(pistes["voix"])


# --- Réglages, équilibrage, repérage de la chanson ----------------------------------------


def test_reglages_de_volume(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"Chœurs": 0.5, "voix": 1.2, "voix lead": 1.5}), encoding="utf-8")
    gains = charger_config(chemin)
    assert (gains["lead"], gains["choeurs"], gains["voix"]) == (1.5, 0.5, 1.2)
    chemin.write_text(json.dumps({"voix": 0.8}), encoding="utf-8")  # « voix » règle les deux
    gains = charger_config(chemin)
    assert gains["lead"] == gains["choeurs"] == 0.8


def test_les_choeurs_suivent_le_lead_a_l_equilibrage():
    sonies = {"lead": -20.0, "choeurs": -30.0, "piano": -26.0, "batterie": -14.0}
    gains = {r.nom: r.gain_equilibrage_db for r in calculer_equilibrage(sonies, {}, ParametresMixage())}
    assert gains["piano"] == pytest.approx(6) and gains["batterie"] == pytest.approx(-6)
    assert gains["lead"] == pytest.approx(0)
    assert gains["choeurs"] == gains["lead"]  # 10 dB sous le lead, comme chanté


def test_choeurs_equilibres_normalement_sans_lead():
    sonies = {"lead": -math.inf, "choeurs": -30.0, "piano": -20.0}
    gains = {r.nom: r.gain_equilibrage_db for r in calculer_equilibrage(sonies, {}, ParametresMixage())}
    assert gains["choeurs"] == pytest.approx(5) and gains["lead"] == 0


def test_lead_et_choeurs_comptent_pour_une_seule_source(tmp_path):
    # Même repérage de la chanson avec une piste voix ou avec lead + chœurs séparés.
    pistes, debut, _ = repetition()
    rng = np.random.default_rng(3)
    choeurs = chant(np.zeros_like(pistes["voix"]), 60, 90, rng, niveau=-24).astype(np.float32)
    separees = {**{n: v for n, v in pistes.items() if n != "voix"}, "lead": pistes["voix"], "choeurs": choeurs}
    chemins = {}
    for nom, audio in separees.items():
        chemins[nom] = tmp_path / f"{nom}.wav"
        sf.write(chemins[nom], audio, SR, subtype="PCM_16")
    extrait = detecter_chanson(chemins)
    assert extrait.debut == pytest.approx(debut - 0.5, abs=0.6)
    assert 95.5 <= extrait.fin <= 99.5


# --- Chaîne complète --------------------------------------------------------------------


def test_chaine_complete_avec_lead_et_choeurs(tmp_path, monkeypatch):
    pistes, _, _ = repetition()
    entree, sortie = tmp_path / "entree", tmp_path / "sortie"
    entree.mkdir()
    sf.write(entree / "repet.wav", np.repeat(sum(pistes.values())[:, None], 2, axis=1), SR, subtype="PCM_24")
    monkeypatch.setattr("musique_repet.pipeline.date_creation", lambda chemin: "20260312")
    transcrit = []
    monkeypatch.setattr(
        "musique_repet.pipeline.reconnaitre_titre", lambda voix, *a, **k: transcrit.append(voix.name)
    )
    options = Options(
        entree=entree, sortie=sortie, travail=sortie / "_travail", facteurs=charger_config(None),
        mixage=ParametresMixage(), anti_echo=ParametresAntiEcho(attenuation_max_db=0), tempo=None,
    )

    class Transcripteur:
        identifiant = "inutilise"

    assert executer(options, SeparateurScenario(pistes), Transcripteur(), None, FauxSeparateurVoix()) == 0
    dossier = sortie / "_travail" / "pistes" / "repet" / "lead_choeurs"
    assert (dossier / "lead.wav").exists() and (dossier / "choeurs.wav").exists()
    assert transcrit == ["lead.wav"]  # les paroles sont lues sur la voix lead seule
    assert (sortie / "20260312-repet.mp3").exists()
