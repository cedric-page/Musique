# Musique

Traitement d'enregistrements de répétition (2 chanteuses, piano, batterie acoustique, basse, guitare) :

1. **Découpe** : si un fichier `<nom>.csv` accompagne `<nom>.mp3` (ou `.wav`, `.flac`…), l'audio est découpé en morceaux enregistrés en `.wav`. Sans CSV, l'enregistrement complet est gardé en `.wav`.
2. **Séparation** des instruments avec [Demucs](https://github.com/facebookresearch/demucs) (modèle `htdemucs_6s`) : `voix`, `piano`, `batterie`, `basse`, `guitare`, `autre`. Les pistes sont **conservées sur disque** : une relance ne refait la séparation que pour les morceaux qui ont changé.
3. **Atténuation de l'écho** de la salle sur chaque piste séparée (voir plus bas). Le résultat est aussi gardé sur disque.
4. **Équilibrage** : la sonie (LUFS, norme ITU-R BS.1770) de chaque piste est mesurée, puis les pistes sont ramenées au même niveau.
5. **Volumes personnalisés** : un facteur par instrument, lu dans un fichier JSON, est appliqué.
6. **Chorus léger sur les voix**, pour épaissir et élargir un peu le chant.
7. **Mixage** de toutes les pistes, normalisé à -14 LUFS (crête max. -1 dBFS), puis export en **MP3** nommé `AAAAMMJJ-titre.mp3`.

> Les deux chanteuses sont dans la même piste `voix` : Demucs sépare les instruments, pas les voix entre elles.

## Installation

Il faut [uv](https://docs.astral.sh/uv/). `ffmpeg` est seulement nécessaire pour les formats que libsndfile ne lit pas (`.m4a`, `.aac`, `.wma`).

```bash
uv sync
```

Au premier lancement, Demucs télécharge les poids du modèle (~ 50 Mo) depuis `dl.fbaipublicfiles.com`.

## Utilisation

```bash
uv run musique-repet <dossier_entree> <dossier_sortie> -c config.json
```

Générer un fichier de configuration d'exemple :

```bash
uv run musique-repet --generer-config config.json
```

### Fichier CSV de découpe

Même nom que l'audio (`repet_12_mars.mp3` → `repet_12_mars.csv`), une ligne par morceau, `titre;debut;fin` :

```csv
titre;debut;fin
Hallelujah;0:45;5:12
Le Sud;6:30;10:02.5
Hallelujah;1:02:10;1:06:40
```

- Temps acceptés : `83`, `83.5`, `1:23`, `1:23.5`, `1:02:03`.
- La ligne d'en-tête est facultative ; les lignes vides ou commençant par `#` sont ignorées.
- Un titre en double devient `Titre (2)`.

### Fichier de configuration des volumes

```json
{
  "voix": 1.0,
  "piano": 1.0,
  "batterie": 0.8,
  "basse": 1.0,
  "guitare": 1.0,
  "autre": 0.5
}
```

Facteurs **linéaires**, appliqués après l'équilibrage : `1.0` = inchangé, `0.5` ≈ -6 dB, `2.0` ≈ +6 dB, `0` = piste coupée (pratique pour travailler sa partie). Un instrument absent du fichier reste à `1.0`. Noms anglais acceptés aussi (`vocals`, `drums`, `bass`, `guitar`, `other`).

### Organisation des fichiers produits

Chaque MP3 est nommé `AAAAMMJJ-titre.mp3` :

- `AAAAMMJJ` est la **date de création du fichier audio d'origine** ;
- `titre` est le titre du morceau dans le CSV, ou le nom du fichier audio s'il n'y a pas de CSV.

```
sortie/
├── 20260312-Hallelujah.mp3   # repet_12_mars.mp3 + CSV : un MP3 par morceau
├── 20260312-Le Sud.mp3
├── 20260315-jam_libre.mp3    # jam_libre.wav sans CSV : un seul MP3
└── _travail/                 # fichiers intermédiaires (réutilisés à la relance)
    ├── decoupes/             # morceaux découpés en .wav
    └── pistes/               # pistes séparées, une par instrument
        └── …/sans_echo/      # les mêmes après atténuation de l'écho
```

Si deux extraits donnent le même nom (même morceau joué deux fois le même jour), le suivant devient `20260312-Hallelujah (2).mp3`.

> La date de création est fournie par macOS et Windows. Sous Linux, elle n'est disponible que sur certains systèmes de fichiers ; sinon la date de dernière modification est utilisée. Attention : copier un fichier remet souvent sa date de création à la date de la copie.

### Atténuation de l'écho

La réverbération d'une grande salle décroît de façon exponentielle. Pour chaque piste, le programme estime à chaque instant et pour chaque fréquence la part de son qui n'est que la traîne des notes précédentes, puis la retire (soustraction spectrale, méthode de Lebart / Habets). Les 50 premières millisecondes après chaque son sont conservées, car elles donnent du corps au son ; seule la traîne est atténuée.

Deux réglages :

- `--rt60` : durée de réverbération de la salle, en secondes (défaut `1.8`). Pour l'estimer, frappe un coup sec (clap, caisse claire) et compte le temps jusqu'à ce que l'écho ne s'entende plus : c'est à peu près le RT60.
- `--anti-echo` : atténuation maximale en dB (défaut `12`, `0` pour désactiver). Au-delà d'environ 15 dB, le son peut devenir « métallique » ou « sous l'eau ».

Changer ces réglages ne refait pas la séparation, seulement l'atténuation (quelques secondes par piste).

### Options utiles

| Option | Rôle |
| --- | --- |
| `--travail DIR` | dossier des fichiers intermédiaires (défaut : `<sortie>/_travail`) |
| `--appareil cuda` | séparation sur GPU (beaucoup plus rapide ; auto-détecté par défaut) |
| `--shifts 2` | séparation plus propre mais plus lente |
| `--forcer-separation` | ignore le cache et refait la séparation |
| `--lufs-cible -16` | sonie du MP3 final |
| `--anti-echo 12` | atténuation max. de l'écho de la salle en dB (`0` = désactivé) |
| `--rt60 1.8` | durée de réverbération de la salle en secondes |
| `--chorus 0.3` | dosage du chorus sur les voix (`0` = désactivé, `0.5` = plus marqué) |
| `--debit 256` | débit MP3 en kb/s |
| `--gain-max-db 12` | correction maximale de l'équilibrage, pour ne pas trop amplifier un instrument qui joue peu |
| `--exclure-equilibrage autre` | pistes laissées à leur niveau d'origine par l'équilibrage (défaut : `autre`, qui contient surtout des résidus de séparation) |

Une piste quasi muette (moins de -50 LUFS, instrument absent du morceau) n'est jamais amplifiée par l'équilibrage.

## Tests

```bash
uv run pytest
```

Les tests utilisent un faux séparateur : ils ne téléchargent pas le modèle.
