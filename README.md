# Musique

Traitement d'enregistrements de répétition (2 chanteuses, piano, batterie acoustique, basse, guitare) :

1. **Découpe** : si un fichier `<nom>.csv` accompagne `<nom>.mp3` (ou `.wav`, `.flac`…), l'audio est découpé en morceaux enregistrés en `.wav`. Sans CSV, l'enregistrement est traité en entier : ce qui précède et suit la chanson (discussions, accordage, silences) est retiré, et le titre est reconnu d'après les paroles (voir plus bas).
2. **Séparation** des instruments avec [Demucs](https://github.com/facebookresearch/demucs) (modèle `htdemucs_6s`) : voix, `piano`, `batterie`, `basse`, `guitare`, `autre`. La voix est ensuite séparée en deux pistes, **`lead`** (voix principale) et **`choeurs`**, avec un modèle [audio-separator](https://pypi.org/project/audio-separator/) (voir plus bas). Les pistes sont **conservées sur disque** : une relance ne refait la séparation que pour les morceaux qui ont changé.
3. **Atténuation de l'écho** de la salle sur chaque piste séparée (voir plus bas). Le résultat est aussi gardé sur disque.
4. **Équilibrage** : la sonie (LUFS, norme ITU-R BS.1770) de chaque piste est mesurée, puis les pistes sont ramenées au même niveau.
5. **Volumes personnalisés** : un facteur par instrument, lu dans un fichier JSON, est appliqué.
6. **Chorus léger sur les voix (lead et chœurs) et la basse**, pour épaissir et élargir un peu le son. Sur la basse, seules les fréquences au-dessus de 250 Hz reçoivent le chorus : les graves restent nets et centrés.
7. **Mixage** de toutes les pistes, normalisé à -14 LUFS (crête max. -1 dBFS), puis export en **MP3** nommé `AAAAMMJJ-titre.mp3`.
8. **Piste groove** : `AAAAMMJJ-titre-groove.mp3` ne contient que la batterie, la basse et un **click** qui suit le tempo du groupe, réajusté toutes les 4 mesures (voir plus bas).
9. **Analyse du tempo** : un graphique `AAAAMMJJ-titre-tempo.jpg` montre les variations de BPM au cours du morceau (voir plus bas).

> Le lead et les chœurs sont séparés selon leur **rôle** (qui chante la mélodie, qui l'accompagne), pas selon la chanteuse : si elles échangent les rôles en cours de morceau, chaque piste suit le rôle.

## Installation

Il faut [uv](https://docs.astral.sh/uv/). `ffmpeg` est seulement nécessaire pour les formats que libsndfile ne lit pas (`.m4a`, `.aac`, `.wma`).

```bash
uv sync
```

Il faut Python 3.10 à 3.13 (uv l'installe s'il manque) : les bibliothèques de calcul utilisées ne suivent pas encore les versions plus récentes.

Au premier lancement, les modèles sont téléchargés : Demucs (~ 50 Mo) depuis `dl.fbaipublicfiles.com`, le modèle lead / chœurs (~ 900 Mo) depuis GitHub dans `~/.cache/musique-repet/modeles`, et Whisper (reconnaissance des paroles, ~ 500 Mo) depuis `huggingface.co`.

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

### Enregistrement sans CSV : chanson repérée et titre reconnu

Quand un audio n'a pas de CSV, on suppose qu'il contient **une chanson** :

**1. Ce qui précède et suit la chanson est retiré**, d'après les pistes séparées :

- la chanson commence et finit là où le groupe joue ensemble (par exemple 3 sources sur 5 : batterie, basse, guitare, piano, voix) ;
- un instrument seul collé à la chanson est gardé (intro au piano, résonance du dernier accord), sur 30 s au plus ;
- une discussion (voix seule), un accordage ou un échauffement séparés de la chanson par un silence sont retirés ;
- un court fondu adoucit le début et la fin ; les pauses à l'intérieur de la chanson (break) sont gardées.

`--sans-rognage` garde l'enregistrement entier.

> Limite : une introduction *a cappella* (voix seules, sans instrument) ressemble à une discussion et serait coupée. Dans ce cas, utilise un CSV ou `--sans-rognage`.

**2. Le titre est reconnu** pour nommer le MP3 (`20260312-Hallelujah.mp3` au lieu de `20260312-repet_mardi.mp3`), dans cet ordre :

1. **façon Shazam** : trois extraits de 12 s de la chanson sont soumis à Shazam (avec la bibliothèque [shazamio](https://pypi.org/project/shazamio/)). Le morceau est retenu s'il est trouvé sur deux extraits, ou sur un seul avec une correspondance exacte ; les mentions de version (« Remastered », « Live »...) sont retirées du titre, et l'orthographe du répertoire est reprise si le morceau y figure ;
2. sinon, **d'après les paroles**, transcrites en local avec [Whisper](https://github.com/SYSTRAN/faster-whisper) :
   - **avec un répertoire** (recommandé) : le titre du répertoire chanté pendant la chanson, ou annoncé dans la minute qui précède (« allez, on fait Le Sud »), est retenu ;
   - **sans répertoire** : la phrase qui revient le plus dans le refrain est proposée comme titre (« Hallelujah », « Let it be », « On dirait le Sud »...) ;
3. sinon le nom du fichier est gardé.

> Shazam reconnaît des **enregistrements** : il trouve le morceau quand l'original passe dans la pièce (on joue par-dessus le disque) ou sur une reprise très fidèle, mais souvent **pas sur la version du groupe**, avec ses propres sons, tempo et tonalité. C'est alors la reconnaissance par les paroles qui prend le relais. Pour Shazam, il faut une connexion Internet ; seule une **empreinte** des extraits est envoyée, pas l'audio. Shazam n'ayant pas d'interface publique, shazamio utilise celle de son application : elle peut cesser de fonctionner du jour au lendemain, et la reconnaissance par les paroles reste alors disponible. `--sans-shazam` désactive cette étape.

Le répertoire est un fichier `repertoire.txt` placé dans le dossier d'entrée (ou indiqué avec `--repertoire`), une chanson par ligne. Des extraits de paroles facultatifs, séparés par `;`, aident à reconnaître une chanson dont le titre n'est pas chanté :

```text
# Répertoire du groupe
Hallelujah
Le Sud ; on dirait le sud ; le temps dure longtemps
Bohemian Rhapsody ; is this the real life ; is this just fantasy
```

La réponse de Shazam et la transcription (environ une minute par chanson sur processeur) sont gardées dans `_travail` : elles ne sont pas refaites à la relance. `--modele-paroles medium` (ou `large-v3`) reconnaît mieux le chant, plus lentement ; `--sans-titre` désactive la reconnaissance.

### Voix lead et chœurs

Demucs isole les voix ensemble. Un second modèle sépare ensuite cette piste voix en deux :

- **`lead`** : la voix principale, celle qui chante la mélodie ;
- **`choeurs`** : les voix d'accompagnement (harmonies, chœurs).

Le modèle par défaut, `mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` (modèle « karaoké » Mel-Roformer d'aufr33 et viperx), extrait la voix lead ; tout le reste de la piste voix forme les chœurs. Il sépare aussi bien les voix avec l'écho d'une grande salle que sans.

Les deux pistes se règlent séparément dans le fichier de volumes (`"lead"`, `"choeurs"`), ce qui permet par exemple de couper le lead pour que la chanteuse travaille sa partie par-dessus les chœurs. À l'équilibrage, les chœurs reçoivent la **même correction que le lead** : leur écart de volume reste celui de la répétition.

- Pour chaque morceau, le journal indique le niveau des chœurs trouvés, par exemple « Chœurs : 8 dB sous le lead ».
- Si le modèle ne trouve pas de voix lead nette (lead plus de 10 dB sous les chœurs sur toute la chanson), la séparation est jugée ratée : la voix reste en une seule piste `voix`, avec un avertissement.
- La séparation prend deux à quatre fois la durée du morceau sur processeur, nettement moins sur la puce graphique d'un Mac Apple Silicon ou sur une carte graphique. Elle est gardée sur disque et n'est pas refaite à la relance (sauf si on change de modèle).
- `--sans-choeurs` garde une seule piste `voix`.
- `--modele-choeurs` permet d'essayer un autre modèle d'audio-separator. Les modèles « BVE » (`UVR-BVE-4B_SN-44100-2.pth`) sont à éviter sur une prise de salle : ils reconnaissent les chœurs à leur écho et, quand la salle résonne, rangent toutes les voix dans les chœurs.

> Limite : quand les deux chanteuses chantent ensemble au même volume, une partie d'une voix peut passer dans l'autre piste. À écouter : si le résultat ne convient pas, `--sans-choeurs` revient à une seule piste voix.

### Fichier de configuration des volumes

```json
{
  "lead": 1.0,
  "choeurs": 0.9,
  "piano": 1.0,
  "batterie": 0.8,
  "basse": 1.0,
  "guitare": 1.0,
  "autre": 0.5
}
```

Facteurs **linéaires**, appliqués après l'équilibrage : `1.0` = inchangé, `0.5` ≈ -6 dB, `2.0` ≈ +6 dB, `0` = piste coupée (pratique pour travailler sa partie). Un instrument absent du fichier reste à `1.0`. `"voix"` règle à la fois le lead et les chœurs (sauf s'ils ont leur propre réglage). Noms anglais acceptés aussi (`vocals`, `drums`, `bass`, `guitar`, `other`), ainsi que `"chœurs"`, `"harmonies"` ou `"voix lead"`.

### Organisation des fichiers produits

Chaque MP3 est nommé `AAAAMMJJ-titre.mp3` :

- `AAAAMMJJ` est la **date de création du fichier audio d'origine** ;
- `titre` est le titre du morceau dans le CSV ; sans CSV, le titre reconnu d'après les paroles, ou à défaut le nom du fichier audio.

```
sortie/
├── 20260312-Hallelujah.mp3            # repet_12_mars.mp3 + CSV : un MP3 par morceau
├── 20260312-Hallelujah-groove.mp3     # … batterie, basse et click
├── 20260312-Hallelujah-tempo.jpg      # … le graphique de son tempo
├── 20260312-Le Sud.mp3
├── 20260312-Le Sud-tempo.jpg
├── 20260315-Let it be.mp3             # repet_15_mars.wav sans CSV : la chanson seule, titre reconnu
├── …
└── _travail/                 # fichiers intermédiaires (réutilisés à la relance)
    ├── decoupes/             # morceaux découpés en .wav
    └── pistes/               # pistes séparées, une par instrument
        ├── …/lead_choeurs/   # la voix séparée en lead et chœurs
        └── …/sans_echo/      # les mêmes après atténuation de l'écho
```

Si deux extraits donnent le même nom (même morceau joué deux fois le même jour), le suivant devient `20260312-Hallelujah (2).mp3`.

> La date de création est fournie par macOS et Windows. Sous Linux, elle n'est disponible que sur certains systèmes de fichiers ; sinon la date de dernière modification est utilisée. Attention : copier un fichier remet souvent sa date de création à la date de la copie.

### Atténuation de l'écho

La réverbération d'une grande salle décroît de façon exponentielle. Pour chaque piste, le programme estime à chaque instant et pour chaque fréquence la part de son qui n'est que la traîne des notes précédentes, puis la retire (soustraction spectrale, méthode de Lebart / Habets). Les 50 premières millisecondes après chaque son sont conservées, car elles donnent du corps au son ; seule la traîne est atténuée.

Deux réglages :

- `--rt60` : durée de réverbération de la salle, en secondes (défaut `0.4`). Pour l'estimer, frappe un coup sec (clap, caisse claire) et compte le temps jusqu'à ce que l'écho ne s'entende plus : c'est à peu près le RT60.
- `--anti-echo` : atténuation maximale en dB (défaut `12`, `0` pour désactiver). Au-delà d'environ 15 dB, le son peut devenir « métallique » ou « sous l'eau ».

Changer ces réglages ne refait pas la séparation, seulement l'atténuation (quelques secondes par piste).

### Analyse du tempo

Pour chaque morceau, le graphique `…-tempo.jpg` montre comment le tempo a évolué : est-ce que le groupe presse dans les refrains, ralentit dans les passages calmes, finit plus vite qu'il n'a commencé ?

![Exemple de graphique du tempo](docs/exemple-tempo.jpg)

*Exemple obtenu sur une batterie synthétique : le groupe presse jusqu'à 120 BPM vers 1:30, s'arrête pendant un break (courbe interrompue), puis ralentit sur la fin.*

- **Trait épais** : tempo moyen sur 8 temps (deux mesures à 4 temps), la courbe à lire.
- **Trait fin** : tempo mesuré entre chaque temps et le suivant ; ses zigzags montrent la précision rythmique.
- **Bande bleutée** : ±3 % autour du tempo médian. Hors de la bande, l'écart s'entend.
- **En-tête** : tempo médian, plage, part du morceau joué dans la bande, et tempo du début et de la fin.

Le tempo est mesuré sur la batterie, la basse, la guitare et le piano séparés (pas sur les voix, souvent plus libres), après atténuation de l'écho. Les passages sans pulsation régulière (silence, break, intro jouée librement) ne sont pas tracés plutôt que d'afficher un tempo inventé.

Si une ballade est mesurée au double de son vrai tempo (par exemple 144 au lieu de 72, à cause d'un charleston en croches), baisse le tempo maximal recherché : `--bpm-max 120`. À l'inverse, pour un morceau très rapide lu à moitié, monte `--bpm-min`.

### Piste groove : batterie, basse et click

À côté de chaque MP3, `…-groove.mp3` ne contient que la **batterie**, la **basse** et un **click**, pour travailler la mise en place de la section rythmique.

- La batterie et la basse sont réglées comme dans le MP3 complet (équilibrage, volumes du fichier de configuration, chorus de la basse), mais la **basse est renforcée de +3 dB** pour bien ressortir face à la batterie : ses notes graves passent mal sur de petits haut-parleurs.
- Le click marque chaque temps : un son aigu sur le premier temps de la mesure, plus grave et plus doux sur les autres.
- Son tempo est **constant pendant 4 mesures**, calé sur les temps réellement joués pendant ces 4 mesures, puis il se réajuste pour les 4 suivantes. Il colle à la musique sans la suivre note à note : à l'intérieur d'un bloc, on entend si le groupe presse ou ralentit par rapport au click.
- Le premier temps de la mesure est repéré sur les pistes séparées : la grosse caisse frappe surtout sur les temps 1 et 3, la caisse claire sur 2 et 4, et la basse change de note en début de mesure.
- Les clicks sont calés sur le début des frappes de la batterie, pour sonner en même temps qu'elle.

Réglages : `--basse-groove 6` pour une basse encore plus présente (`0` = même équilibre que le MP3 complet), `--temps-par-mesure 3` pour une valse (4 par défaut), `--mesures-click 8` pour réajuster le tempo toutes les 8 mesures, `--volume-click 6` pour un click plus fort (ou `-6` plus doux). `--sans-groove` ne produit pas cette piste.

> Sans pulsation régulière (morceau joué librement), la piste groove est produite sans click. Le repérage du premier temps peut se tromper sur un rythme inhabituel : l'accent tombe alors sur un autre temps, mais le click reste sur les temps.

### Options utiles

| Option | Rôle |
| --- | --- |
| `--travail DIR` | dossier des fichiers intermédiaires (défaut : `<sortie>/_travail`) |
| `--appareil cuda` | séparation sur GPU (beaucoup plus rapide ; auto-détecté par défaut) |
| `--shifts 2` | séparation plus propre mais plus lente |
| `--forcer-separation` | ignore le cache et refait la séparation |
| `--sans-choeurs` | garde une seule piste voix au lieu de séparer lead et chœurs |
| `--modele-choeurs NOM` | autre modèle d'audio-separator pour séparer lead et chœurs |
| `--lufs-cible -16` | sonie du MP3 final |
| `--anti-echo 12` | atténuation max. de l'écho de la salle en dB (`0` = désactivé) |
| `--rt60 0.4` | durée de réverbération de la salle en secondes |
| `--chorus 0.3` | dosage du chorus sur les voix, lead et chœurs (`0` = désactivé, `0.5` = plus marqué) |
| `--chorus-basse 0.4` | dosage du chorus sur la basse, au-dessus de 250 Hz (`0` = désactivé) |
| `--sans-rognage` | sans CSV : garde tout l'enregistrement |
| `--sans-titre` | sans CSV : garde le nom du fichier au lieu de reconnaître le titre |
| `--sans-shazam` | sans CSV : reconnaît le titre seulement d'après les paroles, sans Shazam |
| `--repertoire FICHIER` | liste des titres du groupe (défaut : `repertoire.txt` du dossier d'entrée) |
| `--modele-paroles medium` | modèle Whisper pour les paroles : `tiny`, `base`, `small` (défaut), `medium`, `large-v3` |
| `--sans-tempo` | ne produit pas le graphique du tempo |
| `--sans-groove` | ne produit pas la piste `-groove` (batterie, basse et click) |
| `--temps-par-mesure 3` | temps par mesure pour le click (4 par défaut) |
| `--mesures-click 4` | le tempo du click est réajusté toutes les N mesures |
| `--volume-click 6` | click plus fort (en dB, `-6` = plus doux) |
| `--basse-groove 6` | renfort de la basse dans la piste `-groove`, en dB (+3 par défaut) |
| `--bpm-min 50` / `--bpm-max 200` | plage de tempos recherchée par l'analyse |
| `--debit 256` | débit MP3 en kb/s |
| `--gain-max-db 12` | correction maximale de l'équilibrage, pour ne pas trop amplifier un instrument qui joue peu |
| `--exclure-equilibrage autre` | pistes laissées à leur niveau d'origine par l'équilibrage (défaut : `autre`, qui contient surtout des résidus de séparation) |

Une piste quasi muette (moins de -50 LUFS, instrument absent du morceau) n'est jamais amplifiée par l'équilibrage.

## Tests

```bash
uv run pytest
```

Les tests utilisent un faux séparateur : ils ne téléchargent pas le modèle.
