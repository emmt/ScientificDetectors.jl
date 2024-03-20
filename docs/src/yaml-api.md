## Feuille de route pour l'API YAML

ScientificDetectors va accepter une nouvelle structure d'entrée pour créer un CalibrationData:
il s'agit d'un fichier YAML dont voici un exemple:

```yaml
title: "an IRDIS config"
generated: 2024-03-18T17:18:45.160
roi:
    - offset: 0
      step: 1
      bin: 1
      length: 2048
    - length: 1024
      offset: 0
      step: 1
      bin: 1

categories:

  DARK:
    sources: dark
    files:
        0.837464:
            - "/tmp/toto.fits"
            - "../data/tata.fits"
        4.001856:
            - "../data/tutu.fits"

  FLAT:
    datahdu: 1
    sources: dark + flat
    files:
        0.837464:
            - "../data/tete.fits"
        4.001856:
            - "/tmp/titi.fits"
            - "/tmp/tyty.fits"
```

Pour chaque catégorie, on a son nom (ex `FLAT`), ses sources (ex `dark + flat`) ainsi que la liste
des chemins de fichiers pour chaque temps d'exposition donné.

Le motclé `datahdu` indique l'index (ou le HDUNAME ou le EXTNAME, à préciser) du HDU Fits qui
doit être lu pour trouver les données.

## Détails

Le motclé yaml `roi` permet de préciser une ROI pour les données. Selon le fichier différents cas
seront possibles:
- si le motclé yaml `roi` vaut `FULL` (valeur par défaut), ou `(:,:)`, toutes les données seront
  lues
- si le motclé yaml `roi` vaut autre chose par exemple `(512:1024, :)`
  - si les données du fichier sont de taille différente
    - si le fichier contient des motsclés fits indiquant que les données ont déjà été
      découpées avec une ROI différente -> ERREUR
    - sinon
      - si on peut appliquer la ROI du motclé yaml `roi` sur les données -> on le fait
      - sinon -> ERREUR
  - sinon (les données sont déjà de taille égale au motclé yaml `roi`)
    - si le fichier contient des motsclés fits indiquant que les données ont déjà été
      découpées avec une ROI
      - si la ROI fits est égale à la ROI yaml -> on prend toutes les données
      - sinon -> ERREUR
    - sinon -> on prend les données et on affiche un warning

Il y a aussi la question du dernier axe. La plupart du temps, le dernier axe des fichiers sert pour
les frames. Ainsi, la ROI peut indiquer un nombre d'axe égal à celui des données moins un.
Donc, avec une ROI à deux axes, on accepte les fichiers à axes, le troisième devenant les frames,
mais on accepte aussi les fichiers à deux axes, c'est à dire les fichiers à une seule frame.\
Lorsque l'utilisateur indique la ROI `FULL`, le programme assume qu'il y a un axe des frames:
- si le nombre d'axe du fichier est 2, 
