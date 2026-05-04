# ArcSub

ArcSub est une station de travail de traduction de sous-titres de bout en bout qui traite les services cloud et les modèles locaux OpenVINO comme des solutions de premier plan équivalentes.

## Langues

- English: [README.md](./README.md)
- 繁體中文: [README.zh-TW.md](./README.zh-TW.md)
- 日本語: [README.ja.md](./README.ja.md)
- Deutsch: [README.de.md](./README.de.md)
- Français: [README.fr.md](./README.fr.md)

## Documentation

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## Captures d'écran

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Aperçu du tableau de bord: gérez les projets de sous-titres, surveillez les ressources système, et parcourez l'ensemble du workflow.

![ArcSub video fetcher workflow](./docs/assets/screenshots/video-fetcher-overview.png)

Récupérateur de vidéo: analysez les métadonnées sources, sélectionnez les formats téléchargeables, et préparez les assets pour la transcription.

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

Reconnaissance vocale: choisissez un modèle de reconnaissance cloud ou en local, configurez les fonctionnalités avancées, et générez la transcription.

![ArcSub text translation workflow](./docs/assets/screenshots/text-translation-overview.png)

Traduction de texte: choisissez un modèle de traduction cloud ou en local, configurez les options de langue, et comparez les sous-titres originaux et traduits.

![ArcSub video player workflow](./docs/assets/screenshots/video-player-overview.png)

Lecteur vidéo: visionnez les sous-titres produits avec la vidéo et ajustez leur style d'affichage.

## Liens rapides

### Téléchargements

Si les assets packagés sont disponibles pour ce dépôt, téléchargez la dernière archive correspondant à votre système d'exploitation depuis les [Releases](../../releases/latest).

### Version packagée

Pour une utilisation standard, lancez ArcSub à partir de la version packagée :

- Windows
  - `deploy.ps1`
  - `start.production.ps1`
- Linux
  - `deploy.sh`
  - `start.production.sh`

Débutez avec:

- [docs/en/installation.md](./docs/en/installation.md)
- [docs/en/usage.md](./docs/en/usage.md)
- [docs/en/faq.md](./docs/en/faq.md)

### Développement source

Si vous travaillez depuis ce dépôt:

- Windows
  - `npm install`
  - `.\start.ps1`
- Linux
  - `npm install`
  - `./start.sh`

Les programmes `start.ps1` et `start.sh` nettoient les processus obsolètes puis lancent `npm run dev`

## Portée du dépôt

Ce dépôt contient les codes sources de l'application et la documentation publique.

Il n'inclut pas:

- les données d'exécution locales sous `runtime/`
- les modèles locaux ASR ou de traduction
- les environnements d'exécution portables tels que `.arcsub-bootstrap/`
- les identifiants personnels tels que `.env`

## Principales fonctionnalités

- importer du contenu local ou télécharger du contenu en ligne
- lancer la reconnaissance vocale avec les services ASR cloud ou avec des modèles ASR OpenVINO locaux
- alignement des mots, VAD et outils liés à la diarisation
- traduire des sous-titres avec des services de traduction cloud ou des modèles OpenVINO locaux
- visionner les résultats des sous-titres avec la vidéo et ajuster leur style

## Modèles cloud et locaux

ArcSub est conçu pour permettre à chaque projet de choisir les modèles les plus pratiques parmi les services cloud et les environnements d'exécution OpenVINO locaux:

- les modèles cloud d'ASR et de traduction sont configurés dans `Settings` avec les points de terminaison d'API, les clés et les options du fournisseur.
- les modèles locaux d'ASR et de traduction sont installés depuis `Settings` et s'exécutent via l'environnement OpenVINO intégré.
- l'ordre des modèles dans `Settings` contrôle le modèle affiché par défaut dans les sections "Speech to Text" et "Text Translation"
- la diarisation du locuteur pyannote utilise les ressources Hugging Face lorsqu'elle est activée; l'absence d'un token ne bloque pas le démarrage normal ni les flux de travail cloud.

## Plus de documents

- Index de documentation: [docs/README.md](./docs/README.md)
- Releases: [Releases](../../releases/latest)
- Discussions: [Discussions](../../discussions)
- Contribuer: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Code de conduite: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- Sécurité: [SECURITY.md](./SECURITY.md)

## Licence

Ce projet est sous licence [MIT](./LICENSE).
