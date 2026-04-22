# ArcSub

ArcSub is a local-first subtitle workstation for downloading media, converting speech to text, translating subtitles, reviewing results, and exporting finished subtitles.

## Languages

- English: [README.md](./README.md)
- 繁體中文: [README.zh-TW.md](./README.zh-TW.md)
- 日本語: [README.ja.md](./README.ja.md)

## Documentation

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## Screenshots

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Dashboard overview: manage subtitle projects, monitor system resources, and move through the full workflow.

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

Speech to Text: choose a recognition model, configure advanced features, and prepare subtitles for translation.

## Quick Paths

### Downloads

When packaged assets are published for this repository, download them from [Releases](../../releases/latest).

Current `v0.9.1` assets:

- `ArcSub-v0.9.1-windows-x64.zip`
- `ArcSub-v0.9.1-linux-x64.tar.gz`

### Packaged Release

For normal end-user use, start ArcSub from the packaged release:

- Windows
  - `deploy.ps1`
  - `start.production.ps1`
- Linux
  - `deploy.sh`
  - `start.production.sh`

Start with:

- [docs/en/installation.md](./docs/en/installation.md)
- [docs/en/usage.md](./docs/en/usage.md)
- [docs/en/faq.md](./docs/en/faq.md)

### Source Development

If you are working from this repository:

- Windows
  - `npm install`
  - `.\start.ps1`
- Linux
  - `npm install`
  - `./start.sh`

The `start.ps1` and `start.sh` helpers clean up stale dev processes and then launch `npm run dev`.

## Repository Scope

This repository contains the application source code and public documentation.

It does not include:

- local runtime data under `runtime/`
- downloaded local ASR or translation models
- portable bootstrap runtimes such as `.arcsub-bootstrap/`
- personal credentials such as `.env`

## Main Capabilities

- import local media or download online media
- run speech to text with local models or cloud services
- use word alignment, VAD, and diarization-related helpers
- translate subtitles with local models or cloud services
- review and export results in the player

## Local Models

ArcSub supports local models, but installation is split into stages:

- deploy prepares tools and baseline assets
- local ASR and local translation models are installed later from `Settings`
- pyannote requires a Hugging Face token, but missing it does not block startup

## More Documents

- Docs index: [docs/README.md](./docs/README.md)
- Releases: [Releases](../../releases/latest)
- Discussions: [Discussions](../../discussions)
- Contributing: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](./SECURITY.md)

## License

This project is licensed under the [MIT License](./LICENSE).
