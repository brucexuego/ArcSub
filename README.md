# ArcSub

ArcSub is an end-to-end subtitle translation workstation that treats cloud services and local OpenVINO models as equal first-class paths. It covers media intake, speech to text, subtitle translation, and watching the finished subtitles with the video.

## Languages

- English: [README.md](./README.md)
- 繁體中文: [README.zh-TW.md](./README.zh-TW.md)
- 日本語: [README.ja.md](./README.ja.md)
- Deutsch: [README.de.md](./README.de.md)
- Français: [README.fr.md](./README.fr.md)

## Documentation

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## Screenshots

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Dashboard overview: manage subtitle projects, monitor system resources, and move through the full workflow.

![ArcSub video fetcher workflow](./docs/assets/screenshots/video-fetcher-overview.png)

Video Fetcher: parse source metadata, select downloadable formats, and prepare assets for transcription.

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

Speech to Text: choose a cloud or local recognition model, configure advanced features, and generate transcript output.

![ArcSub text translation workflow](./docs/assets/screenshots/text-translation-overview.png)

Text Translation: choose a cloud or local translation model, configure language options, and compare source and translated subtitles.

![ArcSub video player workflow](./docs/assets/screenshots/video-player-overview.png)

Video Player: watch the finished subtitles with the video and fine-tune subtitle styling for the viewing page.

## Quick Paths

### Downloads

When packaged assets are published for this repository, download the latest archive for your operating system from [Releases](../../releases/latest).

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
- run speech to text with cloud ASR services or local OpenVINO ASR models
- use word alignment, VAD, and diarization-related helpers
- translate subtitles with cloud translation services or local OpenVINO translation models
- watch subtitle results with the video and tune styling for the viewing page

## Cloud and Local Models

ArcSub is designed to let each project choose the most practical model path across cloud services and local OpenVINO runtimes:

- cloud ASR and translation models are configured in `Settings` with API endpoints, keys, and provider options
- local ASR and local translation models are installed from `Settings` and run through the bundled OpenVINO runtime path
- model order in `Settings` controls the default model shown in Speech to Text and Text Translation
- pyannote speaker diarization uses Hugging Face assets when enabled; a missing token does not block normal startup or cloud workflows

## More Documents

- Docs index: [docs/README.md](./docs/README.md)
- Releases: [Releases](../../releases/latest)
- Discussions: [Discussions](../../discussions)
- Contributing: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Code of Conduct: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](./SECURITY.md)

## License

This project is licensed under the [MIT License](./LICENSE).
