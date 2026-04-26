# Getting Started

ArcSub is a subtitle workstation for:

- downloading or importing media
- speech to text
- subtitle translation
- review and export

## Choose a Path

### Packaged Release

Windows

1. Extract the packaged release
2. Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

Linux

1. Extract the packaged release
2. Run:

```bash
./deploy.sh
./start.production.sh
```

### Source Repository

If you are working from this repository instead of a packaged release:

Windows

```powershell
npm install
.\start.ps1
```

Linux

```bash
npm install
./start.sh
```

The source start scripts clean up stale dev processes and then launch `npm run dev`.

## First-Time Setup

After ArcSub opens:

1. Go to **Settings**
2. Prepare at least one speech-to-text source
3. Prepare at least one translation source
4. If you want pyannote diarization or gated/private Hugging Face local models, enter your `HF_TOKEN`
5. Install any local ASR or local translation models you want to use from their Hugging Face model ids

## Normal Workflow

1. Import or download media
2. Run **Speech to Text**
3. Run **Text Translation**
4. Review in **Player**
5. Export subtitles

## Read Next

- [installation.md](./installation.md)
- [usage.md](./usage.md)
- [faq.md](./faq.md)
- [env-configuration.md](./env-configuration.md)
