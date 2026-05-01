# Installation

ArcSub can be used from a packaged release or from source.

For most users, the packaged release is recommended because it includes the application files and startup scripts needed for normal use.

## Packaged Release

1. Download the release archive for your operating system.
2. Extract it to a folder with a short path.
3. Start ArcSub:
   - Windows: run `start.ps1`
   - Linux or macOS: run `start.sh`
4. Open the URL shown in the terminal.

If you plan to use local OpenVINO models, install them later from **Settings**. Cloud ASR and cloud translation models can be configured in **Settings** without installing local models.

## Run from Source

Install Node.js, then run:

```bash
npm install
npm run dev
```

The development server prints the local URL after it starts.

## Build a Release Locally

Windows:

```powershell
.\deploy.ps1
```

Linux or macOS:

```bash
./deploy.sh
```

The deploy script prepares the application components needed by the packaged release.

## Optional Model Access

Set `HF_TOKEN` only when you need:

- pyannote speaker diarization
- a Hugging Face model that requires access approval
- a private Hugging Face model

Public Hugging Face models usually do not need a token.

## What Is Not Included in Source Control

The repository does not include generated project data, downloaded media, installed models, personal API keys, or local `.env` secrets.
