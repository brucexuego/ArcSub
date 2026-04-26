# Installation

This page is mainly for normal packaged use.

## Download

If release assets are available for this repository, start from [Releases](../../releases/latest).

Current `v0.9.1` assets:

- `ArcSub-v0.9.1-windows-x64.zip`
- `ArcSub-v0.9.1-linux-x64.tar.gz`

## Recommended Platform

- Windows
- Linux

## Windows

1. Extract the packaged folder
2. Open PowerShell in that folder
3. Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

`deploy.ps1` prepares the runtime, tools, and baseline assets.

## Linux

1. Extract the packaged folder
2. Open a terminal in that folder
3. Run:

```bash
./deploy.sh
./start.production.sh
```

`deploy.sh` prepares the runtime, tools, and baseline assets.

## Notes

- The source repository does not ship local runtime data, downloaded models, or personal `.env` values
- ArcSub can start even if OpenVINO is not installed
- without OpenVINO, local-model install is unavailable, but cloud usage can still work
- local ASR and local translation models are installed later from **Settings** by Hugging Face model id
- long local-model downloads/conversions continue as background install tasks and show status in **Settings**
- `HF_TOKEN` is used for pyannote and gated/private Hugging Face model downloads

## Source Development

If you are working from the source repository, use the dev startup helpers instead:

```powershell
npm install
.\start.ps1
```

```bash
npm install
./start.sh
```
