# Installation

This page is mainly for normal packaged use.

## Download

If release assets are available for this repository, start from [Releases](../../releases/latest).

## Recommended Platform

- Windows
- Linux

## Windows

1. Extract the packaged folder
2. Open PowerShell in that folder
3. Run:

```powershell
.\deploy.ps1
.\start.production.ps1
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
- local ASR and local translation models are installed later from **Settings**
- pyannote requires a Hugging Face access token

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
