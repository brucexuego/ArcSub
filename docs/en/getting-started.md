# Getting Started

ArcSub is an end-to-end subtitle translation workstation for cloud and local workflows.

Use it to:

- prepare media from online links, local video uploads, or existing project files
- convert speech to text with a cloud ASR service or a local model
- translate subtitles with a cloud translation service or a local model
- watch the translated subtitles together with the video in Video Player

## Choose a Workflow

You can start with a cloud-only workflow if you already have ASR and translation API credentials.

You can also install local OpenVINO models from **Settings** if you want speech recognition or translation to run on your own machine. Local models may require a larger download and suitable hardware.

Mixed workflows are supported. For example, you can use cloud ASR with local translation, or local ASR with cloud translation.

## First-Time Setup

1. Open **Settings**.
2. Add at least one cloud ASR model or install one local ASR model.
3. Add at least one cloud translation model or install one local translation model.
4. Drag model cards in **Settings** to choose the default order. The first model is used by default.
5. Add `HF_TOKEN` only if you use pyannote speaker diarization or Hugging Face models that require access approval.

## Normal Workflow

1. Create or open a project.
2. Prepare media in **Video Downloader**.
3. Run **Speech to Text**.
4. Run **Text Translation**.
5. Open **Video Player** to watch the translated subtitles with the video.

## Next Documents

- [Installation](./installation.md)
- [Usage](./usage.md)
- [FAQ](./faq.md)
- [Environment Configuration](./env-configuration.md)
