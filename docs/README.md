# ArcSub Documentation

`docs/` contains the public documentation for ArcSub. ArcSub is an end-to-end subtitle translation workstation that supports both cloud services and local OpenVINO models.

## Languages

- English: [./en/getting-started.md](./en/getting-started.md)
- Traditional Chinese: [./zh-TW/getting-started.md](./zh-TW/getting-started.md)
- Japanese: [./ja/getting-started.md](./ja/getting-started.md)

## Document Set

- `getting-started.md`
  A quick overview of the ArcSub workflow.
- `installation.md`
  How to install and start ArcSub from a packaged release or from source.
- `usage.md`
  The main workflow for preparing media, running **Speech to Text**, translating subtitles in **Text Translation**, and watching the result in **Video Player**.
- `faq.md`
  Common questions about cloud services, local models, pyannote, and project files.
- `env-configuration.md`
  Public `.env` settings that are useful during installation or advanced setup.

## Notes

- These documents are written for public users and avoid internal implementation details.
- Cloud ASR and cloud translation models are configured in **Settings**.
- Local ASR and local translation models can be installed from **Settings** when you want the work to run on your own machine.
- Dragging model cards in **Settings** changes the default model order. The first model is used by default in the workflow pages.
- **Video Player** is for watching translated subtitles together with the video and adjusting the on-page viewing style.
