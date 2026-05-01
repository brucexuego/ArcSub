# FAQ

## Is ArcSub a cloud app or a local app?

It supports both. You can use cloud ASR and translation services, local OpenVINO models, or a mix of both.

## Can I use ArcSub without OpenVINO?

Yes. Cloud ASR and cloud translation workflows can be used without installing local OpenVINO models.

OpenVINO is needed when you want local ASR or local translation to run on your own machine.

## Do I need to install local models during deployment?

No. Local models are installed later from **Settings**. This keeps the initial setup smaller and lets each user choose only the models they need.

## Why can a local model install take a long time?

Local model installation may download large files and prepare them for your computer. The **Settings** page shows the install status while the task is running.

## Why is pyannote unavailable?

pyannote speaker diarization needs:

- a valid `HF_TOKEN`
- accepted Hugging Face model access
- a completed pyannote install in **Settings**

The same `HF_TOKEN` can also be used when a private or access-controlled Hugging Face model requires authentication.

## Where are my files stored?

Project files, downloaded media, generated subtitles, and installed local models are stored on the machine running ArcSub. Personal API keys and local secrets should stay in your local `.env` or in **Settings**, not in source control.

## What is Video Player for?

Video Player lets you watch the translated subtitles together with the video and adjust the subtitle style for that viewing page.

## What should I do if something only fails in a packaged release?

Use the diagnostics script included in the packaged folder and keep the generated archive for inspection.
