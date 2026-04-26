# FAQ

## Can I use ArcSub without OpenVINO?

Yes.

The service can still start, but local-model installation will be unavailable until OpenVINO is ready.

## Do I need to install local models during deployment?

No.

Baseline tools and assets are prepared during deploy, but local ASR and local translation models are installed later from **Settings**.

## Why can a local model install take a long time?

ArcSub downloads and converts local Hugging Face models as background install tasks.

The **Settings** page shows the current task status, so a large model should not fail just because the browser request would have taken too long.

## Why is pyannote unavailable?

pyannote needs:

- a valid `HF_TOKEN`
- accepted Hugging Face gated-model access
- a completed pyannote install in **Settings** or during deploy

The same `HF_TOKEN` is also used when a gated or private Hugging Face local model requires authentication.

## Where are my files stored?

ArcSub stores runtime data under the local `runtime/` directory.

## What should I do if something only fails in a packaged VM?

Use the diagnostics script in the packaged folder and keep the generated archive for inspection.
