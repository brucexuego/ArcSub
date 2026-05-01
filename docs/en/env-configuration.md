# Environment Configuration

Most users can configure ArcSub from **Settings**. Use `.env` for startup values, security keys, and a small set of advanced options.

After editing `.env`, restart ArcSub.

## Common Keys

| Key | Purpose |
|---|---|
| `HOST` | Server bind address. Keep `127.0.0.1` for local-only access. |
| `PORT` | Server port. Change it only if the default port is already used. |
| `ENCRYPTION_KEY` | Key used to protect stored credentials. Set your own value before regular use. |
| `HF_TOKEN` | Optional token for pyannote or Hugging Face models that require access approval. |

## Local Model Options

| Key | Purpose |
|---|---|
| `OPENVINO_LOCAL_MODEL_PRELOAD_ENABLED` | Enables local model preloading when set to `1` or `true`. The default is off so ArcSub starts lighter. |
| `OPENVINO_BASELINE_DEVICE` | Device used by shared OpenVINO helper models. Common values are `CPU`, `GPU`, or `AUTO`, depending on your environment. |
| `PYANNOTE_DEVICE` | Device used by pyannote speaker diarization. Use `CPU` unless your local setup supports another device. |
| `OPENVINO_QWEN3_ASR_FORCED_ALIGNER_DEVICE` | Device used by the Qwen3 ASR word-alignment helper when that workflow is used. |
| `TEN_VAD_PROVIDER` | Execution provider used by Ten VAD. Most users should keep `cpu`; change it only when your installed environment supports another provider. |

## Cloud Request Options

Cloud ASR and cloud translation model endpoints, API keys, and model names should normally be configured in **Settings**.

`TRANSLATE_CLOUD_REQUEST_TIMEOUT_MS` controls how long ArcSub waits for a cloud translation request before treating it as timed out. Increase it only when your provider or model needs more time for long subtitles.

If you use provider rate limits, configure them on the model card in **Settings** so each model can have its own limits.

## Recommendations

- Keep `.env` private.
- Do not commit API keys, tokens, or local machine paths.
- Leave advanced device settings unchanged unless you know your hardware and installed environment support the selected device.
