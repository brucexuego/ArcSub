# Environment Configuration

Most users only need a few `.env` values.

## Common Keys

| Key | Purpose |
|---|---|
| `HOST` | Server bind address |
| `PORT` | Server port |
| `ENCRYPTION_KEY` | Backend encryption key |
| `HF_TOKEN` | Needed for pyannote assets and gated/private Hugging Face model downloads |

## Recommendations

- keep `ENCRYPTION_KEY` set to your own value
- only set `HF_TOKEN` if you want pyannote or gated/private Hugging Face local models
- after editing `.env`, restart ArcSub

## Advanced Tuning

ArcSub also includes advanced settings for local runtime behavior.

In most cases, you can leave those values unchanged.

For local OpenVINO ASR, `OPENVINO_ASR_TIMEOUT_MODE=auto` estimates the transcription timeout from audio duration, device type, and word-alignment mode. Set `OPENVINO_ASR_TIMEOUT_MODE=fixed` only if you need the legacy fixed `OPENVINO_ASR_TIMEOUT_MS` behavior.
