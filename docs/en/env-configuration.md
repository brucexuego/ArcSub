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

## Cloud Translation

Cloud translation request timeout is controlled by `TRANSLATE_CLOUD_REQUEST_TIMEOUT_MS`.

Quota limits should usually be set per cloud translation model in Settings -> Advanced Request Options:

```json
{
  "translation": {
    "quota": { "rpm": 10, "tpm": 250000, "rpd": 500, "maxConcurrency": 1 },
    "batching": {
      "enabled": true,
      "targetLines": 24,
      "minTargetLines": 6,
      "charBudget": 2400,
      "maxOutputTokens": 2048
    }
  }
}
```

ArcSub persists cloud translation quota state under `runtime/cache` by default so RPM/TPM/RPD windows survive restarts. Set `TRANSLATE_CLOUD_QUOTA_PERSIST=0` to keep quota state in memory only. When providers return rate-limit headers or HTTP 429, ArcSub records that feedback and delays the next request window before retrying.

The `TRANSLATE_NVIDIA_CLOUD_*` keys remain as compatibility defaults for NVIDIA hosted OpenAI-compatible endpoints. New providers should prefer per-model `translation.batching` options so provider tuning stays isolated.
