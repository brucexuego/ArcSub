# Environment Configuration

Most users only need a few `.env` values.

## Common Keys

| Key | Purpose |
|---|---|
| `HOST` | Server bind address |
| `PORT` | Server port |
| `ENCRYPTION_KEY` | Backend encryption key |
| `HF_TOKEN` | Needed for pyannote gated assets |

## Recommendations

- keep `ENCRYPTION_KEY` set to your own value
- only set `HF_TOKEN` if you want pyannote
- after editing `.env`, restart ArcSub

## Advanced Tuning

ArcSub also includes advanced settings for local runtime behavior.

In most cases, you can leave those values unchanged.
