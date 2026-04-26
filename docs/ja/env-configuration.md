# 環境設定

多くの利用者が気にする必要がある `.env` は少数です。

## よく使う項目

| 変数 | 用途 |
|---|---|
| `HOST` | サーバーの待受アドレス |
| `PORT` | サーバーポート |
| `ENCRYPTION_KEY` | バックエンド暗号鍵 |
| `HF_TOKEN` | pyannote アセットと gated/private Hugging Face モデルのダウンロードに必要 |

## メモ

- `ENCRYPTION_KEY` は自分の値にしてください
- pyannote または gated/private Hugging Face ローカルモデルを使わないなら `HF_TOKEN` は不要です
- `.env` を変更したら ArcSub を再起動してください

## 詳細設定

通常は変更不要です。

ローカル OpenVINO ASR は既定で `OPENVINO_ASR_TIMEOUT_MODE=auto` を使い、音声の長さ、実行デバイス、word alignment の有無から transcription timeout を見積もります。旧来の固定上限が必要な場合だけ `OPENVINO_ASR_TIMEOUT_MODE=fixed` にして `OPENVINO_ASR_TIMEOUT_MS` を使ってください。
