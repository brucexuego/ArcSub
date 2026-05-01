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

## クラウド翻訳

クラウド翻訳リクエストのタイムアウトは `TRANSLATE_CLOUD_REQUEST_TIMEOUT_MS` で制御します。

RPM、TPM、RPD、同時実行数の制限は、設定画面の「高度なリクエストオプション」でクラウド翻訳モデルごとに個別設定することを推奨します。

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

ArcSub は既定でクラウド翻訳の quota 状態を `runtime/cache` に保存し、再起動後も RPM、TPM、RPD のウィンドウを引き継ぎます。メモリ上だけで管理したい場合は `TRANSLATE_CLOUD_QUOTA_PERSIST=0` を設定してください。プロバイダーが rate-limit headers または HTTP 429 を返した場合、ArcSub はその情報を記録し、次のリクエストウィンドウを遅らせてから再試行します。

`TRANSLATE_NVIDIA_CLOUD_*` は NVIDIA hosted OpenAI-compatible endpoint 向けの互換既定値として残しています。他のプロバイダーを追加する場合は、プロバイダーごとの調整が互いに影響しないよう、各モデルの `translation.batching` 設定を使うことを推奨します。
