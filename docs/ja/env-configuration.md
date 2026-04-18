# 環境設定

多くの利用者が気にする必要がある `.env` は少数です。

## よく使う項目

| 変数 | 用途 |
|---|---|
| `HOST` | サーバーの待受アドレス |
| `PORT` | サーバーポート |
| `ENCRYPTION_KEY` | バックエンド暗号鍵 |
| `HF_TOKEN` | pyannote に必要 |

## メモ

- `ENCRYPTION_KEY` は自分の値にしてください
- pyannote を使わないなら `HF_TOKEN` は不要です
- `.env` を変更したら ArcSub を再起動してください
