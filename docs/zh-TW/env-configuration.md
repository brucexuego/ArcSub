# 環境設定

多數使用者只需要了解少數幾個 `.env` 變數。

## 常用項目

| 變數 | 用途 |
|---|---|
| `HOST` | 服務綁定位置 |
| `PORT` | 服務埠號 |
| `ENCRYPTION_KEY` | 後端加密金鑰 |
| `HF_TOKEN` | pyannote gated assets 需要 |

## 建議

- `ENCRYPTION_KEY` 請使用你自己的值
- 只有在要用 pyannote 時才需要設定 `HF_TOKEN`
- 修改 `.env` 後請重新啟動 ArcSub

## 進階設定

ArcSub 還有一些本地推論相關的進階參數。

多數情況下不需要調整它們。
