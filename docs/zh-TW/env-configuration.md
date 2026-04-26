# 環境設定

多數使用者只需要了解少數幾個 `.env` 變數。

## 常用項目

| 變數 | 用途 |
|---|---|
| `HOST` | 服務綁定位置 |
| `PORT` | 服務埠號 |
| `ENCRYPTION_KEY` | 後端加密金鑰 |
| `HF_TOKEN` | pyannote 資產與 gated/private Hugging Face 模型下載需要 |

## 建議

- `ENCRYPTION_KEY` 請使用你自己的值
- 只有在要用 pyannote 或 gated/private Hugging Face 本地模型時才需要設定 `HF_TOKEN`
- 修改 `.env` 後請重新啟動 ArcSub

## 進階設定

ArcSub 還有一些本地推論相關的進階參數。

多數情況下不需要調整它們。

本地 OpenVINO ASR 預設使用 `OPENVINO_ASR_TIMEOUT_MODE=auto`，會依照音檔長度、執行裝置與 word alignment 模式估算轉錄 timeout。只有在需要舊版固定上限時，才改成 `OPENVINO_ASR_TIMEOUT_MODE=fixed` 並使用 `OPENVINO_ASR_TIMEOUT_MS`。
