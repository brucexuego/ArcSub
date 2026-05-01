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

## 雲端翻譯

雲端翻譯請求逾時由 `TRANSLATE_CLOUD_REQUEST_TIMEOUT_MS` 控制。

RPM、TPM、RPD 與並行限制建議在設定頁的「進階請求參數」針對每個雲端翻譯模型個別設定：

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

ArcSub 預設會把雲端翻譯 quota 狀態保存到 `runtime/cache`，讓 RPM、TPM、RPD 視窗在重啟後仍可延續。若只想使用記憶體狀態，可設定 `TRANSLATE_CLOUD_QUOTA_PERSIST=0`。當供應商回傳 rate-limit headers 或 HTTP 429 時，ArcSub 會記錄該回饋並延後下一個請求視窗再重試。

`TRANSLATE_NVIDIA_CLOUD_*` 仍保留作為 NVIDIA hosted OpenAI-compatible endpoint 的相容預設。新增其他供應商時，建議改用每個模型自己的 `translation.batching` 設定，避免不同供應商彼此影響。
