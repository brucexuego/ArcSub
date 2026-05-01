# 環境設定

多數設定都可以在 **設定** 頁面完成。`.env` 主要用於啟動參數、安全金鑰，以及少數進階執行選項。

修改 `.env` 後，請重新啟動 ArcSub。

## 常用項目

| 變數 | 用途 |
|---|---|
| `HOST` | 伺服器綁定位置。只在本機使用時可保留 `127.0.0.1`。 |
| `PORT` | 伺服器連接埠。只有預設連接埠已被使用時才需要調整。 |
| `ENCRYPTION_KEY` | 用來保護已儲存憑證的金鑰。正式使用前請改成自己的值。 |
| `HF_TOKEN` | 選用。使用 pyannote 或需要授權的 Hugging Face 模型時才需要。 |

## 本機模型選項

| 變數 | 用途 |
|---|---|
| `OPENVINO_LOCAL_MODEL_PRELOAD_ENABLED` | 設為 `1` 或 `true` 時啟用本機模型預載。預設關閉，讓 ArcSub 啟動更輕量。 |
| `OPENVINO_BASELINE_DEVICE` | 共用 OpenVINO 輔助模型使用的裝置。常見值為 `CPU`、`GPU` 或 `AUTO`，實際可用性取決於你的環境。 |
| `PYANNOTE_DEVICE` | pyannote 語者分離使用的裝置。除非本機環境已確認支援其他裝置，否則建議使用 `CPU`。 |
| `OPENVINO_QWEN3_ASR_FORCED_ALIGNER_DEVICE` | 使用 Qwen3 ASR 字詞時間對齊流程時，對齊輔助模型使用的裝置。 |
| `TEN_VAD_PROVIDER` | Ten VAD 使用的執行提供者。多數使用者建議保留 `cpu`；只有在已安裝的執行環境支援其他提供者時才需要調整。 |

## 雲端請求選項

雲端 ASR 與雲端翻譯模型的 API 網址、金鑰與模型名稱，通常應在 **設定** 中設定。

`TRANSLATE_CLOUD_REQUEST_TIMEOUT_MS` 控制 ArcSub 等待雲端翻譯請求的時間。只有在服務或模型處理長字幕需要更久時間時，才建議提高此值。

如果你需要限制服務請求頻率，建議在 **設定** 的模型小卡中設定，讓每個模型可以有自己的限制。

## 建議

- 保持 `.env` 私密。
- 不要提交 API 金鑰、token 或本機路徑。
- 除非你確認硬體與執行環境支援，否則不要任意調整進階裝置設定。
