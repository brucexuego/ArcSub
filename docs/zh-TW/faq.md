# 常見問題

## 沒有 OpenVINO 可以用 ArcSub 嗎？

可以。

服務仍然可以啟動，但本地模型安裝功能會先不可用，之後再補上 OpenVINO 即可。

## 本地模型需要在部署時就安裝嗎？

不用。

部署階段會先準備 baseline 工具與資產，本地 ASR 與本地翻譯模型之後再到 **Settings** 安裝即可。

## 為什麼本地模型安裝可能會很久？

ArcSub 會把 Hugging Face 模型下載與轉換放在背景安裝任務中執行。

**Settings** 會顯示目前任務狀態，所以大型模型不應該因為單次瀏覽器請求太久而被誤判失敗。

## 為什麼 pyannote 不能用？

pyannote 需要：

- 有效的 `HF_TOKEN`
- 你已經在 Hugging Face 同意 gated model 存取
- 在 **Settings** 或部署階段完成安裝

同一組 `HF_TOKEN` 也會用於需要授權的 gated 或 private Hugging Face 本地模型下載。

## 我的資料放在哪裡？

ArcSub 的 runtime 資料預設會放在本地的 `runtime/` 目錄。

## 如果只有打包後的 VM 才出問題怎麼辦？

請直接在打包資料夾內執行診斷收集腳本，保留產生的診斷壓縮檔。
