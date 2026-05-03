# ArcSub

ArcSub 是一套雲端服務與本機 OpenVINO 模型並重的全流程字幕翻譯工作站。它能處理素材取得、語音轉文字、字幕翻譯，以及搭配影片觀賞字幕成果。

## 語言

- English: [README.md](./README.md)
- 繁體中文: [README.zh-TW.md](./README.zh-TW.md)
- 日本語: [README.ja.md](./README.ja.md)
- Deutsch: [README.de.md](./README.de.md)

## 文件

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## 畫面截圖

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Dashboard 總覽：管理字幕專案、查看系統資源，並串起整個工作流程。

![ArcSub video fetcher workflow](./docs/assets/screenshots/video-fetcher-overview.png)

影片獲取器：解析來源中繼資料、選擇下載格式，並準備後續語音辨識素材。

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

語音轉文字：選擇雲端或本機辨識模型、設定進階選項，並產生逐字稿輸出。

![ArcSub text translation workflow](./docs/assets/screenshots/text-translation-overview.png)

文字精準翻譯：選擇雲端或本機翻譯模型、設定目標語言，並檢視原文與翻譯字幕的對照結果。

![ArcSub video player workflow](./docs/assets/screenshots/video-player-overview.png)

影片撥放器：搭配影片觀賞完成的字幕成果，並微調頁面中的字幕樣式。

## 快速入口

### 下載

如果這個 repository 已經發布安裝包，請先到 [Releases](../../releases/latest) 下載符合你作業系統的最新檔案。

### 正式使用

如果你是一般使用者，請從 packaged release 啟動 ArcSub：

- Windows
  - `deploy.ps1`
  - `start.production.ps1`
- Linux
  - `deploy.sh`
  - `start.production.sh`

建議先看：

- [docs/zh-TW/installation.md](./docs/zh-TW/installation.md)
- [docs/zh-TW/usage.md](./docs/zh-TW/usage.md)
- [docs/zh-TW/faq.md](./docs/zh-TW/faq.md)

### 從原始碼開發

如果你是在這個 repository 中開發：

- Windows
  - `npm install`
  - `.\start.ps1`
- Linux
  - `npm install`
  - `./start.sh`

`start.ps1` 和 `start.sh` 會先清理殘留的 dev process，再啟動 `npm run dev`。

## Repository 範圍

這個 repository 包含 ArcSub 的應用程式原始碼與公開文件。

不包含：

- `runtime/` 下的本地執行資料
- 已下載的本地 ASR / 翻譯模型
- `.arcsub-bootstrap/` 這類 app-local bootstrap runtime
- `.env` 之類的個人憑證

## 主要功能

- 匯入本地媒體或下載線上媒體
- 使用雲端 ASR 服務或本機 OpenVINO ASR 模型進行語音轉文字
- 使用 word alignment、VAD 與 diarization 相關輔助能力
- 使用雲端翻譯服務或本機 OpenVINO 翻譯模型翻譯字幕
- 搭配影片觀賞字幕成果，並調整頁面中的字幕樣式

## 雲端與本機模型

ArcSub 讓每個專案依需求在雲端服務與本機 OpenVINO runtime 之間選擇最合適的模型路徑：

- 雲端 ASR 與翻譯模型可在 `設定` 中設定 API endpoint、金鑰與供應商參數
- 本機 ASR 與本機翻譯模型可從 `設定` 安裝，並透過 OpenVINO runtime 執行
- `設定` 中的模型排序會決定語音轉文字與文字精準翻譯頁面的預設模型
- pyannote 語者分離啟用時會使用 Hugging Face 資產；缺少 token 不會影響一般啟動或雲端流程

## 更多文件

- 文件總覽: [docs/README.md](./docs/README.md)
- 版本發布: [Releases](../../releases/latest)
- 討論區: [Discussions](../../discussions)
- 貢獻指南: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 行為準則: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- 安全性: [SECURITY.md](./SECURITY.md)

## 授權

本專案採用 [MIT License](./LICENSE)。
