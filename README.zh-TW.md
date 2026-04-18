# ArcSub

ArcSub 是一款 local-first 的字幕工作台，用來下載媒體、進行語音轉文字、翻譯字幕、校對結果，並匯出完成的字幕檔。

## 語言

- English: [README.md](./README.md)
- Traditional Chinese: [README.zh-TW.md](./README.zh-TW.md)
- Japanese: [README.ja.md](./README.ja.md)

## 文件

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- Traditional Chinese: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- Japanese: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## 畫面截圖

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Dashboard 總覽：管理字幕專案、查看系統資源，並串起整個工作流程。

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

Speech to Text：選擇辨識模型、設定進階選項，並為後續字幕翻譯做準備。

## 快速入口

### 下載

如果這個 repository 已經發布安裝包，請先到 [Releases](../../releases/latest) 下載。

預期資產：

- `ArcSub-windows-x64.zip`
- `ArcSub-linux-x64.tar.gz`

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
- 使用本地模型或雲端服務進行語音轉文字
- 使用 word alignment、VAD 與 diarization 相關輔助能力
- 使用本地模型或雲端服務翻譯字幕
- 在播放器中校對並匯出結果

## 本地模型

ArcSub 支援本地模型，但安裝分成不同階段：

- deploy 先準備工具與 baseline assets
- 本地 ASR 與本地翻譯模型之後再從 `Settings` 安裝
- pyannote 需要 Hugging Face token，但缺少 token 不會阻止啟動

## 更多文件

- 文件總覽: [docs/README.md](./docs/README.md)
- 貢獻指南: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 行為準則: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- 安全性: [SECURITY.md](./SECURITY.md)

## 授權

本專案採用 [MIT License](./LICENSE)。
