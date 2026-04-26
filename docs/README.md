# ArcSub 文件索引

`docs/` 收錄 ArcSub 的安裝、啟動、操作與常見問題說明。

## 語言

- English: [./en/getting-started.md](./en/getting-started.md)
- 繁體中文: [./zh-TW/getting-started.md](./zh-TW/getting-started.md)
- 日本語: [./ja/getting-started.md](./ja/getting-started.md)

## 文件內容

- `getting-started.md`
  首次認識 ArcSub 與整體使用入口。
- `installation.md`
  安裝方式、部署腳本與啟動流程。
- `usage.md`
  影音下載、語音轉文字、翻譯、校對與匯出操作。
- `faq.md`
  常見問題與排查方向。
- `env-configuration.md`
  環境變數與常用設定項目。

## 說明

- 文件內容以安裝、啟動、操作與常見問題為主。
- 一般使用者請優先閱讀 packaged release 流程。
- 若你是從 source repo 開發，可搭配 `start.ps1` 或 `start.sh` 使用。
- 原始碼 repo 不會附帶本地 runtime 資料、已下載模型與個人憑證。
- 本地 ASR / 翻譯模型會從 **Settings** 透過 Hugging Face 模型 id 安裝；需要授權的模型與 pyannote 共用 `HF_TOKEN`。
- 不保留過時版號、舊流程、一次性驗證筆記與舊性能數字。
