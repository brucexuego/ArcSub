# 使用方式

## 主要流程

1. 建立或開啟專案
2. 在 **Video Downloader** 匯入或下載影音
3. 執行 **Speech to Text**
4. 執行 **Text Translation**
5. 在 **Player** 檢查與匯出

## 語音轉文字建議

- 如果你已經知道音檔語言，直接指定語言通常會比自動偵測更穩定
- 如果你要用 pyannote 語者分離，先到 **Settings** 安裝
- 如果 pyannote 尚未安裝，ArcSub 會保留經典引擎作為可用選項

## 翻譯建議

- 翻譯前先快速檢查原文是否有明顯辨識錯誤
- 如果專有名詞很多，建議使用 glossary
- 如果只是一般使用，先用預設設定即可

## 本地模型

如果你要使用本地 ASR 或本地翻譯模型：

1. 先進入 **Settings**
2. 選擇 **ASR 模型** 或 **翻譯模型**
3. 輸入 Hugging Face 模型 id 並先檢查
4. 安裝模型；大型模型下載或轉換會以背景任務繼續進行
5. 回到工作流程頁面後再選用

ArcSub 會在可用時讀取可信的 Hugging Face metadata，產生模型專屬的本地預設值，例如 runtime hints、chat template 支援與 token-aware 翻譯批次。

`HF_TOKEN` 會同時提供給 pyannote 與需要授權的 gated/private Hugging Face 模型下載。公開模型通常不需要設定。
