# 安裝說明

ArcSub 可以從封裝好的版本使用，也可以從原始碼啟動。

多數使用者建議使用封裝版本，因為它已經包含一般使用需要的應用程式檔案與啟動腳本。

## 封裝版本

1. 下載符合你作業系統的發行檔。
2. 解壓縮到路徑較短的資料夾。
3. 啟動 ArcSub：
   - Windows：執行 `start.ps1`
   - Linux 或 macOS：執行 `start.sh`
4. 開啟終端機中顯示的網址。

如果你要使用本機 OpenVINO 模型，可以之後再到 **設定** 安裝。雲端 ASR 與雲端翻譯模型可以直接在 **設定** 中設定，不需要先安裝本機模型。

## 從原始碼啟動

先安裝 Node.js，然後執行：

```bash
npm install
npm run dev
```

開發伺服器啟動後，終端機會顯示本機網址。

## 自行建立封裝版本

Windows：

```powershell
.\deploy.ps1
```

Linux 或 macOS：

```bash
./deploy.sh
```

部署腳本會準備封裝版本需要的應用程式元件。

## 選用的模型存取權限

只有在下列情況才需要設定 `HF_TOKEN`：

- 使用 pyannote 語者分離
- 使用需要申請存取權限的 Hugging Face 模型
- 使用私有 Hugging Face 模型

公開的 Hugging Face 模型通常不需要 token。

## 不會放進原始碼的內容

專案產生的資料、下載的影音檔、已安裝的模型、個人 API 金鑰與本機 `.env` 機密，都不應放進原始碼版本控制。
