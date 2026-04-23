# 快速開始

ArcSub 是一套用來做以下工作的字幕工具：

- 下載或匯入影音
- 語音轉文字
- 字幕翻譯
- 檢查與匯出

## 選擇使用方式

### 正式封包

Windows

1. 解壓正式封包
2. 在資料夾內執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

Linux

1. 解壓正式封包
2. 在資料夾內執行：

```bash
./deploy.sh
./start.production.sh
```

### 原始碼 repo

如果你是直接從這個 repo 開發：

Windows

```powershell
npm install
.\start.ps1
```

Linux

```bash
npm install
./start.sh
```

`start.ps1` 與 `start.sh` 會先清理舊的開發程序，再啟動 `npm run dev`。

## 第一次使用建議

ArcSub 開啟後：

1. 先進入 **Settings**
2. 準備至少一個語音轉文字來源
3. 準備至少一個翻譯來源
4. 如果你要用 pyannote 語者分離，再輸入 `HF_TOKEN` 並安裝

## 標準流程

1. 匯入或下載影音
2. 執行 **Speech to Text**
3. 執行 **Text Translation**
4. 在 **Player** 檢查
5. 匯出字幕

## 下一步

- [installation.md](./installation.md)
- [usage.md](./usage.md)
- [faq.md](./faq.md)
- [env-configuration.md](./env-configuration.md)
