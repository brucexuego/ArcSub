# 安裝

本頁主要針對 packaged release 的一般使用方式。

## 下載

如果這個 repository 已經發布安裝包，請先到 [Releases](../../releases/latest) 下載。

目前 `v0.9.1` 的資產：

- `ArcSub-v0.9.1-windows-x64.zip`
- `ArcSub-v0.9.1-linux-x64.tar.gz`

## 建議平台

- Windows
- Linux

## Windows

1. 解壓 packaged folder
2. 在該資料夾中開啟 PowerShell
3. 執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

`deploy.ps1` 會準備 runtime、工具與 baseline assets。

## Linux

1. 解壓 packaged folder
2. 在該資料夾中開啟 terminal
3. 執行：

```bash
./deploy.sh
./start.production.sh
```

`deploy.sh` 會準備 runtime、工具與 baseline assets。

## 注意事項

- source repository 不包含本地 runtime 資料、已下載模型與個人 `.env`
- 即使尚未安裝 OpenVINO，ArcSub 仍可啟動
- 若沒有 OpenVINO，本地模型安裝功能不可用，但雲端路徑仍可使用
- 本地 ASR 與本地翻譯模型會在之後從 **Settings** 安裝
- pyannote 需要 Hugging Face access token

## 從原始碼開發

如果你是從 source repository 開發，請改用 dev startup helpers：

```powershell
npm install
.\start.ps1
```

```bash
npm install
./start.sh
```
