# インストール

このページは主に packaged release を使う通常利用向けです。

## ダウンロード

この repository で release asset が公開されている場合は、まず [Releases](../../releases/latest) から取得してください。

現在の `v0.9.1` アセット：

- `ArcSub-v0.9.1-windows-x64.zip`
- `ArcSub-v0.9.1-linux-x64.tar.gz`

## 推奨プラットフォーム

- Windows
- Linux

## Windows

1. packaged folder を展開
2. そのフォルダで PowerShell を開く
3. 次を実行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

`deploy.ps1` は runtime、ツール、baseline assets を準備します。

## Linux

1. packaged folder を展開
2. そのフォルダで terminal を開く
3. 次を実行：

```bash
./deploy.sh
./start.production.sh
```

`deploy.sh` は runtime、ツール、baseline assets を準備します。

## 注意

- source repository にはローカル runtime データ、ダウンロード済みモデル、個人用 `.env` は含まれません
- OpenVINO が未導入でも ArcSub は起動できます
- OpenVINO がない場合、ローカルモデル導入は使えませんが、クラウド経路は利用できます
- ローカル ASR とローカル翻訳モデルは後から **Settings** で導入します
- pyannote には Hugging Face access token が必要です

## ソースから開発する場合

source repository で作業する場合は、dev startup helpers を使ってください：

```powershell
npm install
.\start.ps1
```

```bash
npm install
./start.sh
```
