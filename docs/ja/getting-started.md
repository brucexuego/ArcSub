# はじめに

ArcSub は次の作業に使う字幕ツールです。

- 動画や音声の取り込み
- 音声文字起こし
- 字幕翻訳
- 確認と書き出し

## 利用方法を選ぶ

### packaged release

Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.production.ps1
```

Linux

```bash
./deploy.sh
./start.production.sh
```

### source repo

このリポジトリから開発する場合:

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

`start.ps1` と `start.sh` は古い開発プロセスを止めてから `npm run dev` を起動します。

## 最初にやること

1. **Settings** を開く
2. 音声文字起こしの利用先を 1 つ用意する
3. 翻訳の利用先を 1 つ用意する
4. pyannote または gated/private Hugging Face ローカルモデルを使う場合は `HF_TOKEN` を入力する
5. 使いたいローカル ASR またはローカル翻訳モデルを Hugging Face モデル id からインストールする

## 通常の流れ

1. 動画や音声を取り込む
2. **Speech to Text** を実行する
3. **Text Translation** を実行する
4. **Player** で確認する
5. 字幕を書き出す
