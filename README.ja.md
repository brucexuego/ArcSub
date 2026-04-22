# ArcSub

ArcSub は、メディアの取得、音声文字起こし、字幕翻訳、レビュー、字幕書き出しまでを一台の環境で進められる local-first な字幕ワークステーションです。

## 言語

- English: [README.md](./README.md)
- 繁體中文: [README.zh-TW.md](./README.zh-TW.md)
- 日本語: [README.ja.md](./README.ja.md)

## ドキュメント

- English: [docs/en/getting-started.md](./docs/en/getting-started.md)
- 繁體中文: [docs/zh-TW/getting-started.md](./docs/zh-TW/getting-started.md)
- 日本語: [docs/ja/getting-started.md](./docs/ja/getting-started.md)

## スクリーンショット

![ArcSub dashboard overview](./docs/assets/screenshots/dashboard-overview.png)

Dashboard overview: 字幕プロジェクトの管理、システムリソースの確認、全体フローの移動を一画面で行えます。

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

Speech to Text: 認識モデルの選択、詳細オプションの設定、翻訳前の字幕準備を行います。

## クイックガイド

### ダウンロード

この repository で配布用アセットが公開されている場合は、まず [Releases](../../releases/latest) から取得してください。

現在の `v0.9.1` アセット：

- `ArcSub-v0.9.1-windows-x64.zip`
- `ArcSub-v0.9.1-linux-x64.tar.gz`

### パッケージ版

通常利用では、packaged release から ArcSub を起動してください。

- Windows
  - `deploy.ps1`
  - `start.production.ps1`
- Linux
  - `deploy.sh`
  - `start.production.sh`

最初に読むもの：

- [docs/ja/installation.md](./docs/ja/installation.md)
- [docs/ja/usage.md](./docs/ja/usage.md)
- [docs/ja/faq.md](./docs/ja/faq.md)

### ソースから開発する場合

この repository で開発する場合：

- Windows
  - `npm install`
  - `.\start.ps1`
- Linux
  - `npm install`
  - `./start.sh`

`start.ps1` と `start.sh` は、古い dev process を整理してから `npm run dev` を起動します。

## Repository の範囲

この repository には ArcSub のアプリケーションソースコードと公開ドキュメントが含まれます。

含まれないもの：

- `runtime/` 配下のローカル実行データ
- ダウンロード済みのローカル ASR / 翻訳モデル
- `.arcsub-bootstrap/` のような app-local bootstrap runtime
- `.env` のような個人用資格情報

## 主な機能

- ローカルメディアの読み込み、またはオンラインメディアの取得
- ローカルモデルまたはクラウドサービスによる音声文字起こし
- word alignment、VAD、diarization 関連の補助機能
- ローカルモデルまたはクラウドサービスによる字幕翻訳
- プレイヤー上でのレビューと書き出し

## ローカルモデル

ArcSub はローカルモデルをサポートしますが、導入は段階的です。

- deploy ではツールと baseline assets を準備
- ローカル ASR とローカル翻訳モデルは後から `Settings` で導入
- pyannote は Hugging Face token が必要だが、未設定でも起動自体は可能

## 追加ドキュメント

- ドキュメント一覧: [docs/README.md](./docs/README.md)
- リリース: [Releases](../../releases/latest)
- ディスカッション: [Discussions](../../discussions)
- コントリビューション: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 行動規範: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- セキュリティ: [SECURITY.md](./SECURITY.md)

## ライセンス

このプロジェクトは [MIT License](./LICENSE) の下で公開されています。
