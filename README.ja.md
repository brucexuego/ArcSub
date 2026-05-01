# ArcSub

ArcSub は、クラウドサービスとローカル OpenVINO モデルをどちらも主要な実行経路として扱う、エンドツーエンドの字幕翻訳ワークステーションです。メディア取得、音声文字起こし、字幕翻訳、動画と字幕を組み合わせた視聴までを扱えます。

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

![ArcSub video fetcher workflow](./docs/assets/screenshots/video-fetcher-overview.png)

動画取得: ソースのメタデータ解析、ダウンロード形式の選択、音声認識用素材の準備を行います。

![ArcSub speech to text workflow](./docs/assets/screenshots/speech-to-text-overview.png)

音声文字変換: クラウドまたはローカルの認識モデルを選び、詳細オプションを設定して文字起こし結果を生成します。

![ArcSub text translation workflow](./docs/assets/screenshots/text-translation-overview.png)

高精度翻訳: クラウドまたはローカルの翻訳モデルを選び、言語設定を調整して原文と翻訳字幕を並べて確認できます。

![ArcSub video player workflow](./docs/assets/screenshots/video-player-overview.png)

動画プレーヤー: 完成した字幕を動画と一緒に視聴し、ページ内の字幕スタイルを微調整できます。

## クイックガイド

### ダウンロード

この repository で配布用アセットが公開されている場合は、まず [Releases](../../releases/latest) から利用するOS向けの最新ファイルを取得してください。

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
- クラウド ASR サービスまたはローカル OpenVINO ASR モデルによる音声文字起こし
- word alignment、VAD、diarization 関連の補助機能
- クラウド翻訳サービスまたはローカル OpenVINO 翻訳モデルによる字幕翻訳
- 動画と字幕を組み合わせて視聴し、ページ内の字幕スタイルを調整

## クラウドモデルとローカルモデル

ArcSub は、プロジェクトごとにクラウドサービスとローカル OpenVINO runtime のどちらが適しているかを選べる設計です。

- クラウド ASR / 翻訳モデルは `設定` で API endpoint、key、プロバイダー設定を登録
- ローカル ASR / 翻訳モデルは `設定` からインストールし、OpenVINO runtime 経路で実行
- `設定` 上のモデル順序が音声文字変換と高精度翻訳の既定モデルになります
- pyannote 話者分離を有効にする場合は Hugging Face アセットを使いますが、token 未設定でも通常の起動やクラウド経路は利用できます

## 追加ドキュメント

- ドキュメント一覧: [docs/README.md](./docs/README.md)
- リリース: [Releases](../../releases/latest)
- ディスカッション: [Discussions](../../discussions)
- コントリビューション: [CONTRIBUTING.md](./CONTRIBUTING.md)
- 行動規範: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- セキュリティ: [SECURITY.md](./SECURITY.md)

## ライセンス

このプロジェクトは [MIT License](./LICENSE) の下で公開されています。
