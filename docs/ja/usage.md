# 使い方

## 基本の流れ

1. プロジェクトを作る
2. **Video Downloader** で動画や音声を用意する
3. **Speech to Text** を実行する
4. **Text Translation** を実行する
5. **Player** で確認して書き出す

## 文字起こしのコツ

- 言語が分かっているなら、自動検出より直接指定したほうが安定しやすいです
- pyannote 話者分離を使う場合は、先に **Settings** でインストールしてください

## 翻訳のコツ

- 先に文字起こし結果を軽く確認してください
- 固有名詞が多い場合は glossary を使うと安定します

## ローカルモデル

ローカル ASR やローカル翻訳を使いたい場合：

1. **Settings** を開く
2. **ASR モデル** または **翻訳モデル** を選ぶ
3. Hugging Face モデル id を入力して確認する
4. モデルをインストールする。大きなダウンロードや変換はバックグラウンドタスクとして継続されます
5. ワークフロー画面に戻って選択する

ArcSub は利用可能な信頼できる Hugging Face metadata を読み取り、runtime hints、chat template 対応、token-aware 翻訳バッチなどのモデル別ローカル既定値を導出します。

`HF_TOKEN` は pyannote と gated/private Hugging Face モデルのダウンロードで共有されます。公開モデルでは通常不要です。
