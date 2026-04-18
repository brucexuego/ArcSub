# FAQ

## OpenVINO がなくても使えますか？

はい。

サービスは起動できますが、ローカルモデルのインストールは使えません。

## pyannote が使えないのはなぜですか？

pyannote には次が必要です。

- 有効な `HF_TOKEN`
- Hugging Face の gated model 承認
- インストール完了

## データはどこに保存されますか？

通常はローカルの `runtime/` に保存されます。
