# RecurrentGemma

RecurrentGemmaは、[Google DeepMind](https://deepmind.google/)による、新しい[Griffinアーキテクチャ](https://arxiv.org/abs/2402.19427)に基づいたオープンウェイトの言語モデルファミリーです。このアーキテクチャは、グローバルアテンションをローカルアテンションと線形再帰の混合に置き換えることで、長いシーケンスを生成する際の高速推論を実現しています。

このリポジトリには、モデルの実装とサンプリングおよびファインチューニングの例が含まれています。ほとんどのユーザーには、高度に最適化された[Flax](https://github.com/google/flax)実装の採用をお勧めします。また、参考のために、最適化されていない[PyTorch](https://github.com/pytorch/pytorch)実装も提供しています。

<a href="README_JA.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
<a href="../README.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>

### RecurrentGemmaの詳細

- [RecurrentGemma技術レポート](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf)では、RecurrentGemmaのトレーニングと評価に関する具体的な詳細を説明しています。
- [Griffin論文](https://arxiv.org/abs/2402.19427)では、基礎となるモデルアーキテクチャについて説明しています。

## クイックスタート

### インストール

#### Poetryを使用する場合
RecurrentGemmaは、依存関係の管理に[Poetry](https://python-poetry.org/docs/)を使用しています。

プロジェクト全体の依存関係をインストールするには:
* コードをチェックアウトします。
* `poetry install -E full`を実行して、すべての依存関係を含む仮想環境を作成します。
* `poetry shell`を実行して、作成した仮想環境をアクティブにします。

依存関係のサブセットのみをインストールする必要がある場合は、以下の代替のライブラリ固有のコマンドのいずれかを使用してください。

#### pipを使用する場合
Poetryの代わりに`pip`を使用する場合は、仮想環境を作成し（`python -m venv recurrentgemma-demo`と`. recurrentgemma-demo/bin/activate`を実行）、以下を実行します。

* コードをチェックアウトします。
* `pip install .[full]`を実行します。

#### ライブラリ固有のパッケージのインストール

##### JAX
JAXパスウェイの依存関係のみをインストールするには、`poetry install -E jax`（または`pip install .[jax]`）を使用します。

##### PyTorch
PyTorchパスウェイの依存関係のみをインストールするには、`poetry install -E torch`（または`pip install .[torch]`）を使用します。

##### テスト
単体テストの実行に必要な依存関係をインストールするには、`poetry install -E test`（または`pip install .[test]`）を使用します。

### モデルのダウンロード

モデルのチェックポイントは、Kaggleのhttp://kaggle.com/models/google/recurrentgemmaから入手できます。
**Flax**または**PyTorch**のモデルバリエーションを選択し、⤓ボタンをクリックしてモデルアーカイブをダウンロードし、内容をローカルディレクトリに展開します。

どちらの場合も、アーカイブにはモデルの重みとトークナイザの両方が含まれています。

### 単体テストの実行

テストを実行するには、ソースツリーのルートから`[test]`オプションの依存関係をインストールし（例えば、`pip install .[test]`を使用）、次のコマンドを実行します。

```
pytest .
```

## 例

サンプリングスクリプトの例を実行するには、重みディレクトリとトークナイザへのパスを渡します。

```
python examples/sampling_jax.py \
--path_checkpoint=/path/to/archive/contents/2b/ \
--path_tokenizer=/path/to/archive/contents/tokenizer.model
```

### Colabノートブックチュートリアル

- [`colabs/sampling_tutorial_jax.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/sampling_tutorial_jax.ipynb)には、JAXを使用したサンプリングの例を含む[Colab](http://colab.google)ノートブックが含まれています。

- [`colabs/sampling_tutorial_pytorch.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/sampling_tutorial_pytorch.ipynb)には、PyTorchを使用したサンプリングの例を含む[Colab](http://colab.google)ノートブックが含まれています。

- [`colabs/fine_tuning_tutorial_jax.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/fine_tuning_tutorial_jax.ipynb)には、JAXを使用して、英語からフランス語への翻訳などのタスクのためにRecurrentGemmaをファインチューニングする方法に関する基本的なチュートリアルを含む[Colab](http://colab.google)が含まれています。

これらのノートブックを実行するには、Kaggleアカウントが必要であり、最初に[RecurrentGemmaページ](http://kaggle.com/models/google/recurrentgemma)からGemmaのライセンス条項を読んで受け入れる必要があります。
その後、ノートブックを実行すると、そこから自動的に重みとトークナイザがダウンロードされます。

現在、異なるノートブックは以下のハードウェアでサポートされています。

| ハードウェア | T4 | P100 | V100 | A100 | TPUv2 | TPUv3+ |
|------------|:--:|:----:|:----:|:----:|:-----:|:------:|
| JaxでのSampling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyTorchでのSampling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Jaxでのファインチューニング | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |


## システム要件

RecurrentGemmaのコードは、CPU、GPU、またはTPUで実行できます。
このコードは、Flax実装を使用してTPUで実行するように最適化されており、再帰層の線形スキャンを実行する低レベルの[Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html)カーネルが含まれています。

## 貢献

バグ報告や問題提起を歓迎します。プルリクエストの詳細については、[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。

## ライセンス

Copyright 2024 DeepMind Technologies Limited

このコードは、Apache License, Version 2.0（以下「ライセンス」）に基づいてライセンスされています。
このファイルを除き、ライセンスに準拠しない限り、このコードを使用することはできません。ライセンスのコピーは、http://www.apache.org/licenses/LICENSE-2.0で入手できます。

ライセンスに基づいて書面で明示的に合意されない限り、ライセンスに基づいて配布されるソフトウェアは、明示的または黙示的を問わず、いかなる種類の保証や条件もなしに「現状のまま」で配布されます。ライセンスに基づいた許可と制限については、ライセンスを参照してください。

## 免責事項

これは公式のGoogleの製品ではありません。
