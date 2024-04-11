# RecurrentGemma

RecurrentGemma is a family of open-weights Language Models by [Google DeepMind](https://deepmind.google/), based on the novel [Griffin architecture](https://arxiv.org/abs/2402.19427). This architecture achieves fast inference when generating long sequences by replacing global attention with a mixture of local attention and linear recurrences.

This repository contains the model implementation and examples for sampling and fine-tuning. We recommend most users adopt the [Flax](https://github.com/google/flax) implementation, which is highly optimized. We also provide an un-optimized [PyTorch](https://github.com/pytorch/pytorch) implementation for reference.

### Learn more about RecurrentGemma

-   The [RecurrentGemma technical report](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf) gives specific details on the training and evaluation of RecurrentGemma.
-   The [Griffin paper](https://arxiv.org/abs/2402.19427) describes the underlying model architecture.

## Quick start

### Installation

#### Using Poetry
RecurrentGemma uses [Poetry](https://python-poetry.org/docs/) for dependency
management.

To install dependencies for the full project:
* Checkout the code.
* `poetry install -E full` to create a virtual environment with all dependencies.
* `poetry shell` to activate the created virtual environment.

If you only need to install a subset of dependencies use one of the alternative
library-specific commands below.

#### Using pip
If you want to use `pip` instead of Poetry, 
then create a virtual environment (run `python -m venv recurrentgemma-demo` and `. recurrentgemma-demo/bin/activate`) and:

* Checkout the code.
* `pip install .[full]`

#### Installing library-specific packages

##### JAX
To install dependencies only for the JAX pathway use:
`poetry install -E jax` or (`pip install .[jax]`).

##### PyTorch
To install dependencies only for the PyTorch pathway use:
`poetry install -E torch` (or `pip install .[torch]`).

##### Tests
To install dependencies required for running unit tests use:
`poetry install -E test` (or `pip install .[test]`)

### Downloading the models

The model checkpoints are available through Kaggle at
http://kaggle.com/models/google/recurrentgemma.
Select either the **Flax** or **PyTorch** model variations, click the ⤓ button
to download the model archive, then extract the contents to a local directory.

In both cases, the archive contains both the model weights and
the tokenizer.

### Running the unit tests

To run the tests, install the optional `[test]` dependencies (e.g. using `pip install .[test]`) from the root of the source tree, then:

```
pytest .
```

## Examples

To run the example sampling script, pass the paths to the weights directory and tokenizer:

```
python examples/sampling_jax.py \
  --path_checkpoint=/path/to/archive/contents/2b/ \
  --path_tokenizer=/path/to/archive/contents/tokenizer.model
```

### Colab notebook tutorials

-   [`colabs/sampling_tutorial_jax.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/sampling_tutorial_jax.ipynb)
    contains a [Colab](http://colab.google) notebook with a sampling example using JAX.

-   [`colabs/sampling_tutorial_pytorch.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/sampling_tutorial_pytorch.ipynb)
    contains a [Colab](http://colab.google) notebook with a sampling example using PyTorch.

-   [`colabs/fine_tuning_tutorial_jax.ipynb`](https://colab.sandbox.google.com/github/google-deepmind/recurrentgemma/blob/main/colabs/fine_tuning_tutorial_jax.ipynb)
    contains a [Colab](http://colab.google) with a basic tutorial on how to
    fine-tune RecurrentGemma for a task, such as English to French translation, using JAX.

To run these notebooks you will need to have a Kaggle account and first read and accept
the Gemma license terms and conditions from the [RecurrentGemma page](http://kaggle.com/models/google/recurrentgemma).
After this you can run the notebooks, which will automatically download the weights and tokenizer from there.

Currently different notebooks are supported under the following hardware:

| Hardware            | T4  | P100 | V100 | A100 | TPUv2 | TPUv3+ |
|---------------------|:---:|:----:|:----:|:----:|:-----:|:------:|
| Sampling in Jax     | ✅  | ✅   | ✅   | ✅   | ✅    | ✅    |
| Sampling in PyTorch | ✅  | ✅   | ✅   | ✅   | ✅    | ✅    |
| Finetuning in Jax   | ✅  | ✅   | ✅   | ✅   | ❌    | ✅    |


## System Requirements

RecurrentGemma code can run on CPU, GPU or TPU.
The code has been optimized for running on TPU using the Flax implementation,
which contains a low level [Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) kernel to perform the linear scan in the recurrent layers.

## Contributing

We are open to bug reports and issues. Please see
[CONTRIBUTING.md](CONTRIBUTING.md) for details on PRs.

## License

Copyright 2024 DeepMind Technologies Limited

This code is licensed under the Apache License, Version 2.0 (the \"License\");
you may not use this file except in compliance with the License. You may obtain
a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an AS IS BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## Disclaimer

This is not an official Google product.
