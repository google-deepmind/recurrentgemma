# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Minimal test for sampler."""

from collections.abc import Iterable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from recurrentgemma import common
import recurrentgemma.jax as griffin_lib


class MockVocab:

  def __init__(self):
    self._start_id = 3
    self._mapping_text_to_id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        'input': 3,
        'string': 4,
        'hello': 5,
        'world': 6,
        'Hello': 7,
        '!': 8,
        'How': 9,
        'are': 10,
        'you?': 11,
    }
    self._vocab_size = len(self._mapping_text_to_id)
    self._separator = ' '

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return self._separator.join(reverse_mapping[token] for token in ids)

  def EncodeAsIds(self, text: str) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(self._separator)
    return [self._mapping_text_to_id[word] for word in words]


class SamplerTest(parameterized.TestCase):

  def test_samples(self):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
    )
    model = griffin_lib.Griffin(model_config)

    params = model.init(
        jax.random.PRNGKey(0),
        jnp.array([[1]]),
        jnp.array([[1]]),
    )
    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
        params=params['params'],
    )

    result = sampler(['input string', 'hello world'], total_generation_steps=10)
    self.assertIsNotNone(result)

  @parameterized.product(echo=[True, False], return_logits=[True, False])
  def test_output_shapes(self, echo: bool, return_logits: bool):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
    )
    model = griffin_lib.Griffin(model_config)

    raw_input = 'Hello ! How are you?'
    token_input = jnp.asarray(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))

    batch_size, n_input_tokens = token_input.shape
    params = model.init(
        jax.random.PRNGKey(42),
        jnp.array([[1]]),
        jnp.array([[1]]),
    )

    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
        params=params['params'],
    )

    total_generation_steps = 10
    output_sampler = sampler(
        [raw_input],
        total_generation_steps=total_generation_steps,
        echo=echo,
        return_logits=return_logits,
    )
    total_tokens = total_generation_steps
    if echo:
      total_tokens += n_input_tokens

    if not return_logits:
      self.assertEmpty(output_sampler.logits)
    else:
      self.assertEqual(
          jnp.asarray(output_sampler.logits).shape,
          (batch_size, total_tokens, vocab.GetPieceSize()),
      )

    self.assertEqual(
        jnp.asarray(output_sampler.tokens).shape,
        (batch_size, total_tokens),
    )

  @parameterized.parameters(['bfloat16', 'float32'])
  def test_forward_equivalence(self, dtype: str):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
    )

    model = griffin_lib.Griffin(model_config)
    raw_input = 'Hello ! How are you?'
    token_input = jnp.asarray(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))

    batch_size, n_input_tokens = token_input.shape

    params = model.init(
        jax.random.PRNGKey(42),
        jnp.array([[1]]),
        jnp.array([[1]]),
    )

    params = jax.tree_util.tree_map(
        lambda x: x.astype(dtype),
        params,
    )

    segment_pos = jnp.repeat(
        jnp.arange(n_input_tokens)[None],
        batch_size,
        axis=0,
    )
    output_forward, _ = model.apply(
        params,
        tokens=token_input,
        segment_pos=segment_pos,
    )
    output_forward = output_forward[0, :n_input_tokens]
    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
        params=params['params'],
        deterministic_sampling=False,
    )

    total_generation_steps = 10
    output_sampler = sampler(
        [raw_input],
        rng=jax.random.PRNGKey(231321),
        total_generation_steps=total_generation_steps,
        echo=True,
        return_logits=True,
    )
    total_sampled_tokens = total_generation_steps + token_input.shape[-1]
    chex.assert_shape(
        jnp.array(output_sampler.logits),
        (batch_size, total_sampled_tokens, model_config.vocab_size),
    )
    chex.assert_shape(
        jnp.array(output_sampler.tokens),
        (batch_size, total_sampled_tokens),
    )

    out_logits = jnp.array(output_sampler.logits, dtype=jnp.dtype(dtype))
    out_logits = out_logits[0, :n_input_tokens]

    if dtype == 'bfloat16':
      rtol = 1e-1
      atol = 1e-1
    else:
      rtol = 1e-6
      atol = 1e-6

    chex.assert_trees_all_close(
        output_forward,
        out_logits,
        rtol=rtol,
        atol=atol,
    )


if __name__ == '__main__':
  absltest.main()
