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
"""Tests for full Griffin backbone."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from recurrentgemma import common
from recurrentgemma.jax import griffin


class GriffinTest(absltest.TestCase):

  def test_griffin_output_shape(self):
    # given a model
    batch_size = 4
    seq_len = 8
    block_types = (
        common.TemporalBlockType.RECURRENT,
        common.TemporalBlockType.ATTENTION,
    )

    width = 16
    num_heads = 2
    heads_dim = width // 2

    config = common.GriffinConfig(
        vocab_size=16,
        width=width,
        mlp_expanded_width=3 * width,
        num_heads=num_heads,
        lru_width=2 * width,
        attention_window_size=4,
        block_types=block_types,
    )
    backbone = griffin.Griffin(config)
    key = jax.random.PRNGKey(0)
    # and input
    last_tokens = jnp.full((batch_size, seq_len), 3, dtype=jnp.int32)
    current_token_position = jnp.tile(jnp.arange(seq_len), (batch_size, 1))

    # when
    (logits, new_cache), _ = backbone.init_with_output(
        key,
        last_tokens,
        current_token_position,
    )

    # then
    self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size,),)
    for i, block_type in enumerate(block_types):
      block_cache = new_cache[f'blocks.{i}']

      match block_type:
        case common.TemporalBlockType.RECURRENT:
          assert isinstance(block_cache, griffin.modules.RecurrentBlockCache)
          self.assertEqual(
              block_cache.conv1d_state.shape,
              (batch_size, 3, config.lru_width),
          )
          self.assertEqual(
              block_cache.rg_lru_state.shape,
              (batch_size, config.lru_width)
          )

        case common.TemporalBlockType.ATTENTION:
          assert isinstance(block_cache, griffin.modules.AttentionBlockCache)
          self.assertEqual(
              block_cache.keys.shape,
              (batch_size, config.attention_window_size, 1, heads_dim),
          )
          self.assertEqual(
              block_cache.values.shape,
              (batch_size, config.attention_window_size, 1, heads_dim),
          )
          self.assertEqual(block_cache.num_tokens.shape, (batch_size,))


if __name__ == '__main__':
  absltest.main()
